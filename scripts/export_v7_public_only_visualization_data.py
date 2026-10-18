"""Produce a versioned data package for v7_public_only publication figures.

This script is the only stage that reads model artifacts. The plotting stage
consumes only the files written under results/visualization_data/.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "v7_public_only"
DATA_DIR = ROOT / "results" / "visualization_data" / "v7_public_only"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sources() -> tuple[dict, pd.DataFrame]:
    metrics_path = MODEL_DIR / "v7_public_only_model_metrics.json"
    predictions_path = MODEL_DIR / "v7_test_predictions.csv"
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    predictions = pd.read_csv(predictions_path)
    required = {"actual", "predicted", "residual"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    numeric = predictions[list(required)].apply(pd.to_numeric, errors="raise")
    if not np.allclose(numeric["residual"], numeric["actual"] - numeric["predicted"], atol=1e-6):
        raise ValueError("Prediction residuals do not equal actual - predicted")
    predictions = predictions.copy()
    predictions["absolute_error"] = predictions["residual"].abs()
    predictions["actual_percentile"] = stats.rankdata(predictions["actual"], method="average") / len(predictions)
    predictions["predicted_percentile"] = stats.rankdata(predictions["predicted"], method="average") / len(predictions)
    predictions["residual_rank"] = stats.rankdata(predictions["residual"], method="average")
    return metrics, predictions


def export_split_metrics(metrics: dict, n_test: int) -> None:
    rows = []
    for split, key in (("train", "train_metrics"), ("validation", "val_metrics"), ("test", "test_metrics")):
        block = metrics[key]
        rows.append({
            "split": split,
            "n": n_test if split == "test" else np.nan,
            "r2": block["r2"],
            "rmse": block["rmse"],
            "mae": block["mae"],
            "pearson_r": block["pearson_r"],
            "pearson_p": block["pearson_p"],
            "spearman_r": block["spearman_r"],
            "spearman_p": block["spearman_p"],
        })
    pd.DataFrame(rows).to_csv(DATA_DIR / "split_metrics.csv", index=False)


def export_calibration(predictions: pd.DataFrame) -> None:
    work = predictions.assign(predicted_decile=pd.qcut(predictions["predicted"], q=10, duplicates="drop"))
    grouped = work.groupby("predicted_decile", observed=True).agg(
        predicted_mean=("predicted", "mean"),
        actual_mean=("actual", "mean"),
        actual_sd=("actual", "std"),
        n=("actual", "size"),
        residual_mean=("residual", "mean"),
        absolute_error_mean=("absolute_error", "mean"),
    ).reset_index()
    grouped["sem"] = grouped["actual_sd"] / np.sqrt(grouped["n"].clip(lower=1))
    grouped["ci95_low"] = grouped["actual_mean"] - 1.96 * grouped["sem"]
    grouped["ci95_high"] = grouped["actual_mean"] + 1.96 * grouped["sem"]
    grouped.insert(0, "decile", np.arange(1, len(grouped) + 1))
    grouped.to_csv(DATA_DIR / "calibration_bins.csv", index=False)


def export_ecdf(predictions: pd.DataFrame) -> None:
    absolute_error = np.sort(predictions["absolute_error"].to_numpy())
    ecdf = pd.DataFrame({
        "absolute_error": absolute_error,
        "cumulative_proportion": np.arange(1, len(absolute_error) + 1) / len(absolute_error),
    })
    ecdf.to_csv(DATA_DIR / "absolute_error_ecdf.csv", index=False)


def export_qq(predictions: pd.DataFrame) -> None:
    theoretical, ordered = stats.probplot(predictions["residual"], dist="norm", fit=False)
    pd.DataFrame({
        "theoretical_quantile": theoretical,
        "ordered_residual": ordered,
    }).to_csv(DATA_DIR / "residual_qq.csv", index=False)


def export_training_history() -> bool:
    source = MODEL_DIR / "v7_public_only_training_history.json"
    if not source.exists():
        return False
    history = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(history, dict) or not history:
        raise ValueError(f"Training history is empty or invalid: {source}")
    lengths = {len(values) for values in history.values() if isinstance(values, list)}
    if len(lengths) != 1:
        raise ValueError(f"Training history series have inconsistent lengths: {source}")
    (DATA_DIR / "training_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return True


def export_manifest(metrics: dict, predictions: pd.DataFrame) -> None:
    source_files = {
        "metrics_json": MODEL_DIR / "v7_public_only_model_metrics.json",
        "test_predictions_csv": MODEL_DIR / "v7_test_predictions.csv",
    }
    history_json = MODEL_DIR / "v7_public_only_training_history.json"
    has_history = (DATA_DIR / "training_history.json").exists()
    manifest = {
        "schema_version": "v7_public_only_visualization_data_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": metrics.get("model_version", "v7_public_only"),
        "data_stage": "production_artifact_export",
        "plot_stage_input_only": True,
        "n_test_predictions": int(len(predictions)),
        "source_artifacts": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in source_files.items()
        },
        "derived_files": [
            "test_predictions.csv", "split_metrics.csv", "calibration_bins.csv",
            "absolute_error_ecdf.csv", "residual_qq.csv",
        ] + (["training_history.json"] if has_history else []),
        "training_history": {
            "available": has_history,
            "path": "training_history.json" if has_history else None,
            "note": "Numeric training history was not present in the saved v7 artifact."
            if not history_json.exists() else "Exported by the training production script.",
        },
        "processing_order": [
            "read_saved_model_artifacts",
            "validate_prediction_columns_and_residuals",
            "derive_rank_error_and_calibration_tables",
            "write_versioned_visualization_data",
            "plot_from_visualization_data_only",
        ],
    }
    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metrics, predictions = load_sources()
    predictions.to_csv(DATA_DIR / "test_predictions.csv", index=False)
    export_split_metrics(metrics, len(predictions))
    export_calibration(predictions)
    export_ecdf(predictions)
    export_qq(predictions)
    export_training_history()
    export_manifest(metrics, predictions)
    print(f"Saved v7_public_only visualization data to {DATA_DIR}")


if __name__ == "__main__":
    main()
