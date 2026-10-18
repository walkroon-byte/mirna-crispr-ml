#!/usr/bin/env python3
"""Train a leakage-aware, grouped-split successor to v7_public_only."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data/processed_public_datasets/combined_v4_training_data_cleaned_exclude_unverified80.csv"
OUTPUT_DIR = PROJECT_ROOT / "models/v7_public_only_optimized"
RANDOM_STATE = 42
EXPECTED_ROWS = 9894
BASES = "ACGU"


def encode_row(spacer, target):
    """Mirror v7's first-20-nt encoding and add an explicit mismatch mask."""
    spacer = str(spacer).upper().replace("T", "U").replace(" ", "")[:20]
    target = str(target).upper().replace("T", "U").replace(" ", "")[:20]
    if len(spacer) != 20 or len(target) != 20:
        raise ValueError(f"Expected at least 20 nt per sequence, got {len(spacer)} and {len(target)}")
    if set(spacer + target) - set(BASES):
        raise ValueError("Sequence contains bases outside A/C/G/U/T")

    encoded = np.zeros((2, 20, 4), dtype=np.float32)
    for strand, sequence in enumerate((spacer, target)):
        for position, base in enumerate(sequence):
            encoded[strand, position, BASES.index(base)] = 1.0
    mismatch_mask = np.fromiter((a != b for a, b in zip(spacer, target)), dtype=np.float32, count=20)
    # Make mismatch identity and simple composition explicit so tree models do
    # not have to infer pairwise interactions from two independent one-hot blocks.
    pair_features = np.zeros((20, 16), dtype=np.float32)
    for position, (spacer_base, target_base) in enumerate(zip(spacer, target)):
        pair_features[position, BASES.index(spacer_base) * 4 + BASES.index(target_base)] = 1.0
    summary = np.array([
        spacer.count("G") + spacer.count("C"),
        target.count("G") + target.count("C"),
        mismatch_mask.sum(),
        np.flatnonzero(mismatch_mask)[0] if mismatch_mask.any() else -1,
        np.flatnonzero(mismatch_mask)[-1] if mismatch_mask.any() else -1,
    ], dtype=np.float32)
    return np.concatenate((encoded.reshape(-1), mismatch_mask, pair_features.reshape(-1), summary)), spacer


def metrics(y_true, y_pred):
    pearson = pearsonr(y_true, y_pred) if np.std(y_pred) > 0 else (float("nan"), float("nan"))
    spearman = spearmanr(y_true, y_pred) if np.std(y_pred) > 0 else (float("nan"), float("nan"))
    return {
        "n": int(len(y_true)),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def split_indices(groups):
    first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(first.split(np.zeros(len(groups)), groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.1764705882, random_state=RANDOM_STATE)
    train_rel, val_rel = next(second.split(
        np.zeros(len(train_val_idx)), groups=groups[train_val_idx]
    ))
    return train_val_idx[train_rel], train_val_idx[val_rel], test_idx


def candidate_models():
    return {
        "extra_trees_leaf2": ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=2, max_features=0.8,
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "extra_trees_leaf5": ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=5, max_features=0.8,
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "random_forest_leaf2": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, max_features=0.8,
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "hist_gradient_300": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=RANDOM_STATE,
        ),
        "hist_gradient_600": HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.04, max_leaf_nodes=15,
            l2_regularization=2.0, random_state=RANDOM_STATE,
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_FILE)
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} cleaned public rows, found {len(df)}")
    required = {"spacer_seq", "target_seq", "relative_activity"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(df.columns))}")

    encoded = [encode_row(s, t) for s, t in zip(df.spacer_seq, df.target_seq)]
    X = np.stack([item[0] for item in encoded])
    canonical_spacers = np.array([item[1] for item in encoded])
    y = df.relative_activity.to_numpy(dtype=np.float64)
    groups = canonical_spacers
    train_idx, val_idx, test_idx = split_indices(groups)

    split_sets = [set(groups[idx]) for idx in (train_idx, val_idx, test_idx)]
    if any(split_sets[a] & split_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Spacer leakage detected across grouped splits")

    print(f"Rows: {len(df)}; unique spacer groups: {len(np.unique(groups))}")
    print(f"Split rows: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")
    print("Fitting candidate regressors on spacer-disjoint training data...")

    validation_results = {}
    fitted = {}
    for name, estimator in candidate_models().items():
        estimator.fit(X[train_idx], y[train_idx])
        validation_results[name] = metrics(y[val_idx], estimator.predict(X[val_idx]))
        fitted[name] = estimator
        print(f"{name}: validation Spearman={validation_results[name]['spearman_r']:.4f}, "
              f"RMSE={validation_results[name]['rmse']:.4f}")

    selected_name = max(validation_results, key=lambda key: validation_results[key]["spearman_r"])
    selected = candidate_models()[selected_name]
    trainval_idx = np.concatenate((train_idx, val_idx))
    selected.fit(X[trainval_idx], y[trainval_idx])
    test_prediction = selected.predict(X[test_idx])
    test_metrics = metrics(y[test_idx], test_prediction)

    joblib.dump(selected, OUTPUT_DIR / "v7_public_only_optimized_model.joblib")
    pd.DataFrame({
        "spacer_seq": canonical_spacers[test_idx],
        "actual": y[test_idx],
        "predicted": test_prediction,
        "residual": y[test_idx] - test_prediction,
    }).to_csv(OUTPUT_DIR / "v7_public_only_optimized_test_predictions.csv", index=False)

    split_frame = pd.DataFrame({
        "row_index": np.arange(len(df)),
        "spacer_seq": canonical_spacers,
        "split": "",
    })
    split_frame.loc[train_idx, "split"] = "train"
    split_frame.loc[val_idx, "split"] = "validation"
    split_frame.loc[test_idx, "split"] = "test"
    split_frame.to_csv(OUTPUT_DIR / "v7_public_only_optimized_split.csv", index=False)

    result = {
        "model_version": "v7_public_only_optimized",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_file": str(DATA_FILE.relative_to(PROJECT_ROOT)),
        "n_rows": int(len(df)),
        "n_spacer_groups": int(len(np.unique(groups))),
        "n_features": int(X.shape[1]),
        "split_method": "Nested GroupShuffleSplit by canonical first-20-nt spacer; seed 42",
        "split_rows": {"train": int(len(train_idx)), "validation": int(len(val_idx)), "test": int(len(test_idx))},
        "split_groups": {name: int(len(np.unique(groups[idx]))) for name, idx in
                          (("train", train_idx), ("validation", val_idx), ("test", test_idx))},
        "candidate_validation_metrics": validation_results,
        "selected_model": selected_name,
        "test_metrics": test_metrics,
        "feature_definition": "v7-compatible first 20 nt spacer/target one-hot (160) + positional mismatch mask (20) + mismatch-pair one-hot (320) + composition summary (5)",
        "limitations": [
            "Performance estimates generalization to unseen spacer sequences in this dataset; external validation remains necessary.",
            "Dataset cleaning excludes the identified 80 rows but does not independently audit all remaining labels or source definitions.",
            "The test set was used once after model selection on validation data.",
        ],
    }
    with (OUTPUT_DIR / "v7_public_only_optimized_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"Selected: {selected_name}")
    print(f"Grouped test: Spearman={test_metrics['spearman_r']:.4f}, "
          f"R2={test_metrics['r2']:.4f}, RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}")
    print(f"Saved artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
