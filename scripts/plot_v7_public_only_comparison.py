"""Publication-style comparison plots for the public-only v7 models.

The script consumes saved metrics and test predictions. It never retrains a
model, so the figures remain tied to the exact artifacts used for evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = PROJECT_ROOT / "results" / "figures" / "v7_public_only_comparison"

MODELS = {
    "v7_public_only": {
        "label": "v7 public-only",
        "metrics": PROJECT_ROOT / "models/v7_public_only/v7_public_only_model_metrics.json",
        "predictions": PROJECT_ROOT / "models/v7_public_only/v7_test_predictions.csv",
        "color": "#4C72B0",
    },
    "deep_optimized": {
        "label": "deep optimized",
        "metrics": PROJECT_ROOT / "models/v7_public_only_deep_optimized/v7_public_only_deep_optimized_metrics.json",
        "predictions": PROJECT_ROOT / "models/v7_public_only_deep_optimized/v7_public_only_deep_optimized_test_predictions.csv",
        "color": "#DD8452",
    },
    "deep_optimized_v2": {
        "label": "deep optimized v2",
        "metrics": PROJECT_ROOT / "models/v7_public_only_deep_optimized_v2/metrics.json",
        "predictions": PROJECT_ROOT / "models/v7_public_only_deep_optimized_v2/test_predictions.csv",
        "color": "#55A868",
    },
}


def load_model_results() -> dict[str, dict]:
    results = {}
    for key, spec in MODELS.items():
        with spec["metrics"].open(encoding="utf-8") as handle:
            payload = json.load(handle)
        raw = pd.read_csv(spec["predictions"])
        columns = {name.lower(): name for name in raw.columns}
        actual_col = columns.get("actual")
        predicted_col = columns.get("predicted")
        if actual_col is None or predicted_col is None:
            raise ValueError(f"{spec['predictions']} must contain actual and predicted columns")
        actual = raw[actual_col].astype(float).to_numpy()
        predicted = raw[predicted_col].astype(float).to_numpy()
        residual = actual - predicted
        metric_block = payload.get("test_metrics", payload.get("metrics", {}).get("test"))
        if metric_block is None:
            raise ValueError(f"No test metrics found in {spec['metrics']}")
        results[key] = {
            **spec,
            "actual": actual,
            "predicted": predicted,
            "residual": residual,
            "absolute_error": np.abs(residual),
            "metrics": metric_block,
        }
    return results


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


def shared_limits(results: dict[str, dict]) -> tuple[float, float]:
    values = np.concatenate([
        np.concatenate([item["actual"], item["predicted"]]) for item in results.values()
    ])
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    pad = 0.04 * (hi - lo)
    return lo - pad, hi + pad


def parity_and_residual_figure(results: dict[str, dict]) -> None:
    lo, hi = shared_limits(results)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), sharex="row")
    fig.suptitle("v7 public-only models: test-set prediction diagnostics", y=0.99,
                 fontsize=16, fontweight="bold")

    for col, (key, item) in enumerate(results.items()):
        metrics = item["metrics"]
        ax = axes[0, col]
        ax.scatter(item["actual"], item["predicted"], s=18, alpha=0.56,
                   color=item["color"], edgecolors="#222222", linewidths=0.25)
        ax.plot([lo, hi], [lo, hi], color="#B23A48", linestyle="--", linewidth=1.3,
                label="identity")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(item["label"], fontsize=13, fontweight="bold")
        ax.set_xlabel("Actual relative activity")
        ax.set_ylabel("Predicted relative activity")
        ax.text(0.04, 0.96,
                f"n = {int(metrics.get('n', len(item['actual']))):,}\n"
                f"R² = {metrics['r2']:.3f}\n"
                f"ρ = {metrics['spearman_r']:.3f}\n"
                f"RMSE = {metrics['rmse']:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
                bbox={"facecolor": "white", "edgecolor": "#BDBDBD", "alpha": 0.88,
                      "boxstyle": "round,pad=0.35"})
        ax.legend(loc="lower right", frameon=True, fontsize=8)

        ax = axes[1, col]
        ax.scatter(item["predicted"], item["residual"], s=18, alpha=0.56,
                   color=item["color"], edgecolors="#222222", linewidths=0.25)
        ax.axhline(0, color="#B23A48", linestyle="--", linewidth=1.3)
        ax.set_xlim(lo, hi)
        rmax = max(np.max(np.abs(item["residual"])), 0.1) * 1.08
        ax.set_ylim(-rmax, rmax)
        ax.set_xlabel("Predicted relative activity")
        ax.set_ylabel("Residual (actual − predicted)")
        ax.text(0.04, 0.96,
                f"MAE = {metrics['mae']:.3f}\n"
                f"median error = {np.median(item['residual']):+.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
                bbox={"facecolor": "white", "edgecolor": "#BDBDBD", "alpha": 0.88,
                      "boxstyle": "round,pad=0.35"})

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURE_DIR / "v7_public_only_prediction_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def error_distribution_figure(results: dict[str, dict]) -> None:
    long_rows = []
    for key, item in results.items():
        long_rows.extend({
            "model": item["label"],
            "residual": float(value),
            "absolute_error": float(abs(value)),
        } for value in item["residual"])
    frame = pd.DataFrame(long_rows)
    order = [item["label"] for item in results.values()]
    palette = {item["label"]: item["color"] for item in results.values()}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    fig.suptitle("v7 public-only models: test-set error distributions", y=1.02,
                 fontsize=16, fontweight="bold")

    sns.violinplot(data=frame, x="model", y="residual", hue="model", order=order,
                   hue_order=order, palette=palette, legend=False, inner="quartile",
                   cut=0, linewidth=0.8, ax=axes[0])
    axes[0].axhline(0, color="#B23A48", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Residual (actual − predicted)")
    axes[0].tick_params(axis="x", rotation=12)

    sns.boxplot(data=frame, x="model", y="absolute_error", hue="model", order=order,
                hue_order=order, palette=palette, legend=False, width=0.48, fliersize=2.2,
                linewidth=0.9, ax=axes[1])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Absolute error")
    axes[1].tick_params(axis="x", rotation=12)
    for index, item in enumerate(results.values()):
        axes[1].text(index, np.quantile(item["absolute_error"], 0.75),
                     f"MAE {item['metrics']['mae']:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "v7_public_only_error_distributions.png", bbox_inches="tight")
    plt.close(fig)


def metrics_figure(results: dict[str, dict]) -> None:
    labels = [item["label"] for item in results.values()]
    metrics = ["rmse", "mae", "r2", "pearson_r", "spearman_r"]
    titles = ["RMSE ↓", "MAE ↓", "R² ↑", "Pearson r ↑", "Spearman ρ ↑"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.9))
    fig.suptitle("v7 public-only models: held-out test metrics", y=1.03,
                 fontsize=16, fontweight="bold")
    colors = [item["color"] for item in results.values()]
    for ax, metric, title in zip(axes, metrics, titles):
        values = [item["metrics"][metric] for item in results.values()]
        bars = ax.bar(labels, values, color=colors, edgecolor="#222222", linewidth=0.6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=38)
        ax.set_ylim(0, max(values) * 1.24 if max(values) > 0 else 1)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.025,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "v7_public_only_test_metrics.png", bbox_inches="tight")
    plt.close(fig)


def save_summary(results: dict[str, dict]) -> None:
    rows = []
    for key, item in results.items():
        metrics = item["metrics"]
        rows.append({
            "model": key,
            "label": item["label"],
            "n": int(metrics.get("n", len(item["actual"]))),
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "pearson_r": metrics["pearson_r"],
            "spearman_r": metrics["spearman_r"],
            "median_residual": float(np.median(item["residual"])),
            "median_absolute_error": float(np.median(item["absolute_error"])),
        })
    pd.DataFrame(rows).to_csv(FIGURE_DIR / "v7_public_only_test_metrics_summary.csv", index=False)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    results = load_model_results()
    parity_and_residual_figure(results)
    error_distribution_figure(results)
    metrics_figure(results)
    save_summary(results)
    print(f"Saved plots and summary to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
