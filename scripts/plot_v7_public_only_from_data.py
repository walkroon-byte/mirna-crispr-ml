"""Render v7_public_only figures from a standalone visualization data package.

No model files, training code, or training data are read here. Edit this file
when publication layout, typography, colors, or annotations need tuning.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "results" / "visualization_data" / "v7_public_only"
OUT = ROOT / "results" / "figures" / "v7_public_only_from_data"
BLUE = "#4C72B0"
RED = "#C44E52"
GREEN = "#55A868"
GRAY = "#555555"


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.12)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8, "grid.color": "#D9D9D9", "grid.linewidth": 0.6,
        "figure.dpi": 120, "savefig.dpi": 300,
    })


def load_data() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(DATA_DIR / "test_predictions.csv")
    split_metrics = pd.read_csv(DATA_DIR / "split_metrics.csv")
    calibration = pd.read_csv(DATA_DIR / "calibration_bins.csv")
    ecdf = pd.read_csv(DATA_DIR / "absolute_error_ecdf.csv")
    qq = pd.read_csv(DATA_DIR / "residual_qq.csv")
    return manifest, predictions, split_metrics, calibration, ecdf, qq


def training_history_figure() -> None:
    history_path = DATA_DIR / "training_history.json"
    if not history_path.exists():
        return
    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].plot(epochs, history["loss"], color=BLUE, label="Training")
    axes[0].plot(epochs, history["val_loss"], color=RED, label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Loss (MSE)", title="Training history: loss")
    axes[0].legend(frameon=True)
    axes[1].plot(epochs, history["mae"], color=BLUE, label="Training")
    axes[1].plot(epochs, history["val_mae"], color=RED, label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="MAE", title="Training history: MAE")
    axes[1].legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "v7_public_only_training_history.png", bbox_inches="tight")
    plt.close(fig)


def diagnostic_figure(pred: pd.DataFrame, split_metrics: pd.DataFrame) -> None:
    test = split_metrics.loc[split_metrics["split"] == "test"].iloc[0]
    actual, predicted, residual = pred["actual"], pred["predicted"], pred["residual"]
    lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    pad = (hi - lo) * 0.04
    lo, hi = lo - pad, hi + pad
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 10.5))
    fig.suptitle("v7 public-only: held-out test diagnostics", fontsize=16, fontweight="bold", y=0.99)
    ax[0, 0].scatter(actual, predicted, s=20, alpha=0.58, color=BLUE, edgecolors="#222222", linewidths=0.25)
    ax[0, 0].plot([lo, hi], [lo, hi], "--", color=RED, linewidth=1.3)
    ax[0, 0].set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Actual relative activity", ylabel="Predicted relative activity", title="Parity plot")
    ax[0, 0].text(0.04, 0.96, f"n = {len(pred):,}\nR² = {test['r2']:.3f}\nPearson r = {test['pearson_r']:.3f}\nSpearman ρ = {test['spearman_r']:.3f}", transform=ax[0, 0].transAxes, va="top", fontsize=9.5, bbox={"facecolor": "white", "edgecolor": "#BDBDBD", "alpha": 0.9, "boxstyle": "round,pad=0.35"})
    ax[0, 1].scatter(predicted, residual, s=20, alpha=0.58, color=BLUE, edgecolors="#222222", linewidths=0.25)
    ax[0, 1].axhline(0, linestyle="--", color=RED, linewidth=1.3)
    ax[0, 1].set(xlabel="Predicted relative activity", ylabel="Residual (actual − predicted)", title="Residuals versus predictions")
    ax[0, 1].text(0.04, 0.96, f"RMSE = {test['rmse']:.3f}\nMAE = {test['mae']:.3f}\nmedian = {residual.median():+.3f}", transform=ax[0, 1].transAxes, va="top", fontsize=9.5, bbox={"facecolor": "white", "edgecolor": "#BDBDBD", "alpha": 0.9, "boxstyle": "round,pad=0.35"})
    sns.histplot(residual, bins=28, kde=True, color=BLUE, edgecolor="white", ax=ax[1, 0])
    ax[1, 0].axvline(0, linestyle="--", color=RED, linewidth=1.3)
    ax[1, 0].set(xlabel="Residual (actual − predicted)", ylabel="Count", title="Residual distribution")
    sns.kdeplot(actual, fill=True, color=GRAY, alpha=0.24, label="Actual", ax=ax[1, 1])
    sns.kdeplot(predicted, fill=True, color=GREEN, alpha=0.32, label="Predicted", ax=ax[1, 1])
    ax[1, 1].set(xlabel="Relative activity", ylabel="Density", title="Actual and predicted distributions")
    ax[1, 1].legend(frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "v7_public_only_test_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def error_figure(pred: pd.DataFrame, ecdf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.8))
    fig.suptitle("v7 public-only: test error structure", fontsize=16, fontweight="bold", y=1.02)
    sns.violinplot(y=pred["residual"], inner="quartile", color=BLUE, cut=0, linewidth=0.8, ax=ax[0])
    ax[0].axhline(0, linestyle="--", color=RED, linewidth=1.2)
    ax[0].set(xlabel="", ylabel="Residual (actual − predicted)", title="Residual violin")
    ax[1].plot(ecdf["absolute_error"], ecdf["cumulative_proportion"], color=BLUE, linewidth=2)
    ax[1].set(xlabel="Absolute error", ylabel="Cumulative proportion", title="Absolute-error ECDF", ylim=(0, 1.02))
    sns.boxplot(y=pred["absolute_error"], color=GREEN, width=0.42, fliersize=2.2, ax=ax[2])
    ax[2].set(xlabel="", ylabel="Absolute error", title="Absolute-error boxplot")
    ax[2].text(0, pred["absolute_error"].quantile(0.75), f"MAE = {pred['absolute_error'].mean():.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "v7_public_only_error_structure.png", bbox_inches="tight")
    plt.close(fig)


def calibration_figure(calibration: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    x, y = calibration["predicted_mean"], calibration["actual_mean"]
    ax.errorbar(x, y, yerr=1.96 * calibration["sem"], fmt="o-", color=BLUE, ecolor=BLUE, elinewidth=1, capsize=3, markersize=6, label="Observed mean ± 95% CI")
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = (hi - lo) * 0.08
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color=RED, linewidth=1.3, label="Perfect calibration")
    for _, row in calibration.iterrows():
        ax.annotate(f"n={int(row['n'])}", (row["predicted_mean"], row["actual_mean"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad), xlabel="Mean predicted relative activity", ylabel="Mean actual relative activity", title="v7 public-only: decile calibration")
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "v7_public_only_calibration.png", bbox_inches="tight")
    plt.close(fig)


def rank_qq_figure(pred: pd.DataFrame, qq: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.1))
    fig.suptitle("v7 public-only: rank agreement and residual normality", fontsize=16, fontweight="bold", y=1.02)
    ax[0].scatter(pred["actual_percentile"], pred["predicted_percentile"], s=15, alpha=0.48, color=BLUE, edgecolors="none")
    ax[0].plot([0, 1], [0, 1], "--", color=RED, linewidth=1.3)
    ax[0].set(xlabel="Actual percentile rank", ylabel="Predicted percentile rank", title="Percentile-rank agreement")
    ax[1].scatter(qq["theoretical_quantile"], qq["ordered_residual"], s=15, alpha=0.48, color=BLUE, edgecolors="none")
    slope, intercept = np.polyfit(qq["theoretical_quantile"], qq["ordered_residual"], 1)
    x = np.array([qq["theoretical_quantile"].min(), qq["theoretical_quantile"].max()])
    ax[1].plot(x, intercept + slope * x, color=RED, linewidth=1.3)
    ax[1].set(xlabel="Theoretical quantiles", ylabel="Ordered residuals", title="Residual Q–Q plot")
    fig.tight_layout()
    fig.savefig(OUT / "v7_public_only_rank_qq.png", bbox_inches="tight")
    plt.close(fig)


def metrics_figure(split_metrics: pd.DataFrame) -> None:
    specs = [("r2", "R²"), ("rmse", "RMSE"), ("mae", "MAE"), ("pearson_r", "Pearson r"), ("spearman_r", "Spearman ρ")]
    names = {"train": "Train", "validation": "Validation", "test": "Test"}
    fig, axes = plt.subplots(1, len(specs), figsize=(16, 4.9))
    fig.suptitle("v7 public-only: train, validation and test metrics", fontsize=16, fontweight="bold", y=1.03)
    colors = [BLUE, "#DD8452", GREEN]
    for axis, (key, title) in zip(axes, specs):
        values = split_metrics.set_index("split").loc[["train", "validation", "test"], key].to_numpy()
        bars = axis.bar([names[x] for x in ["train", "validation", "test"]], values, color=colors, edgecolor="#222222", linewidth=0.6)
        axis.set_title(title, fontweight="bold")
        axis.set_ylabel("Value")
        axis.set_ylim(0, max(values) * 1.25)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "v7_public_only_split_metrics.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    manifest, pred, split_metrics, calibration, ecdf, qq = load_data()
    if manifest.get("schema_version") != "v7_public_only_visualization_data_v1":
        raise ValueError("Unsupported visualization data schema")
    diagnostic_figure(pred, split_metrics)
    error_figure(pred, ecdf)
    calibration_figure(calibration)
    rank_qq_figure(pred, qq)
    metrics_figure(split_metrics)
    training_history_figure()
    print(f"Rendered v7_public_only figures from {DATA_DIR} to {OUT}")


if __name__ == "__main__":
    main()
