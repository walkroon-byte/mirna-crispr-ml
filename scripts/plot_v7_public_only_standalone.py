"""Standalone figure set for the saved v7_public_only evaluation artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "v7_public_only"
OUT = ROOT / "results" / "figures" / "v7_public_only_standalone"
BLUE = "#4C72B0"
RED = "#C44E52"
GREEN = "#55A868"
GRAY = "#555555"


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.12)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.6,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


def load_artifacts() -> tuple[dict, pd.DataFrame]:
    with (MODEL_DIR / "v7_public_only_model_metrics.json").open(encoding="utf-8") as fh:
        metrics = json.load(fh)
    pred = pd.read_csv(MODEL_DIR / "v7_test_predictions.csv")
    required = {"actual", "predicted", "residual"}
    missing = required.difference(pred.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    pred = pred.astype(float)
    pred["absolute_error"] = pred["residual"].abs()
    return metrics, pred


def save_training_history() -> None:
    source = MODEL_DIR / "v7_public_only_training_history.png"
    if source.exists():
        shutil.copy2(source, OUT / "v7_public_only_training_history.png")


def diagnostic_figure(pred: pd.DataFrame, metrics: dict) -> None:
    actual, predicted, residual = pred.actual, pred.predicted, pred.residual
    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    pad = (hi - lo) * 0.04
    lo, hi = lo - pad, hi + pad
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 10.5))
    fig.suptitle("v7 public-only: held-out test diagnostics", fontsize=16,
                 fontweight="bold", y=0.99)

    ax[0, 0].scatter(actual, predicted, s=20, alpha=0.58, color=BLUE,
                     edgecolors="#222222", linewidths=0.25)
    ax[0, 0].plot([lo, hi], [lo, hi], "--", color=RED, linewidth=1.3)
    ax[0, 0].set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Actual relative activity",
                 ylabel="Predicted relative activity", title="Parity plot")
    test = metrics["test_metrics"]
    ax[0, 0].text(0.04, 0.96, f"n = {len(pred):,}\nR² = {test['r2']:.3f}\n"
                  f"Pearson r = {test['pearson_r']:.3f}\nSpearman ρ = {test['spearman_r']:.3f}",
                  transform=ax[0, 0].transAxes, va="top", fontsize=9.5,
                  bbox={"facecolor": "white", "edgecolor": "#BDBDBD", "alpha": 0.9,
                        "boxstyle": "round,pad=0.35"})

    ax[0, 1].scatter(predicted, residual, s=20, alpha=0.58, color=BLUE,
                     edgecolors="#222222", linewidths=0.25)
    ax[0, 1].axhline(0, linestyle="--", color=RED, linewidth=1.3)
    ax[0, 1].set(xlabel="Predicted relative activity", ylabel="Residual (actual − predicted)",
                 title="Residuals versus predictions")
    ax[0, 1].text(0.04, 0.96, f"RMSE = {test['rmse']:.3f}\nMAE = {test['mae']:.3f}\n"
                  f"median = {residual.median():+.3f}", transform=ax[0, 1].transAxes,
                  va="top", fontsize=9.5,
                  bbox={"facecolor": "white", "edgecolor": "#BDBDBD", "alpha": 0.9,
                        "boxstyle": "round,pad=0.35"})

    sns.histplot(residual, bins=28, kde=True, color=BLUE, edgecolor="white", ax=ax[1, 0])
    ax[1, 0].axvline(0, linestyle="--", color=RED, linewidth=1.3)
    ax[1, 0].set(xlabel="Residual (actual − predicted)", ylabel="Count",
                 title="Residual distribution")

    sns.kdeplot(actual, fill=True, color=GRAY, alpha=0.24, label="Actual", ax=ax[1, 1])
    sns.kdeplot(predicted, fill=True, color=GREEN, alpha=0.32, label="Predicted", ax=ax[1, 1])
    ax[1, 1].set(xlabel="Relative activity", ylabel="Density", title="Actual and predicted distributions")
    ax[1, 1].legend(frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "v7_public_only_test_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def error_figure(pred: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.8))
    fig.suptitle("v7 public-only: test error structure", fontsize=16, fontweight="bold", y=1.02)
    sns.violinplot(y=pred.residual, inner="quartile", color=BLUE, cut=0, linewidth=0.8, ax=ax[0])
    ax[0].axhline(0, linestyle="--", color=RED, linewidth=1.2)
    ax[0].set(xlabel="", ylabel="Residual (actual − predicted)", title="Residual violin")
    sns.ecdfplot(pred.absolute_error, color=BLUE, linewidth=2, ax=ax[1])
    ax[1].set(xlabel="Absolute error", ylabel="Cumulative proportion", title="Absolute-error ECDF")
    ax[1].set_ylim(0, 1.02)
    sns.boxplot(y=pred.absolute_error, color=GREEN, width=0.42, fliersize=2.2, ax=ax[2])
    ax[2].set(xlabel="", ylabel="Absolute error", title="Absolute-error boxplot")
    ax[2].text(0, pred.absolute_error.quantile(0.75), f"MAE = {pred.absolute_error.mean():.3f}",
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "v7_public_only_error_structure.png", bbox_inches="tight")
    plt.close(fig)


def calibration_figure(pred: pd.DataFrame) -> None:
    work = pred.assign(bin=pd.qcut(pred.predicted, q=10, duplicates="drop"))
    grouped = work.groupby("bin", observed=True).agg(
        predicted_mean=("predicted", "mean"), actual_mean=("actual", "mean"),
        actual_sd=("actual", "std"), n=("actual", "size"),
    ).reset_index()
    grouped["sem"] = grouped.actual_sd / np.sqrt(grouped.n.clip(lower=1))
    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    ax.errorbar(grouped["predicted_mean"], grouped["actual_mean"], yerr=1.96 * grouped["sem"],
                fmt="o-", color=BLUE, ecolor=BLUE, elinewidth=1, capsize=3,
                markersize=6, label="Observed mean ± 95% CI")
    lo = min(grouped.predicted_mean.min(), grouped.actual_mean.min())
    hi = max(grouped.predicted_mean.max(), grouped.actual_mean.max())
    pad = (hi - lo) * 0.08
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color=RED,
            linewidth=1.3, label="Perfect calibration")
    for _, row in grouped.iterrows():
        ax.annotate(f"n={int(row.n)}", (row.predicted_mean, row.actual_mean),
                    xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad),
           xlabel="Mean predicted relative activity", ylabel="Mean actual relative activity",
           title="v7 public-only: decile calibration")
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "v7_public_only_calibration.png", bbox_inches="tight")
    plt.close(fig)
    grouped.to_csv(OUT / "v7_public_only_calibration_bins.csv", index=False)


def rank_qq_figure(pred: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.1))
    fig.suptitle("v7 public-only: rank agreement and residual normality", fontsize=16,
                 fontweight="bold", y=1.02)
    actual_rank = stats.rankdata(pred.actual) / len(pred)
    predicted_rank = stats.rankdata(pred.predicted) / len(pred)
    ax[0].scatter(actual_rank, predicted_rank, s=15, alpha=0.48, color=BLUE,
                  edgecolors="none")
    ax[0].plot([0, 1], [0, 1], "--", color=RED, linewidth=1.3)
    ax[0].set(xlabel="Actual percentile rank", ylabel="Predicted percentile rank",
              title="Percentile-rank agreement")
    stats.probplot(pred.residual, dist="norm", plot=ax[1])
    ax[1].get_lines()[0].set_markerfacecolor(BLUE)
    ax[1].get_lines()[0].set_markeredgecolor("none")
    ax[1].get_lines()[0].set_alpha(0.52)
    ax[1].get_lines()[1].set_color(RED)
    ax[1].set_title("Residual Q–Q plot")
    ax[1].set_xlabel("Theoretical quantiles")
    ax[1].set_ylabel("Ordered residuals")
    fig.tight_layout()
    fig.savefig(OUT / "v7_public_only_rank_qq.png", bbox_inches="tight")
    plt.close(fig)


def metrics_figure(metrics: dict, n_test: int) -> None:
    split_names = ["Train", "Validation", "Test"]
    split_keys = ["train_metrics", "val_metrics", "test_metrics"]
    specs = [("r2", "R²"), ("rmse", "RMSE"), ("mae", "MAE"),
             ("pearson_r", "Pearson r"), ("spearman_r", "Spearman ρ")]
    fig, axes = plt.subplots(1, len(specs), figsize=(16, 4.9))
    fig.suptitle("v7 public-only: train, validation and test metrics", fontsize=16,
                 fontweight="bold", y=1.03)
    colors = [BLUE, "#DD8452", GREEN]
    for axis, (key, title) in zip(axes, specs):
        values = [metrics[k][key] for k in split_keys]
        bars = axis.bar(split_names, values, color=colors, edgecolor="#222222", linewidth=0.6)
        axis.set_title(title, fontweight="bold")
        axis.set_ylabel("Value")
        axis.set_ylim(0, max(values) * 1.25)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.025,
                      f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "v7_public_only_split_metrics.png", bbox_inches="tight")
    plt.close(fig)
    rows = []
    for name, key in zip(split_names, split_keys):
        row = {"split": name, "n": n_test if name == "Test" else np.nan}
        row.update({metric: metrics[key][metric] for metric, _ in specs})
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "v7_public_only_split_metrics.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    metrics, pred = load_artifacts()
    save_training_history()
    diagnostic_figure(pred, metrics)
    error_figure(pred)
    calibration_figure(pred)
    rank_qq_figure(pred)
    metrics_figure(metrics, len(pred))
    print(f"Saved standalone v7_public_only figures to {OUT}")


if __name__ == "__main__":
    main()
