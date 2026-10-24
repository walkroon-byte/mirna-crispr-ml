#!/usr/bin/env python3
"""Train a 25-nt EasyDesign-compatible Cas12a activity model.

This is a new model line. It does not overwrite models/v7_public_only.
The input data are the cleaned public-only records, whose EasyDesign guides
and targets are 25 nt. Splitting is grouped by complete spacer sequence.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import RNA
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data/processed_public_datasets/combined_v4_training_data_cleaned_exclude_unverified80.csv"
OUTPUT_DIR = PROJECT_ROOT / "models/v7_public_only_25nt"
RANDOM_STATE = 42
SEQ_LENGTH = 25
BASES = "ACGU"


def normalize(seq: str) -> str:
    value = str(seq).upper().replace("T", "U").replace(" ", "")
    if len(value) != SEQ_LENGTH or set(value) - set(BASES):
        raise ValueError(f"Expected a valid {SEQ_LENGTH}-nt RNA/DNA sequence: {seq!r}")
    return value


def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGU", "UGCA"))[::-1]


def duplex_energy(spacer: str, target: str) -> float:
    return float(RNA.duplexfold(spacer, reverse_complement(target)).energy)


def longest_run(seq: str) -> int:
    best = run = 1
    for left, right in zip(seq, seq[1:]):
        run = run + 1 if left == right else 1
        best = max(best, run)
    return best if seq else 0


def extract_features_25nt(spacer_raw: str, target_raw: str) -> np.ndarray:
    """Return features that retain all 25 aligned positions.

    Layout: 25*8 strand one-hot + 25 mismatch flags + 25*16 mismatch-pair
    one-hot + 18 summary/thermodynamic features = 643 dimensions.
    """
    spacer = normalize(spacer_raw)
    target = normalize(target_raw)
    encoded = np.zeros((2, SEQ_LENGTH, 4), dtype=np.float32)
    pair = np.zeros((SEQ_LENGTH, 16), dtype=np.float32)
    mismatch = np.zeros(SEQ_LENGTH, dtype=np.float32)
    mismatch_positions = []
    for i, (a, b) in enumerate(zip(spacer, target)):
        encoded[0, i, BASES.index(a)] = 1.0
        encoded[1, i, BASES.index(b)] = 1.0
        pair[i, BASES.index(a) * 4 + BASES.index(b)] = 1.0
        if a != b:
            mismatch[i] = 1.0
            mismatch_positions.append(i)

    first = float(mismatch_positions[0]) if mismatch_positions else -1.0
    last = float(mismatch_positions[-1]) if mismatch_positions else -1.0
    dg_pair = duplex_energy(spacer, target)
    dg_perfect = duplex_energy(spacer, spacer)
    dg_delta = dg_pair - dg_perfect if mismatch_positions else 0.0
    spacer_gc = (spacer.count("G") + spacer.count("C")) / SEQ_LENGTH
    target_gc = (target.count("G") + target.count("C")) / SEQ_LENGTH
    mismatch_count = float(len(mismatch_positions))
    summary = np.array([
        spacer_gc,
        target_gc,
        abs(spacer_gc - target_gc),
        mismatch_count,
        first,
        last,
        last - first if mismatch_positions else 0.0,
        float(sum(i < 8 for i in mismatch_positions)),
        float(sum(i < 5 for i in mismatch_positions)),
        float(sum(8 <= i < 17 for i in mismatch_positions)),
        float(sum(i >= 17 for i in mismatch_positions)),
        dg_pair,
        dg_perfect,
        dg_delta,
        float(RNA.fold(spacer)[1]),
        float(RNA.fold(target)[1]),
        float(longest_run(spacer)),
        float(longest_run(target)),
    ], dtype=np.float32)
    return np.concatenate((encoded.reshape(-1), mismatch, pair.reshape(-1), summary))


def encode_dataset(df: pd.DataFrame):
    features = []
    keep = []
    groups = []
    for idx, row in df.iterrows():
        try:
            spacer = normalize(row.spacer_seq)
            features.append(extract_features_25nt(row.spacer_seq, row.target_seq))
            groups.append(spacer)
            keep.append(idx)
        except (ValueError, RuntimeError):
            continue
    clean = df.loc[keep].reset_index(drop=True)
    return np.asarray(features, dtype=np.float32), clean.relative_activity.to_numpy(np.float32), np.asarray(groups), clean


def metrics(y_true, y_pred):
    pearson = pearsonr(y_true, y_pred)
    spearman = spearmanr(y_true, y_pred)
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


def split_groups(groups):
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    train_val, test = next(outer.split(np.zeros(len(groups)), groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1764705882, random_state=RANDOM_STATE)
    train_rel, val_rel = next(inner.split(np.zeros(len(train_val)), groups=groups[train_val]))
    return train_val[train_rel], train_val[val_rel], test


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_FILE)
    X, y, groups, clean = encode_dataset(df)
    if len(clean) != 9894 or X.shape[1] != 643:
        raise RuntimeError(f"Unexpected encoded data: rows={len(clean)}, features={X.shape[1]}")

    train_idx, val_idx, test_idx = split_groups(groups)
    group_sets = [set(groups[i]) for i in (train_idx, val_idx, test_idx)]
    if any(group_sets[a] & group_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Spacer leakage detected")

    scaler = StandardScaler()
    X_scaled = np.zeros_like(X)
    X_scaled[train_idx] = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_scaled[val_idx] = scaler.transform(X[val_idx]).astype(np.float32)
    X_scaled[test_idx] = scaler.transform(X[test_idx]).astype(np.float32)
    with (OUTPUT_DIR / "feature_scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)

    models = {
        "hist_gradient_300": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=RANDOM_STATE,
        ),
        "hist_gradient_500": HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.04, max_leaf_nodes=15,
            l2_regularization=2.0, random_state=RANDOM_STATE,
        ),
        "random_forest_leaf3": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=0.7,
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
    }
    validation = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_scaled[train_idx], y[train_idx])
        fitted[name] = model
        validation[name] = metrics(y[val_idx], model.predict(X_scaled[val_idx]))

    selected_name = max(validation, key=lambda name: validation[name]["spearman_r"])
    selected = fitted[selected_name]
    test_pred = selected.predict(X_scaled[test_idx])
    test_metrics = metrics(y[test_idx], test_pred)
    selected.fit(X_scaled[np.concatenate((train_idx, val_idx))], y[np.concatenate((train_idx, val_idx))])

    import joblib
    joblib.dump(selected, OUTPUT_DIR / "v7_public_only_25nt_model.joblib")
    pred = pd.DataFrame({
        "row_index": test_idx,
        "spacer_seq": groups[test_idx],
        "target_seq": clean.iloc[test_idx].target_seq.to_numpy(),
        "actual": y[test_idx],
        "predicted": test_pred,
        "residual": y[test_idx] - test_pred,
    })
    pred.to_csv(OUTPUT_DIR / "v7_public_only_25nt_test_predictions.csv", index=False)
    split = pd.DataFrame({"row_index": np.arange(len(clean)), "spacer_seq": groups, "split": ""})
    split.loc[train_idx, "split"] = "train"
    split.loc[val_idx, "split"] = "validation"
    split.loc[test_idx, "split"] = "test"
    split.to_csv(OUTPUT_DIR / "v7_public_only_25nt_split.csv", index=False)

    sha256 = hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()
    report = {
        "model_version": "v7_public_only_25nt",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_file": str(DATA_FILE.relative_to(PROJECT_ROOT)),
        "data_sha256": sha256,
        "n_rows": int(len(clean)),
        "n_features": int(X.shape[1]),
        "sequence_length": SEQ_LENGTH,
        "feature_definition": "25-nt spacer/target one-hot (200) + 25-position mismatch mask (25) + 25-position mismatch-pair one-hot (400) + 18 composition/position/ViennaRNA features = 643",
        "actual_feature_count": int(X.shape[1]),
        "split_method": "Nested GroupShuffleSplit by complete 25-nt spacer; seed 42",
        "split_rows": {"train": int(len(train_idx)), "validation": int(len(val_idx)), "test": int(len(test_idx))},
        "split_groups": {"train": len(group_sets[0]), "validation": len(group_sets[1]), "test": len(group_sets[2])},
        "validation_metrics": validation,
        "selected_model": selected_name,
        "test_metrics": test_metrics,
        "limitations": [
            "This evaluates public labels after excluding the identified unverified records; remaining labels still require source-level audit.",
            "The model predicts the dataset's relative_activity endpoint, not RT-RPA-Cas12a assay signal or miR-29a selectivity directly.",
            "External wet-lab validation on miR-29a/b/c and correct PAM/amplicon architecture remains required.",
        ],
    }
    (OUTPUT_DIR / "v7_public_only_25nt_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
