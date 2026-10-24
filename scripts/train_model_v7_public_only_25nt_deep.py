#!/usr/bin/env python3
"""Train a 25-nt CNN/BiLSTM Cas12a activity model on EasyDesign data.

This is a separate deep-learning model line. It preserves both v7_public_only
and the previously trained 25-nt gradient-boosting model.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import Model, layers


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data/processed_public_datasets/combined_v4_training_data_cleaned_exclude_unverified80.csv"
OUTPUT_DIR = PROJECT_ROOT / "models/v7_public_only_25nt_deep"
RANDOM_STATE = 42
SEQ_LENGTH = 25
BASES = "ACGU"

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_model_v7_public_only_25nt as base  # noqa: E402


def encode_aligned_pair(spacer_raw: str, target_raw: str) -> tuple[np.ndarray, str]:
    """Encode all 25 positions as spacer(4)+target(4)+pair(16)+mismatch(1)."""
    spacer = base.normalize(spacer_raw)
    target = base.normalize(target_raw)
    encoded = np.zeros((SEQ_LENGTH, 25), dtype=np.float32)
    for i, (a, b) in enumerate(zip(spacer, target)):
        encoded[i, BASES.index(a)] = 1.0
        encoded[i, 4 + BASES.index(b)] = 1.0
        encoded[i, 8 + BASES.index(a) * 4 + BASES.index(b)] = 1.0
        encoded[i, 24] = float(a != b)
    return encoded, spacer


def load_data():
    df = pd.read_csv(DATA_FILE)
    if len(df) != 9894:
        raise ValueError(f"Expected 9894 EasyDesign rows, found {len(df)}")
    sequences, features, groups, keep = [], [], [], []
    for idx, row in df.iterrows():
        try:
            seq, group = encode_aligned_pair(row.spacer_seq, row.target_seq)
            full_features = base.extract_features_25nt(row.spacer_seq, row.target_seq)
            sequences.append(seq)
            features.append(full_features[-18:])
            groups.append(group)
            keep.append(idx)
        except (ValueError, RuntimeError):
            continue
    clean = df.loc[keep].reset_index(drop=True)
    if len(clean) != 9894:
        raise ValueError(f"Some rows failed 25-nt encoding: retained {len(clean)}")
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(features, dtype=np.float32),
        clean.relative_activity.to_numpy(np.float32),
        np.asarray(groups),
        clean,
    )


def split_indices(groups):
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    train_val, test = next(outer.split(np.zeros(len(groups)), groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1764705882, random_state=RANDOM_STATE)
    train_rel, val_rel = next(inner.split(np.zeros(len(train_val)), groups=groups[train_val]))
    return train_val[train_rel], train_val[val_rel], test


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


def build_model():
    pair_input = layers.Input((SEQ_LENGTH, 25), name="aligned_25nt_pair")
    x = layers.Conv1D(64, 3, padding="same", activation="swish")(pair_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(96, 5, padding="same", activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.SpatialDropout1D(0.15)(x)
    x = layers.Bidirectional(layers.LSTM(48, return_sequences=True, dropout=0.10))(x)
    x = layers.GlobalMaxPooling1D()(x)
    seq_branch = layers.Dense(96, activation="swish")(x)
    seq_branch = layers.Dropout(0.25)(seq_branch)

    feature_input = layers.Input((18,), name="thermodynamic_features")
    f = layers.Dense(48, activation="swish")(feature_input)
    f = layers.LayerNormalization()(f)
    f = layers.Dropout(0.15)(f)
    feat_branch = layers.Dense(24, activation="swish")(f)

    z = layers.Concatenate()([seq_branch, feat_branch])
    z = layers.Dense(96, activation="swish")(z)
    z = layers.LayerNormalization()(z)
    z = layers.Dropout(0.25)(z)
    z = layers.Dense(48, activation="swish")(z)
    output = layers.Dense(1, name="relative_activity_prediction")(z)
    model = Model([pair_input, feature_input], output, name="v7_public_only_25nt_deep")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_seq, X_feat, y, groups, clean = load_data()
    train_idx, val_idx, test_idx = split_indices(groups)
    group_sets = [set(groups[idx]) for idx in (train_idx, val_idx, test_idx)]
    if any(group_sets[a] & group_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Spacer leakage detected")

    scaler = StandardScaler()
    X_feat_scaled = np.zeros_like(X_feat)
    X_feat_scaled[train_idx] = scaler.fit_transform(X_feat[train_idx]).astype(np.float32)
    X_feat_scaled[val_idx] = scaler.transform(X_feat[val_idx]).astype(np.float32)
    X_feat_scaled[test_idx] = scaler.transform(X_feat[test_idx]).astype(np.float32)
    with (OUTPUT_DIR / "feature_scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)

    model = build_model()
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            OUTPUT_DIR / "best_model.keras", monitor="val_loss", save_best_only=True, verbose=0
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=18, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=2e-6, verbose=1
        ),
    ]
    history = model.fit(
        [X_seq[train_idx], X_feat_scaled[train_idx]], y[train_idx],
        validation_data=([X_seq[val_idx], X_feat_scaled[val_idx]], y[val_idx]),
        epochs=140,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )

    results = {}
    predictions = {}
    for name, idx in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        pred = model.predict([X_seq[idx], X_feat_scaled[idx]], verbose=0).reshape(-1)
        predictions[name] = pred
        results[name] = metrics(y[idx], pred)

    pd.DataFrame({
        "row_index": test_idx,
        "spacer_seq": groups[test_idx],
        "target_seq": clean.iloc[test_idx].target_seq.to_numpy(),
        "actual": y[test_idx],
        "predicted": predictions["test"],
        "residual": y[test_idx] - predictions["test"],
    }).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    split = pd.DataFrame({"row_index": np.arange(len(y)), "spacer_seq": groups, "split": ""})
    split.loc[train_idx, "split"] = "train"
    split.loc[val_idx, "split"] = "validation"
    split.loc[test_idx, "split"] = "test"
    split.to_csv(OUTPUT_DIR / "split.csv", index=False)
    model.save(OUTPUT_DIR / "full_model.keras")
    with (OUTPUT_DIR / "history.json").open("w", encoding="utf-8") as handle:
        json.dump({key: [float(v) for v in values] for key, values in history.history.items()}, handle, indent=2)

    report = {
        "model_version": "v7_public_only_25nt_deep",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "tensorflow_version": tf.__version__,
        "data_file": str(DATA_FILE.relative_to(PROJECT_ROOT)),
        "data_sha256": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
        "n_rows": int(len(y)),
        "sequence_length": SEQ_LENGTH,
        "sequence_input_shape": [SEQ_LENGTH, 25],
        "sequence_encoding": "25 aligned positions: spacer one-hot (4) + target one-hot (4) + base-pair identity one-hot (16) + mismatch flag (1)",
        "feature_input_shape": [18],
        "feature_definition": "GC/composition, mismatch count and position bins, duplex delta-G, RNAfold MFE, homopolymer summaries",
        "split_method": "Nested GroupShuffleSplit by complete 25-nt spacer; seed 42",
        "split_rows": {"train": int(len(train_idx)), "validation": int(len(val_idx)), "test": int(len(test_idx))},
        "split_groups": {"train": len(group_sets[0]), "validation": len(group_sets[1]), "test": len(group_sets[2])},
        "architecture": "25-position aligned-pair CNN + BiLSTM branch fused with 18-feature thermodynamic branch; regression output",
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "metrics": results,
        "limitations": [
            "Metrics estimate generalization to unseen spacers within the cleaned EasyDesign dataset.",
            "The target is the dataset relative_activity label, not RT-RPA-Cas12a signal or miR-29a selectivity.",
            "Remaining public labels require source-level audit; wet-lab validation using miR-29a/b/c is required.",
        ],
    }
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
