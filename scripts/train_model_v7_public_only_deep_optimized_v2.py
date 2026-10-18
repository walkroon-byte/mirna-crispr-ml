#!/usr/bin/env python3
"""Aligned-pair CNN/BiLSTM optimization for cleaned v7 public data."""

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data/processed_public_datasets/combined_v4_training_data_cleaned_exclude_unverified80.csv"
OUTPUT_DIR = PROJECT_ROOT / "models/v7_public_only_deep_optimized_v2"
RANDOM_STATE = 42
BASES = "ACGU"
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import train_model_v7_public_only as base  # noqa: E402


def encode_pair(spacer, target):
    spacer = str(spacer).upper().replace("T", "U").replace(" ", "")[:20]
    target = str(target).upper().replace("T", "U").replace(" ", "")[:20]
    if len(spacer) != 20 or len(target) != 20 or set(spacer + target) - set(BASES):
        raise ValueError("Invalid sequence")
    channels = np.zeros((20, 25), dtype=np.float32)
    for i, (a, b) in enumerate(zip(spacer, target)):
        channels[i, BASES.index(a)] = 1.0
        channels[i, 4 + BASES.index(b)] = 1.0
        channels[i, 8 + BASES.index(a) * 4 + BASES.index(b)] = 1.0
        channels[i, 24] = float(a != b)
    return channels, spacer


def metric(y, pred):
    p = pearsonr(y, pred) if np.std(pred) else (np.nan, np.nan)
    s = spearmanr(y, pred) if np.std(pred) else (np.nan, np.nan)
    return {"n": int(len(y)), "spearman_r": float(s.statistic), "spearman_p": float(s.pvalue),
            "pearson_r": float(p.statistic), "pearson_p": float(p.pvalue),
            "r2": float(r2_score(y, pred)), "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred))}


def load_data():
    df = pd.read_csv(DATA_FILE)
    if len(df) != 9894:
        raise ValueError(f"Expected 9894 rows, found {len(df)}")
    seq, feat, groups, keep = [], [], [], []
    for idx, row in df.iterrows():
        try:
            x, group = encode_pair(row.spacer_seq, row.target_seq)
            seq.append(x)
            feat.append(base.extract_features_v6(row.spacer_seq, row.target_seq, row.mismatch_pos))
            groups.append(group)
            keep.append(idx)
        except Exception:
            continue
    df = df.loc[keep].reset_index(drop=True)
    return np.asarray(seq, np.float32), np.asarray(feat, np.float32), df.relative_activity.to_numpy(np.float32), np.asarray(groups)


def splits(groups):
    outer = GroupShuffleSplit(n_splits=1, test_size=.15, random_state=RANDOM_STATE)
    train_val, test = next(outer.split(np.zeros(len(groups)), groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=.1764705882, random_state=RANDOM_STATE)
    tr, va = next(inner.split(np.zeros(len(train_val)), groups=groups[train_val]))
    return train_val[tr], train_val[va], test


def build_model():
    pair_input = layers.Input((20, 25), name="aligned_pair_input")
    x = layers.Conv1D(96, 3, padding="same", activation="swish")(pair_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 5, padding="same", activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.SpatialDropout1D(.12)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=.08))(x)
    x = layers.GlobalMaxPooling1D()(x)
    seq_branch = layers.Dense(128, activation="swish")(x)
    seq_branch = layers.Dropout(.20)(seq_branch)

    feature_input = layers.Input((20,), name="feature_input")
    f = layers.Dense(64, activation="swish")(feature_input)
    f = layers.LayerNormalization()(f)
    f = layers.Dropout(.12)(f)
    feat_branch = layers.Dense(32, activation="swish")(f)

    z = layers.Concatenate()([seq_branch, feat_branch])
    z = layers.Dense(128, activation="swish")(z)
    z = layers.LayerNormalization()(z)
    z = layers.Dropout(.20)(z)
    z = layers.Dense(64, activation="swish")(z)
    output = layers.Dense(1)(z)
    model = Model([pair_input, feature_input], output, name="v7_public_only_deep_optimized_v2")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
                  loss="mse", metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return model


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, F, y, groups = load_data()
    tr, va, te = splits(groups)
    group_sets = [set(groups[i]) for i in (tr, va, te)]
    if any(group_sets[a] & group_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Spacer leakage")
    scaler = StandardScaler()
    F_scaled = np.zeros_like(F)
    F_scaled[tr] = scaler.fit_transform(F[tr]).astype(np.float32)
    F_scaled[va] = scaler.transform(F[va]).astype(np.float32)
    F_scaled[te] = scaler.transform(F[te]).astype(np.float32)
    with (OUTPUT_DIR / "feature_scaler.pkl").open("wb") as h:
        pickle.dump(scaler, h)
    model = build_model()
    callbacks = [
        keras.callbacks.ModelCheckpoint(OUTPUT_DIR / "best_model.h5", monitor="val_loss", save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=18, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5, patience=6, min_lr=2e-6, verbose=1),
    ]
    history = model.fit([X[tr], F_scaled[tr]], y[tr], validation_data=([X[va], F_scaled[va]], y[va]),
                        epochs=140, batch_size=64, callbacks=callbacks, verbose=2)
    model.save(OUTPUT_DIR / "full_model.h5")
    results = {}
    for name, idx in (("train", tr), ("validation", va), ("test", te)):
        pred = model.predict([X[idx], F_scaled[idx]], verbose=0).ravel()
        results[name] = metric(y[idx], pred)
        if name == "test":
            pd.DataFrame({"spacer_seq": groups[idx], "actual": y[idx], "predicted": pred,
                          "residual": y[idx] - pred}).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    split = pd.DataFrame({"row_index": np.arange(len(y)), "spacer_seq": groups, "split": ""})
    split.loc[tr, "split"] = "train"; split.loc[va, "split"] = "validation"; split.loc[te, "split"] = "test"
    split.to_csv(OUTPUT_DIR / "split.csv", index=False)
    report = {"model_version": "v7_public_only_deep_optimized_v2", "trained_at_utc": datetime.now(timezone.utc).isoformat(),
              "data_file": str(DATA_FILE.relative_to(PROJECT_ROOT)), "n_rows": int(len(y)),
              "n_spacer_groups": int(len(np.unique(groups))), "split_method": "Grouped by spacer, seed 42",
              "split_rows": {"train": len(tr), "validation": len(va), "test": len(te)},
              "architecture": "aligned 20-position pair input: spacer 4 + target 4 + pair type 16 + mismatch flag; CNN+BiLSTM+20 features",
              "training": "MSE, Adam 5e-4, batch 64, L2-free capacity-controlled network, early stopping and LR reduction",
              "metrics": results,
              "limitations": ["Grouped test estimates unseen-spacer generalization; external experimental validation remains necessary.", "Remaining public labels were not independently re-audited beyond removing the identified 80 rows."]}
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as h:
        json.dump(report, h, indent=2, ensure_ascii=False)
    with (OUTPUT_DIR / "history.json").open("w", encoding="utf-8") as h:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, h, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
