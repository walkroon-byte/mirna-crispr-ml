#!/usr/bin/env python3
"""Deep-learning optimization of v7_public_only on the cleaned public data."""

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
OUTPUT_DIR = PROJECT_ROOT / "models/v7_public_only_deep_optimized"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Reuse the project's established sequence normalization and 20-feature definition.
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import train_model_v7_public_only as base  # noqa: E402


def metrics(y_true, y_pred):
    p = pearsonr(y_true, y_pred) if np.std(y_pred) > 0 else (np.nan, np.nan)
    s = spearmanr(y_true, y_pred) if np.std(y_pred) > 0 else (np.nan, np.nan)
    return {
        "n": int(len(y_true)),
        "spearman_r": float(s.statistic), "spearman_p": float(s.pvalue),
        "pearson_r": float(p.statistic), "pearson_p": float(p.pvalue),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def load_data():
    df = pd.read_csv(DATA_FILE)
    if len(df) != 9894:
        raise ValueError(f"Expected 9894 cleaned rows, found {len(df)}")
    sequences, features, valid_rows = [], [], []
    for index, row in df.iterrows():
        try:
            spacer = base.one_hot_encode_sequence(row["spacer_seq"], 20)
            target = base.one_hot_encode_sequence(row["target_seq"], 20)
            sequences.append(np.concatenate([spacer, target], axis=0))
            features.append(base.extract_features_v6(row["spacer_seq"], row["target_seq"], row["mismatch_pos"]))
            valid_rows.append(index)
        except Exception:
            continue
    df = df.loc[valid_rows].reset_index(drop=True)
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(features, dtype=np.float32),
        df.relative_activity.to_numpy(dtype=np.float32),
        df,
    )


def grouped_split(groups):
    outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    train_val, test = next(outer.split(np.zeros(len(groups)), groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1764705882, random_state=RANDOM_STATE)
    train_rel, validation_rel = next(inner.split(np.zeros(len(train_val)), groups=groups[train_val]))
    return train_val[train_rel], train_val[validation_rel], test


def build_model():
    sequence_input = layers.Input((40, 4), name="sequence_input")
    x = layers.Conv1D(64, 3, padding="same", activation="relu",
                      kernel_regularizer=keras.regularizers.l2(2e-4))(sequence_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 5, padding="same", activation="relu",
                      kernel_regularizer=keras.regularizers.l2(2e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False,
                                         dropout=0.10,
                                         kernel_regularizer=keras.regularizers.l2(2e-4)))(x)
    x = layers.Dropout(0.25)(x)
    sequence_branch = layers.Dense(128, activation="relu")(x)
    sequence_branch = layers.Dropout(0.25)(sequence_branch)

    feature_input = layers.Input((20,), name="feature_input")
    f = layers.Dense(64, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(2e-4))(feature_input)
    f = layers.BatchNormalization()(f)
    f = layers.Dropout(0.20)(f)
    feature_branch = layers.Dense(32, activation="relu")(f)

    z = layers.Concatenate()([sequence_branch, feature_branch])
    z = layers.Dense(128, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(2e-4))(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.25)(z)
    z = layers.Dense(64, activation="relu")(z)
    z = layers.Dropout(0.15)(z)
    output = layers.Dense(1, activation="linear", name="output")(z)
    model = Model([sequence_input, feature_input], output, name="v7_public_only_deep_optimized")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss=keras.losses.Huber(delta=0.15),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_seq, X_feat, y, df = load_data()
    groups = df.spacer_seq.astype(str).str.upper().str.replace("T", "U", regex=False).str[:20].to_numpy()
    train_idx, validation_idx, test_idx = grouped_split(groups)
    split_groups = [set(groups[i]) for i in (train_idx, validation_idx, test_idx)]
    if any(split_groups[a] & split_groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Spacer leakage detected")

    scaler = StandardScaler()
    X_feat_scaled = np.zeros_like(X_feat, dtype=np.float32)
    X_feat_scaled[train_idx] = scaler.fit_transform(X_feat[train_idx]).astype(np.float32)
    X_feat_scaled[validation_idx] = scaler.transform(X_feat[validation_idx]).astype(np.float32)
    X_feat_scaled[test_idx] = scaler.transform(X_feat[test_idx]).astype(np.float32)
    with (OUTPUT_DIR / "v7_public_only_deep_optimized_feature_scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)

    model = build_model()
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            OUTPUT_DIR / "v7_public_only_deep_optimized_best.h5",
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=18,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=7, min_lr=1e-6, verbose=1),
    ]
    history = model.fit(
        [X_seq[train_idx], X_feat_scaled[train_idx]], y[train_idx],
        validation_data=([X_seq[validation_idx], X_feat_scaled[validation_idx]], y[validation_idx]),
        epochs=120, batch_size=64, callbacks=callbacks, verbose=2,
    )
    model.save(OUTPUT_DIR / "v7_public_only_deep_optimized_full.h5")

    predictions = {}
    for name, indices in (("train", train_idx), ("validation", validation_idx), ("test", test_idx)):
        pred = model.predict([X_seq[indices], X_feat_scaled[indices]], verbose=0).ravel()
        predictions[name] = metrics(y[indices], pred)
        if name == "test":
            pd.DataFrame({"spacer_seq": groups[indices], "actual": y[indices],
                          "predicted": pred, "residual": y[indices] - pred}).to_csv(
                              OUTPUT_DIR / "v7_public_only_deep_optimized_test_predictions.csv", index=False)

    split = pd.DataFrame({"row_index": np.arange(len(df)), "spacer_seq": groups, "split": ""})
    split.loc[train_idx, "split"] = "train"
    split.loc[validation_idx, "split"] = "validation"
    split.loc[test_idx, "split"] = "test"
    split.to_csv(OUTPUT_DIR / "v7_public_only_deep_optimized_split.csv", index=False)
    result = {
        "model_version": "v7_public_only_deep_optimized",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_file": str(DATA_FILE.relative_to(PROJECT_ROOT)),
        "n_rows": int(len(df)), "n_spacer_groups": int(len(np.unique(groups))),
        "split_method": "Nested GroupShuffleSplit by canonical first-20-nt spacer; seed 42",
        "split_rows": {k: int(len(v)) for k, v in (("train", train_idx), ("validation", validation_idx), ("test", test_idx))},
        "split_groups": {k: int(len(np.unique(v))) for k, v in (("train", groups[train_idx]), ("validation", groups[validation_idx]), ("test", groups[test_idx]))},
        "architecture": "v7 CNN(64,128)+BiLSTM(64)+20 hand-crafted features; lower LR, Huber loss, clipnorm, L2/dropout",
        "metrics": predictions,
        "limitations": ["Grouped test estimates unseen-spacer generalization; external experimental validation remains necessary.", "Remaining public labels were not independently re-audited beyond removing the identified 80 rows."],
    }
    with (OUTPUT_DIR / "v7_public_only_deep_optimized_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    with (OUTPUT_DIR / "v7_public_only_deep_optimized_history.json").open("w", encoding="utf-8") as handle:
        json.dump({k: [float(x) for x in vals] for k, vals in history.history.items()}, handle, indent=2)
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
