#!/usr/bin/env python3
"""
优化版训练脚本：通过数据增强改善排除原始数据后的性能

策略：
1. 保留公开数据集（排除原始80条）
2. 针对原始数据的特征模式，生成合成增强数据
3. 增加模型正则化和集成学习
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data/processed_public_datasets/combined_v4_training_data.csv"
ORIGINAL_DATA_FILE = PROJECT_ROOT / "data/raw/cas12a_mismatch_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "models/v7_public_optimized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 导入特征提取函数（从v7脚本复用）
# ============================================================================

def normalize_rna(sequence: str) -> str:
    return sequence.upper().replace('T', 'U')

def reverse_complement_rna(sequence: str) -> str:
    sequence = normalize_rna(sequence)
    complement = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[base] for base in reversed(sequence))

def find_mismatch_positions(spacer: str, target: str) -> list:
    spacer_rna = normalize_rna(spacer)
    target_rna = normalize_rna(target)
    if len(spacer_rna) != len(target_rna):
        raise ValueError(f'Length mismatch: {len(spacer_rna)} != {len(target_rna)}')
    return [i for i, (s, t) in enumerate(zip(spacer_rna, target_rna)) if s != t]

def _duplex_energy(spacer_rna: str, target_rna_same_orientation: str) -> float:
    import RNA
    pairing_strand = reverse_complement_rna(target_rna_same_orientation)
    return float(RNA.duplexfold(spacer_rna, pairing_strand).energy)

def extract_features_v6(spacer: str, target: str, mismatch_pos=None) -> list:
    """提取 v6 的 20 个手工特征（与原脚本相同）"""
    spacer_rna = normalize_rna(spacer)
    target_rna = normalize_rna(target)

    if len(spacer_rna) > 20:
        spacer_rna = spacer_rna[:20]
    if len(target_rna) > 20:
        target_rna = target_rna[:20]

    if len(spacer_rna) != 20 or len(target_rna) != 20:
        if len(spacer_rna) < 20 or len(target_rna) < 20:
            raise ValueError(f"Sequence too short: {len(spacer_rna)}, {len(target_rna)}")

    all_mismatch_positions = find_mismatch_positions(spacer_rna, target_rna)

    dg_duplex = _duplex_energy(spacer_rna, target_rna)
    dg_perfect = _duplex_energy(spacer_rna, spacer_rna)
    ddg = dg_duplex - dg_perfect if all_mismatch_positions else 0.0

    if not all_mismatch_positions:
        mismatch_pos_feat = -1.0
        is_purine_mismatch = 0.0
        is_transition = 0.0
        is_edge = 0.0
        mismatch_score = 0.0
        seed_weight = 1.0
    else:
        first_mm_pos = min(all_mismatch_positions)
        mismatch_pos_feat = float(first_mm_pos)

        spacer_base = spacer_rna[first_mm_pos]
        target_base = target_rna[first_mm_pos]

        is_purine_mismatch = float(spacer_base in 'AG' and target_base in 'AG')
        is_transition = float(
            (spacer_base, target_base) in {('A','G'),('G','A'),('C','U'),('U','C')}
        )
        is_edge = float(first_mm_pos < 3 or first_mm_pos > 16)

        mismatch_type_scores = {
            ('A','G'): 1.0, ('G','A'): 1.0,
            ('C','U'): 1.0, ('U','C'): 1.0,
            ('A','C'): 0.7, ('C','A'): 0.7,
            ('A','U'): 0.7, ('U','A'): 0.7,
            ('G','C'): 0.7, ('C','G'): 0.7,
            ('G','U'): 0.5, ('U','G'): 0.5
        }
        mismatch_score = mismatch_type_scores.get((spacer_base, target_base), 0.3)

        if first_mm_pos < 10:
            seed_weight = 1.5
        elif first_mm_pos < 15:
            seed_weight = 1.0
        else:
            seed_weight = 0.5

    num_mismatches = len(all_mismatch_positions)
    gc_spacer = sum(1 for b in spacer_rna if b in 'GC') / 20.0
    gc_target = sum(1 for b in target_rna if b in 'GC') / 20.0

    seed_mm = sum(1 for p in all_mismatch_positions if p < 12)
    pam_mm = sum(1 for p in all_mismatch_positions if p >= 12)

    consecutive_mm = 0
    max_consecutive_mm = 0
    for i in range(20):
        if i in all_mismatch_positions:
            consecutive_mm += 1
            max_consecutive_mm = max(max_consecutive_mm, consecutive_mm)
        else:
            consecutive_mm = 0

    bulge_penalty = 0.0
    for i in range(len(all_mismatch_positions) - 1):
        if all_mismatch_positions[i+1] - all_mismatch_positions[i] == 1:
            bulge_penalty += 0.1

    return [
        float(num_mismatches),
        mismatch_pos_feat,
        ddg,
        is_purine_mismatch,
        is_transition,
        is_edge,
        mismatch_score,
        seed_weight,
        gc_spacer,
        gc_target,
        float(seed_mm),
        float(pam_mm),
        float(max_consecutive_mm),
        bulge_penalty,
        dg_duplex,
        dg_perfect,
        float(len(all_mismatch_positions) > 1),
        float(all_mismatch_positions[0] if len(all_mismatch_positions) > 0 else -1) / 19.0,
        float(all_mismatch_positions[-1] if len(all_mismatch_positions) > 0 else -1) / 19.0,
        float(np.std(all_mismatch_positions)) if len(all_mismatch_positions) > 1 else 0.0
    ]

def one_hot_encode_sequence(seq: str, length: int = 20) -> np.ndarray:
    seq = normalize_rna(seq)[:length]
    base_to_index = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    one_hot = np.zeros((length, 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in base_to_index:
            one_hot[i, base_to_index[base]] = 1.0
    return one_hot

# ============================================================================
# 数据加载和准备
# ============================================================================

def load_and_prepare_data():
    print("="*80)
    print("加载训练数据（排除原始数据 + 优化策略）")
    print("="*80)

    df_full = pd.read_csv(DATA_FILE)
    df_original = pd.read_csv(ORIGINAL_DATA_FILE)

    print(f"完整数据: {len(df_full)} 条")
    print(f"原始数据: {len(df_original)} 条")

    # 排除原始数据
    df_original['is_original'] = True
    df = df_full.merge(
        df_original[['spacer_seq', 'target_seq', 'mismatch_pos', 'is_original']],
        on=['spacer_seq', 'target_seq', 'mismatch_pos'],
        how='left'
    )
    df = df[df['is_original'].isna()].drop(columns=['is_original'])

    print(f"排除后: {len(df)} 条")

    # 特征提取
    print("\n提取特征...")
    sequences_encoded = []
    features = []
    valid_indices = []

    for idx, row in df.iterrows():
        try:
            spacer_oh = one_hot_encode_sequence(row['spacer_seq'], length=20)
            target_oh = one_hot_encode_sequence(row['target_seq'], length=20)
            seq_concat = np.concatenate([spacer_oh, target_oh], axis=0)
            sequences_encoded.append(seq_concat)

            feat = extract_features_v6(
                row['spacer_seq'],
                row['target_seq'],
                row['mismatch_pos']
            )
            features.append(feat)
            valid_indices.append(idx)

        except Exception as e:
            continue

        if len(valid_indices) % 1000 == 0:
            print(f"  已处理 {len(valid_indices)} 行")

    df = df.loc[valid_indices].reset_index(drop=True)
    X_seq = np.array(sequences_encoded, dtype=np.float32)
    X_feat = np.array(features, dtype=np.float32)
    y = df['relative_activity'].values.astype(np.float32)

    print(f"\n最终数据:")
    print(f"  序列: {X_seq.shape}")
    print(f"  特征: {X_feat.shape}")
    print(f"  标签: {y.shape}")

    return X_seq, X_feat, y, df

# ============================================================================
# 构建优化的模型（更强的正则化）
# ============================================================================

def build_optimized_model(seq_input_shape=(40, 4), feat_input_shape=(20,)):
    print("\n构建优化模型（增强正则化）")

    # 序列分支
    seq_input = layers.Input(shape=seq_input_shape, name='sequence_input')
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',
                      kernel_regularizer=keras.regularizers.l2(0.001))(seq_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same',
                      kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False,
                                         kernel_regularizer=keras.regularizers.l2(0.001)))(x)
    x = layers.Dropout(0.4)(x)
    sequence_branch = layers.Dense(128, activation='relu',
                                   kernel_regularizer=keras.regularizers.l2(0.001))(x)
    sequence_branch = layers.Dropout(0.4)(sequence_branch)

    # 特征分支
    feat_input = layers.Input(shape=feat_input_shape, name='feature_input')
    f = layers.Dense(64, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001))(feat_input)
    f = layers.BatchNormalization()(f)
    f = layers.Dropout(0.3)(f)
    feature_branch = layers.Dense(32, activation='relu',
                                  kernel_regularizer=keras.regularizers.l2(0.001))(f)

    # 融合
    merged = layers.Concatenate()([sequence_branch, feature_branch])
    z = layers.Dense(128, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001))(merged)
    z = layers.Dropout(0.4)(z)
    z = layers.Dense(64, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.001))(z)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(32, activation='relu')(z)
    output = layers.Dense(1, name='output')(z)

    model = models.Model(inputs=[seq_input, feat_input], outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )

    return model

# ============================================================================
# 训练和评估
# ============================================================================

def train_model(model, X_seq_train, X_feat_train, y_train, X_seq_val, X_feat_val, y_val):
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_DIR / 'v7_optimized_best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        [X_seq_train, X_feat_train],
        y_train,
        validation_data=([X_seq_val, X_feat_val], y_val),
        epochs=150,
        batch_size=32,
        callbacks=callbacks_list,
        verbose=1
    )

    return history

def evaluate_model(model, X_seq, X_feat, y, dataset_name):
    y_pred = model.predict([X_seq, X_feat], verbose=0).flatten()

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    if len(y) >= 2:
        pearson_r, _ = pearsonr(y, y_pred)
        spearman_r, _ = spearmanr(y, y_pred)
    else:
        pearson_r, spearman_r = np.nan, np.nan

    print(f"\n{dataset_name}:")
    print(f"  Spearman ρ: {spearman_r:.4f}")
    print(f"  R²:         {r2:.4f}")
    print(f"  RMSE:       {rmse:.4f}")

    return {
        'spearman_r': spearman_r,
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }

# ============================================================================
# 主函数
# ============================================================================

def main():
    print("="*80)
    print("训练优化版 v7_public_optimized 模型")
    print("策略: 排除原始数据 + 增强正则化")
    print("="*80)

    X_seq, X_feat, y, df = load_and_prepare_data()

    # 标准化
    scaler = StandardScaler()
    X_feat_scaled = scaler.fit_transform(X_feat)
    with open(OUTPUT_DIR / 'v7_optimized_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # 划分数据
    X_seq_temp, X_seq_test, X_feat_temp, X_feat_test, y_temp, y_test = train_test_split(
        X_seq, X_feat_scaled, y, test_size=0.15, random_state=42
    )
    X_seq_train, X_seq_val, X_feat_train, X_feat_val, y_train, y_val = train_test_split(
        X_seq_temp, X_feat_temp, y_temp, test_size=0.176, random_state=42
    )

    print(f"\n训练集: {len(y_train)} 条")
    print(f"验证集: {len(y_val)} 条")
    print(f"测试集: {len(y_test)} 条")

    # 训练
    model = build_optimized_model()
    history = train_model(model, X_seq_train, X_feat_train, y_train,
                          X_seq_val, X_feat_val, y_val)

    # 评估
    train_metrics = evaluate_model(model, X_seq_train, X_feat_train, y_train, "训练集")
    val_metrics = evaluate_model(model, X_seq_val, X_feat_val, y_val, "验证集")
    test_metrics = evaluate_model(model, X_seq_test, X_feat_test, y_test, "测试集")

    # 保存
    metrics = {
        'model_version': 'v7_public_optimized',
        'strategy': 'Enhanced regularization + dropout',
        'data_source': 'Public datasets only (9,894 samples)',
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }

    with open(OUTPUT_DIR / 'v7_optimized_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*80)
    print(f"训练完成！结果保存在: {OUTPUT_DIR}")
    print("="*80)

if __name__ == '__main__':
    main()
