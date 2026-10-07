#!/usr/bin/env python3
"""
训练 v7 模型：深度学习方法（简化版，无ViennaRNA依赖）- 仅使用公开数据集

架构：
- 序列编码器：One-hot + CNN + Bi-LSTM
- 特征编码器：简化的10个手工特征（无RNA二级结构）
- 融合层：拼接 + 全连接
- 数据：排除原始50条数据，仅使用公开数据集（约9,894条）

作者: Evelyn
日期: 2026-10-03
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# 设置路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data/processed_public_datasets/combined_v4_training_data.csv"
ORIGINAL_DATA_FILE = PROJECT_ROOT / "data/raw/cas12a_mismatch_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "models/v7_public_only_simple"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 序列编码和简化特征提取（无ViennaRNA）
# ============================================================================

def one_hot_encode_sequence(seq: str, length: int = 20) -> np.ndarray:
    """One-hot编码DNA/RNA序列"""
    seq = seq.upper().replace('T', 'U')[:length]
    mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

    encoded = np.zeros((length, 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if i >= length:
            break
        if base in mapping:
            encoded[i, mapping[base]] = 1.0

    return encoded

def extract_simple_features(spacer: str, target: str, mismatch_pos=None) -> list:
    """
    提取简化的10个特征（不依赖ViennaRNA）
    """
    spacer = spacer.upper().replace('T', 'U')[:20]
    target = target.upper().replace('T', 'U')[:20]

    # 确保长度为20
    if len(spacer) < 20:
        spacer = spacer + 'N' * (20 - len(spacer))
    if len(target) < 20:
        target = target + 'N' * (20 - len(target))

    # 找出所有错配位置
    mismatches = [i for i in range(20) if spacer[i] != target[i]]
    num_mismatches = len(mismatches)

    # 特征1: 错配数量
    feat_num_mm = float(num_mismatches)

    # 特征2: 第一个错配位置（归一化到0-1）
    feat_first_mm_pos = float(min(mismatches)) / 19.0 if mismatches else 0.0

    # 特征3: GC含量（spacer）
    gc_spacer = sum(1 for b in spacer if b in 'GC') / 20.0

    # 特征4: GC含量（target）
    gc_target = sum(1 for b in target if b in 'GC') / 20.0

    # 特征5: Seed区域错配（前12bp）
    seed_mm = sum(1 for i in range(min(12, 20)) if spacer[i] != target[i])

    # 特征6: PAM-proximal错配（后8bp）
    pam_proximal_mm = sum(1 for i in range(12, 20) if spacer[i] != target[i])

    # 特征7: 连续匹配最大长度
    max_match_len = 0
    current_match = 0
    for i in range(20):
        if spacer[i] == target[i]:
            current_match += 1
            max_match_len = max(max_match_len, current_match)
        else:
            current_match = 0
    feat_max_match = float(max_match_len) / 20.0

    # 特征8: 嘌呤-嘧啶错配
    purine_pyrimidine_mm = 0
    for i in mismatches:
        if (spacer[i] in 'AG' and target[i] in 'CU') or \
           (spacer[i] in 'CU' and target[i] in 'AG'):
            purine_pyrimidine_mm += 1

    # 特征9: 错配分布均匀度（标准差）
    if len(mismatches) > 1:
        mm_std = np.std(mismatches) / 20.0
    else:
        mm_std = 0.0

    # 特征10: Seed区域GC含量
    seed_gc = sum(1 for b in spacer[:12] if b in 'GC') / 12.0

    return [
        feat_num_mm,
        feat_first_mm_pos,
        gc_spacer,
        gc_target,
        float(seed_mm),
        float(pam_proximal_mm),
        feat_max_match,
        float(purine_pyrimidine_mm),
        mm_std,
        seed_gc
    ]

# ============================================================================
# 数据加载和准备
# ============================================================================

def load_and_prepare_data():
    """加载数据并准备序列编码和手工特征"""
    print("="*80)
    print("加载训练数据（仅公开数据集）")
    print("="*80)

    # 加载完整数据
    df_full = pd.read_csv(DATA_FILE)
    print(f"完整数据量: {len(df_full)} 条")

    # 加载原始数据并排除
    df_original = pd.read_csv(ORIGINAL_DATA_FILE)
    print(f"原始数据量: {len(df_original)} 条")

    # 标记原始数据
    df_original['is_original'] = True
    df = df_full.merge(
        df_original[['spacer_seq', 'target_seq', 'mismatch_pos', 'is_original']],
        on=['spacer_seq', 'target_seq', 'mismatch_pos'],
        how='left'
    )

    # 排除原始数据
    df = df[df['is_original'].isna()].copy()
    df = df.drop(columns=['is_original'])

    print(f"\n排除原始数据后: {len(df)} 条")
    print(f"唯一 spacer 数量: {df['spacer_seq'].nunique()}")

    print("\n准备深度学习输入...")
    print("1. 序列 one-hot 编码")
    print("2. 提取简化特征（无ViennaRNA依赖）")

    sequences_encoded = []
    features = []
    valid_indices = []

    for idx, row in df.iterrows():
        try:
            # One-hot 编码序列
            spacer_onehot = one_hot_encode_sequence(row['spacer_seq'], length=20)
            target_onehot = one_hot_encode_sequence(row['target_seq'], length=20)
            seq_concat = np.concatenate([spacer_onehot, target_onehot], axis=0)
            sequences_encoded.append(seq_concat)

            # 提取简化特征
            feat = extract_simple_features(
                row['spacer_seq'],
                row['target_seq'],
                row['mismatch_pos']
            )
            features.append(feat)
            valid_indices.append(idx)

        except Exception as e:
            print(f"  警告: 第 {idx} 行处理失败: {e}")
            continue

        if (len(valid_indices)) % 1000 == 0:
            print(f"  已处理 {len(valid_indices)} 行")

    df = df.loc[valid_indices].reset_index(drop=True)
    X_seq = np.array(sequences_encoded, dtype=np.float32)
    X_feat = np.array(features, dtype=np.float32)
    y = df['relative_activity'].values.astype(np.float32)

    print(f"\n最终数据:")
    print(f"  序列矩阵: {X_seq.shape}")
    print(f"  特征矩阵: {X_feat.shape}")
    print(f"  活性值: {y.shape}, 范围 [{y.min():.4f}, {y.max():.4f}]")

    return X_seq, X_feat, y, df

# ============================================================================
# 构建深度学习模型
# ============================================================================

def build_hybrid_model(seq_input_shape=(40, 4), feat_input_shape=(10,)):
    """构建混合深度学习模型（简化版）"""
    print("\n" + "="*80)
    print("构建 v7 混合深度学习模型（简化版）")
    print("="*80)

    # 序列输入分支
    seq_input = layers.Input(shape=seq_input_shape, name='sequence_input')
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(seq_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)
    sequence_branch = layers.Dense(128, activation='relu', name='seq_encoding')(x)
    sequence_branch = layers.Dropout(0.3)(sequence_branch)

    # 特征输入分支
    feat_input = layers.Input(shape=feat_input_shape, name='feature_input')
    f = layers.Dense(64, activation='relu')(feat_input)
    f = layers.BatchNormalization()(f)
    f = layers.Dropout(0.3)(f)
    feature_branch = layers.Dense(32, activation='relu', name='feat_encoding')(f)

    # 融合层
    merged = layers.Concatenate()([sequence_branch, feature_branch])
    z = layers.Dense(128, activation='relu')(merged)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(64, activation='relu')(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(32, activation='relu')(z)
    output = layers.Dense(1, name='output')(z)

    model = models.Model(inputs=[seq_input, feat_input], outputs=output, name='v7_hybrid_simple')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    print(f"\n模型总结:")
    model.summary()

    return model

# ============================================================================
# 训练、评估、保存（其余代码与原版相同）
# ============================================================================

def train_model(model, X_seq_train, X_feat_train, y_train, X_seq_val, X_feat_val, y_val):
    """训练模型"""
    print("\n" + "="*80)
    print("训练模型")
    print("="*80)

    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_DIR / 'v7_simple_best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        [X_seq_train, X_feat_train],
        y_train,
        validation_data=([X_seq_val, X_feat_val], y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks_list,
        verbose=1
    )

    return history

def evaluate_model(model, X_seq, X_feat, y, dataset_name):
    """评估模型"""
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
        'mae': mae,
        'predictions': y_pred
    }

# ============================================================================
# 主函数
# ============================================================================

def main():
    print(f"{'='*80}")
    print(f"训练 v7_public_only_simple 模型")
    print(f"{'='*80}")

    # 1. 加载数据
    X_seq, X_feat, y, df = load_and_prepare_data()

    # 2. 标准化特征
    scaler = StandardScaler()
    X_feat_scaled = scaler.fit_transform(X_feat)

    with open(OUTPUT_DIR / 'v7_simple_feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # 3. 划分数据集
    X_seq_temp, X_seq_test, X_feat_temp, X_feat_test, y_temp, y_test = train_test_split(
        X_seq, X_feat_scaled, y, test_size=0.15, random_state=42
    )
    X_seq_train, X_seq_val, X_feat_train, X_feat_val, y_train, y_val = train_test_split(
        X_seq_temp, X_feat_temp, y_temp, test_size=0.176, random_state=42
    )

    print(f"\n训练集: {len(X_seq_train)} 条")
    print(f"验证集: {len(X_seq_val)} 条")
    print(f"测试集: {len(X_seq_test)} 条")

    # 4. 构建并训练模型
    model = build_hybrid_model(seq_input_shape=(40, 4), feat_input_shape=(10,))
    history = train_model(model, X_seq_train, X_feat_train, y_train, X_seq_val, X_feat_val, y_val)

    # 5. 评估
    train_metrics = evaluate_model(model, X_seq_train, X_feat_train, y_train, "训练集")
    val_metrics = evaluate_model(model, X_seq_val, X_feat_val, y_val, "验证集")
    test_metrics = evaluate_model(model, X_seq_test, X_feat_test, y_test, "测试集")

    # 6. 保存结果
    model.save(OUTPUT_DIR / "v7_simple_full_model.h5")

    metrics = {
        'model_version': 'v7_public_only_simple',
        'data_source': 'Public datasets only (no ViennaRNA features)',
        'num_features': 10,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }

    with open(OUTPUT_DIR / "v7_simple_metrics.json", 'w') as f:
        json.dump({k: (v if not isinstance(v, np.ndarray) else v.tolist())
                   for k, v in metrics.items() if k != 'predictions'}, f, indent=2)

    print(f"\n{'='*80}")
    print(f"训练完成！结果保存在: {OUTPUT_DIR}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
