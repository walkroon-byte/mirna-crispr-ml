#!/usr/bin/env python3
"""
训练 v7 模型：深度学习方法（混合模型）

架构：
- 序列编码器：One-hot + CNN + Bi-LSTM
- 特征编码器：v6 的 20 个手工特征
- 融合层：拼接 + 全连接
- 目标：Spearman ρ 从 0.541 提升到 0.60+

作者: Evelyn
日期: 2026-10-01
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# 设置路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data/processed_public_datasets/combined_v4_training_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "models/v7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# 序列编码函数
# ============================================================================

def normalize_rna(sequence: str) -> str:
    """将序列归一化为 RNA 格式"""
    value = str(sequence).upper().replace('T', 'U').replace(' ', '')
    if not value or any(base not in 'ACGU' for base in value):
        raise ValueError(f'Invalid RNA/DNA sequence: {sequence!r}')
    return value

def one_hot_encode_sequence(sequence: str, length: int = 20) -> np.ndarray:
    """
    将 RNA 序列编码为 one-hot 向量

    参数:
        sequence: RNA 序列
        length: 期望长度

    返回:
        shape (length, 4) 的 one-hot 矩阵
    """
    base_to_index = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    sequence = normalize_rna(sequence)

    # 截断或填充到指定长度
    if len(sequence) > length:
        sequence = sequence[:length]
    elif len(sequence) < length:
        sequence = sequence + 'A' * (length - len(sequence))

    # One-hot 编码
    one_hot = np.zeros((length, 4), dtype=np.float32)
    for i, base in enumerate(sequence):
        if base in base_to_index:
            one_hot[i, base_to_index[base]] = 1.0

    return one_hot

# ============================================================================
# 特征提取（复用 v6 的函数）
# ============================================================================

def reverse_complement_rna(sequence: str) -> str:
    """计算 RNA 反向互补序列"""
    sequence = normalize_rna(sequence)
    complement = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[base] for base in reversed(sequence))

def find_mismatch_positions(spacer: str, target: str) -> list:
    """找出错配位置"""
    spacer_rna = normalize_rna(spacer)
    target_rna = normalize_rna(target)
    if len(spacer_rna) != len(target_rna):
        raise ValueError(f'Length mismatch: {len(spacer_rna)} != {len(target_rna)}')
    return [i for i, (s, t) in enumerate(zip(spacer_rna, target_rna)) if s != t]

def _duplex_energy(spacer_rna: str, target_rna_same_orientation: str) -> float:
    """计算 RNA-DNA 双链自由能"""
    import RNA
    pairing_strand = reverse_complement_rna(target_rna_same_orientation)
    return float(RNA.duplexfold(spacer_rna, pairing_strand).energy)

def extract_features_v6(spacer: str, target: str, mismatch_pos=None) -> list:
    """提取 v6 的 20 个手工特征"""
    spacer_rna = normalize_rna(spacer)
    target_rna = normalize_rna(target)

    # 处理长度
    if len(spacer_rna) > 20:
        spacer_rna = spacer_rna[:20]
    if len(target_rna) > 20:
        target_rna = target_rna[:20]

    if len(spacer_rna) != 20 or len(target_rna) != 20:
        if len(spacer_rna) < 20 or len(target_rna) < 20:
            raise ValueError(f'Sequences too short: spacer={len(spacer_rna)}, target={len(target_rna)}')

    # 找出所有错配位置
    all_mismatch_positions = find_mismatch_positions(spacer_rna, target_rna)

    # 计算 ΔΔG
    dg_duplex = _duplex_energy(spacer_rna, target_rna)
    dg_perfect = _duplex_energy(spacer_rna, spacer_rna)
    ddg = dg_duplex - dg_perfect if all_mismatch_positions else 0.0

    # 第一个错配位置特征
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
        is_edge = float(first_mm_pos <= 2 or first_mm_pos >= 17)
        mismatch_score = {
            ('A','C'): 3.0, ('C','A'): 3.0,
            ('G','A'): 2.0, ('A','G'): 2.0,
            ('C','U'): 1.0, ('U','C'): 1.0,
            ('G','U'): 0.0, ('U','G'): 0.0,
        }.get((spacer_base, target_base), 2.0)
        seed_weight = 1.5 if first_mm_pos <= 8 else 1.0

    pos_ddg_interaction = mismatch_pos_feat * ddg if mismatch_pos_feat >= 0 else 0.0

    # 序列组成
    spacer_gc_content = (spacer_rna.count('G') + spacer_rna.count('C')) / len(spacer_rna)
    target_gc_content = (target_rna.count('G') + target_rna.count('C')) / len(target_rna)
    gc_difference = abs(spacer_gc_content - target_gc_content)

    poly_base_score = 0.0
    for base in 'ACGU':
        for length in [3, 4, 5, 6]:
            pattern = base * length
            if pattern in spacer_rna or pattern in target_rna:
                poly_base_score += (length - 2) * 0.5

    # 局部区域错配
    seed_region_mismatches = sum(1 for pos in all_mismatch_positions if pos < 8)
    pam_proximal_mismatches = sum(1 for pos in all_mismatch_positions if pos < 5)
    middle_region_mismatches = sum(1 for pos in all_mismatch_positions if 5 <= pos < 15)
    distal_region_mismatches = sum(1 for pos in all_mismatch_positions if pos >= 15)

    # 二级结构
    import RNA
    spacer_self_structure = float(RNA.fold(spacer_rna)[1])
    target_self_structure = float(RNA.fold(target_rna)[1])

    # 多错配
    total_mismatches = float(len(all_mismatch_positions))
    if len(all_mismatch_positions) >= 2:
        mismatch_density = max(all_mismatch_positions) - min(all_mismatch_positions)
    else:
        mismatch_density = 0.0

    return [
        mismatch_pos_feat, ddg, is_purine_mismatch, is_transition,
        is_edge, pos_ddg_interaction, mismatch_score, seed_weight,
        spacer_gc_content, target_gc_content, gc_difference, poly_base_score,
        seed_region_mismatches, pam_proximal_mismatches,
        middle_region_mismatches, distal_region_mismatches,
        spacer_self_structure, target_self_structure,
        total_mismatches, mismatch_density
    ]

FEATURE_NAMES = [
    'mismatch_pos', 'ddg', 'is_purine_mismatch', 'is_transition',
    'is_edge', 'pos_ddg_interaction', 'mismatch_score', 'seed_weight',
    'spacer_gc_content', 'target_gc_content', 'gc_difference', 'poly_base_score',
    'seed_region_mismatches', 'pam_proximal_mismatches',
    'middle_region_mismatches', 'distal_region_mismatches',
    'spacer_self_structure', 'target_self_structure',
    'total_mismatches', 'mismatch_density'
]

# ============================================================================
# 数据加载和准备
# ============================================================================

def load_and_prepare_data():
    """加载数据并准备序列编码和手工特征"""
    print("="*80)
    print("加载训练数据")
    print("="*80)

    df = pd.read_csv(DATA_FILE)
    print(f"总数据量: {len(df)} 条")
    print(f"唯一 spacer 数量: {df['spacer_seq'].nunique()}")

    print("\n准备深度学习输入...")
    print("1. 序列 one-hot 编码")
    print("2. 提取手工特征（v6）")

    sequences_encoded = []
    features = []
    failed_indices = []

    for idx, row in df.iterrows():
        try:
            # One-hot 编码序列
            spacer_onehot = one_hot_encode_sequence(row['spacer_seq'], length=20)
            target_onehot = one_hot_encode_sequence(row['target_seq'], length=20)
            # 拼接 spacer + target = (40, 4)
            seq_concat = np.concatenate([spacer_onehot, target_onehot], axis=0)
            sequences_encoded.append(seq_concat)

            # 提取手工特征
            feat = extract_features_v6(
                row['spacer_seq'],
                row['target_seq'],
                row['mismatch_pos']
            )
            features.append(feat)

        except Exception as e:
            print(f"  警告: 第 {idx} 行处理失败: {e}")
            failed_indices.append(idx)
            sequences_encoded.append(np.zeros((40, 4), dtype=np.float32))
            features.append([0] * 20)

        if (idx + 1) % 1000 == 0:
            print(f"  已处理 {idx + 1} / {len(df)} 行")

    if failed_indices:
        print(f"\n⚠️  共有 {len(failed_indices)} 行处理失败，已从数据集中移除")
        df = df.drop(failed_indices).reset_index(drop=True)
        sequences_encoded = [s for i, s in enumerate(sequences_encoded) if i not in failed_indices]
        features = [f for i, f in enumerate(features) if i not in failed_indices]

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

def build_hybrid_model(seq_input_shape=(40, 4), feat_input_shape=(20,)):
    """
    构建混合深度学习模型

    架构：
    - 序列分支：One-hot → CNN → Bi-LSTM
    - 特征分支：Dense layers
    - 融合：Concatenate → Dense → Output
    """
    print("\n" + "="*80)
    print("构建 v7 混合深度学习模型")
    print("="*80)

    # ========== 序列输入分支 ==========
    seq_input = layers.Input(shape=seq_input_shape, name='sequence_input')
    print(f"序列输入: {seq_input_shape}")

    # 1D CNN - 提取局部 motif
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(seq_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    print("  CNN 层: 64 → 128 filters")

    # Bidirectional LSTM - 捕捉长距离依赖
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)
    print("  Bi-LSTM: 64 units")

    sequence_branch = layers.Dense(128, activation='relu', name='seq_encoding')(x)
    sequence_branch = layers.Dropout(0.3)(sequence_branch)
    print("  序列编码输出: 128 维")

    # ========== 特征输入分支 ==========
    feat_input = layers.Input(shape=feat_input_shape, name='feature_input')
    print(f"\n特征输入: {feat_input_shape}")

    # Dense layers
    y = layers.Dense(64, activation='relu')(feat_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(32, activation='relu')(y)
    y = layers.Dropout(0.2)(y)
    print("  特征编码: 64 → 32 维")

    feature_branch = layers.Dense(32, activation='relu', name='feat_encoding')(y)
    print("  特征编码输出: 32 维")

    # ========== 融合层 ==========
    print("\n融合层:")
    concat = layers.Concatenate(name='fusion')([sequence_branch, feature_branch])
    print(f"  拼接: 128 + 32 = 160 维")

    # 全连接层
    z = layers.Dense(128, activation='relu')(concat)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(64, activation='relu')(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(32, activation='relu')(z)
    print("  全连接: 128 → 64 → 32")

    # ========== 输出层 ==========
    output = layers.Dense(1, activation='linear', name='output')(z)
    print("  输出: 1 (回归)")

    # 构建模型
    model = Model(inputs=[seq_input, feat_input], outputs=output, name='v7_hybrid_model')

    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    print("\n模型总结:")
    print(f"  总参数: {model.count_params():,}")
    return model

# ============================================================================
# 训练模型
# ============================================================================

def train_model(model, X_seq_train, X_feat_train, y_train,
                X_seq_val, X_feat_val, y_val, output_dir):
    """训练深度学习模型"""
    print("\n" + "="*80)
    print("训练模型")
    print("="*80)

    # 回调函数
    callbacks = [
        # 早停
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        # 学习率衰减
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        # 保存最佳模型
        ModelCheckpoint(
            filepath=str(output_dir / 'v7_best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    print(f"\n训练集大小: {len(X_seq_train)}")
    print(f"验证集大小: {len(X_seq_val)}")
    print(f"开始训练... (最多 100 epochs)")

    # 训练
    history = model.fit(
        [X_seq_train, X_feat_train],
        y_train,
        validation_data=([X_seq_val, X_feat_val], y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    print("\n✓ 训练完成!")
    return history

# ============================================================================
# 评估模型
# ============================================================================

def evaluate_model(model, X_seq, X_feat, y, dataset_name):
    """评估模型性能"""
    print(f"\n{dataset_name}集性能:")

    # 预测
    y_pred = model.predict([X_seq, X_feat], verbose=0).flatten()

    # 计算指标
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    if len(y) >= 2 and np.std(y) > 0 and np.std(y_pred) > 0:
        pearson_r, pearson_p = pearsonr(y, y_pred)
        spearman_r, spearman_p = spearmanr(y, y_pred)
    else:
        pearson_r, pearson_p = np.nan, np.nan
        spearman_r, spearman_p = np.nan, np.nan

    print(f"  RMSE:      {rmse:.4f}")
    print(f"  MAE:       {mae:.4f}")
    print(f"  R²:        {r2:.4f}")
    print(f"  Pearson:   {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman:  {spearman_r:.4f} (p={spearman_p:.2e})")

    return {
        'rmse': rmse, 'mae': mae, 'r2': r2,
        'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'spearman_r': spearman_r, 'spearman_p': spearman_p,
        'predictions': y_pred
    }

# ============================================================================
# 可视化和保存
# ============================================================================

def plot_training_history(history, output_dir):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    ax1 = axes[0]
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    ax1.set_title('Training History - Loss', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # MAE 曲线
    ax2 = axes[1]
    ax2.plot(history.history['mae'], label='Training MAE', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax2.set_title('Training History - MAE', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'v7_training_history.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 保存训练历史图: {output_dir / 'v7_training_history.png'}")
    plt.close()

def plot_predictions(y_test, y_pred, test_metrics, output_dir):
    """绘制预测结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 散点图
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.5, s=30, edgecolors='black', linewidth=0.3)
    lim = [min(y_test.min(), y_pred.min()) - 0.05,
           max(y_test.max(), y_pred.max()) + 0.05]
    ax1.plot(lim, lim, 'r--', linewidth=2, label='Perfect prediction')
    ax1.set_xlabel('Actual Relative Activity', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Relative Activity', fontsize=12, fontweight='bold')
    ax1.set_title(f'v7 Model Performance (Test Set)\n' +
                  f'Spearman ρ = {test_metrics["spearman_r"]:.3f}, ' +
                  f'R² = {test_metrics["r2"]:.3f}', fontsize=13)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 残差图
    ax2 = axes[1]
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors='black', linewidth=0.3)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Relative Activity', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residual', fontsize=12, fontweight='bold')
    ax2.set_title(f'Residual Plot\n' +
                  f'RMSE = {test_metrics["rmse"]:.4f}, ' +
                  f'MAE = {test_metrics["mae"]:.4f}', fontsize=13)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'v7_model_performance.png', dpi=300, bbox_inches='tight')
    print(f"✓ 保存性能图: {output_dir / 'v7_model_performance.png'}")
    plt.close()

def save_results(model, train_metrics, val_metrics, test_metrics, output_dir):
    """保存模型和结果"""
    print("\n" + "="*80)
    print("保存模型和结果")
    print("="*80)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存模型架构（JSON）
    model_json = model.to_json()
    with open(output_dir / "v7_model_architecture.json", 'w') as f:
        f.write(model_json)
    print("✓ 保存模型架构")

    # 保存模型权重
    model.save(output_dir / "v7_full_model.h5")
    print("✓ 保存完整模型")

    # 保存指标
    metrics = {
        'model_version': 'v7',
        'model_type': 'deep_learning_hybrid',
        'training_date': timestamp,
        'architecture': {
            'sequence_branch': 'One-hot → CNN (64,128) → Bi-LSTM (64) → Dense (128)',
            'feature_branch': 'Dense (64,32)',
            'fusion': 'Concatenate → Dense (128,64,32) → Output (1)'
        },
        'train_metrics': {k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
                         for k, v in train_metrics.items() if k != 'predictions'},
        'val_metrics': {k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
                       for k, v in val_metrics.items() if k != 'predictions'},
        'test_metrics': {k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
                        for k, v in test_metrics.items() if k != 'predictions'},
        'improvements_from_v6': {
            'method': 'Deep learning with sequence encoding + hand-crafted features',
            'key_differences': [
                'Direct sequence input via one-hot encoding',
                'CNN for local motif extraction',
                'Bi-LSTM for long-range dependencies',
                'Hybrid architecture combining DL + traditional features'
            ]
        }
    }

    with open(output_dir / "v7_model_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("✓ 保存指标")

    # 保存测试集预测
    results_df = pd.DataFrame({
        'actual': test_metrics['predictions'],  # 注意这里存的是 y_test
        'predicted': test_metrics['predictions'],
        'residual': test_metrics['predictions'] - test_metrics['predictions']
    })
    # 修正：需要从外部传入 y_test
    print("✓ 保存预测结果")

    print(f"\n所有文件已保存到: {output_dir}")

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print(f"{'='*80}")
    print(f"训练 Cas12a 活性预测模型 v7 (深度学习)")
    print(f"{'='*80}")
    print(f"数据文件: {DATA_FILE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 加载数据
    X_seq, X_feat, y, df = load_and_prepare_data()

    # 2. 标准化特征（只标准化手工特征，序列已经是 one-hot）
    print("\n标准化手工特征...")
    scaler = StandardScaler()
    X_feat_scaled = scaler.fit_transform(X_feat)

    # 保存 scaler
    with open(OUTPUT_DIR / 'v7_feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ 特征标准化完成")

    # 3. 划分数据集 (70% train, 15% val, 15% test)
    print("\n划分数据集 (70% train, 15% val, 15% test)...")
    X_seq_temp, X_seq_test, X_feat_temp, X_feat_test, y_temp, y_test = train_test_split(
        X_seq, X_feat_scaled, y, test_size=0.15, random_state=42
    )
    X_seq_train, X_seq_val, X_feat_train, X_feat_val, y_train, y_val = train_test_split(
        X_seq_temp, X_feat_temp, y_temp, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
    )

    print(f"训练集: {len(X_seq_train)} 条 ({len(X_seq_train)/len(X_seq)*100:.1f}%)")
    print(f"验证集: {len(X_seq_val)} 条 ({len(X_seq_val)/len(X_seq)*100:.1f}%)")
    print(f"测试集: {len(X_seq_test)} 条 ({len(X_seq_test)/len(X_seq)*100:.1f}%)")

    # 4. 构建模型
    model = build_hybrid_model(seq_input_shape=(40, 4), feat_input_shape=(20,))
    model.summary()

    # 5. 训练模型
    history = train_model(
        model, X_seq_train, X_feat_train, y_train,
        X_seq_val, X_feat_val, y_val, OUTPUT_DIR
    )

    # 6. 评估模型
    print("\n" + "="*80)
    print("模型评估")
    print("="*80)

    train_metrics = evaluate_model(model, X_seq_train, X_feat_train, y_train, "训练")
    val_metrics = evaluate_model(model, X_seq_val, X_feat_val, y_val, "验证")
    test_metrics = evaluate_model(model, X_seq_test, X_feat_test, y_test, "测试")

    # 7. 可视化
    plot_training_history(history, OUTPUT_DIR)
    plot_predictions(y_test, test_metrics['predictions'], test_metrics, OUTPUT_DIR)

    # 8. 保存结果（修正版）
    test_metrics_with_y = test_metrics.copy()
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': test_metrics['predictions'],
        'residual': y_test - test_metrics['predictions']
    })
    results_df.to_csv(OUTPUT_DIR / "v7_test_predictions.csv", index=False)

    save_results(model, train_metrics, val_metrics, test_metrics_with_y, OUTPUT_DIR)

    # 9. 性能对比
    print(f"\n{'='*80}")
    print(f"✓ v7 模型训练完成!")
    print(f"{'='*80}")
    print(f"\n性能对比:")
    print(f"  v6 (传统ML):  Spearman ρ = 0.541")
    print(f"  v7 (深度学习): Spearman ρ = {test_metrics['spearman_r']:.3f}")
    if test_metrics['spearman_r'] > 0.541:
        improvement = ((test_metrics['spearman_r'] - 0.541) / 0.541 * 100)
        print(f"  改进: +{improvement:.1f}% ✅")
    else:
        degradation = ((0.541 - test_metrics['spearman_r']) / 0.541 * 100)
        print(f"  变化: -{degradation:.1f}% ⚠️")

    print(f"\n下一步:")
    if test_metrics['spearman_r'] >= 0.60:
        print(f"✅ 性能目标达成！可以使用 v7 模型进行预测")
    elif test_metrics['spearman_r'] > 0.541:
        print(f"✅ 性能有提升，可以使用 v7 或考虑模型融合（v6+v7）")
    else:
        print(f"⚠️ 性能未提升，建议使用 v6 或尝试模型融合")

if __name__ == "__main__":
    main()
