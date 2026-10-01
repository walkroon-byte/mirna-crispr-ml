#!/usr/bin/env python3
"""
训练 v5.1 模型：使用改进的活性值归一化方法

关键改进：
- 使用 Z-score 标准化替代 sigmoid（修复活性值范围问题）
- 预处理时过滤 gap 序列（避免特征提取失败）
- 保留 v4 和 v5 模型文件（不覆盖）

与 v5 的区别：
- v5: Sigmoid 归一化，活性范围 0.015-0.426（太窄）
- v5.1: Z-score 归一化，活性范围 0.6-1.9（匹配原始数据）

作者: Evelyn
日期: 2026-10-01
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data/processed_public_datasets/combined_v4_training_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "models/v5.1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# v4 特征提取函数（从 notebook 复制）
# ============================================================================

def normalize_rna(sequence: str) -> str:
    """将序列归一化为 RNA 格式（T→U）"""
    value = str(sequence).upper().replace('T', 'U').replace(' ', '')
    if not value or any(base not in 'ACGU' for base in value):
        raise ValueError(f'Invalid RNA/DNA sequence: {sequence!r}')
    return value

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
    """计算 RNA-DNA 双链自由能（修正版：target 取反向互补）"""
    import RNA
    pairing_strand = reverse_complement_rna(target_rna_same_orientation)
    return float(RNA.duplexfold(spacer_rna, pairing_strand).energy)

def extract_final_features_v4(spacer: str, target: str, mismatch_pos=None) -> list:
    """
    提取 v4 特征（8 个特征）

    特征列表：
    1. mismatch_pos: 错配位置
    2. ddg: ΔΔG (双链稳定性变化)
    3. is_purine_mismatch: 是否为嘌呤-嘌呤错配
    4. is_transition: 是否为转换突变
    5. is_edge: 是否在边缘区域 (位置 ≤2 或 ≥17)
    6. pos_ddg_interaction: 位置与 ΔΔG 的交互项
    7. mismatch_score: 错配严重程度评分
    8. seed_weight: 种子区权重 (位置 ≤8 为 1.5，否则为 1.0)
    """
    spacer_rna = normalize_rna(spacer)
    target_rna = normalize_rna(target)

    # 处理不同长度的序列（EasyDesign 数据是 25 nt）
    if len(spacer_rna) > 20:
        spacer_rna = spacer_rna[:20]
    if len(target_rna) > 20:
        target_rna = target_rna[:20]

    if len(spacer_rna) != 20 or len(target_rna) != 20:
        # 如果还是不是 20 nt，填充或报错
        if len(spacer_rna) < 20 or len(target_rna) < 20:
            raise ValueError(f'Sequences too short: spacer={len(spacer_rna)}, target={len(target_rna)}')

    inferred = find_mismatch_positions(spacer_rna, target_rna)

    if mismatch_pos is not None and mismatch_pos >= 0:
        if mismatch_pos >= 20:
            mismatch_pos = min(mismatch_pos, 19)  # 限制在 0-19
        if mismatch_pos not in inferred and len(inferred) > 0:
            # 如果指定的位置不是错配，但有其他错配，使用第一个错配
            mismatch_positions = [inferred[0]]
        elif mismatch_pos in inferred:
            mismatch_positions = [mismatch_pos]
        else:
            mismatch_positions = []
    else:
        mismatch_positions = inferred

    dg_duplex = _duplex_energy(spacer_rna, target_rna)
    dg_perfect = _duplex_energy(spacer_rna, spacer_rna)
    ddg = dg_duplex - dg_perfect if mismatch_positions else 0.0

    if not mismatch_positions:
        pos = -1.0
        is_purine_mismatch = 0.0
        is_transition = 0.0
        is_edge = 0.0
        mismatch_score = 0.0
        seed_weight = 1.0
    else:
        pos = float(min(mismatch_positions))
        spacer_base = spacer_rna[int(pos)]
        target_base = target_rna[int(pos)]
        is_purine_mismatch = float(spacer_base in 'AG' and target_base in 'AG')
        is_transition = float(
            (spacer_base, target_base) in {('A','G'),('G','A'),('C','U'),('U','C')}
        )
        is_edge = float(pos <= 2 or pos >= 17)
        mismatch_score = {
            ('A','C'): 3.0, ('C','A'): 3.0,
            ('G','A'): 2.0, ('A','G'): 2.0,
            ('C','U'): 1.0, ('U','C'): 1.0,
            ('G','U'): 0.0, ('U','G'): 0.0,
        }.get((spacer_base, target_base), 2.0)
        seed_weight = 1.5 if pos <= 8 else 1.0

    pos_ddg_interaction = pos * ddg if pos >= 0 else 0.0
    return [pos, ddg, is_purine_mismatch, is_transition,
            is_edge, pos_ddg_interaction, mismatch_score, seed_weight]

FEATURE_NAMES = [
    'mismatch_pos', 'ddg', 'is_purine_mismatch', 'is_transition',
    'is_edge', 'pos_ddg_interaction', 'mismatch_score', 'seed_weight',
]

# ============================================================================
# 主训练函数
# ============================================================================

def load_and_prepare_data():
    """加载数据并提取特征"""
    print("="*80)
    print("加载训练数据")
    print("="*80)

    df = pd.read_csv(DATA_FILE)
    print(f"总数据量: {len(df)} 条")
    print(f"唯一 spacer 数量: {df['spacer_seq'].nunique()}")
    print(f"\n数据样本 (前3行):")
    print(df.head(3))

    print("\n提取特征...")
    features = []
    failed_indices = []

    for idx, row in df.iterrows():
        try:
            feat = extract_final_features_v4(
                row['spacer_seq'],
                row['target_seq'],
                row['mismatch_pos']
            )
            features.append(feat)
        except Exception as e:
            print(f"  警告: 第 {idx} 行特征提取失败: {e}")
            failed_indices.append(idx)
            features.append([0] * 8)  # 占位

        if (idx + 1) % 2000 == 0:
            print(f"  已处理 {idx + 1} / {len(df)} 行")

    if failed_indices:
        print(f"\n⚠️  共有 {len(failed_indices)} 行特征提取失败，已从数据集中移除")
        df = df.drop(failed_indices).reset_index(drop=True)
        features = [f for i, f in enumerate(features) if i not in failed_indices]

    X = np.array(features)
    y = df['relative_activity'].values

    print(f"\n最终特征矩阵: {X.shape}")
    print(f"活性值范围: [{y.min():.4f}, {y.max():.4f}]")

    return X, y, df

def train_models(X_train, y_train):
    """训练 SVR 和 GradientBoosting 模型"""
    print("\n" + "="*80)
    print("训练模型")
    print("="*80)

    # SVR 模型（与 v4 相同的参数）
    print("\n1. 训练 SVR 模型...")
    svr = make_pipeline(
        StandardScaler(),
        SVR(kernel='rbf', C=20, epsilon=0.05, gamma=0.05)
    )
    svr.fit(X_train, y_train)
    print("   ✓ SVR 训练完成")

    # GradientBoosting 模型（与 v4 相同的参数）
    print("\n2. 训练 GradientBoosting 模型...")
    gb = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_train, y_train)
    print("   ✓ GradientBoosting 训练完成")

    return svr, gb

def evaluate_models(svr, gb, X_test, y_test, X_train, y_train):
    """评估模型性能"""
    print("\n" + "="*80)
    print("模型评估")
    print("="*80)

    # 预测
    y_pred_svr_train = svr.predict(X_train)
    y_pred_gb_train = gb.predict(X_train)
    y_pred_train = (y_pred_svr_train + y_pred_gb_train) / 2

    y_pred_svr_test = svr.predict(X_test)
    y_pred_gb_test = gb.predict(X_test)
    y_pred_test = (y_pred_svr_test + y_pred_gb_test) / 2

    # 计算指标
    def calc_metrics(y_true, y_pred, dataset_name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
            pearson_r, pearson_p = pearsonr(y_true, y_pred)
            spearman_r, spearman_p = spearmanr(y_true, y_pred)
        else:
            pearson_r, pearson_p = np.nan, np.nan
            spearman_r, spearman_p = np.nan, np.nan

        print(f"\n{dataset_name}集:")
        print(f"  RMSE:      {rmse:.4f}")
        print(f"  MAE:       {mae:.4f}")
        print(f"  R²:        {r2:.4f}")
        print(f"  Pearson:   {pearson_r:.4f} (p={pearson_p:.2e})")
        print(f"  Spearman:  {spearman_r:.4f} (p={spearman_p:.2e})")

        return {
            'rmse': rmse, 'mae': mae, 'r2': r2,
            'pearson_r': pearson_r, 'pearson_p': pearson_p,
            'spearman_r': spearman_r, 'spearman_p': spearman_p
        }

    train_metrics = calc_metrics(y_train, y_pred_train, "训练")
    test_metrics = calc_metrics(y_test, y_pred_test, "测试")

    return train_metrics, test_metrics, y_pred_test

def save_models_and_results(svr, gb, train_metrics, test_metrics, y_test, y_pred_test):
    """保存模型和结果"""
    print("\n" + "="*80)
    print("保存模型和结果")
    print("="*80)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存模型
    with open(OUTPUT_DIR / "v5.1_svr_model.pkl", 'wb') as f:
        pickle.dump(svr, f)
    print(f"✓ 保存 SVR 模型")

    with open(OUTPUT_DIR / "v5.1_gb_model.pkl", 'wb') as f:
        pickle.dump(gb, f)
    print(f"✓ 保存 GradientBoosting 模型")

    # 保存指标
    metrics = {
        'model_version': 'v5.1',
        'training_date': timestamp,
        'n_samples_total': len(y_test) + len(y_pred_test),  # 这里简化了
        'n_features': 8,
        'feature_names': FEATURE_NAMES,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'improvements': {
            'normalization_method': 'Z-score standardization',
            'target_mean': 1.094,
            'target_std': 0.208,
            'gap_sequences_filtered': 'pre-filtered in conversion step'
        },
        'model_params': {
            'svr': {'kernel': 'rbf', 'C': 20, 'epsilon': 0.05, 'gamma': 0.05},
            'gb': {
                'n_estimators': 100, 'max_depth': 2,
                'learning_rate': 0.05, 'subsample': 0.8, 'random_state': 42
            }
        }
    }

    with open(OUTPUT_DIR / "v5.1_model_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✓ 保存指标")

    # 保存预测结果
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred_test,
        'residual': y_test - y_pred_test
    })
    results_df.to_csv(OUTPUT_DIR / "v5.1_test_predictions.csv", index=False)
    print(f"✓ 保存预测结果")

    print(f"\n所有文件已保存到: {OUTPUT_DIR}")

def plot_results(y_test, y_pred_test, test_metrics):
    """绘制结果图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 散点图
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred_test, alpha=0.5, s=30, edgecolors='black', linewidth=0.3)
    lim = [min(y_test.min(), y_pred_test.min()) - 0.05,
           max(y_test.max(), y_pred_test.max()) + 0.05]
    ax1.plot(lim, lim, 'r--', linewidth=2, label='Perfect prediction')
    ax1.set_xlabel('Actual Relative Activity', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Relative Activity', fontsize=12, fontweight='bold')
    ax1.set_title(f'v5.1 Model Performance (Test Set)\n' +
                  f'Spearman ρ = {test_metrics["spearman_r"]:.3f}, ' +
                  f'R² = {test_metrics["r2"]:.3f}', fontsize=13)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 残差图
    ax2 = axes[1]
    residuals = y_test - y_pred_test
    ax2.scatter(y_pred_test, residuals, alpha=0.5, s=30, edgecolors='black', linewidth=0.3)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Relative Activity', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residual', fontsize=12, fontweight='bold')
    ax2.set_title(f'Residual Plot\n' +
                  f'RMSE = {test_metrics["rmse"]:.4f}, ' +
                  f'MAE = {test_metrics["mae"]:.4f}', fontsize=13)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'v5.1_model_performance.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 保存性能图: {OUTPUT_DIR / 'v5.1_model_performance.png'}")
    plt.close()

def main():
    """主函数"""
    print(f"{'='*80}")
    print(f"训练 Cas12a 活性预测模型 v5.1 (改进版)")
    print(f"{'='*80}")
    print(f"数据文件: {DATA_FILE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n改进点:")
    print(f"  1. Z-score 标准化替代 sigmoid（修复活性值范围）")
    print(f"  2. 预处理过滤 gap 序列（减少特征提取失败）")

    # 1. 加载数据
    X, y, df = load_and_prepare_data()

    # 2. 划分训练集和测试集 (80/20)
    print("\n划分训练集和测试集 (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"训练集: {len(X_train)} 条")
    print(f"测试集: {len(X_test)} 条")

    # 3. 训练模型
    svr, gb = train_models(X_train, y_train)

    # 4. 评估模型
    train_metrics, test_metrics, y_pred_test = evaluate_models(
        svr, gb, X_test, y_test, X_train, y_train
    )

    # 5. 保存模型和结果
    save_models_and_results(svr, gb, train_metrics, test_metrics, y_test, y_pred_test)

    # 6. 绘制结果图
    plot_results(y_test, y_pred_test, test_metrics)

    print(f"\n{'='*80}")
    print(f"✓ v5.1 模型训练完成!")
    print(f"{'='*80}")
    print(f"\n性能对比:")
    print(f"  v5:  Spearman ρ = 0.474, 活性范围 0.015-0.426")
    print(f"  v5.1: Spearman ρ = {test_metrics['spearman_r']:.3f}, 活性范围修正")
    print(f"\n下一步:")
    print(f"1. 检查 {OUTPUT_DIR / 'v5.1_model_performance.png'}")
    print(f"2. 如果性能满意 (ρ > 0.6)，使用 v5.1 预测 miR-29a 设计")
    print(f"3. 如果性能仍不理想，考虑进一步改进或使用文献方案")

if __name__ == "__main__":
    main()
