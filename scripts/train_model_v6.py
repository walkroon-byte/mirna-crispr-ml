#!/usr/bin/env python3
"""
训练 v6 模型：改进特征工程（从 8 个特征扩展到 20 个）

关键改进：
- 新增 12 个特征：序列组成、局部区域、二级结构、多错配
- 保留 v5.1 的 Z-score 归一化方法
- 保留 v5.1 的模型架构（SVR + GradientBoosting）

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
OUTPUT_DIR = PROJECT_ROOT / "models/v6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 基础函数（从 v5.1 继承）
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
    """计算 RNA-DNA 双链自由能"""
    import RNA
    pairing_strand = reverse_complement_rna(target_rna_same_orientation)
    return float(RNA.duplexfold(spacer_rna, pairing_strand).energy)

# ============================================================================
# v6 新特征提取函数
# ============================================================================

def extract_features_v6(spacer: str, target: str, mismatch_pos=None) -> list:
    """
    提取 v6 特征（20 个特征）

    特征列表：
    === v5.1 原有特征 (1-8) ===
    1. mismatch_pos: 第一个错配位置
    2. ddg: ΔΔG (双链稳定性变化)
    3. is_purine_mismatch: 是否为嘌呤-嘌呤错配
    4. is_transition: 是否为转换突变
    5. is_edge: 是否在边缘区域
    6. pos_ddg_interaction: 位置与 ΔΔG 的交互项
    7. mismatch_score: 错配严重程度评分
    8. seed_weight: 种子区权重

    === v6 新增特征 (9-20) ===
    【A. 序列组成 (9-12)】
    9. spacer_gc_content: spacer 的 GC 含量
    10. target_gc_content: target 的 GC 含量
    11. gc_difference: GC 含量差异
    12. poly_base_score: 连续相同碱基惩罚

    【B. 局部区域错配 (13-16)】
    13. seed_region_mismatches: 种子区 (1-8) 错配数
    14. pam_proximal_mismatches: PAM 近端 (1-5) 错配数
    15. middle_region_mismatches: 中间区 (6-15) 错配数
    16. distal_region_mismatches: 远端区 (16-20) 错配数

    【C. 二级结构 (17-18)】
    17. spacer_self_structure: spacer 自身二级结构 MFE
    18. target_self_structure: target 自身二级结构 MFE

    【D. 多错配模式 (19-20)】
    19. total_mismatches: 总错配数
    20. mismatch_density: 错配分布密度
    """
    spacer_rna = normalize_rna(spacer)
    target_rna = normalize_rna(target)

    # 处理不同长度的序列
    if len(spacer_rna) > 20:
        spacer_rna = spacer_rna[:20]
    if len(target_rna) > 20:
        target_rna = target_rna[:20]

    if len(spacer_rna) != 20 or len(target_rna) != 20:
        if len(spacer_rna) < 20 or len(target_rna) < 20:
            raise ValueError(f'Sequences too short: spacer={len(spacer_rna)}, target={len(target_rna)}')

    # 找出所有错配位置
    all_mismatch_positions = find_mismatch_positions(spacer_rna, target_rna)

    # === 特征 1-8: v5.1 原有特征 ===

    # 计算 ΔΔG
    dg_duplex = _duplex_energy(spacer_rna, target_rna)
    dg_perfect = _duplex_energy(spacer_rna, spacer_rna)
    ddg = dg_duplex - dg_perfect if all_mismatch_positions else 0.0

    # 第一个错配位置的特征
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

    # === 特征 9-12: 序列组成 ===

    # 9. spacer GC 含量
    spacer_gc_content = (spacer_rna.count('G') + spacer_rna.count('C')) / len(spacer_rna)

    # 10. target GC 含量
    target_gc_content = (target_rna.count('G') + target_rna.count('C')) / len(target_rna)

    # 11. GC 差异
    gc_difference = abs(spacer_gc_content - target_gc_content)

    # 12. 连续相同碱基惩罚
    poly_base_score = 0.0
    for base in 'ACGU':
        for length in [3, 4, 5, 6]:  # 检测 3-6 个连续碱基
            pattern = base * length
            if pattern in spacer_rna or pattern in target_rna:
                poly_base_score += (length - 2) * 0.5  # 越长惩罚越重

    # === 特征 13-16: 局部区域错配 ===

    # 13. 种子区 (0-7) 错配数
    seed_region_mismatches = sum(1 for pos in all_mismatch_positions if pos < 8)

    # 14. PAM 近端 (0-4) 错配数
    pam_proximal_mismatches = sum(1 for pos in all_mismatch_positions if pos < 5)

    # 15. 中间区 (5-14) 错配数
    middle_region_mismatches = sum(1 for pos in all_mismatch_positions if 5 <= pos < 15)

    # 16. 远端区 (15-19) 错配数
    distal_region_mismatches = sum(1 for pos in all_mismatch_positions if pos >= 15)

    # === 特征 17-18: 二级结构 ===

    import RNA

    # 17. spacer 自身二级结构 MFE（最小自由能）
    spacer_self_structure = float(RNA.fold(spacer_rna)[1])

    # 18. target 自身二级结构 MFE
    target_self_structure = float(RNA.fold(target_rna)[1])

    # === 特征 19-20: 多错配模式 ===

    # 19. 总错配数
    total_mismatches = float(len(all_mismatch_positions))

    # 20. 错配密度（错配分布的紧密程度）
    if len(all_mismatch_positions) >= 2:
        mismatch_density = max(all_mismatch_positions) - min(all_mismatch_positions)
    else:
        mismatch_density = 0.0

    # 返回所有 20 个特征
    return [
        # 1-8: v5.1 特征
        mismatch_pos_feat, ddg, is_purine_mismatch, is_transition,
        is_edge, pos_ddg_interaction, mismatch_score, seed_weight,
        # 9-12: 序列组成
        spacer_gc_content, target_gc_content, gc_difference, poly_base_score,
        # 13-16: 局部区域错配
        seed_region_mismatches, pam_proximal_mismatches,
        middle_region_mismatches, distal_region_mismatches,
        # 17-18: 二级结构
        spacer_self_structure, target_self_structure,
        # 19-20: 多错配模式
        total_mismatches, mismatch_density
    ]

FEATURE_NAMES = [
    # 1-8
    'mismatch_pos', 'ddg', 'is_purine_mismatch', 'is_transition',
    'is_edge', 'pos_ddg_interaction', 'mismatch_score', 'seed_weight',
    # 9-12
    'spacer_gc_content', 'target_gc_content', 'gc_difference', 'poly_base_score',
    # 13-16
    'seed_region_mismatches', 'pam_proximal_mismatches',
    'middle_region_mismatches', 'distal_region_mismatches',
    # 17-18
    'spacer_self_structure', 'target_self_structure',
    # 19-20
    'total_mismatches', 'mismatch_density'
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

    print("\n提取 v6 特征（20 个特征）...")
    print("注意: 二级结构计算可能需要几分钟...")
    features = []
    failed_indices = []

    for idx, row in df.iterrows():
        try:
            feat = extract_features_v6(
                row['spacer_seq'],
                row['target_seq'],
                row['mismatch_pos']
            )
            features.append(feat)
        except Exception as e:
            print(f"  警告: 第 {idx} 行特征提取失败: {e}")
            failed_indices.append(idx)
            features.append([0] * 20)  # 占位

        if (idx + 1) % 1000 == 0:
            print(f"  已处理 {idx + 1} / {len(df)} 行")

    if failed_indices:
        print(f"\n⚠️  共有 {len(failed_indices)} 行特征提取失败，已从数据集中移除")
        df = df.drop(failed_indices).reset_index(drop=True)
        features = [f for i, f in enumerate(features) if i not in failed_indices]

    X = np.array(features)
    y = df['relative_activity'].values

    print(f"\n最终特征矩阵: {X.shape}")
    print(f"活性值范围: [{y.min():.4f}, {y.max():.4f}]")

    # 显示特征统计
    print(f"\n特征统计（前5行样本）:")
    feature_df = pd.DataFrame(X[:5], columns=FEATURE_NAMES)
    print(feature_df.to_string())

    return X, y, df

def train_models(X_train, y_train):
    """训练 SVR 和 GradientBoosting 模型"""
    print("\n" + "="*80)
    print("训练模型")
    print("="*80)

    # SVR 模型（与 v5.1 相同的参数）
    print("\n1. 训练 SVR 模型...")
    svr = make_pipeline(
        StandardScaler(),
        SVR(kernel='rbf', C=20, epsilon=0.05, gamma=0.05)
    )
    svr.fit(X_train, y_train)
    print("   ✓ SVR 训练完成")

    # GradientBoosting 模型（与 v5.1 相同的参数）
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

    return train_metrics, test_metrics, y_pred_test, gb

def analyze_feature_importance(gb, output_dir):
    """分析特征重要性"""
    print("\n" + "="*80)
    print("特征重要性分析")
    print("="*80)

    importances = gb.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n特征重要性排名（Top 10）:")
    for i in range(min(10, len(FEATURE_NAMES))):
        idx = indices[i]
        print(f"  {i+1}. {FEATURE_NAMES[idx]}: {importances[idx]:.4f}")

    # 绘制特征重要性图
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(FEATURE_NAMES))
    ax.barh(y_pos, importances[indices], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([FEATURE_NAMES[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontweight='bold')
    ax.set_title('v6 Model Feature Importance', fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'v6_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 保存特征重要性图: {output_dir / 'v6_feature_importance.png'}")
    plt.close()

    return importances

def save_models_and_results(svr, gb, train_metrics, test_metrics, y_test, y_pred_test, feature_importances):
    """保存模型和结果"""
    print("\n" + "="*80)
    print("保存模型和结果")
    print("="*80)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存模型
    with open(OUTPUT_DIR / "v6_svr_model.pkl", 'wb') as f:
        pickle.dump(svr, f)
    print(f"✓ 保存 SVR 模型")

    with open(OUTPUT_DIR / "v6_gb_model.pkl", 'wb') as f:
        pickle.dump(gb, f)
    print(f"✓ 保存 GradientBoosting 模型")

    # 保存指标
    metrics = {
        'model_version': 'v6',
        'training_date': timestamp,
        'n_features': 20,
        'feature_names': FEATURE_NAMES,
        'feature_importances': {
            name: float(imp) for name, imp in zip(FEATURE_NAMES, feature_importances)
        },
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'improvements_from_v5.1': {
            'new_features': [
                'spacer_gc_content', 'target_gc_content', 'gc_difference', 'poly_base_score',
                'seed_region_mismatches', 'pam_proximal_mismatches',
                'middle_region_mismatches', 'distal_region_mismatches',
                'spacer_self_structure', 'target_self_structure',
                'total_mismatches', 'mismatch_density'
            ],
            'rationale': 'Enhanced feature engineering with sequence composition, local region analysis, secondary structure, and multi-mismatch support'
        },
        'model_params': {
            'svr': {'kernel': 'rbf', 'C': 20, 'epsilon': 0.05, 'gamma': 0.05},
            'gb': {
                'n_estimators': 100, 'max_depth': 2,
                'learning_rate': 0.05, 'subsample': 0.8, 'random_state': 42
            }
        }
    }

    with open(OUTPUT_DIR / "v6_model_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✓ 保存指标")

    # 保存预测结果
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred_test,
        'residual': y_test - y_pred_test
    })
    results_df.to_csv(OUTPUT_DIR / "v6_test_predictions.csv", index=False)
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
    ax1.set_title(f'v6 Model Performance (Test Set)\n' +
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
    plt.savefig(OUTPUT_DIR / 'v6_model_performance.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ 保存性能图: {OUTPUT_DIR / 'v6_model_performance.png'}")
    plt.close()

def main():
    """主函数"""
    print(f"{'='*80}")
    print(f"训练 Cas12a 活性预测模型 v6 (特征工程改进)")
    print(f"{'='*80}")
    print(f"数据文件: {DATA_FILE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n改进点:")
    print(f"  1. 特征数量: 8 → 20 个")
    print(f"  2. 新增序列组成、局部区域、二级结构、多错配特征")
    print(f"  3. 保留 v5.1 的 Z-score 归一化和模型架构")

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
    train_metrics, test_metrics, y_pred_test, gb_model = evaluate_models(
        svr, gb, X_test, y_test, X_train, y_train
    )

    # 5. 特征重要性分析
    feature_importances = analyze_feature_importance(gb_model, OUTPUT_DIR)

    # 6. 保存模型和结果
    save_models_and_results(svr, gb, train_metrics, test_metrics, y_test, y_pred_test, feature_importances)

    # 7. 绘制结果图
    plot_results(y_test, y_pred_test, test_metrics)

    print(f"\n{'='*80}")
    print(f"✓ v6 模型训练完成!")
    print(f"{'='*80}")
    print(f"\n性能对比:")
    print(f"  v5.1: 8 特征, Spearman ρ = 0.515")
    print(f"  v6:   20 特征, Spearman ρ = {test_metrics['spearman_r']:.3f}")
    print(f"  改进: {((test_metrics['spearman_r'] - 0.515) / 0.515 * 100):.1f}%")
    print(f"\n下一步:")
    print(f"1. 检查特征重要性图，了解哪些新特征最有用")
    print(f"2. 如果性能提升明显，使用 v6 预测 miR-29a 设计")
    print(f"3. 如果某些特征不重要，考虑特征选择（创建 v6.1）")

if __name__ == "__main__":
    main()
