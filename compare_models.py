#!/usr/bin/env python3
"""
对比 v7 原始版本 vs v7_simple（仅公开数据集）的性能
"""

import json
from pathlib import Path

def compare_models():
    print("="*70)
    print("模型性能对比分析")
    print("="*70)

    # 读取v7原始版本
    v7_orig_path = Path("models/v7/v7_model_metrics.json")
    with open(v7_orig_path) as f:
        v7_orig = json.load(f)

    # 读取v7_simple版本
    v7_simple_path = Path("models/v7_public_only_simple/v7_simple_metrics.json")
    with open(v7_simple_path) as f:
        v7_simple = json.load(f)

    print("\n【v7 原始版本】")
    print("-" * 70)
    print("数据量: 9,974条 (包含所有数据)")
    print("特征: 20个 (包含ViennaRNA计算的RNA二级结构特征)")

    v7_test = v7_orig['test_metrics']
    print(f"\n测试集性能:")
    print(f"  Spearman ρ:  {v7_test['spearman_r']:.4f}")
    print(f"  R²:          {v7_test['r2']:.4f}")
    print(f"  RMSE:        {v7_test['rmse']:.4f}")
    print(f"  MAE:         {v7_test['mae']:.4f}")

    print("\n" + "="*70)
    print("【v7_simple 版本】仅公开数据集，无ViennaRNA")
    print("-" * 70)
    print("数据量: 9,894条 (排除原始80条数据)")
    print("特征: 10个简化特征 (不依赖ViennaRNA)")

    v7s_test = v7_simple['test_metrics']
    print(f"\n测试集性能:")
    print(f"  Spearman ρ:  {v7s_test['spearman_r']:.4f}")
    print(f"  R²:          {v7s_test['r2']:.4f}")
    print(f"  RMSE:        {v7s_test['rmse']:.4f}")
    print(f"  MAE:         {v7s_test['mae']:.4f}")

    print("\n" + "="*70)
    print("【性能差异分析】")
    print("-" * 70)

    delta_rho = v7s_test['spearman_r'] - v7_test['spearman_r']
    delta_r2 = v7s_test['r2'] - v7_test['r2']
    delta_rmse = v7s_test['rmse'] - v7_test['rmse']

    pct_rho = (delta_rho / v7_test['spearman_r']) * 100
    pct_r2 = (delta_r2 / v7_test['r2']) * 100
    pct_rmse = (delta_rmse / v7_test['rmse']) * 100

    print(f"\n数据量变化: -80条 (-0.8%)")
    print(f"特征变化: 20个 → 10个 (移除ViennaRNA特征)")

    print(f"\n性能指标变化:")
    print(f"  Δ Spearman ρ: {delta_rho:+.4f}  ({pct_rho:+.2f}%)")
    print(f"  Δ R²:         {delta_r2:+.4f}  ({pct_r2:+.2f}%)")
    print(f"  Δ RMSE:       {delta_rmse:+.4f}  ({pct_rmse:+.2f}%)")

    print("\n" + "="*70)
    print("【结论】")
    print("="*70)

    if abs(pct_rho) < 2:
        print("\n✓ 排除原始80条数据对模型性能影响很小")
        print("✓ 即使移除ViennaRNA复杂特征，简化的10个特征仍然有效")
        print("✓ 深度学习模型主要从大规模公开数据中学习")
        print("\n→ 这说明:")
        print("  1. 原始小样本数据的贡献有限")
        print("  2. 简单的序列特征+深度学习就能达到相近性能")
        print("  3. RNA二级结构特征不是必需的（对这个任务）")
    elif pct_rho < -5:
        print("\n⚠ 性能下降较明显，可能原因:")
        print("  1. ViennaRNA特征对模型有重要贡献")
        print("  2. 原始80条数据提供了关键信息")
    else:
        print("\n✓ 性能略有差异，但整体可接受")

    print("\n" + "="*70)

if __name__ == '__main__':
    compare_models()
