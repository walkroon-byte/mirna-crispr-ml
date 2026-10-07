#!/usr/bin/env python3
"""
检查 v7_public_only 训练状态并展示结果对比
"""

import json
import time
from pathlib import Path

def check_and_display():
    metrics_file = Path("models/v7_public_only/v7_public_only_model_metrics.json")

    if not metrics_file.exists():
        print("⏳ 训练尚未完成，请稍后再运行此脚本查看结果")
        return False

    print("="*70)
    print("✓ 训练完成！正在生成结果对比...")
    print("="*70)

    # 读取两个版本的指标
    with open(metrics_file) as f:
        public_data = json.load(f)

    with open('models/v7/v7_model_metrics.json') as f:
        orig_data = json.load(f)

    test_public = public_data['test_metrics']
    test_orig = orig_data['test_metrics']

    # 显示结果
    print("\n【模型版本对比】")
    print("="*70)

    print("\n1. v7 原始版本 (包含所有数据)")
    print("-" * 70)
    print(f"   数据量: 9,974 条")
    print(f"   Spearman ρ:  {test_orig['spearman_r']:.4f}")
    print(f"   R²:          {test_orig['r2']:.4f}")
    print(f"   RMSE:        {test_orig['rmse']:.4f}")

    print("\n2. v7_public_only (仅公开数据集)")
    print("-" * 70)
    print(f"   数据量: 9,894 条 (排除原始标注的80条)")
    print(f"   Spearman ρ:  {test_public['spearman_r']:.4f}")
    print(f"   R²:          {test_public['r2']:.4f}")
    print(f"   RMSE:        {test_public['rmse']:.4f}")

    # 计算差异
    delta_rho = test_public['spearman_r'] - test_orig['spearman_r']
    delta_r2 = test_public['r2'] - test_orig['r2']
    delta_rmse = test_public['rmse'] - test_orig['rmse']

    pct_rho = (delta_rho / test_orig['spearman_r']) * 100
    pct_r2 = (delta_r2 / test_orig['r2']) * 100
    pct_rmse = (delta_rmse / test_orig['rmse']) * 100

    print("\n【性能差异分析】")
    print("="*70)
    print(f"\n数据量变化: -80条 (-0.8%)")
    print(f"\n性能指标变化:")
    print(f"   Δ Spearman ρ: {delta_rho:+.4f}  ({pct_rho:+.2f}%)")
    print(f"   Δ R²:         {delta_r2:+.4f}  ({pct_r2:+.2f}%)")
    print(f"   Δ RMSE:       {delta_rmse:+.4f}  ({pct_rmse:+.2f}%)")

    # 结论
    print("\n【结论】")
    print("="*70)

    if abs(pct_rho) < 1:
        print("✓ 排除原始80条数据对模型性能影响极小 (<1%)")
        print("✓ 说明深度学习模型主要从大规模公开数据集中学习特征")
        print("✓ 原始小样本标注数据的贡献可以忽略不计")
        print("\n→ 这验证了深度学习在大数据场景下的优势：")
        print("  模型从近万条公开数据中自动学习的模式")
        print("  远比几十条手工标注数据的贡献更重要")
    elif pct_rho < -2:
        print("⚠ 排除原始数据后性能有明显下降")
        print("  原始80条标注数据对模型有一定贡献")
    elif pct_rho > 0:
        print("✓ 排除原始数据后性能略有提升")
        print("  可能原始数据存在一定噪声")

    print("\n" + "="*70)
    return True

if __name__ == '__main__':
    check_and_display()
