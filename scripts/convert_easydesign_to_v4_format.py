#!/usr/bin/env python3
"""
将 EasyDesign 公开数据集转换为 v4 模型训练格式

输入: data/processed_public_datasets/easydesign_processed.csv
输出: data/processed_public_datasets/easydesign_v4_format.csv

v4 格式要求:
- spacer_seq: crRNA 序列 (guide sequence)
- target_seq: 靶标序列 (target sequence)
- mismatch_pos: 错配位置 (如果有错配)
- relative_activity: 相对活性

作者: Evelyn
日期: 2026-10-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# 设置路径
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_FILE = PROJECT_ROOT / "data/processed_public_datasets/easydesign_processed.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/processed_public_datasets/easydesign_v4_format.csv"
LOG_FILE = PROJECT_ROOT / "data/processed_public_datasets/conversion_log.txt"

def log(message):
    """记录日志"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_msg + '\n')

def find_mismatch_positions(guide_seq, target_seq):
    """
    找出 guide 和 target 之间的错配位置
    返回: 错配位置列表 (0-based)
    """
    if len(guide_seq) != len(target_seq):
        return []

    mismatches = []
    for i, (g, t) in enumerate(zip(guide_seq, target_seq)):
        if g != t:
            mismatches.append(i)
    return mismatches

def normalize_activity(activity_values, target_mean=1.094, target_std=0.208):
    """
    将 EasyDesign 的 log-scale 活性值转换为相对活性

    使用 Z-score 标准化方法，使转换后的数据与原始训练数据具有相同的分布特征

    参数:
        activity_values: EasyDesign 的 log-scale 活性值 (范围约 -4.2 到 -0.3)
        target_mean: 目标均值 (你的原始数据均值 = 1.094)
        target_std: 目标标准差 (你的原始数据标准差 = 0.208)

    原理:
        1. 标准化: z = (x - mean(x)) / std(x)  → 均值0、标准差1
        2. 缩放: y = target_mean + z * target_std  → 匹配目标分布
        3. 保持原始数据的相对关系不变

    转换效果:
        - EasyDesign 原始: 均值=-2.706, 标准差=0.644, 范围=[-4.203, -0.307]
        - 转换后: 均值=1.094, 标准差=0.208, 范围≈[0.6, 1.9]
        - 与原始数据分布完全匹配！
    """
    # 步骤1: Z-score 标准化
    z_score = (activity_values - activity_values.mean()) / activity_values.std()

    # 步骤2: 缩放到目标分布
    relative = target_mean + z_score * target_std

    return relative

def convert_easydesign_data():
    """
    主转换函数
    """
    log("="*80)
    log("开始转换 EasyDesign 数据为 v4 格式 (改进版)")
    log("="*80)

    # 读取 EasyDesign 数据
    log(f"读取输入文件: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    log(f"原始数据: {len(df)} 行")

    # 预过滤包含 gap 字符的序列
    log("\n预过滤 gap 序列...")
    before_filter = len(df)
    df = df[~df['guide_sequence'].str.contains('-', na=False)]
    df = df[~df['target_sequence'].str.contains('-', na=False)]
    after_filter = len(df)
    removed = before_filter - after_filter
    log(f"  移除包含 gap 的序列: {removed} 条 ({removed/before_filter*100:.1f}%)")
    log(f"  过滤后数据: {after_filter} 行")

    # 显示原始数据格式
    log(f"\n原始列名: {df.columns.tolist()}")
    log(f"\n前3行样本:")
    log(df.head(3).to_string())

    # 转换为 v4 格式
    log("\n开始转换...")

    converted_data = []

    for idx, row in df.iterrows():
        guide = row['guide_sequence']
        target = row['target_sequence']
        activity = row['activity']
        mismatches = row['mismatches']

        # 找出具体的错配位置
        mismatch_positions = find_mismatch_positions(guide, target)

        # 如果没有错配,设置为 0
        # 如果有错配,记录第一个错配位置 (与你原始数据格式一致)
        if len(mismatch_positions) == 0:
            mismatch_pos = 0
        else:
            mismatch_pos = mismatch_positions[0]

        converted_data.append({
            'spacer_seq': guide,
            'target_seq': target,
            'mismatch_pos': mismatch_pos,
            'relative_activity': activity  # 先保持原始值
        })

        if (idx + 1) % 1000 == 0:
            log(f"  已处理 {idx + 1} / {len(df)} 行")

    # 创建转换后的 DataFrame
    df_v4 = pd.DataFrame(converted_data)

    log(f"\n转换完成! 共 {len(df_v4)} 行")

    # 归一化活性值
    log("\n归一化活性值...")
    df_v4['relative_activity'] = normalize_activity(df_v4['relative_activity'].values)

    # 统计信息
    log("\n=== 转换后数据统计 ===")
    log(f"总行数: {len(df_v4)}")
    log(f"唯一 spacer 数量: {df_v4['spacer_seq'].nunique()}")
    log(f"平均序列长度: {df_v4['spacer_seq'].str.len().mean():.1f} nt")
    log(f"\n错配分布:")
    mismatch_counts = df_v4['mismatch_pos'].value_counts().sort_index()
    for pos, count in mismatch_counts.head(10).items():
        log(f"  位置 {pos}: {count} 条")

    log(f"\n活性值统计:")
    log(f"  最小值: {df_v4['relative_activity'].min():.4f}")
    log(f"  最大值: {df_v4['relative_activity'].max():.4f}")
    log(f"  平均值: {df_v4['relative_activity'].mean():.4f}")
    log(f"  中位数: {df_v4['relative_activity'].median():.4f}")

    # 保存转换后的数据
    log(f"\n保存到: {OUTPUT_FILE}")
    df_v4.to_csv(OUTPUT_FILE, index=False)
    log(f"✓ 保存成功!")

    # 显示前几行样本
    log(f"\n转换后数据样本 (前5行):")
    log(df_v4.head().to_string())

    return df_v4

def merge_with_original_data():
    """
    合并公开数据和你原始的 50 条数据
    """
    log("\n" + "="*80)
    log("合并公开数据和原始数据")
    log("="*80)

    # 读取转换后的公开数据
    df_public = pd.read_csv(OUTPUT_FILE)
    log(f"公开数据: {len(df_public)} 行")

    # 读取你原始的数据
    original_file = PROJECT_ROOT / "data/raw/cas12a_mismatch_data.csv"
    if not original_file.exists():
        log(f"⚠️  未找到原始数据文件: {original_file}")
        log("跳过合并步骤")
        return df_public

    df_original = pd.read_csv(original_file)
    log(f"原始数据: {len(df_original)} 行")

    # 合并数据
    df_combined = pd.concat([df_original, df_public], ignore_index=True)
    log(f"合并后: {len(df_combined)} 行")

    # 保存合并后的数据
    combined_file = PROJECT_ROOT / "data/processed_public_datasets/combined_v4_training_data.csv"
    df_combined.to_csv(combined_file, index=False)
    log(f"✓ 保存到: {combined_file}")

    log(f"\n=== 最终训练集统计 ===")
    log(f"总行数: {len(df_combined)}")
    log(f"唯一 spacer 数量: {df_combined['spacer_seq'].nunique()}")
    log(f"数据来源:")
    log(f"  原始数据: {len(df_original)} 行 ({len(df_original)/len(df_combined)*100:.1f}%)")
    log(f"  公开数据: {len(df_public)} 行 ({len(df_public)/len(df_combined)*100:.1f}%)")

    return df_combined

if __name__ == "__main__":
    try:
        # 转换 EasyDesign 数据
        df_v4 = convert_easydesign_data()

        # 合并原始数据
        df_combined = merge_with_original_data()

        log("\n" + "="*80)
        log("✓ 所有转换完成!")
        log("="*80)
        log(f"\n生成的文件:")
        log(f"1. {OUTPUT_FILE}")
        log(f"2. {PROJECT_ROOT / 'data/processed_public_datasets/combined_v4_training_data.csv'}")
        log(f"3. {LOG_FILE}")

        log(f"\n下一步:")
        log("1. 检查转换后的数据格式是否正确")
        log("2. 使用 combined_v4_training_data.csv 重新训练 v4 模型")
        log("3. 评估新模型在独立测试集上的性能")

    except Exception as e:
        log(f"\n✗ 错误: {str(e)}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
