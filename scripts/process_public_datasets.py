#!/usr/bin/env python3
"""
处理公开数据集并提取 Cas12a 活性数据

从三个主要来源提取数据：
1. NucleaSeq - Supplemental_File_1.xlsx
2. Sashital Lab - KMlib*.xlsx 文件
3. EasyDesign - Table S1-S4.xlsx

输出统一格式的 CSV 文件用于后续模型训练

作者: Evelyn
日期: 2026-10-01
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# 设置路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_public_datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 日志文件
LOG_FILE = OUTPUT_DIR / f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log(message):
    """记录日志"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_msg + '\n')

# ============================================================================
# 数据集 1: NucleaSeq
# ============================================================================
def process_nucleaseq():
    """
    处理 NucleaSeq 数据
    Supplemental_File_1.xlsx 包含超过1万条靶标的切割动力学数据

    注意：这个文件是 Cas9 数据，不是 Cas12a！
    第一行是说明文字，第二行才是真正的列名
    """
    log("="*80)
    log("处理 NucleaSeq 数据集")
    log("="*80)

    input_file = DATA_DIR / "nucleaseq" / "Supplemental_File_1.xlsx"

    if not input_file.exists():
        log(f"✗ 文件不存在: {input_file}")
        return None

    try:
        # 读取 Excel 文件，跳过第一行说明
        log(f"读取文件: {input_file.name}")
        df = pd.read_excel(input_file, sheet_name=0, header=1)

        log(f"原始数据: {len(df)} 行, {len(df.columns)} 列")
        log(f"列名: {list(df.columns)[:5]}")

        # ⚠️ 重要发现：NucleaSeq 是 Cas9 数据，不是 Cas12a 数据！
        # Cas9 使用 NGG PAM (3' 端)
        # Cas12a 使用 TTTV PAM (5' 端)
        # 这个数据集不适合我们的 Cas12a 项目

        log("⚠️  警告: NucleaSeq 数据集是 Cas9 数据，不是 Cas12a 数据")
        log("   Cas9 PAM: NGG (3' 端)")
        log("   Cas12a PAM: TTTV (5' 端)")
        log("   这个数据集不能直接用于 Cas12a 模型训练")
        log("   跳过此数据集")

        return None

    except Exception as e:
        log(f"✗ 处理失败: {str(e)}")
        return None

# ============================================================================
# 数据集 2: Sashital Lab
# ============================================================================
def process_sashital():
    """
    处理 Sashital Lab 数据
    KMlib*.xlsx 包含体外质粒文库数据

    列名:
    - MM: 错配模式 (mismatch pattern)，如 'r1:C7A' 表示位置7的C→A错配
    - N: 读数计数 (read count)
    - SC: 选择系数 (selection coefficient)，即活性分数
    """
    log("="*80)
    log("处理 Sashital Lab 数据集")
    log("="*80)

    data_dir = DATA_DIR / "Cas12a_nickase" / "other-files"

    if not data_dir.exists():
        log(f"✗ 目录不存在: {data_dir}")
        return None

    all_data = []

    # 处理所有 KMlib*.xlsx 文件
    for xlsx_file in sorted(data_dir.glob("KMlib*.xlsx")):
        log(f"\n读取文件: {xlsx_file.name}")

        try:
            df = pd.read_excel(xlsx_file, sheet_name=0)
            log(f"  - {len(df)} 行, {len(df.columns)} 列")

            # 确保有必需的列
            if 'SC' not in df.columns:
                log(f"  ✗ 缺少 'SC' (selection coefficient) 列")
                continue

            # 提取有效数据
            valid_rows = 0
            for idx, row in df.iterrows():
                try:
                    # 获取错配信息和活性
                    mismatch_pattern = row.get('MM', row.get('Unnamed: 0', ''))
                    selection_coef = row['SC']
                    read_count = row.get('N', 0)

                    # 跳过无效数据
                    if pd.isna(selection_coef) or pd.isna(mismatch_pattern):
                        continue

                    # 解析错配模式
                    # 例如: 'r1:C7A' = replicate 1, position 7, C→A
                    mismatch_info = str(mismatch_pattern)

                    data_entry = {
                        'dataset': 'Sashital_' + xlsx_file.stem,
                        'mismatch_pattern': mismatch_info,
                        'activity': float(selection_coef),
                        'read_count': int(read_count) if not pd.isna(read_count) else 0,
                        'source_file': xlsx_file.name
                    }

                    all_data.append(data_entry)
                    valid_rows += 1

                except Exception as e:
                    continue

            log(f"  ✓ 提取了 {valid_rows} 条有效数据")

        except Exception as e:
            log(f"  ✗ 读取失败: {str(e)}")
            continue

    log(f"\n✓ Sashital Lab 处理完成: {len(all_data)} 条有效数据")

    if all_data:
        df_result = pd.DataFrame(all_data)
        # 保存中间结果
        output_file = OUTPUT_DIR / "sashital_processed.csv"
        df_result.to_csv(output_file, index=False)
        log(f"✓ 已保存到: {output_file}")
        return df_result

    return None

# ============================================================================
# 数据集 3: EasyDesign
# ============================================================================
def process_easydesign():
    """
    处理 EasyDesign 训练数据

    Table S1.xlsx - 病原体列表 (不含活性数据)
    Table S2.xlsx - 训练数据！包含:
        - guide_seq: crRNA 序列 (25 nt)
        - target_at_guide: 靶标序列
        - guide_target_hamming_dist: 错配数
        - 30 min / 20 min normalized: 活性分数 (log scale)
    Table S3.xlsx - 病原体基因组序列
    Table S4.xlsx - 实验验证的 crRNA 和 RPA 引物
    """
    log("="*80)
    log("处理 EasyDesign 数据集")
    log("="*80)

    data_dir = DATA_DIR / "EasyDesign" / "data"

    if not data_dir.exists():
        log(f"✗ 目录不存在: {data_dir}")
        return None

    all_data = []

    # 重点处理 Table S2 - 这是训练数据
    table_s2 = data_dir / "Table S2.xlsx"

    if not table_s2.exists():
        log(f"✗ Table S2.xlsx 不存在")
        return None

    log(f"\n读取文件: Table S2.xlsx (训练数据)")

    try:
        df = pd.read_excel(table_s2, sheet_name=0)
        log(f"  - {len(df)} 行, {len(df.columns)} 列")
        log(f"  - 列名: {list(df.columns)}")

        # 提取训练数据
        valid_rows = 0
        for idx, row in df.iterrows():
            try:
                guide_seq = row.get('guide_seq', '')
                target_seq = row.get('target_at_guide', '')
                mismatches = row.get('guide_target_hamming_dist', 0)
                activity_30min = row.get('30 min', np.nan)
                activity_20min = row.get('20 min normalized', np.nan)
                pathogen_type1 = row.get('type1', '')
                pathogen_type2 = row.get('type2', '')

                # 跳过无效数据
                if pd.isna(guide_seq) or pd.isna(target_seq):
                    continue

                # 使用 30 min 的活性数据（如果有）
                activity = activity_30min if not pd.isna(activity_30min) else activity_20min

                if pd.isna(activity):
                    continue

                # 提取 PAM (Cas12a PAM 在 5' 端，通常是 TTTV)
                # guide_seq 是 25 nt，实际靶标前面应该有 PAM
                # 这里记录 guide 序列，PAM 需要从原始数据或文献中获取

                data_entry = {
                    'dataset': 'EasyDesign_TableS2',
                    'guide_sequence': str(guide_seq),
                    'target_sequence': str(target_seq),
                    'mismatches': int(mismatches),
                    'activity': float(activity),
                    'pathogen_type': f"{pathogen_type1}|{pathogen_type2}",
                    'guide_length': len(str(guide_seq))
                }

                all_data.append(data_entry)
                valid_rows += 1

            except Exception as e:
                continue

        log(f"  ✓ 提取了 {valid_rows} 条有效数据")

    except Exception as e:
        log(f"  ✗ 读取失败: {str(e)}")
        return None

    log(f"\n✓ EasyDesign 处理完成: {len(all_data)} 条有效数据")

    if all_data:
        df_result = pd.DataFrame(all_data)
        # 保存中间结果
        output_file = OUTPUT_DIR / "easydesign_processed.csv"
        df_result.to_csv(output_file, index=False)
        log(f"✓ 已保存到: {output_file}")

        # 显示统计信息
        log(f"\n统计信息:")
        log(f"  - 平均错配数: {df_result['mismatches'].mean():.2f}")
        log(f"  - 活性范围: [{df_result['activity'].min():.2f}, {df_result['activity'].max():.2f}]")
        log(f"  - Guide 长度: {df_result['guide_length'].value_counts().to_dict()}")

        return df_result

    return None

# ============================================================================
# 检查数据集内容
# ============================================================================
def inspect_datasets():
    """
    检查数据集的实际内容和列名
    为后续处理提供信息
    """
    log("="*80)
    log("检查数据集内容")
    log("="*80)

    # 1. NucleaSeq
    log("\n【数据集 1: NucleaSeq】")
    nucleaseq_file = DATA_DIR / "nucleaseq" / "Supplemental_File_1.xlsx"
    if nucleaseq_file.exists():
        try:
            df = pd.read_excel(nucleaseq_file, sheet_name=0, nrows=5)
            log(f"文件大小: {nucleaseq_file.stat().st_size / 1024 / 1024:.2f} MB")
            log(f"行数 (前5行): {len(df)} 行")
            log(f"列数: {len(df.columns)} 列")
            log(f"列名:")
            for i, col in enumerate(df.columns, 1):
                log(f"  {i}. {col}")
            log(f"\n前3行数据:")
            log(df.head(3).to_string())
        except Exception as e:
            log(f"✗ 读取失败: {str(e)}")

    # 2. Sashital Lab
    log("\n【数据集 2: Sashital Lab】")
    sashital_dir = DATA_DIR / "Cas12a_nickase" / "other-files"
    if sashital_dir.exists():
        for xlsx_file in sorted(sashital_dir.glob("KMlib*.xlsx")):
            log(f"\n文件: {xlsx_file.name}")
            try:
                df = pd.read_excel(xlsx_file, sheet_name=0, nrows=3)
                log(f"  文件大小: {xlsx_file.stat().st_size / 1024:.2f} KB")
                log(f"  列数: {len(df.columns)} 列")
                log(f"  列名: {', '.join(df.columns[:10])}")
            except Exception as e:
                log(f"  ✗ 读取失败: {str(e)}")

    # 3. EasyDesign
    log("\n【数据集 3: EasyDesign】")
    easydesign_dir = DATA_DIR / "EasyDesign" / "data"
    if easydesign_dir.exists():
        for xlsx_file in sorted(easydesign_dir.glob("Table*.xlsx")):
            log(f"\n文件: {xlsx_file.name}")
            try:
                df = pd.read_excel(xlsx_file, sheet_name=0, nrows=3)
                log(f"  文件大小: {xlsx_file.stat().st_size / 1024:.2f} KB")
                log(f"  行数: {len(df)} 行")
                log(f"  列数: {len(df.columns)} 列")
                log(f"  列名: {', '.join(df.columns[:10])}")
                if len(df) > 0:
                    log(f"  前1行数据:")
                    log(f"  {df.iloc[0].to_dict()}")
            except Exception as e:
                log(f"  ✗ 读取失败: {str(e)}")

    log("\n" + "="*80)
    log("检查完成")
    log("="*80)

# ============================================================================
# 主函数
# ============================================================================
def main():
    log("="*80)
    log("开始处理公开 Cas12a 活性数据集")
    log(f"项目根目录: {PROJECT_ROOT}")
    log(f"输出目录: {OUTPUT_DIR}")
    log("="*80)

    # 第一步：检查数据集内容
    # inspect_datasets()  # 已经检查过了，注释掉

    # 第二步：处理各个数据集
    log("\n" + "="*80)
    log("开始处理数据集")
    log("="*80)

    results = {}

    # 1. NucleaSeq (Cas9 数据，跳过)
    results['nucleaseq'] = process_nucleaseq()

    # 2. Sashital Lab (Cas12a 数据)
    results['sashital'] = process_sashital()

    # 3. EasyDesign (Cas12a 数据)
    results['easydesign'] = process_easydesign()

    # 第三步：合并所有有效数据集
    log("\n" + "="*80)
    log("合并数据集")
    log("="*80)

    all_datasets = []
    for name, df in results.items():
        if df is not None and len(df) > 0:
            log(f"✓ {name}: {len(df)} 条数据")
            all_datasets.append(df)
        else:
            log(f"✗ {name}: 无数据")

    if all_datasets:
        # 找到共同的列
        common_cols = set(all_datasets[0].columns)
        for df in all_datasets[1:]:
            common_cols = common_cols.intersection(set(df.columns))

        log(f"\n共同的列: {common_cols}")

        # 如果没有共同列，保留所有列（会有 NaN）
        if len(common_cols) == 0:
            log("警告: 数据集之间没有共同列，将保留所有列")
            combined_df = pd.concat(all_datasets, ignore_index=True, sort=False)
        else:
            # 只保留共同列
            aligned_datasets = [df[list(common_cols)] for df in all_datasets]
            combined_df = pd.concat(aligned_datasets, ignore_index=True)

        log(f"\n合并后的数据集: {len(combined_df)} 条")

        # 保存合并结果
        output_file = OUTPUT_DIR / "combined_cas12a_dataset.csv"
        combined_df.to_csv(output_file, index=False)
        log(f"✓ 已保存到: {output_file}")

        # 显示统计信息
        log("\n" + "="*80)
        log("数据集统计")
        log("="*80)
        log(f"总数据量: {len(combined_df)} 条")
        log(f"数据来源分布:")
        if 'dataset' in combined_df.columns:
            for dataset, count in combined_df['dataset'].value_counts().items():
                log(f"  - {dataset}: {count} 条")

        if 'activity' in combined_df.columns:
            log(f"\n活性统计:")
            log(f"  - 平均值: {combined_df['activity'].mean():.4f}")
            log(f"  - 标准差: {combined_df['activity'].std():.4f}")
            log(f"  - 范围: [{combined_df['activity'].min():.4f}, {combined_df['activity'].max():.4f}]")

    else:
        log("\n✗ 没有可用的数据集")

    log("\n" + "="*80)
    log("下一步:")
    log("="*80)
    log("1. 查看生成的 CSV 文件，确认数据格式正确")
    log("2. 将数据转换为你的模型训练格式 (与 v4 模型兼容)")
    log("3. 合并公开数据集 + 你自己的 50 条数据")
    log("4. 重新训练 v4 模型")
    log("5. 使用新模型重新评估 miR-29a RT-cDNA 构建")

    log(f"\n完整日志已保存到: {LOG_FILE}")

if __name__ == "__main__":
    main()
