#!/usr/bin/env python3
"""
下载公开 Cas12a 活性数据集以扩充训练集

按导师建议的优先级：
1. NucleaSeq 平台数据 - 超过1万条含错配的靶标
2. 体外质粒文库数据 - Sashital lab
3. EasyDesign 训练数据 - GitHub
4. 人类细胞评估数据 - NCBI BioProject

作者: Evelyn
日期: 2026-10-01
"""

import os
import sys
import requests
import subprocess
from pathlib import Path
import json
from datetime import datetime

# 设置路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "public_datasets"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# 创建目录
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 日志文件
LOG_FILE = DATA_DIR / f"download_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log(message):
    """记录日志"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_msg + '\n')

def download_file(url, output_path, description):
    """下载文件"""
    log(f"开始下载: {description}")
    log(f"URL: {url}")
    log(f"保存到: {output_path}")

    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r进度: {progress:.1f}%", end='', flush=True)

        print()  # 换行
        log(f"✓ 下载完成: {output_path.name} ({downloaded / 1024 / 1024:.2f} MB)")
        return True

    except Exception as e:
        log(f"✗ 下载失败: {str(e)}")
        return False

def clone_github_repo(repo_url, output_dir, description):
    """克隆 GitHub 仓库"""
    log(f"开始克隆仓库: {description}")
    log(f"URL: {repo_url}")

    try:
        if output_dir.exists():
            log(f"目录已存在，跳过克隆: {output_dir}")
            return True

        subprocess.run(['git', 'clone', repo_url, str(output_dir)],
                      check=True, capture_output=True, text=True)
        log(f"✓ 克隆完成: {output_dir.name}")
        return True

    except subprocess.CalledProcessError as e:
        log(f"✗ 克隆失败: {e.stderr}")
        return False
    except FileNotFoundError:
        log("✗ 找不到 git 命令，请确保已安装 Git")
        return False

# ============================================================================
# 数据集 1: NucleaSeq 平台数据
# ============================================================================
def download_nucleaseq():
    """
    NucleaSeq: 超过1万条含错配的 Cas12a 靶标，测量了切割动力学
    GitHub: https://github.com/finkelsteinlab/nucleaseq
    """
    log("="*80)
    log("数据集 1: NucleaSeq 平台数据")
    log("="*80)

    repo_url = "https://github.com/finkelsteinlab/nucleaseq.git"
    output_dir = RAW_DIR / "nucleaseq"

    if clone_github_repo(repo_url, output_dir, "NucleaSeq 平台数据"):
        # 查找数据文件
        data_files = list(output_dir.rglob("*.csv")) + list(output_dir.rglob("*.tsv"))
        log(f"找到 {len(data_files)} 个数据文件")
        for f in data_files[:10]:  # 只显示前10个
            log(f"  - {f.relative_to(output_dir)}")
        if len(data_files) > 10:
            log(f"  ... 还有 {len(data_files) - 10} 个文件")

        return True
    return False

# ============================================================================
# 数据集 2: 体外质粒文库 + 高通量测序 (Sashital Lab)
# ============================================================================
def download_sashital_data():
    """
    体外质粒文库 + 高通量测序数据
    代码: https://github.com/sashital-lab/Cas12a_nickase
    数据: https://iastate.figshare.com/articles/dataset/...
    """
    log("="*80)
    log("数据集 2: 体外质粒文库 + 高通量测序 (Sashital Lab)")
    log("="*80)

    # 克隆代码仓库
    repo_url = "https://github.com/sashital-lab/Cas12a_nickase.git"
    code_dir = RAW_DIR / "sashital_cas12a_nickase"
    clone_github_repo(repo_url, code_dir, "Sashital Lab Cas12a nickase 代码")

    # 下载 Figshare 数据
    # 注意：Figshare 数据集很大，这里提供下载说明
    figshare_url = "https://iastate.figshare.com/articles/dataset/High-throughput_sequencing_HTS_data_of_plasmid_library_subjected_to_Cas12a_cleavage/8178938/3"

    log("\n⚠️  Figshare 数据集需要手动下载")
    log(f"URL: {figshare_url}")
    log(f"请手动下载并解压到: {RAW_DIR / 'sashital_data'}")
    log("文件列表应该包含:")
    log("  - Plasmid_library.fastq.gz")
    log("  - Cas12a_treated_samples.fastq.gz")
    log("  - 等等")

    # 创建占位目录和说明文件
    data_dir = RAW_DIR / "sashital_data"
    data_dir.mkdir(exist_ok=True)

    readme = data_dir / "DOWNLOAD_INSTRUCTIONS.txt"
    with open(readme, 'w', encoding='utf-8') as f:
        f.write(f"Sashital Lab 数据下载说明\n")
        f.write(f"=" * 80 + "\n\n")
        f.write(f"数据源: {figshare_url}\n\n")
        f.write(f"下载步骤:\n")
        f.write(f"1. 访问上述 URL\n")
        f.write(f"2. 点击 'Download all' 下载全部文件\n")
        f.write(f"3. 解压到当前目录: {data_dir}\n\n")
        f.write(f"预期文件:\n")
        f.write(f"- Plasmid_library.fastq.gz (质粒文库测序数据)\n")
        f.write(f"- Cas12a_treated_samples.fastq.gz (Cas12a 处理后的样本)\n")
        f.write(f"- 其他相关文件\n")

    log(f"✓ 已创建下载说明: {readme}")
    return True

# ============================================================================
# 数据集 3: EasyDesign 训练数据
# ============================================================================
def download_easydesign_data():
    """
    EasyDesign 的训练数据
    GitHub: https://github.com/scRNA-Compt/EasyDesign
    """
    log("="*80)
    log("数据集 3: EasyDesign 训练数据")
    log("="*80)

    repo_url = "https://github.com/scRNA-Compt/EasyDesign.git"
    output_dir = RAW_DIR / "easydesign_data"

    if clone_github_repo(repo_url, output_dir, "EasyDesign 训练数据"):
        # 查找训练数据文件
        data_files = list(output_dir.rglob("*train*")) + \
                    list(output_dir.rglob("*data*")) + \
                    list(output_dir.rglob("*.csv")) + \
                    list(output_dir.rglob("*.tsv"))

        log(f"找到 {len(data_files)} 个可能的数据文件")
        for f in data_files[:15]:
            log(f"  - {f.relative_to(output_dir)}")
        if len(data_files) > 15:
            log(f"  ... 还有 {len(data_files) - 15} 个文件")

        return True
    return False

# ============================================================================
# 数据集 4: 人类细胞评估数据 (NCBI BioProject)
# ============================================================================
def download_ncbi_data():
    """
    人类细胞评估数据 - 20多个 Cas12a 变体、数千条靶标
    NCBI: https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1074843
    Code Ocean: https://codeocean.com/capsule/9398276/tree/v1
    """
    log("="*80)
    log("数据集 4: 人类细胞评估数据 (NCBI BioProject)")
    log("="*80)

    ncbi_url = "https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1074843"
    codeocean_url = "https://codeocean.com/capsule/9398276/tree/v1"

    log("\n⚠️  NCBI 数据集需要使用 SRA Toolkit 下载")
    log(f"BioProject: {ncbi_url}")
    log(f"Code Ocean: {codeocean_url}")
    log(f"\n下载步骤:")
    log("1. 安装 SRA Toolkit: https://github.com/ncbi/sra-tools")
    log("2. 使用 prefetch 和 fasterq-dump 下载测序数据")
    log("3. 或者直接访问 Code Ocean 获取处理后的数据")

    # 创建说明文件
    data_dir = RAW_DIR / "ncbi_cas12a_human_cells"
    data_dir.mkdir(exist_ok=True)

    readme = data_dir / "DOWNLOAD_INSTRUCTIONS.txt"
    with open(readme, 'w', encoding='utf-8') as f:
        f.write(f"NCBI BioProject 数据下载说明\n")
        f.write(f"=" * 80 + "\n\n")
        f.write(f"BioProject ID: PRJNA1074843\n")
        f.write(f"URL: {ncbi_url}\n")
        f.write(f"Code Ocean: {codeocean_url}\n\n")
        f.write(f"选项 1 - 从 Code Ocean 下载处理后的数据 (推荐):\n")
        f.write(f"1. 访问 Code Ocean URL\n")
        f.write(f"2. 下载 'results' 或 'data' 文件夹\n")
        f.write(f"3. 解压到当前目录\n\n")
        f.write(f"选项 2 - 从 NCBI SRA 下载原始数据:\n")
        f.write(f"1. 安装 SRA Toolkit\n")
        f.write(f"2. prefetch PRJNA1074843\n")
        f.write(f"3. fasterq-dump --split-files SRR*\n")

    log(f"✓ 已创建下载说明: {readme}")
    return True

# ============================================================================
# 主函数
# ============================================================================
def main():
    log("="*80)
    log("开始下载公开 Cas12a 活性数据集")
    log(f"项目根目录: {PROJECT_ROOT}")
    log(f"数据保存目录: {DATA_DIR}")
    log("="*80)

    results = {}

    # 按导师建议的优先级下载
    results['nucleaseq'] = download_nucleaseq()
    print("\n")

    results['sashital'] = download_sashital_data()
    print("\n")

    results['easydesign'] = download_easydesign_data()
    print("\n")

    results['ncbi'] = download_ncbi_data()
    print("\n")

    # 总结
    log("="*80)
    log("下载总结")
    log("="*80)
    for name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        log(f"{status}: {name}")

    log(f"\n完整日志已保存到: {LOG_FILE}")
    log("\n下一步:")
    log("1. 按照说明文件手动下载 Sashital 和 NCBI 数据集")
    log("2. 运行 process_public_datasets.py 处理下载的数据")
    log("3. 运行 merge_datasets.py 合并所有数据集")
    log("4. 重新训练你的 v4 模型")

if __name__ == "__main__":
    main()
