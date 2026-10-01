# 项目文件整理记录

## 整理时间
2026年10月1日

## 整理目标
1. 将分散在根目录的文件移动到规范的文件夹结构中
2. 将中文文件名改为英文
3. 删除重复文件（mirna-crispr 文件夹）

## 当前状态

### ⚠️ 需要手动处理的项目

由于文件权限保护，以下文件夹需要你手动删除（它们是重复或临时文件）：

1. **`mirna-crispr/`** - 这是完全重复的文件夹，所有内容已经移动到正确位置，可以安全删除
2. **`.ipynb_checkpoints/`** - Jupyter 的临时检查点文件，可以删除
3. **`notebooks/.ipynb_checkpoints/`** - Jupyter 的临时文件，可以删除
4. **`environment_records/.ipynb_checkpoints/`** - Jupyter 的临时文件，可以删除

### ✅ 已完成的整理

所有数据文件、notebook 和结果图表已经移动到正确的位置。

## 新的文件夹结构

```
mirna-crispr-ml/
├── data/
│   ├── raw/                        # 原始数据
│   │   ├── cas12a_mismatch_data.csv
│   │   └── candidate_sequences_corrected.csv
│   └── processed/                  # 处理后的数据
│       ├── final_predictions.csv
│       ├── model_predictions.csv
│       ├── ranked_crRNA.csv
│       ├── ranked_crRNA_v4.csv
│       ├── ranked_crRNA_selectivity_v4.csv
│       └── ranked_crRNA_selectivity_v4_corrected.csv
├── notebooks/                      # Jupyter notebooks
│   ├── 01_final_model_svr_gb_8features_v3.ipynb  (从中文名改过来)
│   ├── 02_v4_audited.ipynb
│   ├── 03_v4_run_notebook_version.ipynb
│   ├── 04_final_model_audit_v2.ipynb
│   └── 05_rpa_forward_primer.ipynb
├── results/                        # 结果输出
│   ├── figures/                    # 图表
│   │   ├── model_performance.png
│   │   ├── model_v2_performance.png
│   │   ├── model_v3_performance.png
│   │   ├── model_v4_performance.png
│   │   └── model_v4_audit.png
│   └── metrics/                    # 评估指标
│       ├── model_v4_audit_metrics.csv
│       └── model_v4_error_by_mismatch_position.csv
├── environment_records/            # 环境配置记录
│   ├── model_v4_environment_manifest.csv
│   ├── model_v4_environment_manifest.json
│   ├── model_v4_environment_manifest.md
│   └── model_v4_pip_freeze.txt
├── easy_design_mir29a/            # EasyDesign 分析结果
│   ├── design_parameters.json
│   ├── EASYDESIGN_RUN_REPORT.md
│   ├── README.md
│   ├── easydesign_mir29a_rpa.tsv
│   ├── mir29_rt_cdna_constructs.fasta
│   ├── mir29a_primary_rt_cdna.fasta
│   └── mir29bc_exclusion_rt_cdna.fasta
├── tools/                         # 第三方工具
│   └── EasyDesign/
├── scripts/                       # 脚本文件夹（预留）
└── anaconda_projects/             # Anaconda 项目配置

## 待删除的重复文件夹（需手动删除）
└── mirna-crispr/                  # ⚠️ 完全重复，可以删除
```

## 详细变更记录表

| 操作类型 | 原始路径 | 原始文件名 | 新路径 | 新文件名 | 说明 |
|---------|---------|-----------|-------|---------|------|
| 移动+重命名 | (根目录) | 最终模型（SVR+GB 8特征+新参数)_v3.ipynb | notebooks/ | 01_final_model_svr_gb_8features_v3.ipynb | 中文名改为英文 |
| 移动+重命名 | (根目录) | v4_audited_2.ipynb | notebooks/ | 02_v4_audited.ipynb | 统一编号 |
| 移动+重命名 | (根目录) | v4_run_notebook_version.ipynb | notebooks/ | 03_v4_run_notebook_version.ipynb | 统一编号 |
| 移动+重命名 | (根目录) | RPA Forward Primer.ipynb | notebooks/ | 05_rpa_forward_primer.ipynb | 去除空格，统一编号 |
| 移动 | (根目录) | cas12a_mismatch_data.csv | data/raw/ | cas12a_mismatch_data.csv | 原始数据归档 |
| 移动 | (根目录) | candidate_sequences_corrected.csv | data/raw/ | candidate_sequences_corrected.csv | 原始数据归档 |
| 移动 | (根目录) | final_predictions.csv | data/processed/ | final_predictions.csv | 处理后数据归档 |
| 移动 | (根目录) | model_predictions.csv | data/processed/ | model_predictions.csv | 处理后数据归档 |
| 移动 | (根目录) | ranked_crRNA.csv | data/processed/ | ranked_crRNA.csv | 处理后数据归档 |
| 移动 | (根目录) | ranked_crRNA_v4.csv | data/processed/ | ranked_crRNA_v4.csv | 处理后数据归档 |
| 移动 | (根目录) | ranked_crRNA_selectivity_v4.csv | data/processed/ | ranked_crRNA_selectivity_v4.csv | 处理后数据归档 |
| 移动 | (根目录) | ranked_crRNA_selectivity_v4_corrected.csv | data/processed/ | ranked_crRNA_selectivity_v4_corrected.csv | 处理后数据归档 |
| 移动 | (根目录) | model_performance.png | results/figures/ | model_performance.png | 结果图表归档 |
| 移动 | (根目录) | model_v2_performance.png | results/figures/ | model_v2_performance.png | 结果图表归档 |
| 移动 | (根目录) | model_v3_performance.png | results/figures/ | model_v3_performance.png | 结果图表归档 |
| 移动 | (根目录) | model_v4_performance.png | results/figures/ | model_v4_performance.png | 结果图表归档 |
| 移动 | (根目录) | model_v4_audit.png | results/figures/ | model_v4_audit.png | 结果图表归档 |
| 移动 | (根目录) | model_v4_audit_metrics.csv | results/metrics/ | model_v4_audit_metrics.csv | 评估指标归档 |
| 移动 | (根目录) | model_v4_error_by_mismatch_position.csv | results/metrics/ | model_v4_error_by_mismatch_position.csv | 评估指标归档 |
| 待删除 | mirna-crispr/ | (整个文件夹) | - | - | ⚠️ 完全重复的文件夹 |
| 待删除 | .ipynb_checkpoints/ | (整个文件夹) | - | - | Jupyter 临时文件 |
| 待删除 | notebooks/.ipynb_checkpoints/ | (整个文件夹) | - | - | Jupyter 临时文件 |
| 待删除 | environment_records/.ipynb_checkpoints/ | (整个文件夹) | - | - | Jupyter 临时文件 |

## 文件路径对照表（用于更新代码中的路径引用）

### 原始路径 → 新路径映射

| 原始路径 | 新路径 |
|---------|-------|
| `C:\Users\paul\Desktop\mirna-crispr-ml\最终模型（SVR+GB 8特征+新参数)_v3.ipynb` | `D:\Evelyn\mirna-crispr-ml\notebooks\01_final_model_svr_gb_8features_v3.ipynb` |
| `C:\Users\paul\Desktop\mirna-crispr-ml\cas12a_mismatch_data.csv` | `D:\Evelyn\mirna-crispr-ml\data\raw\cas12a_mismatch_data.csv` |
| `C:\Users\paul\Desktop\mirna-crispr-ml\candidate_sequences_corrected.csv` | `D:\Evelyn\mirna-crispr-ml\data\raw\candidate_sequences_corrected.csv` |
| `C:\Users\paul\Desktop\mirna-crispr-ml\final_predictions.csv` | `D:\Evelyn\mirna-crispr-ml\data\processed\final_predictions.csv` |
| `C:\Users\paul\Desktop\mirna-crispr-ml\model_predictions.csv` | `D:\Evelyn\mirna-crispr-ml\data\processed\model_predictions.csv` |
| `C:\Users\paul\Desktop\mirna-crispr-ml\ranked_crRNA.csv` | `D:\Evelyn\mirna-crispr-ml\data\processed\ranked_crRNA.csv` |
| `C:\Users\paul\Desktop\mirna-crispr-ml\model_v4_audit.png` | `D:\Evelyn\mirna-crispr-ml\results\figures\model_v4_audit.png` |
| `C:\Users\paul\Desktop\mirna-crispr-ml\model_v4_audit_metrics.csv` | `D:\Evelyn\mirna-crispr-ml\results\metrics\model_v4_audit_metrics.csv` |

## 后续建议

1. **手动删除重复文件夹**：
   - 删除 `mirna-crispr/` 文件夹（大小约 21MB）
   - 删除所有 `.ipynb_checkpoints/` 文件夹

2. **更新代码中的路径引用**：
   - 打开 notebooks 中的文件
   - 将所有旧路径（`C:\Users\paul\Desktop\mirna-crispr-ml\`）更新为新路径（`D:\Evelyn\mirna-crispr-ml\`）
   - 更新为相对路径会更好，例如 `../data/raw/cas12a_mismatch_data.csv`

3. **创建 README.md**：
   - 建议在项目根目录创建一个 README 文件
   - 说明项目结构、如何运行 notebooks 等

4. **Git 提交**：
   - 文件整理后建议提交到 Git
   - 提交信息：`Reorganize project structure and rename Chinese filenames to English`
