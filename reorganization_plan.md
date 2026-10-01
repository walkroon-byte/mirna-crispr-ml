# Project Reorganization Plan

## Current Issues Identified

1. **Duplicate content**: The `mirna-crispr/` subfolder contains duplicate files from the root directory
2. **Mixed Chinese/English naming**: Files like `最终模型（SVR+GB 8特征+新参数)_v3.ipynb`
3. **Scattered files**: Notebooks, CSV files, and images are mixed in root directory
4. **Empty directories**: `data/` and `results/` folders are empty or underutilized

## Proposed New Structure

```
mirna-crispr-ml/
├── README.md (to be created)
├── data/
│   ├── raw/
│   │   ├── cas12a_mismatch_data.csv
│   │   └── candidate_sequences_corrected.csv
│   └── processed/
│       ├── final_predictions.csv
│       ├── model_predictions.csv
│       ├── ranked_crRNA.csv
│       ├── ranked_crRNA_v4.csv
│       ├── ranked_crRNA_selectivity_v4.csv
│       └── ranked_crRNA_selectivity_v4_corrected.csv
├── notebooks/
│   ├── 01_final_model_svr_gb_8features_v3.ipynb (renamed from 最终模型)
│   ├── 02_v4_audited.ipynb
│   ├── 03_v4_run_notebook_version.ipynb
│   ├── 04_final_model_audit_v2.ipynb
│   └── 05_rpa_forward_primer.ipynb
├── results/
│   ├── figures/
│   │   ├── model_performance.png
│   │   ├── model_v2_performance.png
│   │   ├── model_v3_performance.png
│   │   ├── model_v4_performance.png
│   │   └── model_v4_audit.png
│   └── metrics/
│       ├── model_v4_audit_metrics.csv
│       └── model_v4_error_by_mismatch_position.csv
├── environment_records/
│   ├── model_v4_environment_manifest.csv
│   ├── model_v4_environment_manifest.json
│   ├── model_v4_environment_manifest.md
│   └── model_v4_pip_freeze.txt
├── easy_design_mir29a/
│   ├── design_parameters.json
│   ├── EASYDESIGN_RUN_REPORT.md
│   └── README.md
├── tools/
│   └── EasyDesign/ (keep as is)
└── scripts/ (empty, ready for future use)
```

## Actions to Take

1. **Remove duplicate `mirna-crispr/` folder** - it's a complete duplicate
2. **Rename Chinese files to English**
3. **Reorganize files into logical directories**
4. **Create a comprehensive mapping table**

## Next Step

Proceeding with reorganization...
