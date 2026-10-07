# 清理版训练数据说明

## 文件

- 输入合并数据：`combined_v4_training_data.csv`（9,974 行）
- 排除记录：`../raw/cas12a_mismatch_data.csv`（80 行）
- 输出文件：`combined_v4_training_data_cleaned_exclude_unverified80.csv`（9,894 行）
- 生成脚本：`../../scripts/create_clean_training_data.py`

## 清理规则

将排除文件中的 80 行视为一个多重集合，从合并数据中逐条扣除。匹配键包含 `spacer_seq`、`target_seq`、`mismatch_pos` 和 `relative_activity`；活性数值按十进制数值比较，因此 `0.798600` 与 `0.7986` 视为相同。每个匹配项只扣除一次，避免因同一序列键对应不同活性值而整组删除记录。

`mismatch_pos` 在此仅作为旧记录识别字段，不表示其位置含义已验证。清理操作没有修复或重算任何标签，也没有变更剩余记录。

## 解释范围

该文件只表示排除了已知的 80 条无逐序列活性依据的记录，不代表其余所有公开数据已完成来源、实验定义、标签公式或序列位置的全面审计。用于单碱基错配活性建模前，仍需核实剩余记录的逐序列测量依据及标签定义。

## 复现

在项目根目录运行：

```powershell
python scripts/create_clean_training_data.py
```

脚本默认拒绝覆盖已存在的输出文件；如需重新生成，应先将旧输出另行归档，再指定新的 `--output` 路径。
