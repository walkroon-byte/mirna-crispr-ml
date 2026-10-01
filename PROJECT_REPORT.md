# miR-29 家族选择性 Cas12a 检测系统：机器学习辅助设计

**项目状态报告 | 2026年10月1日**

---

## 执行摘要

本项目开发了一个基于机器学习的 LbCas12a crRNA 设计系统，用于区分 miR-29a/b/c 家族成员。项目完成了从数据处理、特征工程、模型训练到候选筛选的完整流程，并在 v4 版本中主动发现并修正了 v3 中的两处能量计算错误。

**关键成果：**
- 训练了 SVR+GradientBoost 集成模型（Spearman ρ = 0.692, p = 2.66×10⁻⁸）
- 识别出全局最优候选 CRRNA-CORR-008（选择性比率 1.29）
- 完成 RPA 引物设计和 EasyDesign 平台集成测试
- 建立了完整的可复现环境记录

**当前局限：**
- 训练数据仅 50 条样本，且 crRNA 识别序列始终为同一条
- 模型未见过"换一条 crRNA 会怎样"，对新序列泛化能力未经验证
- 对完全匹配靶标，特征退化导致预测分数完全相同

---

## 1. 项目背景

### 1.1 生物学背景

miR-29 家族包含三个高度同源的成员（29a/b/c），它们在多种生理和病理过程中发挥重要作用。开发能够选择性检测单个家族成员的方法对于精准医学和基础研究具有重要意义。

**序列比对：**
```
miR-29a-3p: UAGCACCAU-CUGAAAUCGGUUA
miR-29b-3p: UAGCACCAUUUGAAAUCAGUGUU
miR-29c-3p: UAGCACCAUUUGAAAUCGGUUA
            ********* *********  **
```

关键差异位点：
- 位置 9：29a 为 C，29b/c 为 U
- 位置 18：29a/c 为 G，29b 为 A
- 3' 末端：29a/c 为 UA，29b 为 UU

### 1.2 技术路线

采用 RT-RPA-Cas12a 三联检测策略：
1. **逆转录（RT）**：将 miRNA 转化为 cDNA
2. **重组酶聚合酶扩增（RPA）**：等温扩增 cDNA
3. **Cas12a 切割检测**：利用错配敏感性实现选择性识别

---

## 2. 数据集描述

### 2.1 训练数据

**来源：** 50 条 LbCas12a 错配实验数据

**数据特征：**
- Spacer 序列：`CGGUUCGCGUGGAUUAAAGG`（20 nt，固定不变）
- Target 序列：在不同位置引入单碱基错配（位置 0-16）
- 相对活性：测量值范围 0.967-1.537

**数据结构示例：**
```python
{
    "spacer_seq": "CGGUUCGCGUGGAUUAAAGG",
    "target_seq": "AGGTTCGCGTGGATTAAAGG",  # 位置 0 错配
    "mismatch_pos": 0,
    "relative_activity": 1.244067
}
```

**数据覆盖：**
- 错配位置：0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
- 每个位置 2-3 个不同碱基替换
- **重要局限**：所有 50 条数据使用同一条 spacer 序列

### 2.2 候选序列

**设计策略：** 针对 miR-29a/b/c 三个靶标，通过滑动窗口生成 10 条候选 crRNA：
- 针对 miR-29a：3 条（CRRNA-CORR-001, 002, 003）
- 针对 miR-29b：4 条（CRRNA-CORR-004, 005, 006, 007）
- 针对 miR-29c：3 条（CRRNA-CORR-008, 009, 010）

---

## 3. 方法学

### 3.1 特征工程（v4 修正版）

设计了 8 个特征捕捉 crRNA-target 相互作用：

| 特征名称 | 类型 | 描述 | 取值范围 |
|---------|------|------|---------|
| `mismatch_pos` | 数值 | 最左错配位置（0-19，完全匹配为 -1） | [-1, 19] |
| `ddg` | 数值 | 双链能量变化（ΔΔG = ΔG_duplex - ΔG_perfect） | 实数 |
| `is_purine_mismatch` | 二值 | spacer 和 target 均为嘌呤（A/G） | {0, 1} |
| `is_transition` | 二值 | 转换型错配（A↔G, C↔U） | {0, 1} |
| `is_edge` | 二值 | 边缘错配（≤2 或 ≥17） | {0, 1} |
| `pos_ddg_interaction` | 数值 | 位置与能量交互项 | 实数 |
| `mismatch_score` | 分类 | 错配严重程度（A-C:3, G-A:2, C-U:1, G-U:0） | {0,1,2,3} |
| `seed_weight` | 数值 | seed 区权重（≤8 为 1.5，否则 1.0） | {1.0, 1.5} |

**v3→v4 关键修正：**

1. **RNA-DNA 双链方向修正**
   ```python
   # v3 错误做法（直接传入 target 原链）
   dg_duplex = RNA.duplexfold(spacer_rna, target_rna).energy
   
   # v4 修正（target 取 reverse-complement）
   pairing_strand = reverse_complement_rna(target_rna)
   dg_duplex = RNA.duplexfold(spacer_rna, pairing_strand).energy
   ```
   
   **原理：** Cas12a 的 crRNA 与 target DNA 反义链配对，计算杂交能量时必须取 reverse-complement。

2. **序列归一化**
   ```python
   # v4 新增：统一转换为 RNA（T→U）
   def normalize_rna(sequence: str) -> str:
       return str(sequence).upper().replace('T', 'U').replace(' ', '')
   ```
   
   **原因：** v3 混用 DNA/RNA 字母，导致碱基比较不一致。

### 3.2 模型架构

**集成学习策略：** SVR + Gradient Boosting 平均

#### 模型 1：支持向量回归（SVR）
```python
SVR(kernel='rbf', C=20, epsilon=0.05, gamma=0.05)
```
- 核函数：径向基函数（RBF）
- 正则化：C=20（允许一定拟合）
- ε-不敏感损失：0.05

#### 模型 2：梯度提升回归（GradientBoost）
```python
GradientBoostingRegressor(
    n_estimators=100,
    max_depth=2,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
```
- 100 棵浅层树（max_depth=2）
- 低学习率（0.05）+ 子采样（0.8）防止过拟合

#### 集成方式
```python
y_pred = (y_pred_svr + y_pred_gb) / 2
```

### 3.3 交叉验证

**Leave-One-Out（LOO）交叉验证：**
- 每次留一条样本作为测试集
- 其余 49 条训练
- 重复 50 次，每条样本都作为一次测试

**重要说明：** LOO 主要验证了"对这一条固定 spacer，错配位置行为的学习"，**不能**证明模型对未见过的 crRNA 序列具有泛化能力。

---

## 4. 结果

### 4.1 模型性能

**v4（修正版）表现：**
- **Spearman ρ = 0.692**（p = 2.66×10⁻⁸）
- **Pearson r = 0.703**（p = 1.42×10⁻⁸）
- **RMSE = 0.087**
- **MAE = 0.069**
- **R² = 0.474**

**v3（对比）表现：**
- Spearman ρ = 0.721（p = 3.64×10⁻⁹）
- *注：v3 表面性能更好，但计算方法有误*

**性能可视化：**
- 预测值 vs 实际值散点图显示良好线性关系
- 残差图显示无明显系统性偏差
- 按错配位置分组的 MAE 分析显示不同位置误差分布均匀

### 4.2 候选 crRNA 排名

**选择性排名（修正算法）：**

| 排名 | 候选ID | 目标 | 对29a活性 | 对29b活性 | 对29c活性 | 选择性比率 |
|-----|--------|-----|----------|----------|----------|----------|
| 1 | CRRNA-CORR-008 | 29c | 1.191 | 1.170 | **1.536** | **1.290** |
| 2 | CRRNA-CORR-004 | 29b | 1.259 | **1.536** | 1.358 | 1.132 |
| 3 | CRRNA-CORR-001 | 29a | **1.536** | 1.368 | 1.228 | 1.124 |
| 4 | CRRNA-CORR-010 | 29c | 1.188 | 1.132 | 1.194 | 1.005 |
| 5 | CRRNA-CORR-009 | 29c | 1.194 | 1.119 | 1.193 | 0.999 |

**选择性比率计算（v4 修正）：**
```python
selectivity_ratio = activity_on_target / max(activity_off_target_1, activity_off_target_2)
```

**最优候选 CRRNA-CORR-008：**
- **序列**：`UAGCACCAUUUGAAAUCGGU`（20 nt）
- **目标**：miR-29c
- **选择性**：对 29c 活性比对 29a/b 高 29%
- **设计理念**：利用位置 9 和 18 的单碱基差异实现区分

### 4.3 完全匹配问题

**关键发现：** 对于完全匹配的靶标，所有特征退化为相同值，导致模型预测分数完全一致（小数点后十几位）。

**实测数据：**
```
针对 29a 的 3 条候选：activity = 1.536494（完全相同）
针对 29b 的 4 条候选：部分相同
针对 29c 的 3 条候选：部分相同
```

**原因分析：**
- 8 个特征全部基于"错配位置和类型"
- 没有任何一个特征描述"这条 crRNA 的序列是什么"
- 完全匹配时，特征向量变为固定值：`[-1, 0, 0, 0, 0, 0, 0, 1.0]`

**影响：** 排名实际上是"窗口平移后脱靶错配落在哪个位置"，而不是"哪条 crRNA 本身更好"。

---

## 5. RPA 引物设计

### 5.1 设计策略

**Forward 引物：**
```
5'-[PAM-TTTV]-[miRNA-derived sequence]-3'
```

**要求：**
- 长度 30 nt
- 5' 端包含 LbCas12a PAM（TTTV）
- 3' 端与 RT-cDNA 互补

**实际设计（针对 miR-29a）：**
```
Forward: TTTGTAGCACCAUCUGAAAUCGGUUAGCCA
         ^^^^                         
         PAM (TTTG)
```

### 5.2 EasyDesign 集成测试

**测试配置：**
- 输入：66 nt RT-cDNA 构建
- Guide 长度：25 nt
- Primer 长度：30 nt
- 错配容忍：0

**结果：**
```
Zero targets were found. The number of total primer pairs found was 36 
and the number of them that were suitable (passing basic criteria, e.g., 
on length) was 0.
```

**原因分析：**
- EasyDesign 需要非重叠设计空间：2×30（primers）+ 25（guide）≈ 85 nt
- 66 nt 构建无法容纳
- **这不是bug，是预期结果**

**解决方案（见"下一步计划"）：**
- 重新设计至少 100 nt 的 RT-cDNA 构建
- 加入 adapter/handle 提供额外空间

---

## 6. 当前局限性与风险

### 6.1 数据局限

**问题 1：训练集过小（50 条）**
- 对于机器学习，50 个样本是非常小的训练集
- 容易过拟合，泛化能力受限

**问题 2：单一 spacer 序列**
- 所有 50 条数据使用同一条 spacer：`CGGUUCGCGUGGAUUAAAGG`
- 模型从未见过"换一条 crRNA 会怎么样"
- **无法学习序列本身的影响**

**问题 3：缺失序列特征**
- 8 个特征全部描述"错配的位置和类型"
- 没有任何特征描述"crRNA 自身序列组成"
- GC 含量、二级结构、序列基序等信息未纳入

### 6.2 模型局限

**问题 4：完全匹配预测失效**
- 对完全匹配靶标，特征向量固定
- 不同候选打出完全相同的分数
- 排名依赖脱靶错配位置，而非 crRNA 本身质量

**问题 5：外部验证缺失**
- LOO 交叉验证只是内部验证
- 未在独立数据集上测试
- 未在真实新 crRNA 上实验验证

### 6.3 实验风险

**问题 6：RT-cDNA 构建过短**
- 66 nt 无法容纳 EasyDesign 推荐的标准布局
- 需要重新设计更长构建（≥100 nt）

**问题 7：选择性未经湿实验验证**
- 当前排名完全基于计算预测
- 实际特异性需要合成 miR-29a/b/c 标准品测试
- 可能存在意外交叉反应

---

## 7. 下一步计划

### 7.1 短期（1-2周）：湿实验启动

**优先级：最高**

导师明确指出：**"实验方案这块不必等计算部分出结果，先按现成工具的推荐把序列定下来，把体系搭起来，边做边调。"**

**具体行动：**

1. **重新设计 RT-cDNA 构建（≥100 nt）**
   ```
   [5' adapter/handle] - [miRNA-derived] - [3' adapter/handle]
   
   要求：
   - 总长 ≥100 nt
   - 包含 TTTV PAM
   - 跨越 miR-29a/c 差异位点
   - 允许 25 nt guide + 2×30 nt primers 布局
   ```

2. **运行 EasyDesign 获取推荐序列**
   ```bash
   python design.py \
     --input mir29a_extended_construct.fasta \
     --specific-against mir29bc_exclusion.fasta \
     --guide-length 25 \
     --primer-length 30 \
     --mode rpa
   ```

3. **订购试剂**
   - 合成 miR-29a/b/c 标准品（各 100 pmol）
   - 订购 EasyDesign 推荐的 crRNA 和 RPA 引物
   - 准备 LbCas12a 酶和 RPA 试剂盒

4. **建立检测体系**
   - 优化 RT 反应条件
   - 优化 RPA 扩增条件（温度、时间）
   - 优化 Cas12a 切割反应（酶浓度、反应时间）

### 7.2 中期（2-4周）：扩充训练数据

**优先级：高**

这是**导师最推荐的计算方向**："最值得投入的方向是去找公开数据。"

**公开数据源（按优先级）：**

1. **EasyDesign 训练数据**
   - 仓库：https://github.com/scRNA-Compt/EasyDesign
   - 优势：同样针对 Cas12a，深度学习方法
   - 数据规模：远超 50 条

2. **NucleaSeq 平台数据**
   - 仓库：https://github.com/finkelsteinlab/nucleaseq
   - 规模：>10,000 条含错配靶标
   - 特色：切割动力学数据（不仅是终点活性）

3. **体外质粒文库数据**
   - 代码：https://github.com/sashital-lab/Cas12a_nickase
   - 数据：https://iastate.figshare.com/articles/dataset/8178938/3
   - 优势：场景与本项目高度接近

4. **NCBI PRJNA1074843**
   - 数据：https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1074843
   - 代码：https://codeocean.com/capsule/9398276/tree/v1
   - 规模：20+ Cas12a 变体，数千条靶标

**数据整合策略：**
```python
# 伪代码示意
combined_data = []
combined_data.extend(load_easydesign_data())   # 假设 5000 条
combined_data.extend(load_nucleaseq_data())    # 假设 10000 条
combined_data.extend(current_50_samples)       # 原有 50 条

# 划分训练/测试集（按 spacer 序列分组）
unique_spacers = get_unique_spacers(combined_data)
train_spacers, test_spacers = train_test_split(unique_spacers, test_size=0.2)

# 确保测试集包含完全未见过的 spacer
train_data = filter_by_spacers(combined_data, train_spacers)
test_data = filter_by_spacers(combined_data, test_spacers)
```

**新特征探索：**
- GC 含量（整体 + 局部）
- 序列基序（k-mer 频率）
- 二级结构预测（折叠自由能）
- crRNA 长度变化（16 nt vs 20 nt）

### 7.3 长期（1-2个月）：模型改进

**优先级：中**

1. **增加序列描述特征**
   ```python
   # 新增特征示例
   def extract_sequence_features(spacer, target):
       gc_content = (spacer.count('G') + spacer.count('C')) / len(spacer)
       seed_gc = (spacer[:8].count('G') + spacer[:8].count('C')) / 8
       self_fold_dg = RNA.fold(spacer)[1]  # 自折叠能量
       ...
   ```

2. **探索深度学习模型**
   - CNN：捕捉局部序列模式
   - RNN/LSTM：捕捉位置依赖
   - Attention：学习关键位置权重

3. **crRNA 长度优化**
   - 测试 16 nt、18 nt、20 nt、22 nt
   - 文献表明 16 nt 可能提高单碱基区分能力

4. **多目标优化**
   - 同时优化：活性、选择性、合成难度
   - Pareto 前沿分析

---

## 8. 项目管理建议

### 8.1 文件组织（已完成）

当前项目结构清晰规范：
```
mirna-crispr-ml/
├── data/
│   ├── raw/                 # 原始数据
│   └── processed/           # 处理后数据
├── notebooks/               # Jupyter notebooks (01-05编号)
├── results/
│   ├── figures/            # 图表
│   └── metrics/            # 评估指标
├── environment_records/    # 环境配置
├── easy_design_mir29a/    # EasyDesign 分析
├── tools/                  # 第三方工具
└── scripts/                # 脚本（预留）
```

**待清理：**
- 手动删除 `mirna-crispr/`（重复文件夹）
- 删除所有 `.ipynb_checkpoints/`

### 8.2 版本控制建议

**Git 提交策略：**
```bash
# 提交当前整理
git add .
git commit -m "docs: Add comprehensive project report and reorganize structure"

# 分支管理
git checkout -b feature/public-data-integration  # 数据整合分支
git checkout -b feature/wet-lab-redesign        # 实验设计分支
```

### 8.3 国际化建议

如果参加国际活动，需要：
1. ✅ 文件名已改为英文（notebooks 01-05）
2. ⚠️ 注释仍为中文，需要改为英文
3. ⚠️ 变量名部分为拼音，建议改为英文

**示例改进：**
```python
# 当前（中文注释）
def extract_final_features_v4(spacer, target, mismatch_pos):
    """提取最终特征"""
    # 计算双链能量
    
# 建议（英文注释）
def extract_final_features_v4(spacer, target, mismatch_pos):
    """Extract v4 corrected features for LbCas12a activity prediction."""
    # Calculate duplex folding energy with corrected orientation
```

---

## 9. 结论

本项目成功建立了一个 miR-29 家族选择性检测的计算设计框架，包含完整的数据处理、特征工程、模型训练和候选筛选流程。v4 版本主动发现并修正了 v3 中的能量计算错误，体现了良好的科学自查意识。

**项目亮点：**
1. ✅ 完整的机器学习 pipeline
2. ✅ 主动发现并修正计算错误（v3→v4）
3. ✅ 完善的环境记录和可复现性文档
4. ✅ 清晰的文件组织和版本管理

**当前限制：**
1. ⚠️ 训练数据量小（50条）且单一 spacer
2. ⚠️ 模型对新 crRNA 序列泛化能力未验证
3. ⚠️ 完全匹配预测失效（特征退化问题）
4. ⚠️ RT-cDNA 构建过短，无法运行 EasyDesign

**推荐行动（优先级排序）：**
1. **立即**：重新设计 ≥100 nt RT-cDNA 构建，运行 EasyDesign，启动湿实验
2. **短期（2-4周）**：下载并整合公开数据集，重新训练模型
3. **中期（1-2月）**：增加序列特征，探索深度学习方法

**最重要的一点**：如导师所说，**"计算这部分的价值更多在于把设计思路讲清楚，写在报告里会是一个完整的亮点，但它不必成为实验的前置条件。"** 现在应该让湿实验和计算改进并行推进，而不是等计算"完美"了才动手实验。

---

## 10. 参考文献与资源

### 10.1 公开数据资源

1. EasyDesign: https://github.com/scRNA-Compt/EasyDesign
2. NucleaSeq: https://github.com/finkelsteinlab/nucleaseq
3. Cas12a nickase library: https://github.com/sashital-lab/Cas12a_nickase
4. NCBI BioProject: https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1074843

### 10.2 关键工具

1. ViennaRNA: RNA 二级结构和热力学计算
2. scikit-learn: 机器学习模型
3. EasyDesign: Cas12a/RPA 一体化设计平台

### 10.3 项目文件

| 文件 | 路径 | 描述 |
|-----|------|------|
| 训练数据 | `data/raw/cas12a_mismatch_data.csv` | 50 条 LbCas12a 错配数据 |
| 候选序列 | `data/raw/candidate_sequences_corrected.csv` | 10 条候选 crRNA |
| v4 模型 | `notebooks/04_final_model_audit_v2.ipynb` | 完整训练和评估代码 |
| 排名结果 | `data/processed/ranked_crRNA_selectivity_v4_corrected.csv` | 选择性排名 |
| 环境记录 | `environment_records/` | 完整可复现环境 |
| EasyDesign 报告 | `easy_design_mir29a/EASYDESIGN_RUN_REPORT.md` | EasyDesign 运行记录 |

---

**报告生成时间：** 2026年10月1日  
**项目路径：** `D:\Evelyn\mirna-crispr-ml`  
**报告版本：** v1.0
