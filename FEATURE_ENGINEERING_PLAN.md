# v6 模型特征工程改进计划

**目标**: 从 8 个特征扩展到 ~20 个特征，提升 Spearman ρ 从 0.515 到 0.60+

---

## 🔍 当前特征分析（v5.1）

### 8 个现有特征
1. `mismatch_pos`: 错配位置
2. `ddg`: ΔΔG（双链稳定性）
3. `is_purine_mismatch`: 嘌呤-嘌呤错配
4. `is_transition`: 转换突变
5. `is_edge`: 边缘区域
6. `pos_ddg_interaction`: 位置×ΔΔG
7. `mismatch_score`: 错配评分
8. `seed_weight`: 种子区权重

### 问题
- ❌ 缺少序列组成特征（GC含量）
- ❌ 缺少局部区域特征（PAM、seed、edge详细信息）
- ❌ 缺少二级结构特征
- ❌ 二元特征太多（信息量低）
- ❌ 只考虑单个错配

---

## ✨ 新增特征（12个）

### A. 序列组成特征（4个）

**9. `spacer_gc_content`** - spacer 的 GC 含量
- 计算: `(G数量 + C数量) / 总长度`
- 原理: GC含量影响双链稳定性和切割效率
- 范围: 0.0 - 1.0

**10. `target_gc_content`** - target 的 GC 含量
- 同上
- 原理: target GC含量影响靶标可及性

**11. `gc_difference`** - spacer 和 target 的 GC 差异
- 计算: `|spacer_gc - target_gc|`
- 原理: GC 差异越大，错配可能越不稳定

**12. `poly_base_score`** - 连续相同碱基惩罚
- 计算: 检测连续3+个相同碱基（如 AAAA, GGGG）
- 惩罚: 每段连续碱基 +1
- 原理: poly-A/T/C/G 影响二级结构和切割效率

---

### B. 局部区域特征（4个）

**13. `seed_region_mismatches`** - 种子区（1-8 nt）错配数
- 计算: 统计前8个碱基的错配数
- 原理: Cas12a 的种子区（PAM 侧）对错配最敏感

**14. `pam_proximal_mismatches`** - PAM 近端（1-5 nt）错配数
- 计算: 统计前5个碱基的错配数
- 原理: PAM 近端错配影响最大

**15. `middle_region_mismatches`** - 中间区（6-15 nt）错配数
- 计算: 统计中间10个碱基的错配数

**16. `distal_region_mismatches`** - 远端区（16-20 nt）错配数
- 计算: 统计后5个碱基的错配数
- 原理: 远端错配容忍度较高

---

### C. 二级结构特征（2个）

**17. `spacer_self_structure`** - spacer 自身二级结构稳定性
- 计算: 使用 ViennaRNA `RNA.fold(spacer)` 计算最小自由能
- 单位: kcal/mol
- 原理: spacer 自身形成发卡结构会降低活性

**18. `target_self_structure`** - target 自身二级结构稳定性
- 计算: `RNA.fold(target)` 的最小自由能
- 原理: target 二级结构强会阻碍 guide 结合

---

### D. 错配模式特征（2个）

**19. `total_mismatches`** - 总错配数
- 计算: 统计所有错配位置
- 原理: 当前只考虑单个错配，但数据中有多错配样本

**20. `mismatch_density`** - 错配密度（针对多错配）
- 计算: 错配最远位置 - 最近位置，衡量错配分布
- 原理: 集中的错配比分散的错配影响更大

---

## 📊 特征重要性预测

### 预期最有用的特征（Top 5）
1. **seed_region_mismatches** ⭐⭐⭐⭐⭐
   - Cas12a 机制决定：PAM 侧种子区最重要
   
2. **spacer_gc_content** ⭐⭐⭐⭐
   - 影响双链稳定性和切割效率
   
3. **total_mismatches** ⭐⭐⭐⭐
   - 当前模型忽略了多错配信息
   
4. **spacer_self_structure** ⭐⭐⭐
   - 二级结构是已知影响因素
   
5. **pam_proximal_mismatches** ⭐⭐⭐
   - PAM 近端最敏感

### 预期中等有用（可能有帮助）
- gc_difference
- target_self_structure
- middle_region_mismatches

### 预期作用较小（但保留以防万一）
- poly_base_score
- distal_region_mismatches
- mismatch_density

---

## 🔧 实施步骤

### 第1步：创建新特征提取函数
- 文件: `scripts/train_model_v6.py`
- 函数: `extract_features_v6(spacer, target, mismatch_pos=None)`
- 输出: 20 个特征的 list

### 第2步：训练 v6 模型
- 使用相同的数据集（9,974 条）
- 使用相同的模型架构（SVR + GradientBoosting）
- 对比 v5.1 vs v6 性能

### 第3步：特征重要性分析
- 使用 GradientBoosting 的 `feature_importances_`
- 绘制特征重要性图
- 找出最有用的特征

### 第4步（如果需要）：特征选择
- 如果某些特征完全无用，可以移除
- 创建 v6.1（精简版）

---

## 🎯 预期性能提升

| 指标 | v5.1 | v6 目标 | 改进 |
|------|------|---------|------|
| Spearman ρ | 0.515 | **0.60-0.65** | ↑ 16-26% |
| R² | 0.236 | **0.35-0.45** | ↑ 48-91% |
| RMSE | 0.185 | **0.16-0.17** | ↓ 8-14% |

### 如果性能提升不明显
- 可能数据本身噪音较大
- 或者特征选择不够好
- 需要考虑深度学习方法

---

## ⏱️ 预计时间
- 编写特征提取函数: 30分钟
- 训练 v6 模型: 10分钟
- 特征重要性分析: 10分钟
- **总计: ~50分钟**

---

## 📝 注意事项

1. **保留 v5.1 文件**：不覆盖，创建新目录 `models/v6/`
2. **特征标准化**：所有特征都会被 StandardScaler 标准化
3. **缺失值处理**：如果某个样本无法计算某特征，填充 0 或均值
4. **ViennaRNA 调用**：二级结构计算可能较慢，已在 v5.1 中优化

---

**准备好了吗？我们开始实施！**
