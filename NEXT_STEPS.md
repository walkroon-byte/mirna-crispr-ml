# v5 模型训练完成 - 下一步建议

**日期**: 2026-10-01  
**当前状态**: v5 模型已训练完成，但性能不如预期

---

## 📊 v5 模型表现总结

### 训练结果
- ✅ 成功训练，模型保存在 `models/v5/`
- ✅ v4 模型已保留（未覆盖）
- ⚠️ 测试集 Spearman ρ = **0.474**（中等相关性）

### 关键问题
1. **数据损失**: 10,714 条 → 7,985 条（25% 包含gap字符）
2. **预测偏保守**: 模型严重低估高活性样本
3. **活性值归一化**: 可能是主要瓶颈

---

## 🎯 三个选项

### 选项 A: 立即使用 v5 模型预测 ⏱️ 30分钟
**适合**: 想快速看看结果，接受一定不确定性

**操作**:
```bash
# 1. 创建预测脚本
python scripts/predict_with_v5.py \
    --input easy_design_mir29a/mir29_rt_cdna_constructs.fasta \
    --output predictions_v5.csv

# 2. 选择 top 5-10 个候选
# 3. 对比 v4 和 v5 的预测（如果序列相同）
```

**优点**: 
- 快速得到结果
- v5 能预测任意 guide-target 组合
- 至少比随机选择好

**缺点**:
- 预测可能偏保守
- 需要选更多候选（5-10个而非2-3个）
- 实验验证风险较高

**成本**: ~$1,330 试剂 + 实验时间

---

### 选项 B: 改进数据处理，训练 v5.1 ⏱️ 1-2小时 ⭐ **推荐**
**适合**: 想要更可靠的预测，愿意投入少量额外时间

**具体改进**:
1. **修复活性值归一化** （核心改进）
   ```python
   # 当前方法（sigmoid）- 问题
   relative_activity = 1 / (1 + np.exp(-activity))
   # 结果范围: 0.015 - 0.426
   
   # 改进方法 - 匹配你的原始数据范围
   # 你的数据: 0.7 - 1.5，中心约 1.0
   z_score = (activity - activity.mean()) / activity.std()
   relative_activity = 1.0 + 0.25 * z_score  # 中心1.0，范围约0.4-1.6
   ```

2. **预处理过滤gap序列**
   ```python
   # 在转换时就过滤，而非特征提取时失败
   df = df[~df['guide_sequence'].str.contains('-')]
   df = df[~df['target_sequence'].str.contains('-')]
   ```

3. **数据质量检查**
   - 移除异常活性值（> 3σ）
   - 验证序列长度和格式

**预期改进**:
- Spearman ρ 从 0.474 → **0.55-0.65** （保守估计）
- 预测值范围更合理
- 高活性样本不再被严重低估

**操作**:
```bash
# 1. 修改转换脚本
python scripts/convert_easydesign_to_v4_format.py --improved

# 2. 重新训练
python scripts/train_model_v5.py --version v5.1

# 3. 对比 v5 vs v5.1 性能
python scripts/compare_models.py --models v5 v5.1

# 4. 选择更好的模型用于预测
```

**成本**: 1-2 小时时间

---

### 选项 C: 保守策略 - 暂缓订购试剂 ⏱️ 数周
**适合**: 对模型性能不满意，想要更高置信度

**操作**:
1. **寻找更多数据**
   - 搜索最新的 Cas12a 文献（2024-2026）
   - 联系其他实验室获取数据
   - 考虑生成少量实验数据来验证模型

2. **使用文献验证的设计**
   - 查找已发表的 miRNA 检测 crRNA
   - 直接使用验证过的 guide 序列
   - 修改用于 miR-29a

3. **重新评估实验设计**
   - 是否必须用 Cas12a？考虑其他方法
   - 简化检测系统
   - 降低特异性要求

**成本**: 数周时间，可能需要文献调研或额外实验

---

## 🔍 详细诊断：为什么 v5 性能不理想？

### 问题 1: 活性值归一化错误 🔥 **最可能**
**证据**:
- EasyDesign 活性值: -4.2 到 -0.3 (log scale)
- Sigmoid 转换后: 0.015 到 0.426
- 你的原始数据: 0.7 到 1.5
- **量级不匹配！**

**影响**:
- 训练数据的活性范围太窄
- 模型学不到高活性和低活性的真正差异
- 预测时自然压缩在低值范围

**解决方案**: 重新归一化使数据分布一致

---

### 问题 2: 25% 数据损失
**证据**:
- 2,729 条序列包含 gap 字符（'-'）
- 这些是序列比对的结果，不是真实序列

**影响**:
- 损失 1/4 训练数据
- 可能包含一些高质量样本

**解决方案**: 预处理时就过滤，避免浪费计算

---

### 问题 3: 特征提取可能不适配 25 nt
**证据**:
- v4 特征设计基于 20 nt
- EasyDesign 数据是 25 nt
- 当前做法：截断到 20 nt

**影响**:
- 丢失末端 5 nt 的信息
- seed region 定义可能不准确

**解决方案**: 调整特征提取函数适配 25 nt

---

## 💡 我的推荐

### 优先级排序
1. **第一优先**: 选项 B（1-2小时）
   - 修复活性值归一化 ← **最关键**
   - 预处理过滤 gap 序列
   - 重新训练 v5.1

2. **如果 v5.1 性能好** (Spearman > 0.6)
   - 使用 v5.1 预测
   - 选择 top 5 候选
   - 订购试剂

3. **如果 v5.1 性能仍不理想** (Spearman < 0.5)
   - 选项 C：暂缓订购
   - 寻找更多数据或使用文献方案

---

## 📝 立即行动清单

### 如果选择选项 B（推荐）

**第 1 步**: 修改 `scripts/convert_easydesign_to_v4_format.py`
```python
# 在 normalize_activity() 函数中修改
def normalize_activity(activity_values):
    # 改进版：匹配原始数据分布
    z_score = (activity_values - activity_values.mean()) / activity_values.std()
    relative = 1.0 + 0.25 * z_score  # 中心在1.0，标准差0.25
    return relative

# 在 convert_easydesign_data() 中添加预过滤
def convert_easydesign_data():
    df = pd.read_csv(INPUT_FILE)
    
    # 添加这两行
    df = df[~df['guide_sequence'].str.contains('-', na=False)]
    df = df[~df['target_sequence'].str.contains('-', na=False)]
    
    log(f"过滤gap后: {len(df)} 行")
    # ... 继续原有逻辑
```

**第 2 步**: 重新运行转换
```bash
python scripts/convert_easydesign_to_v4_format.py
```

**第 3 步**: 训练 v5.1
```bash
# 修改 train_model_v5.py 中的输出目录
OUTPUT_DIR = PROJECT_ROOT / "models/v5.1"
# 然后运行
python scripts/train_model_v5.py
```

**第 4 步**: 对比性能
```bash
# 检查 v5.1 的 Spearman ρ
cat models/v5.1/v5_model_metrics.json | grep "spearman_r"
```

**预期时间**: 1-2 小时

---

## ❓ 需要你的决定

请告诉我你想选择哪个选项：

**A.** 立即用 v5 预测（快速但有风险）  
**B.** 改进数据后训练 v5.1（推荐，1-2小时）  
**C.** 暂缓订购试剂（保守策略）

或者你有其他想法？

---

## 📂 生成的文件

1. ✅ `models/v5/v5_svr_model.pkl` - SVR 模型
2. ✅ `models/v5/v5_gb_model.pkl` - GradientBoosting 模型  
3. ✅ `models/v5/v5_model_metrics.json` - 性能指标
4. ✅ `models/v5/v5_test_predictions.csv` - 测试集预测
5. ✅ `models/v5/v5_model_performance.png` - 性能图
6. ✅ `V5_MODEL_TRAINING_REPORT.md` - 详细报告

---

**备注**: git 提交遇到了锁文件权限问题，但所有文件已经生成。你可以稍后手动提交：
```bash
cd D:\Evelyn\mirna-crispr-ml
# 如果锁文件还在，先关闭所有 git 相关程序，然后：
del .git\index.lock  # Windows
# 或
rm .git/index.lock   # Linux/Mac

git add models/v5/ scripts/ V5_MODEL_TRAINING_REPORT.md NEXT_STEPS.md
git commit -m "训练 v5 模型（使用公开数据集）"
```
