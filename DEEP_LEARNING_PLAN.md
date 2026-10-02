# v7 深度学习模型设计方案

**目标**: 突破 v6 的 Spearman ρ = 0.541 瓶颈，目标 0.60-0.70

---

## 🧠 架构设计

### 整体思路：混合模型（Hybrid Model）

```
输入:
  ├─ 序列输入: spacer (20 nt) + target (20 nt)
  └─ 手工特征: v6 的 20 个特征

编码层:
  ├─ 序列编码器 (Sequence Encoder)
  │   ├─ One-hot 编码: (20, 4) 
  │   ├─ 1D CNN: 提取局部 motif
  │   └─ Bi-LSTM: 捕捉长距离依赖
  │
  └─ 特征编码器 (Feature Encoder)
      └─ Dense layers: 编码手工特征

融合层:
  └─ 拼接 + Attention 机制

输出层:
  └─ Dense → 预测相对活性
```

---

## 🔬 详细架构

### 1. 序列编码器（Sequence Encoder）

**输入**: spacer (20 nt) + target (20 nt) → 拼接成 40 nt

**One-hot 编码**:
```
A → [1, 0, 0, 0]
C → [0, 1, 0, 0]
G → [0, 0, 1, 0]
U → [0, 0, 0, 1]
```
输出 shape: (40, 4)

**1D Convolutional Layers**:
```python
Conv1D(filters=64, kernel_size=3) → 学习 3-mer motif
Conv1D(filters=128, kernel_size=5) → 学习 5-mer motif
MaxPooling1D(pool_size=2)
```
作用: 识别局部序列模式（如 poly-A、GC-rich 区域）

**Bidirectional LSTM**:
```python
Bidirectional(LSTM(64, return_sequences=False))
```
作用: 捕捉前后依赖关系（PAM 侧和远端的相互作用）

**输出**: 128 维向量（序列的抽象表示）

---

### 2. 特征编码器（Feature Encoder）

**输入**: v6 的 20 个手工特征

**Dense Layers**:
```python
Dense(64, activation='relu')
Dropout(0.3)
Dense(32, activation='relu')
```

**输出**: 32 维向量

---

### 3. 融合层（Fusion Layer）

**拼接**:
```python
concat = Concatenate()([sequence_encoding, feature_encoding])
# 128 + 32 = 160 维
```

**Attention 机制** (可选):
```python
attention = Dense(160, activation='softmax')(concat)
weighted = Multiply()([concat, attention])
```
作用: 让模型学习哪些信息最重要

**全连接层**:
```python
Dense(128, activation='relu')
Dropout(0.3)
Dense(64, activation='relu')
Dropout(0.2)
```

---

### 4. 输出层

```python
Dense(1, activation='linear')  # 回归任务
```

---

## 📊 为什么这个架构可能更好？

### 优势 1: 保留序列信息
- v6: spacer → 10个数字（丢失序列模式）
- v7: spacer → one-hot → CNN/LSTM（保留完整信息）

### 优势 2: 多尺度特征
- CNN: 局部模式（3-mer, 5-mer）
- LSTM: 全局依赖
- 手工特征: 领域知识

### 优势 3: 自动发现交互
- 深度网络可以学习任意高阶特征交互
- 不需要手动设计 `pos × ddg` 这样的交互项

---

## 🛠️ 实施细节

### 训练策略

**1. 数据划分**:
- 训练集: 80% (7,979 条)
- 验证集: 10% (997 条) - 用于早停
- 测试集: 10% (998 条) - 最终评估

**2. 损失函数**:
```python
loss = 'mse'  # 均方误差
# 或者组合损失: MSE + Spearman loss
```

**3. 优化器**:
```python
Adam(learning_rate=0.001)
```

**4. 早停（Early Stopping）**:
```python
EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)
```

**5. 学习率衰减**:
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10
)
```

---

## 🎯 预期性能

| 指标 | v6 | v7 目标 | 改进 |
|------|----|---------|----- |
| Spearman ρ | 0.541 | **0.60-0.65** | ↑ 11-20% |
| R² | 0.262 | **0.40-0.50** | ↑ 53-91% |
| RMSE | 0.182 | **0.15-0.17** | ↓ 7-18% |

### 如果性能不理想
- 可能需要更多数据（当前 9,974 条可能不够）
- 或者架构需要调整（更深/更浅）
- 或者数据本身噪音太大

---

## 📦 实施步骤

### 第1步: 准备数据
- 创建序列 one-hot 编码函数
- 创建数据生成器（tf.data.Dataset）

### 第2步: 构建模型
- 定义序列编码器
- 定义特征编码器
- 融合并输出

### 第3步: 训练
- 训练 50-100 epochs
- 监控验证集性能
- 早停防止过拟合

### 第4步: 评估
- 对比 v6 vs v7
- 分析哪些样本预测改善了

### 第5步（可选）: 模型融合
- 集成 v6（传统ML）+ v7（深度学习）
- 可能获得最佳性能

---

## ⏱️ 时间估计

- 编写代码: 40 分钟
- 训练模型: 20-30 分钟（取决于 CPU/GPU）
- 评估分析: 10 分钟
- **总计: 1-1.5 小时**

---

## 🔧 技术栈

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
```

如果没有 GPU:
- CPU 训练也可以，就是慢一点（20-30分钟）
- 可以减小模型规模加速

---

## 📝 注意事项

1. **保留 v6 文件**: 所有新文件放在 `models/v7/`
2. **过拟合风险**: 深度学习容易过拟合，需要 Dropout + 早停
3. **可解释性**: 深度学习是黑盒，难以解释为什么预测某个值
4. **数据量**: 9,974 条对深度学习来说偏少，可能无法发挥全部潜力

---

**准备好了吗？我们开始实现！**
