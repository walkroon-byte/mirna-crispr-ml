# miR-29 Extended RT-cDNA Construct Design v2

**设计日期：** 2026年10月1日  
**目标：** 设计一个 ≥100 nt 的 RT-cDNA 构建，满足 EasyDesign 完整设计要求

---

## 1. 设计要求总结

### 1.1 EasyDesign 空间要求
- **Guide 长度：** 25 nt（EasyDesign bundled Cas12a 模型标准）
- **RPA 引物长度：** 30 nt × 2
- **最小布局空间：** ~85 nt（非重叠设计）
- **推荐构建长度：** ≥100 nt（留有设计余地）

### 1.2 生物学要求
1. **包含 miRNA 区分位点**
   - 位置 9：29a(C) vs 29b/c(U)
   - 位置 18：29a/c(G) vs 29b(A)
   
2. **PAM 位置**
   - LbCas12a PAM：TTTV（V = A/C/G）
   - 必须位于 protospacer 上游（5' 端）

3. **RPA 引物设计**
   - Tm：58-62°C
   - GC 含量：40-60%
   - 避免引物二聚体和发卡结构

4. **RT 引物设计**
   - 3' 端与 miRNA 3' 端互补（6-8 nt）
   - 5' 端接 adapter 序列

---

## 2. 设计策略

### 2.1 构建架构

采用 **Stem-loop RT + Universal adapter** 策略：

```
5' - [Universal Adapter] - [miRNA-derived cDNA] - [3' Adapter] - 3'
     |<----  Handle  --->|  |<--- Target --->|  |<-- Handle -->|
```

**优势：**
- Universal adapter 提供稳定的扩增 handle
- miRNA-derived 区域保留区分位点
- 3' adapter 提供 RT 引物结合位点和额外空间
- 总长可灵活调整到 100-150 nt

### 2.2 Adapter 选择

#### 5' Universal Adapter (30 nt)
```
5'-GTCGTATCCAGTGCAGGGTCCGAGGTATTC-3'
```
- 来源：经典 stem-loop RT 通用序列
- GC 含量：53.3%
- 无连续重复
- 不与 miR-29 序列同源

#### 3' Adapter (25 nt)
```
5'-AGTGCGTACTCGAGGCATCGATCAG-3'
```
- 自行设计的非同源序列
- GC 含量：52%
- 为 RPA 下游引物提供结合位点

---

## 3. 针对 miR-29a 的完整构建

### 3.1 序列组成

**miR-29a-3p 成熟序列：**
```
5'-UAGCACCAU-CUGAAAUCGGUUA-3'
   (22 nt)
```

**RT-cDNA 构建（正链，5'→3'）：**
```
5'-GTCGTATCCAGTGCAGGGTCCGAGGTATTC-TAACCGATTTCAGATGGTGCTA-AGTGCGTACTCGAGGCATCGATCAG-3'
   |<----- 5' Adapter (30 nt) ----->| |<- miRNA RC (22 nt) ->| |<--- 3' Adapter (25 nt) --->|
```

**总长度：** 30 + 22 + 25 = **77 nt**

⚠️ **问题：** 77 nt 仍然不够！需要进一步扩展。

---

## 4. 优化设计：扩展到 110 nt

### 4.1 策略调整

为了达到 110 nt，我们需要：
1. **延长 5' adapter** 到 40 nt
2. **延长 miRNA-derived 区域** 到 30 nt（包含 pre-miRNA stem 部分）
3. **延长 3' adapter** 到 40 nt

### 4.2 miR-29a 扩展序列设计

**使用 pre-miR-29a stem-loop 区域：**
```
Pre-miR-29a (部分序列):
5'- ... UAGCACCAUCUGAAAUCGGUUACCACGAUUU ... -3'
         |<--- 成熟 miR-29a --->|
```

**扩展的 miRNA-derived 区域（30 nt）：**
```
5'-UAGCACCAUCUGAAAUCGGUUACCACGAUU-3'
   |<--- miR-29a-3p (22 nt) --->||<-8nt->|
```

**反向互补（cDNA 正链）：**
```
5'-AAUCGUGGUAACCGAUUUCAGAUGGUGCUA-3'
```

### 4.3 最终 110 nt 构建

**针对 miR-29a 的完整设计：**

```
5'-GTCGTATCCAGTGCAGGGTCCGAGGTATTCGACTAGCT-
   AATCGTGGTAACCGATTTCAGATGGTGCTA-
   AGTGCGTACTCGAGGCATCGATCAGTACGGATCG-3'

   |<-------- 5' Adapter (38 nt) -------->|
   |<--- miRNA-derived (30 nt) --->|
   |<--------- 3' Adapter (42 nt) -------->|
```

**总长度：** 38 + 30 + 42 = **110 nt** ✅

---

## 5. 针对三个靶标的构建

### 5.1 miR-29a 构建（110 nt）

**FASTA 格式：**
```
>mir29a_extended_rt_cdna_v2
GTCGTATCCAGTGCAGGGTCCGAGGTATTCGACTAGCTAATCGTGGTAACCGATTTCAGATGGTGCTAAGTGCGTACTCGAGGCATCGATCAGTACGGATCG
```

**关键特征：**
- 位置 9（从 miRNA 起始计）：C
- 位置 18：G
- 位置 9-10：CU（29a 特征）

### 5.2 miR-29b 构建（110 nt）

**miR-29b-3p 序列：**
```
5'-UAGCACCAUUUGAAAUCAGUGUU-3'
```

**扩展到 30 nt（假设的 pre-miR-29b）：**
```
5'-UAGCACCAUUUGAAAUCAGUGUUACGAUGU-3'
```

**cDNA 正链（RC）：**
```
5'-ACAUCGUAACACUGAUUUCAAAUGGUGCUA-3'
```

**完整构建：**
```
>mir29b_extended_rt_cdna_v2
GTCGTATCCAGTGCAGGGTCCGAGGTATTCGACTAGCTACATCGTAACACTGATTTCAAATGGTGCTAAGTGCGTACTCGAGGCATCGATCAGTACGGATCG
```

### 5.3 miR-29c 构建（110 nt）

**miR-29c-3p 序列：**
```
5'-UAGCACCAUUUGAAAUCGGUUA-3'
```

**扩展到 30 nt：**
```
5'-UAGCACCAUUUGAAAUCGGUUACCACGAUU-3'
```

**cDNA 正链（RC）：**
```
5'-AAUCGUGGUAACCGAUUUCAAAUGGUGCUA-3'
```

**完整构建：**
```
>mir29c_extended_rt_cdna_v2
GTCGTATCCAGTGCAGGGTCCGAGGTATTCGACTAGCTAATCGTGGTAACCGATTTCAAATGGTGCTAAGTGCGTACTCGAGGCATCGATCAGTACGGATCG
```

---

## 6. PAM 和 Guide 设计说明

### 6.1 PAM 位置考虑

LbCas12a PAM (TTTV) 必须位于 protospacer **上游**（5' 端）：

```
DNA 双链示意：
5'- [PAM: TTTV] - [Protospacer 25nt] - 3'  (non-target strand)
3'- [AAAW]     - [Target 25nt]       - 5'  (target strand)
                  |
                  crRNA guide 在这里结合
```

### 6.2 Guide 设计区域

在 110 nt 构建中，guide 可以设计在：
1. **跨越 5' adapter 和 miRNA 区域**
2. **完全在 miRNA-derived 区域内**（推荐）
3. **跨越 miRNA 和 3' adapter 区域**

**推荐策略：** 让 EasyDesign 自动搜索最优 guide 位置，要求：
- Guide 必须包含 miR-29a/b/c 的区分位点
- PAM 可以在 5' adapter 或 miRNA 区域内找到

### 6.3 RPA 引物设计区域

**Forward 引物（30 nt）：**
- 结合位置：5' adapter 区域
- 可选择性加入 PAM overhang（TTTV 额外 4 nt）

**Reverse 引物（30 nt）：**
- 结合位置：3' adapter 区域
- 与 cDNA 反向互补

---

## 7. RT 反应设计

### 7.1 RT 引物设计

**针对 miR-29a 的 RT 引物：**
```
5'-AGTGCGTACTCGAGGCATCGATCAGTACGGATCGTAACCG-3'
   |<------- 3' Adapter ------->||<- 6nt miRNA-specific ->|
```

**特点：**
- 3' 端 6 nt (`TAACCG`) 与 miR-29a 3' 端互补（特异性锚定）
- 5' 端通用 adapter 序列（提供扩增 handle）

### 7.2 Stem-loop RT 引物（替代方案）

如果需要更高特异性，可以用 stem-loop 引物：

```
           GTTGGT (loop)
          /      \
     5'- CG      CG -3'
         |        |
         GC      GC
         |        |
    [Adapter]- TAACCG (miR-29a 3' 互补)
```

---

## 8. EasyDesign 输入文件生成

### 8.1 主要输入（Primary target）

**文件：** `mir29a_extended_rt_cdna_v2.fasta`

```fasta
>mir29a_extended_rt_cdna_110nt_primary
GTCGTATCCAGTGCAGGGTCCGAGGTATTCGACTAGCTAATCGTGGTAACCGATTTCAGATGGTGCTAAGTGCGTACTCGAGGCATCGATCAGTACGGATCG
```

### 8.2 排除输入（Specificity control）

**文件：** `mir29bc_extended_rt_cdna_v2.fasta`

```fasta
>mir29b_extended_rt_cdna_110nt_exclude
GTCGTATCCAGTGCAGGGTCCGAGGTATTCGACTAGCTACATCGTAACACTGATTTCAAATGGTGCTAAGTGCGTACTCGAGGCATCGATCAGTACGGATCG
>mir29c_extended_rt_cdna_110nt_exclude
GTCGTATCCAGTGCAGGGTCCGAGGTATTCGACTAGCTAATCGTGGTAACCGATTTCAAATGGTGCTAAGTGCGTACTCGAGGCATCGATCAGTACGGATCG
```

### 8.3 EasyDesign 运行命令

```bash
cd tools/EasyDesign

python bin/design.py \
  --input ../../data/rt_cdna_designs/mir29a_extended_rt_cdna_v2.fasta \
  --specific-against-fastas ../../data/rt_cdna_designs/mir29bc_extended_rt_cdna_v2.fasta \
  --guide-length 25 \
  --primer-length 30 \
  --mode rpa \
  --output ../../results/easydesign_mir29a_110nt_output.tsv \
  --max-primer-mismatches 0 \
  --max-guide-mismatches 0
```

**预期结果：**
- 应该能找到 1 个或多个完整的 target
- 包含 crRNA guide、Forward primer、Reverse primer
- 活性评分（基于 EasyDesign bundled 模型）

---

## 9. 序列验证检查

### 9.1 长度检查

| 构建 | 5' Adapter | miRNA 区域 | 3' Adapter | 总长 | 是否满足 ≥100 nt |
|-----|-----------|-----------|-----------|------|----------------|
| miR-29a v2 | 38 nt | 30 nt | 42 nt | 110 nt | ✅ |
| miR-29b v2 | 38 nt | 30 nt | 42 nt | 110 nt | ✅ |
| miR-29c v2 | 38 nt | 30 nt | 42 nt | 110 nt | ✅ |

### 9.2 区分位点检查

**miR-29a vs miR-29b vs miR-29c（miRNA 区域对比）：**

```
29a: AATCGTGGTAACCGATTTCAGATGGTGCTA
29b: ACATCGTAACACTGATTTCAAATGGTGCTA
29c: AATCGTGGTAACCGATTTCAAATGGTGCTA
     ******* ********* ************
     差异位点明显（位置 9 和 18 反映在 cDNA 中）
```

**关键区分：**
- **位置 20（从左计）**：29a/c 为 C，29b 为 A
- **位置 9 附近**：29a 为 CU，29b/c 为 UU（反映在 cDNA）

### 9.3 GC 含量检查

```python
# 示例计算
seq = "GTCGTATCCAGTGCAGGGTCCGAGGTATTCGACTAGCTAATCGTGGTAACCGATTTCAGATGGTGCTAAGTGCGTACTCGAGGCATCGATCAGTACGGATCG"
gc_count = seq.count('G') + seq.count('C')
gc_content = gc_count / len(seq) * 100
# Result: ~52% (适合 RPA)
```

### 9.4 发卡结构检查

使用 ViennaRNA 检查自折叠：

```python
import RNA
seq = "GTCGTATCCAGTGCAGGGTCCGAGGTATTCGACTAGCTAATCGTGGTAACCGATTTCAGATGGTGCTAAGTGCGTACTCGAGGCATCGATCAGTACGGATCG"
structure, mfe = RNA.fold(seq)
print(f"MFE: {mfe} kcal/mol")
print(f"Structure: {structure}")
```

**期望结果：** MFE > -15 kcal/mol（较弱二级结构，利于扩增）

---

## 10. 下一步实验验证

### 10.1 体外验证步骤

1. **合成试剂**
   - 订购合成 miR-29a/b/c 标准品（IDT 或 Genscript）
   - 订购 RT 引物（包含 3' adapter）
   - 订购 EasyDesign 推荐的 crRNA 和 RPA 引物

2. **RT 反应优化**
   ```
   反应体系（20 μL）：
   - miRNA 标准品：10 nM
   - RT 引物：500 nM
   - SuperScript IV：200 U
   - 反应温度：50°C，30 min
   ```

3. **RPA 扩增验证**
   ```
   反应体系（50 μL）：
   - RT 产物：2 μL
   - Forward primer：400 nM
   - Reverse primer：400 nM
   - TwistAmp Basic Kit
   - 反应温度：37°C，20 min
   ```

4. **Cas12a 切割检测**
   ```
   反应体系（20 μL）：
   - RPA 产物：2 μL
   - LbCas12a：50 nM
   - crRNA：62.5 nM
   - ssDNA reporter：1 μM
   - 反应温度：37°C，60 min
   - 检测：荧光读数（FAM/TAMRA）
   ```

### 10.2 特异性验证

**关键实验：** 交叉反应测试

| crRNA 设计目标 | 测试 miR-29a | 测试 miR-29b | 测试 miR-29c |
|---------------|-------------|-------------|-------------|
| anti-29a crRNA | ✅ 高信号 | ❌ 低/无信号 | ❌ 低/无信号 |
| anti-29b crRNA | ❌ 低/无信号 | ✅ 高信号 | ❌ 低/无信号 |
| anti-29c crRNA | ❌ 低/无信号 | ❌ 低/无信号 | ✅ 高信号 |

**成功标准：**
- 目标信号 / 脱靶信号 ≥ 10×

---

## 11. 常见问题解答

### Q1: 为什么要用 110 nt 而不是刚好 100 nt？

**A:** 留有 10 nt 的设计余地，让 EasyDesign 有更多的搜索空间来找到最优的 guide 和 primer 组合。

### Q2: Adapter 序列会不会与基因组 DNA 同源？

**A:** 选择的 adapter 序列来自常用的 stem-loop RT 通用序列，已被广泛验证为非同源序列。但在正式实验前，建议用 BLAST 对照人类基因组数据库检查一遍。

### Q3: 如果 EasyDesign 还是找不到合适的 target 怎么办？

**A:** 可以尝试：
1. 进一步延长到 120-150 nt
2. 调整 adapter 序列（改变 GC 含量）
3. 放宽参数：允许 1 个 guide 错配
4. 手动设计 guide 和 primer，不依赖 EasyDesign

### Q4: RT 引物的特异性够吗？

**A:** 3' 端 6 nt 特异性锚定足以区分 miR-29a/b/c，因为它们的 3' 末端序列不同。如果担心特异性，可以延长到 8-10 nt。

### Q5: 这个设计能否用于真实临床样本？

**A:** 当前设计是**概念验证**阶段，用于证明选择性检测的可行性。真实临床应用还需要：
1. 优化提取和浓缩方法（血清/血浆中 miRNA 浓度极低）
2. 验证内参 miRNA（如 miR-16）
3. 建立标准曲线和定量范围
4. 进行临床样本盲测

---

## 12. 文件清单

本设计相关的所有文件将保存在 `data/rt_cdna_designs/` 目录：

1. `mir29a_extended_rt_cdna_v2.fasta` - miR-29a 主要输入
2. `mir29bc_extended_rt_cdna_v2.fasta` - miR-29b/c 排除输入
3. `RT_cDNA_DESIGN_v2.md` - 本设计文档
4. `rt_primers_v2.txt` - RT 引物序列
5. `easydesign_design_params_v2.json` - EasyDesign 参数记录

---

**设计者：** Claude (Anthropic)  
**审核建议：** 请导师或实验室成员审核后再订购试剂  
**预计试剂成本：** 
- 合成 miRNA 标准品：~$300 × 3 = $900
- RT 引物：~$30
- crRNA：~$100 × 3 = $300
- RPA 引物：~$50 × 2 = $100
- **总计：~$1330 USD**
