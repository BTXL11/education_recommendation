# DGCPL 项目详细解析

> **论文**: DGCPL: Dual Graph Distillation for Concept Prerequisite Relation Learning  
> **会议**: IJCAI 2025 Main Track  
> **任务**: 概念先决条件关系预测（Concept Prerequisite Relation Learning）

---

## 一、项目概览

DGCPL 是一个基于**双图蒸馏**的深度学习模型，用于预测教育领域中概念之间的先决条件关系（即学习概念A是否是学习概念B的前置条件）。模型从两个视角建模概念关系：

1. **知识视角（Knowledge Perspective）**：通过概念-资源超图（Concept-Resource HyperGraph, CRHG）捕获概念之间的高阶知识关联
2. **学习行为视角（Learning Behavior Perspective）**：通过学习行为图（Learning Behavior Graph, LBG）捕获学习者点击流中体现的概念依赖关系

最终通过**门控知识蒸馏（Gate Knowledge Distillation, GKD）** 机制融合两个视角的特征，得到全面的概念嵌入，用于准确预测先决条件关系。

### 架构图说明

```
BERT嵌入 (输入特征)
    │
    ├──→ CRHG (HyperGNN) ──→ X_H (知识视角特征)
    │                              │
    ├──→ LBG (DirectedGCN) ──→ X_N (行为视角特征)  
    │                              │
    │         ┌────────────────────┘
    │         ▼
    └──→ GKD (门控知识蒸馏): theta = σ(W3·X_H + W4·X_N)
              │
              ▼
         X_T = theta * X_H + (1-theta) * X_N  (融合特征)
              │
              ├──→ SiameseNet1 (X_H) ──→ logit_H
              ├──→ SiameseNet2 (X_N) ──→ logit_N
              └──→ SiameseNet3 (X_T) ──→ logit_T (教师网络)
                       │
                       ▼
                  平均融合 → 最终预测
```

### 数据集

项目支持三个数据集：
| 数据集 | 说明 |
|--------|------|
| **MOOC** | 大规模在线课程数据 |
| **LectureBank** | 课程讲义数据 |
| **University_Course** | 大学课程数据 |

---

## 二、文件结构总览

```
DGCPL/
├── model.py              # 核心模型：CPL (Concept Prerequisite Learning)
├── HyperGNN.py           # 超图神经网络（处理 CRHG）
├── DirectedGCN.py        # 有向图卷积网络（处理 LBG）
├── SiameseNet.py         # 孪生网络（预测先决条件关系）
├── layers.py             # 底层图卷积层实现
├── train.py              # 训练脚本
├── test.py               # 测试脚本
├── train.sh / test.sh    # Shell 启动脚本
├── load_data/
│   ├── load_HGNN_data.py # 超图数据加载与转换
│   └── load_DGCN_data.py # 有向图邻接矩阵构建
├── utils/
│   ├── eval_performance.py  # 评估指标计算
│   ├── loss_function.py     # 损失函数（BCE + 蒸馏损失）
│   ├── set_logger.py        # 日志配置
│   └── set_seed.py          # 随机种子设置
├── data/                 # 数据集目录
├── best_model/           # 保存的最佳模型权重
├── logs/                 # 训练日志
├── paper/                # 论文 PDF 文件
└── environment.yml / environment.txt  # 环境依赖
```

---

## 三、逐文件详细解析

### 3.1 `layers.py` — 底层图卷积层

**作用**：定义两个最基础的图卷积算子，是整个模型的构建基块。

#### `HGNN_conv` — 超图卷积层

```
前向传播: X' = G @ (X @ W + b)
```

- 输入特征 X 先经过线性变换（权重矩阵 W）
- 再通过超图的关联矩阵 G 进行消息传递
- G 是由关联矩阵 H 构造的归一化超图拉普拉斯矩阵

**关键点**：与普通图卷积不同，超图卷积通过关联矩阵 G 实现了"概念—资源—概念"的二阶信息传递，能捕获概念之间通过共享资源形成的高阶关系。

#### `GraphConvolution` — 标准图卷积层

```
前向传播: X' = A @ (X @ W) + b
```

- 使用稀疏矩阵乘法 `torch.spmm` 高效处理大规模稀疏邻接矩阵 A
- 与 HGNN_conv 的核心区别：使用标准邻接矩阵 A 而非超图关联矩阵 G

**两个层的对比**：

| 特性 | HGNN_conv | GraphConvolution |
|------|-----------|-----------------|
| 消息传递矩阵 | 超图关联矩阵 G（稠密） | 邻接矩阵 A（稀疏） |
| 适用场景 | 概念-资源超图 | 概念-概念有向图 |
| 矩阵运算 | `G.matmul(x)` | `torch.spmm(adj, support)` |

---

### 3.2 `HyperGNN.py` — 超图神经网络

**作用**：基于 `HGNN_conv` 构建两层超图卷积网络，处理概念-资源超图（CRHG），输出知识视角的概念嵌入。

**架构**：
```
输入 X (N×768)
    │
    ├──→ HGNN_conv1(X, G) ──→ ReLU ──┐
    ├──→ Linear1(X) ──────────────────┼──→ + ──→ Dropout ──→ X1 (N×256)
    │                                  │
    ├──→ HGNN_conv2(X1, G) ──→ ReLU ──┐
    ├──→ Linear2(X1) ─────────────────┼──→ + ──→ Dropout ──→ X_H (N×256)
```

**设计亮点 — 残差连接**：每层同时使用图卷积和线性变换，结果相加。这种设计：
- 保留了原始特征信息（通过 Linear 支路）
- 融合了图结构信息（通过 HGNN_conv 支路）
- 起到类似残差连接的作用，缓解过平滑问题

---

### 3.3 `DirectedGCN.py` — 有向图卷积网络

**作用**：基于 `GraphConvolution` 构建两层有向图卷积网络，处理学习行为图（LBG）。

**架构**：与 HyperGNN 完全对称，仅将 `HGNN_conv` 替换为 `GraphConvolution`。

**关键设计 — 有向图的双向处理**：
- LBG 是一个有向图（概念A→概念B表示学习者在学完A后点击了B）
- 模型分别为**出度图**（adj_out）和**入度图**（adj_in）各训练一个 GCN
- `gcn1` 处理出度图，`gcn2` 处理入度图
- 两个方向的嵌入通过 `W1·X_out + W2·X_in` 融合

**为什么需要两个方向**：
- 出度方向：概念A→B 表示"学完A的人倾向于去学B"
- 入度方向：概念B→A 表示"学B的人之前学过A"
- 两个方向提供了互补的先决条件信号

---

### 3.4 `SiameseNet.py` — 孪生网络

**作用**：接收一对概念嵌入 (c1, c2)，预测 c1 是否是 c2 的先决条件。

**架构**：
```
c1, c2 (各256维)
    │
    ├──→ fc_layer(c1) ──→ ReLU ──→ c1' (64维)
    ├──→ fc_layer(c2) ──→ ReLU ──→ c2' (64维)
    │
    ├──→ diff = c1' - c2'           (64维)
    ├──→ multiply = c1' * c2'       (64维)
    │
    └──→ concat(c1', c2', diff, multiply) ──→ Linear(256→1) ──→ logit
```

**特征组合的语义**：
| 特征 | 维度 | 语义 |
|------|------|------|
| c1' | 64 | 概念1的嵌入 |
| c2' | 64 | 概念2的嵌入 |
| diff | 64 | 差异：捕获方向性（先决条件关系是不对称的） |
| multiply | 64 | 乘积：捕获共性（两个概念的关联强度） |

使用 `diff` 和 `multiply` 是孪生网络的经典设计，能同时捕获关系的**不对称性**和**相关性**。

---

### 3.5 `model.py` — 核心模型 CPL

**作用**：将上述所有模块组装成完整的 DGCPL 模型。

**完整数据流**：

1. **输入**：概念对 (c1, c2) 的索引，以及预训练 BERT 嵌入矩阵

2. **CRHG 分支**：
   ```
   feature_matrix → HGNN → X_H (N×256)
   ```
   通过超图卷积获取知识视角的概念嵌入

3. **LBG 分支**：
   ```
   feature_matrix → GCN_out(adj_out) → X_out (N×256)
   feature_matrix → GCN_in(adj_in)   → X_in  (N×256)
   X_N = ReLU(W1·X_out + W2·X_in)    (N×256)
   ```
   通过双向有向图卷积获取行为视角的概念嵌入

4. **门控知识蒸馏（GKD）**：
   ```
   theta = σ(W3·X_H + W4·X_N)
   X_T = theta ⊙ X_H + (1-theta) ⊙ X_N
   ```
   - theta 是一个可学习的门控向量，逐元素控制两个视角的融合比例
   - X_T 是教师嵌入，融合了知识和行为两个视角

5. **Siamese 预测**：
   ```
   logit_H = SiameseNet1(X_H[c1], X_H[c2])  # 知识视角预测
   logit_N = SiameseNet2(X_N[c1], X_N[c2])  # 行为视角预测
   logit_T = SiameseNet3(X_T[c1], X_T[c2])  # 教师（融合）预测
   ```

**三个 SiameseNet 角色的设计思想**：
- `siameseNet1` 和 `siameseNet2` 是**学生网络**，分别从单一视角学习
- `siameseNet3` 是**教师网络**，从融合视角学习
- 通过知识蒸馏损失，将教师的知识迁移到学生网络

---

### 3.6 `train.py` — 训练脚本

**训练流程**：

1. **参数解析**：通过 argparse 接收超参数（通道数、epoch、batch size、学习率等）
2. **数据加载**：
   - 读取 train/val/test 的 CSV 文件（格式：concept1_id, concept2_id, label）
   - 加载超图关联矩阵 H → 生成超图 G
   - 加载点击流数据 → 生成出/入度邻接矩阵
   - 加载 BERT 预训练嵌入
3. **模型初始化**：创建 CPL 模型、Adam 优化器、LossFunc 损失函数
4. **训练循环**：
   - 每个 epoch 随机打乱训练数据
   - 按 batch 前向传播 → 计算损失（BCE + 知识蒸馏）→ 反向传播
   - 每个 epoch 后在训练集和验证集上评估
   - 根据验证集 AUC 保存最佳模型
5. **日志记录**：记录 loss、ACC、F1、Precision、Recall、AUC、AP 等指标

**关键超参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| in_channels | 768 | BERT 嵌入维度 |
| out_channels1 | 256 | 第一层输出维度 |
| out_channels2 | 256 | 第二层输出维度（最终概念嵌入维度） |
| epochs | 50 | 训练轮数 |
| batch_size | 16 | 批次大小 |
| kd_loss_weight | 1E-5 | 蒸馏损失权重（非常小，避免主导训练） |
| lr | 0.0001 | 学习率 |
| dropout_rate | 0.5 | Dropout 比例 |
| T | 0.5 | 蒸馏温度系数 |

---

### 3.7 `test.py` — 测试脚本

**测试流程**：
1. 加载测试数据集
2. 加载数据（超图、邻接矩阵、BERT 嵌入）
3. 加载训练好的最佳模型权重
4. 在测试集上评估，输出 ACC、F1、Precision、Recall、AUC、AP
5. 保存预测结果到 CSV 文件（包含概念对、预测标签、真实标签）

与 `train.py` 的主要区别：
- 不执行反向传播和参数更新
- 只加载已保存的模型而不训练
- `evaluate_test` 额外保存预测结果到 CSV

---

### 3.8 `load_data/load_HGNN_data.py` — 超图数据加载

**核心函数 `generate_G_from_H`**：

将超图关联矩阵 H（concept × resource）转换为超图拉普拉斯矩阵 G：

```
G = D_v^(-1/2) · H · W · D_e^(-1) · H^T · D_v^(-1/2)
```

其中：
- H：概念-资源关联矩阵，H[i][j]=1 表示概念 i 与资源 j 相关
- D_v：顶点度矩阵（每个概念关联多少资源）
- D_e：超边度矩阵（每个资源关联多少概念）
- W：超边权重矩阵

**物理意义**：G[i][j] 表示概念 i 和概念 j 通过共享资源的关联强度，本质上是概念在资源空间中的共现相似度归一化结果。

---

### 3.9 `load_data/load_DGCN_data.py` — 有向图数据加载

**核心函数 `generate_adj_matrices`**：

从点击流数据构建出度和入度邻接矩阵：

1. 遍历每条点击流记录 (head→tail, 点击次数 Num)
2. `adj_out[head][tail] += Num`：累积出度权重
3. `adj_in[tail][head] += Num`：累积入度权重
4. 添加自环（identity matrix）后行归一化
5. 转换为稀疏张量

**归一化方式**：行归一化 `D^(-1) · A`，使每行和为1，实现加权平均聚合。

---

### 3.10 `utils/loss_function.py` — 损失函数

**`LossFunc` 设计**：

总损失由两部分组成：

```
Loss_total = Loss_BCE + λ · Loss_KD
```

**BCE 损失**（主损失）：
```python
Loss_BCE = BCE(y_H, target) + BCE(y_N, target) + BCE(y_T, target)
```
三个分支（知识视角、行为视角、融合视角）各自与真实标签计算二分类交叉熵。

**知识蒸馏损失**（辅助损失）：
```python
p_soft = sigmoid(logit / T)  # 温度缩放后的软标签
Loss_KD = |p_T - p_H|₁ + |p_T - p_N|₁  # L1距离
```
- 教师网络（融合视角）的软预测作为知识，通过 L1 距离传递给学生网络
- 温度 T 控制软标签的平滑程度（T=0.5 使分布更平滑）
- λ (kd_loss_weight) 通常很小（1E-5），因为蒸馏起辅助作用

**最终预测**：三个分支的概率取平均 `p_mean = (y_H + y_N + y_T) / 3`

---

### 3.11 `utils/eval_performance.py` — 评估指标

**`evaluate` 函数**（训练/验证用）：
返回 6 个指标：ACC、F1、Precision、Recall、AUC、AP

**`evaluate_test` 函数**（测试用）：
额外功能：
- 保存每个样本的预测结果到 CSV（含 concept1, concept2, prediction, target）
- 便于后续错误分析和案例研究

**预测融合**：与训练时一致，取三个分支的平均 `p_mean = (y_H + y_N + y_T) / 3`

---

### 3.12 `utils/set_logger.py` — 日志配置

- 自动创建 `logs/{dataset}/` 和 `best_model/{dataset}/` 目录
- 日志文件以时间戳命名：`training_YYYY-MM-DD_HH-MM-SS.log`
- 同时输出到文件和控制台

---

### 3.13 `utils/set_seed.py` — 随机种子

固定所有随机源（Python random、NumPy、PyTorch、CUDA），确保实验可复现。

---

## 四、核心创新点总结

1. **双图建模**：首次同时使用超图（知识视角）和有向图（行为视角）建模概念先决条件关系
2. **门控知识蒸馏（GKD）**：通过可学习的门控机制融合双视角特征，而非简单拼接或平均
3. **双向有向图处理**：分别使用出度和入度图，捕获概念间非对称的先决条件关系
4. **孪生网络预测**：通过 diff 和 multiply 特征组合，同时建模关系的不对称性和相关性
5. **多层级知识蒸馏**：教师网络（融合视角）向两个学生网络（单视角）蒸馏知识

---

## 五、数据文件说明

每个数据集目录下的核心文件：

| 文件 | 说明 |
|------|------|
| `train.csv / val.csv / test.csv` | 训练/验证/测试样本 (c1_id, c2_id, label) |
| `bert_embeddings.csv` | 概念的 BERT 预训练嵌入 (768维) |
| `concepts_index.csv` | 概念索引列表 |
| `resources_index.csv` | 资源索引列表 |
| `Hypergraph_H.csv` | 概念-资源超图关联矩阵 H |
| `clickStreamLink_data_id.csv` | 点击流数据 (head_id, tail_id, Num) |
| `MOOC_data.csv / UC_data.csv / LectureBank_data.csv` | 原始数据集 |
| `predictions_with_concepts.csv` | 测试后生成的预测结果文件 |

---

## 六、依赖关系图

```
model.py
  ├── HyperGNN.py ──────→ layers.py (HGNN_conv)
  ├── DirectedGCN.py ───→ layers.py (GraphConvolution)
  └── SiameseNet.py

train.py
  ├── model.py
  ├── load_data/load_HGNN_data.py
  ├── load_data/load_DGCN_data.py
  ├── utils/loss_function.py
  ├── utils/eval_performance.py
  ├── utils/set_logger.py
  └── utils/set_seed.py

test.py
  ├── model.py
  ├── load_data/load_HGNN_data.py
  ├── load_data/load_DGCN_data.py
  └── utils/eval_performance.py
```

---

## 七、运行方式

### 训练
```bash
# 方式1：Shell 脚本
bash train.sh

# 方式2：Python 命令
python train.py --dataset MOOC --epochs 50 --batch_size 16 --lr 0.0001
```

### 测试
```bash
# 方式1：Shell 脚本
bash test.sh

# 方式2：Python 命令
python test.py --dataset MOOC
```

---

*解析生成日期: 2026-05-21*
