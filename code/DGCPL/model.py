"""
DGCPL 核心模型：CPL (Concept Prerequisite Learning)
==================================================
本文件定义了 DGCPL 的完整模型结构，包含四个核心模块：
  1. CRHG (Concept-Resource HyperGraph)：概念-资源超图 → 知识视角特征
  2. LBG  (Learning Behavior Graph)：学习行为有向图 → 行为视角特征
  3. GKD  (Gate Knowledge Distillation)：门控知识蒸馏 → 双视角融合
  4. SiameseNet ×3：孪生网络 → 预测先决条件关系

数据流：
  BERT嵌入 ─┬─→ HGNN(CRHG) ──→ X_H ──┬──→ GKD ──→ X_T ──→ SiameseNet3 ──→ logit_T (教师)
             │                        │
             └─→ GCN×2(LBG) ──→ X_N ──┘
                  │                              │
                  └──→ SiameseNet1(X_H) ──→ logit_H (学生-知识视角)
                  └──→ SiameseNet2(X_N) ──→ logit_N (学生-行为视角)
"""

import torch.nn as nn
import torch.nn.functional as F
from HyperGNN import HGNN
from DirectedGCN import GCN
from SiameseNet import SiameseNet


class CPL(nn.Module):
    """
    CPL (Concept Prerequisite Learning) 模型
    通过双图蒸馏学习概念先决条件关系
    """

    def __init__(self, in_channels, out_channels1, out_channels2, G, adj_in, adj_out, feature_matrix, dropout_rate=0.5):
        """
        参数说明：
            in_channels:   输入特征维度（BERT嵌入维度，默认768）
            out_channels1: 第一层输出维度（默认256）
            out_channels2: 第二层输出维度，即最终概念嵌入维度（默认256）
            G:             超图的归一化拉普拉斯矩阵 (concept_num × concept_num)
            adj_in:        学习行为图的入度邻接矩阵（稀疏张量）
            adj_out:       学习行为图的出度邻接矩阵（稀疏张量）
            feature_matrix:预训练的BERT概念嵌入矩阵 (concept_num × 768)
            dropout_rate:  Dropout比率（默认0.5）
        """
        super(CPL, self).__init__()
        self.dropout_rate = dropout_rate

        # === 预训练特征矩阵 ===
        # 所有概念的BERT嵌入，作为模型的初始输入特征
        self.feature_matrix = feature_matrix

        # === 图结构数据 ===
        self.G = G              # 概念-资源超图（CRHG）的拉普拉斯矩阵
        self.adj_out = adj_out  # 学习行为图（LBG）的出度邻接矩阵
        self.adj_in = adj_in    # 学习行为图（LBG）的入度邻接矩阵

        # ============================================================
        # 模块1：Concept-Resource HyperGraph (CRHG) — 知识视角
        # ============================================================
        # 两层超图卷积网络，通过"概念—资源—概念"的二阶路径
        # 捕获概念之间基于共享知识资源的高阶关联
        self.hgnn = HGNN(
            in_ch=in_channels,
            n_hid=out_channels1,
            n_class=out_channels2,
            dropout_rate=self.dropout_rate
        )

        # ============================================================
        # 模块2：Learning Behavior Graph (LBG) — 行为视角
        # ============================================================
        # 两个独立的有向图卷积网络，分别处理：
        #   - gcn1: 出度图（概念A→B：学完A的人倾向于去学B）
        #   - gcn2: 入度图（概念B→A：学B的人之前学过A）
        # 两个方向提供互补的先决条件信号
        self.gcn1 = GCN(nfeat=in_channels, nhid=out_channels1, nclass=int(out_channels2))
        self.gcn2 = GCN(nfeat=in_channels, nhid=out_channels1, nclass=int(out_channels2))

        # 融合出度和入度两个方向的嵌入
        self.w1 = nn.Linear(out_channels2, out_channels2)  # 出度方向变换
        self.w2 = nn.Linear(out_channels2, out_channels2)  # 入度方向变换

        # ============================================================
        # 模块3：Gate Knowledge Distillation (GKD) — 门控知识蒸馏
        # ============================================================
        # 通过可学习的门控向量 theta，逐元素控制两个视角的融合比例：
        #   theta = σ(W3·X_H + W4·X_N)
        #   X_T = theta ⊙ X_H + (1-theta) ⊙ X_N
        # theta 接近1 → 更信任知识视角；theta 接近0 → 更信任行为视角
        self.w3 = nn.Linear(out_channels2, out_channels2)  # 知识视角门控权重
        self.w4 = nn.Linear(out_channels2, out_channels2)  # 行为视角门控权重

        # ============================================================
        # 模块4：SiameseNet ×3 — 三组孪生网络
        # ============================================================
        # - siameseNet1: 仅使用知识视角嵌入 X_H 预测 → 学生网络1
        # - siameseNet2: 仅使用行为视角嵌入 X_N 预测 → 学生网络2
        # - siameseNet3: 使用融合视角嵌入 X_T 预测 → 教师网络
        # 训练时，教师网络通过蒸馏损失将融合知识传递给学生网络
        self.siameseNet1 = SiameseNet(out_channels2)
        self.siameseNet2 = SiameseNet(out_channels2)
        self.siameseNet3 = SiameseNet(out_channels2)

        # Sigmoid 激活函数，用于门控机制和输出概率
        self.sigmoid = nn.Sigmoid()

    def forward(self, c1, c2):
        """
        前向传播

        参数：
            c1: 概念1的索引列表（batch内）
            c2: 概念2的索引列表（batch内）

        返回：
            logit_H: 知识视角（CRHG）的预测logit
            logit_N: 行为视角（LBG）的预测logit
            logit_T: 融合视角（Teacher）的预测logit
        """

        # ============================================================
        # 阶段1：Concept-Resource HyperGraph (CRHG) → 知识视角
        # ============================================================
        # 所有概念通过超图卷积获取知识层面的嵌入表示
        # X_H 的形状：(concept_num, out_channels2)，即 (N, 256)
        X_H = self.hgnn(self.feature_matrix, self.G)

        # ============================================================
        # 阶段2：Learning Behavior Graph (LBG) → 行为视角
        # ============================================================
        # 分别用出度图和入度图进行图卷积
        X_out = self.gcn1(self.feature_matrix, self.adj_out)  # 出度方向嵌入
        X_in = self.gcn2(self.feature_matrix, self.adj_in)    # 入度方向嵌入

        # 融合两个方向的嵌入：
        # 使用线性变换后相加 + ReLU激活
        # （注释掉的方案是直接拼接：torch.cat([X_out, X_in], -1)）
        X_N = F.relu(self.w1(X_out) + self.w2(X_in))

        # ============================================================
        # 阶段3：Gate Knowledge Distillation (GKD) → 门控融合
        # ============================================================
        # theta: 门控向量，形状 (concept_num, 256)
        # 每个维度的 theta 值在 [0, 1] 之间，控制该维度上两个视角的贡献比例
        theta = self.sigmoid(self.w3(X_H) + self.w4(X_N))

        # 加权融合：
        # theta ⊙ X_H:     保留知识视角中置信度高的部分
        # (1-theta) ⊙ X_N: 保留行为视角中知识视角不足的部分
        out_h = theta * X_H          # 知识视角的保留部分
        out_n = (1 - theta) * X_N    # 行为视角的补充部分
        X_T = out_h + out_n          # 融合后的教师嵌入

        # ============================================================
        # 阶段4：SiameseNet 预测
        # ============================================================
        # 从完整嵌入矩阵中取出当前batch对应的概念对嵌入，送入孪生网络
        logit_H = self.siameseNet1(X_H[c1], X_H[c2])  # 学生1：纯知识视角预测
        logit_N = self.siameseNet2(X_N[c1], X_N[c2])  # 学生2：纯行为视角预测
        logit_T = self.siameseNet3(X_T[c1], X_T[c2])  # 教师：融合视角预测

        return logit_H, logit_N, logit_T
