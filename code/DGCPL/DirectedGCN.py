"""
有向图卷积网络 (Directed Graph Convolution Network)
=====================================================
基于 GraphConvolution 构建的两层图卷积网络，用于处理学习行为图（LBG）。

网络结构（每层，与 HyperGNN 对称）：
    输入 X
      ├──→ GraphConvolution(X, adj) ──→ ReLU ──┐
      ├──→ Linear(X) ───────────────────────────┼──→ + ──→ Dropout → 输出

关键设计：有向图的双向处理
  学习行为图是一个有向图，概念A→B表示"学习者在学完A后点击了B"。
  模型中的两个GCN实例分别处理：
    - gcn1 (model.py中) + adj_out: 出度方向 → "学完A的人倾向于去学B"
    - gcn2 (model.py中) + adj_in:  入度方向 → "学B的人之前学过A"
  两个方向提供互补的先决条件信号。

输入：BERT嵌入矩阵 (N × 768) + 邻接矩阵 adj (N × N, 稀疏)
输出：行为视角的概念嵌入 (N × 256)
"""

import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    """
    两层有向图卷积网络

    与 HGNN 形成对称结构：
      HGNN  使用 HGNN_conv + 超图 G     → 知识视角
      GCN   使用 GraphConvolution + 邻接矩阵 → 行为视角
    """

    def __init__(self, nfeat, nhid, nclass, dropout_rate=0.5):
        """
        参数：
            nfeat:        输入特征维度（BERT嵌入维度，768）
            nhid:         隐藏层维度（256）
            nclass:       输出维度，即最终概念嵌入维度（256）
            dropout_rate: Dropout比率
        """
        super(GCN, self).__init__()

        # === 图卷积层 ===
        # 第一层：nfeat → nhid (768 → 256)
        self.gc1 = GraphConvolution(nfeat, nhid)
        # 第二层：nhid → nclass (256 → 256)
        self.gc2 = GraphConvolution(nhid, nclass)
        # 可选的第三层（注释掉）
        # self.gc3 = GraphConvolution(nclass, nclass)

        # === 全连接层（残差支路）===
        # 每条Linear支路与对应的GraphConvolution输出相加
        # 作用：保留原始特征，缓解图卷积的过平滑问题
        self.linear1 = nn.Linear(nfeat, nhid)    # 第一层残差：768 → 256
        self.linear2 = nn.Linear(nhid, nclass)   # 第二层残差：256 → 256
        # self.linear3 = nn.Linear(nclass, nclass)

        # === 正则化 ===
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, adj):
        """
        前向传播

        参数：
            x:   输入特征矩阵 (N × nfeat)，即所有概念的BERT嵌入
            adj: 归一化邻接矩阵 (N × N)，稀疏张量
                 adj[i][j] 表示概念 j 对概念 i 的影响权重

        返回：
            概念嵌入矩阵 (N × nclass)，即行为视角的概念表示
        """
        # === 第一层：768 → 256 ===
        # gc1(x, adj): 图卷积，聚合邻居概念的特征
        #   在有向图中，邻接矩阵控制消息传递方向：
        #     adj_out → 沿出边方向聚合（从先决条件到后续概念）
        #     adj_in  → 沿入边方向聚合（从后续概念到先决条件）
        # linear1(x): 线性变换，保留原始BERT语义
        # 两者相加 → ReLU → Dropout
        # 物理意义：X1 融合了BERT语义 + 学习行为中的概念转移模式
        x1 = self.dropout(F.relu(self.gc1(x, adj) + self.linear1(x)))

        # === 第二层：256 → 256 ===
        # 在第一层基础上进一步聚合，获得更深层的行为模式
        x2 = self.dropout(F.relu(self.gc2(x1, adj) + self.linear2(x1)))

        # 可选的第三层（注释掉，避免过拟合和计算开销）
        # x3 = self.dropout(F.relu(self.gc2(x2, adj) + self.linear2(x2)))

        return x2
