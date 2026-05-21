"""
超图神经网络 (Hypergraph Neural Network)
=========================================
基于 HGNN_conv 构建的两层超图卷积网络，用于处理概念-资源超图（CRHG）。

网络结构（每层）：
    输入 X
      ├──→ HGNN_conv(X, G) ──→ ReLU ──┐
      ├──→ Linear(X) ──────────────────┼──→ + ──→ Dropout → 输出

设计思想：
  - 双路结构：图卷积分支捕获图结构信息，线性分支保留原始特征
  - 残差连接效果：两条支路相加，减轻过平滑问题
  - 共两层：第一层 768→256，第二层 256→256

输入：BERT嵌入矩阵 (N × 768) + 超图拉普拉斯矩阵 G (N × N)
输出：知识视角的概念嵌入 X_H (N × 256)
"""

from torch import nn
import torch.nn.functional as F
from layers import HGNN_conv


class HGNN(nn.Module):
    """
    两层超图卷积网络

    与 DirectedGCN (GCN) 形成对称结构，分别处理知识视角和行为视角
    """

    def __init__(self, in_ch, n_hid, n_class, dropout_rate=0.5):
        """
        参数：
            in_ch:        输入特征维度（BERT嵌入维度，768）
            n_hid:        隐藏层维度（256）
            n_class:      输出类别维度，即最终概念嵌入维度（256）
            dropout_rate: Dropout比率
        """
        super(HGNN, self).__init__()

        # === 超图卷积层 ===
        # 第一层：in_ch → n_hid (768 → 256)
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        # 第二层：n_hid → n_class (256 → 256)
        self.hgc2 = HGNN_conv(n_hid, n_class)
        # 可选的第三层（注释掉，避免过拟合和过平滑）
        # self.hgc3 = HGNN_conv(n_class, n_class)

        # === 全连接层（残差支路）===
        # 每条支路与对应的HGNN_conv输出相加，保留原始特征信息
        # 第一层残差：in_ch → n_hid (768 → 256)
        self.linear1 = nn.Linear(in_ch, n_hid)
        # 第二层残差：n_hid → n_class (256 → 256)
        self.linear2 = nn.Linear(n_hid, n_class)
        # self.linear3 = nn.Linear(n_class, n_class)

        # === 正则化 ===
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, G):
        """
        前向传播

        参数：
            x: 输入特征矩阵 (N × in_ch)，即所有概念的BERT嵌入
            G: 超图的拉普拉斯矩阵 (N × N)

        返回：
            概念嵌入矩阵 (N × n_class)，即知识视角的概念表示
        """
        # === 第一层：768 → 256 ===
        # hgc1(x, G): 超图卷积，通过"概念→资源→概念"路径聚合知识信息
        # linear1(x): 线性变换，保留原始BERT特征
        # 两者相加 → ReLU → Dropout
        # 物理意义：X1 融合了BERT语义 + 概念间共享资源的高阶关系
        x1 = self.dropout(F.relu(self.hgc1(x, G) + self.linear1(x)))

        # === 第二层：256 → 256 ===
        # 在第一层基础上进一步聚合，获得更深层的高阶知识关联
        x2 = self.dropout(F.relu(self.hgc2(x1, G) + self.linear2(x1)))

        # 可选的第三层（注释掉）
        # x3 = self.dropout(F.relu(self.hgc2(x2, G) + self.linear2(x2)))

        return x2
