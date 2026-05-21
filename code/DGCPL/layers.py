"""
底层图卷积层实现
================
本文件定义两个基础的图卷积算子：
  1. HGNN_conv:        超图卷积层，用于概念-资源超图的消息传递
  2. GraphConvolution: 标准图卷积层，用于有向概念图的消息传递

两者都实现了"线性变换 + 图消息传递"的计算范式，区别在于：
  - HGNN_conv 使用超图的关联矩阵 G（稠密矩阵乘法）
  - GraphConvolution 使用邻接矩阵 A（稀疏矩阵乘法，适合大规模稀疏图）
"""

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class HGNN_conv(nn.Module):
    """
    超图卷积层 (Hypergraph Convolution Layer)

    前向传播公式:
        X' = G @ (X @ W + b)

    其中：
        X: 输入特征矩阵 (N × in_ft)
        W: 可学习权重矩阵 (in_ft × out_ft)
        b: 可学习偏置向量 (out_ft,)
        G: 超图的归一化拉普拉斯矩阵 (N × N)
           G = D_v^(-1/2) · H · W_e · D_e^(-1) · H^T · D_v^(-1/2)
           H 为概念-资源关联矩阵，G[i][j] 表示概念 i 和 j 通过共享资源的关联强度

    物理意义：
        (X @ W) 将概念特征从 in_ft 维线性映射到 out_ft 维
        G @ (...) 通过"概念→资源→概念"的二阶路径聚合邻居概念的特征
        因为超边连接了概念和资源，两步走完一条"概念-资源-概念"路径
    """

    def __init__(self, in_ft, out_ft, bias=True):
        """
        参数：
            in_ft:  输入特征维度
            out_ft: 输出特征维度
            bias:   是否使用偏置项
        """
        super(HGNN_conv, self).__init__()

        # 可学习的权重矩阵 W，形状 (in_ft, out_ft)
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))

        # 可选的偏置向量 b，形状 (out_ft,)
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)

        # 参数初始化
        self.reset_parameters()

    def reset_parameters(self):
        """
        使用均匀分布初始化权重和偏置
        初始化范围：[-1/√(out_ft), 1/√(out_ft)]
        这种 Xavier 风格的初始化有助于稳定训练
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        """
        前向传播

        参数：
            x: 输入特征 (N × in_ft)，N 为概念数量
            G: 超图拉普拉斯矩阵 (N × N)，稠密张量

        返回：
            输出特征 (N × out_ft)
        """
        # 步骤1：线性变换 X @ W
        x = x.matmul(self.weight)

        # 步骤2：加偏置
        if self.bias is not None:
            x = x + self.bias

        # 步骤3：超图消息传递 G @ (X @ W + b)
        # G[i][j] > 0 意味着概念 i 和 j 通过某些资源相关联
        # 此操作将相关联的概念特征加权聚合
        x = G.matmul(x)
        return x


class GraphConvolution(Module):
    """
    标准图卷积层 (Graph Convolution Layer)

    前向传播公式:
        X' = A @ (X @ W) + b

    其中：
        X: 输入特征矩阵 (N × in_features)
        W: 可学习权重矩阵 (in_features × out_features)
        b: 可选的偏置向量 (out_features,)
        A: 归一化的邻接矩阵 (N × N)，稀疏张量

    与 HGNN_conv 的关键区别：
        - 使用稀疏邻接矩阵 A 而非超图关联矩阵 G
        - 消息传递在概念-概念边上直接进行（一跳），而非通过资源中转（二跳）
        - 使用 torch.spmm 进行稀疏矩阵乘法，内存效率更高
    """

    def __init__(self, in_features, out_features, dropout=0., bias=True):
        """
        参数：
            in_features:  输入特征维度
            out_features: 输出特征维度
            dropout:      Dropout比率（在此层中未实际使用，保留用于扩展）
            bias:         是否使用偏置项
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        # 可学习的权重矩阵 W，形状 (in_features, out_features)
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        # 可选的偏置向量 b
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """
        使用均匀分布初始化权重和偏置
        初始化范围：[-1/√(out_features), 1/√(out_features)]
        """
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        前向传播

        参数：
            input: 输入特征 (N × in_features)
            adj:   稀疏邻接矩阵 (N × N)，COO格式的稀疏张量

        返回：
            输出特征 (N × out_features)
        """
        # 步骤1：线性变换 input @ W
        support = torch.mm(input, self.weight)

        # 步骤2：稀疏图卷积 A @ (input @ W)
        # 使用 torch.spmm（稀疏-稠密矩阵乘法）高效处理大规模稀疏邻接矩阵
        # 每个概念聚合其邻居（在行为图中有关联的概念）的变换后特征
        output = torch.spmm(adj, support)

        # 步骤3：加偏置
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        """层的字符串表示，方便调试和打印模型结构"""
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
