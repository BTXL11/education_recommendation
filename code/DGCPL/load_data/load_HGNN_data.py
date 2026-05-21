"""
超图数据加载与预处理
====================
将概念-资源关联矩阵 H 转换为超图的归一化拉普拉斯矩阵 G。

核心公式：
    G = D_v^(-1/2) · H · W · D_e^(-1) · H^T · D_v^(-1/2)

矩阵维度说明：
    H: 概念-资源关联矩阵 (N_concept × N_resource)
       H[i][j] = 1 表示概念 i 与资源 j 相关
    G: 超图拉普拉斯矩阵 (N_concept × N_concept)
       G[i][j] 表示概念 i 和 j 通过共享资源的关联强度

物理意义：
    概念通过资源间接关联——两个概念共同关联的资源越多，
    它们在知识结构上越相关。G矩阵将这种"概念-资源-概念"
    的二阶关系编码为概念间的直接关联权重。
"""

import numpy as np
import torch
import scipy.sparse as sp
import os

# 环境变量设置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决macOS上OpenMP库冲突
os.environ["OMP_NUM_THREADS"] = "1"           # 限制OpenMP线程数，避免资源竞争


def generate_G_from_H(H, variable_weight=False):
    """
    从关联矩阵 H 生成超图的归一化拉普拉斯矩阵 G

    参数：
        H: 概念-资源关联矩阵，可以是 numpy 数组或 pandas DataFrame
           H[i][j] = 1 表示概念 i 使用了资源 j（如教材、视频、习题等）
        variable_weight: 是否返回分解后的矩阵组件（用于变体实现）

    返回：
        G: 超图的归一化拉普拉斯矩阵 (N_concept × N_concept)，稀疏COO张量
           如果 variable_weight=True，返回 (DV2_H, W, invDE_HT_DV2)

    计算步骤：
        1. D_v: 顶点度 = 每个概念关联了多少资源
        2. D_e: 超边度 = 每个资源关联了多少概念
        3. W:   超边权重（默认全1，即所有资源等权重）
        4. G = D_v^(-1/2) · H · W · D_e^(-1) · H^T · D_v^(-1/2)
    """
    H = np.array(H)

    # n_edge: 超边数 = 资源数量 = H的列数
    n_edge = H.shape[1]

    # W: 超边权重向量，初始化为全1（所有资源等权重）
    W = np.ones(n_edge)

    # DV: 顶点度向量 (N_concept,)
    # DV[i] = Σ_j H[i][j] * W[j] = 概念i关联的资源数（考虑权重）
    DV = np.sum(H * W, axis=1)

    # DE: 超边度向量 (N_resource,)
    # DE[j] = Σ_i H[i][j] = 资源j被多少个概念使用
    DE = np.sum(H, axis=0)

    # invDE: 超边度的逆矩阵 D_e^(-1)，对角矩阵
    # 度大的资源（被很多概念使用）权重降低，避免热门资源主导
    invDE = np.diag(np.power(DE, float(-1)))

    # DV2: 顶点度的-1/2次方矩阵 D_v^(-1/2)，对角矩阵
    # 度大的概念（关联很多资源）权重降低，平衡不同概念的影响力
    DV2 = np.diag(np.power(DV, -0.5))

    W = np.diag(W)      # 将权重向量转为对角矩阵
    HT = H.T            # H的转置 (N_resource × N_concept)

    if variable_weight:
        # 返回分解的组件（用于需要动态调整权重的场景）
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        # 标准路径：计算完整的超图拉普拉斯矩阵 G
        # G = D_v^(-1/2) @ H @ W @ D_e^(-1) @ H^T @ D_v^(-1/2)
        # 计算顺序：
        #   1. DV2 @ H:     归一化概念侧
        #   2. ... @ W:     加权超边
        #   3. ... @ invDE: 归一化资源侧
        #   4. ... @ HT:    通过资源传播回概念
        #   5. ... @ DV2:   再次归一化概念侧
        G = DV2 @ H @ W @ invDE @ HT @ DV2

        # 转换为稀疏COO格式 → PyTorch稀疏张量
        G = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(G))
        return G


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    将scipy稀疏矩阵转换为PyTorch稀疏张量（COO格式）

    参数：
        sparse_mx: scipy稀疏矩阵

    返回：
        PyTorch稀疏COO张量

    COO（Coordinate）格式存储三个数组：
        - indices: 非零元素的行列索引 (2 × nnz)
        - values:  非零元素的值 (nnz,)
        - shape:   张量形状
    """
    # 转换为COO格式并确保数据类型为float32
    sparse_mx = sparse_mx.tocoo().astype(np.float32)

    # 构建索引矩阵：第0行=行索引，第1行=列索引
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )

    # 非零元素的值
    values = torch.from_numpy(sparse_mx.data)

    # 张量形状
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse_coo_tensor(indices, values, shape)
