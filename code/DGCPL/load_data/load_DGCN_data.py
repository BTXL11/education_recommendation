"""
有向图数据加载与预处理
=======================
从点击流数据构建学习行为图（LBG）的出度和入度邻接矩阵。

点击流数据格式（clickStreamLink_data_id.csv）：
    C1(head)_id, C2(tail)_id, Num
    表示：Num个学习者在浏览概念C1后点击了概念C2

图构建逻辑：
    - adj_out[head][tail] += Num：累积出度（从head到tail的转移频次）
    - adj_in[tail][head] += Num：累积入度（从tail看head的逆转移频次）

为什么需要出度和入度两个图：
    出度方向：A→B 表示"学完A的人倾向于去学B"（可能是先决条件信号）
    入度方向：B→A 表示"学B的人之前学过A"（从另一个方向验证先决条件）
    两个方向互补，联合使用能更全面地捕获先决条件关系
"""

import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd


def generate_adj_matrices(data_file, num_nodes):
    """
    从点击流数据生成归一化的出度图和入度图邻接矩阵

    参数：
        data_file: 点击流数据CSV文件路径
                   列：C1(head)_id, C2(tail)_id, Num
        num_nodes: 图中节点数（概念总数）

    返回：
        adj_out: 出度邻接矩阵（稀疏COO张量），adj_out[i][j] ∝ 从i到j的转移权重
        adj_in:  入度邻接矩阵（稀疏COO张量），adj_in[i][j]  ∝ 从j到i的转移权重

    处理步骤：
        1. 遍历点击流记录，累积转移频次
        2. 添加自环（保留节点自身信息）
        3. 行归一化（使每行和为1）
        4. 转换为PyTorch稀疏张量
    """

    # 初始化两个 N×N 的零矩阵
    adj_out = np.zeros((num_nodes, num_nodes))  # 出度矩阵
    adj_in = np.zeros((num_nodes, num_nodes))   # 入度矩阵

    # 读取点击流数据
    df = pd.read_csv(data_file, header=0)
    head_ids = df['C1(head)_id']   # 源概念ID（学习者当前浏览的概念）
    tail_ids = df['C2(tail)_id']   # 目标概念ID（学习者随后点击的概念）
    Nums = df['Num']                # 点击次数（转移频次）

    # 累积点击频次
    # 注意：概念ID从1开始，需要减1转换为0-based索引
    for head, tail, Num in zip(head_ids, tail_ids, Nums):
        # 出度：从head出发到tail的频次
        adj_out[head - 1][tail - 1] += Num
        # 入度：从tail的角度看来自head的频次（即逆方向）
        adj_in[tail - 1][head - 1] += Num

    # 添加自环并归一化
    # 自环（单位矩阵）的作用：
    #   1. 保证每个节点至少有一条边（避免度为零的孤立节点）
    #   2. 在图卷积中保留节点自身的特征信息
    adj_out = normalize(adj_out + sp.eye(num_nodes))
    adj_in = normalize(adj_in + sp.eye(num_nodes))

    # 转换为PyTorch稀疏张量（COO格式）
    adj_out = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj_out))
    adj_in = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj_in))

    return adj_out, adj_in


def normalize(mx):
    """
    行归一化：对矩阵的每一行除以其行和

    公式：
        mx_normalized[i][j] = mx[i][j] / Σ_k mx[i][k]

    物理意义：
        将邻接矩阵转换为随机游走转移概率矩阵
        每行的元素和为1，表示从节点i出发到各邻居的归一化权重

    参数：
        mx: scipy稀疏矩阵或numpy数组

    返回：
        行归一化后的矩阵
    """
    # 计算每一行的和
    rowsum = np.array(mx.sum(1))

    # 取倒数：1/rowsum
    r_inv = np.power(rowsum, -1).flatten()

    # 处理无穷大（度为0的节点，取倒数后为inf）
    r_inv[np.isinf(r_inv)] = 0.

    # 构造对角矩阵 D^(-1)
    r_mat_inv = sp.diags(r_inv)

    # D^(-1) @ mx：每行除以其行和
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    将scipy稀疏矩阵转换为PyTorch稀疏FloatTensor

    参数：
        sparse_mx: scipy稀疏矩阵

    返回：
        PyTorch稀疏张量（COO格式）

    注意：这里使用 torch.sparse.FloatTensor（旧API），
    与 load_HGNN_data.py 中使用 torch.sparse_coo_tensor（新API）功能等价
    """
    # 转换为COO格式并确保数据类型为float32
    sparse_mx = sparse_mx.tocoo().astype(np.float32)

    # 构建索引（行索引和列索引堆叠）
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )

    # 非零元素的值
    values = torch.from_numpy(sparse_mx.data)

    # 张量形状
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)
