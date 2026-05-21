"""
测试脚本
========
加载已训练的最佳模型，在测试集上评估性能。

测试流程：
1. 解析命令行参数
2. 加载测试数据和图结构
3. 加载预训练的模型权重
4. 在测试集上评估，输出6个指标
5. 保存预测结果CSV（含概念对和预测标签）

使用方法：
    python test.py --dataset MOOC
    或 bash test.sh
"""

import time
import argparse
import numpy as np
import pandas as pd
import torch
from model import CPL
from load_data.load_HGNN_data import generate_G_from_H
from load_data.load_DGCN_data import generate_adj_matrices
from utils.eval_performance import evaluate_test


# ============================================================
# 命令行参数配置
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--in_channels', type=int, default=768,
                    help='输入通道数（BERT嵌入维度）')
parser.add_argument('--out_channels1', type=int, default=256,
                    help='第一层输出通道数')
parser.add_argument('--out_channels2', type=int, default=256,
                    help='第二层输出通道数')
parser.add_argument('--epochs', type=int, default=30,
                    help='训练轮数（测试时未使用，仅为兼容）')
parser.add_argument('--batch_size', type=int, default=16,
                    help='批次大小')

# 不同数据集的最佳超参数（论文中调参结果）
# University_Course: kd_loss_weight=1E-6, weight_decay=1E-4, seed=25
# LectureBank:       kd_loss_weight=1E-1, weight_decay=1E-2, seed=25
# MOOC:              kd_loss_weight=1E-5, weight_decay=1E-3, seed=42
parser.add_argument('--kd_loss_weight', type=float, default=1E-5,
                    help='知识蒸馏损失权重（MOOC数据集）')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='学习率（测试时未使用）')
parser.add_argument('--weight_decay', type=float, default=1E-3,
                    help='权重衰减（MOOC数据集）')
parser.add_argument('--dropout_rate', type=float, default=0.5,
                    help='Dropout比率')
parser.add_argument('--T', type=float, default=0.5,
                    help='知识蒸馏温度系数')
parser.add_argument('--seed', type=int, default=42,
                    help='随机种子（MOOC数据集）')
parser.add_argument('--dataset', type=str, default='MOOC',
                    help='数据集名称：MOOC / LectureBank / University_Course')
args = parser.parse_args()


def train(args):
    """
    模型测试流程
    （函数名为train仅为与train.py保持一致，实际执行测试）
    """

    # ============================================================
    # 步骤1：初始化设备
    # ============================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print('Dataset: ' + args.dataset + '\n')

    # ============================================================
    # 步骤2：加载测试数据
    # ============================================================
    print("Read test data complete!")

    # 读取测试集CSV（格式：concept1_id, concept2_id, label）
    test_data_df = pd.read_csv(f'./data/{args.dataset}/test.csv', header=0)
    # 转换为元组列表
    test_data = [tuple(x) for x in test_data_df.to_numpy()]

    # ============================================================
    # 步骤3：加载图结构数据
    # ============================================================

    # 3.1 概念索引 → 获取概念总数
    concept_df = pd.read_csv(
        f'./data/{args.dataset}/concepts_index.csv', header=None
    )
    num_concepts = len(concept_df)

    # 3.2 加载概念-资源超图 (CRHG)
    # 从关联矩阵H生成超图拉普拉斯矩阵G
    adj = generate_G_from_H(
        pd.read_csv(f'./data/{args.dataset}/Hypergraph_H.csv', header=None)
    )
    G = adj.to(device)

    # 3.3 加载学习行为图 (LBG)
    # 生成出度和入度邻接矩阵
    adj_out, adj_in = generate_adj_matrices(
        f'./data/{args.dataset}/clickStreamLink_data_id.csv',
        num_concepts
    )
    adj_in = adj_in.to(device)
    adj_out = adj_out.to(device)

    # ============================================================
    # 步骤4：加载BERT嵌入
    # ============================================================
    bert_embeddings_df = pd.read_csv(
        f'./data/{args.dataset}/bert_embeddings.csv'
    )
    embeddings = np.stack(
        bert_embeddings_df['bert_embedding'].apply(
            lambda x: np.fromstring(x, sep=',')
        ).values
    )
    feature_matrix = torch.tensor(embeddings, dtype=torch.float32).to(device)

    # ============================================================
    # 步骤5：初始化模型并加载预训练权重
    # ============================================================
    model = CPL(
        args.in_channels, args.out_channels1, args.out_channels2,
        G, adj_out, adj_in, feature_matrix,
        dropout_rate=args.dropout_rate
    ).to(device)

    # 加载训练时保存的最佳模型
    model_name = (
        f"./best_model/{args.dataset}/"
        f"{args.dataset}-DGCPL_best_net.pth"
    )
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model'])
    model.eval()  # 切换到评估模式（禁用dropout等）

    # ============================================================
    # 步骤6：测试评估
    # ============================================================
    print("Testing!!!!")

    # evaluate_test 函数：
    #   1. 对测试数据逐batch预测
    #   2. 取三个分支的平均作为最终预测
    #   3. 计算6个评估指标（ACC, F1, Precision, Recall, AUC, AP）
    #   4. 将预测结果保存到 predictions_with_concepts.csv
    test_metrics = evaluate_test(
        model, test_data, args.batch_size, device,
        save_path=f"./data/{args.dataset}/predictions_with_concepts.csv"
    )

    # 输出测试结果
    print(
        f"Test metrics: ACC = {test_metrics['ACC']:.4f}, "
        f"F1 = {test_metrics['F1']:.4f}, "
        f"Precision = {test_metrics['Precision']:.4f}, "
        f"Recall = {test_metrics['Recall']:.4f}, "
        f"AUC = {test_metrics['AUC']:.4f}, "
        f"AP = {test_metrics['AP']:.4f}"
    )


if __name__ == '__main__':
    start_time = time.time()
    train(args)
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Test time: {total_time:.2f} seconds\n')
