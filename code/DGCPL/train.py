"""
训练脚本
========
DGCPL 模型的完整训练流程：

1. 解析命令行超参数
2. 加载三个数据集（train/val/test）和图结构数据
3. 初始化模型、优化器和损失函数
4. 按epoch迭代训练：
   - 随机打乱训练数据
   - 按batch前向传播 → 计算损失 → 反向传播
   - 每epoch后在训练集和验证集评估
   - 根据验证集AUC保存最佳模型
5. 记录训练日志

损失函数组成：
    Total_Loss = BCE_Loss + λ * KD_Loss
    - BCE_Loss: 三个分支（H/N/T）各自的二分类交叉熵
    - KD_Loss:  教师(T)向学生(H, N)蒸馏的L1损失

使用方法：
    python train.py --dataset MOOC --epochs 50 --batch_size 16
    或 bash train.sh
"""

import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch import optim as optima
from sklearn.utils import shuffle
from model import CPL
from load_data.load_HGNN_data import generate_G_from_H
from load_data.load_DGCN_data import generate_adj_matrices
from utils.set_logger import set_logger
from utils.set_seed import set_seed
from utils.loss_function import LossFunc
from utils.eval_performance import evaluate


# ============================================================
# 命令行参数配置
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--in_channels', type=int, default=768,
                    help='输入通道数（BERT嵌入维度）')
parser.add_argument('--out_channels1', type=int, default=256,
                    help='第一层输出通道数')
parser.add_argument('--out_channels2', type=int, default=256,
                    help='第二层输出通道数（最终概念嵌入维度）')
parser.add_argument('--epochs', type=int, default=50,
                    help='训练轮数')
parser.add_argument('--batch_size', type=int, default=16,
                    help='批次大小')
parser.add_argument('--kd_loss_weight', type=float, default=1E-5,
                    help='知识蒸馏损失权重 λ（通常很小，避免主导BCE损失）')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Adam优化器学习率')
parser.add_argument('--weight_decay', type=float, default=1E-3,
                    help='权重衰减（L2正则化系数）')
parser.add_argument('--dropout_rate', type=float, default=0.5,
                    help='Dropout比率')
parser.add_argument('--T', type=float, default=0.5,
                    help='知识蒸馏温度系数（越小，软标签越"硬"）')
parser.add_argument('--seed', type=int, default=42,
                    help='随机种子（确保可复现性）')
parser.add_argument('--dataset', type=str, default='MOOC',
                    help='数据集名称：MOOC / LectureBank / University_Course')
args = parser.parse_args()

# 初始化日志系统
logger = set_logger(args.dataset)


def train(args):
    """
    DGCPL 模型的完整训练流程
    """

    # ============================================================
    # 步骤1：初始化
    # ============================================================

    # 固定随机种子，确保实验可复现
    set_seed(args.seed)

    # 选择设备（优先GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    logger.info('Dataset: ' + args.dataset + '\n')

    # ============================================================
    # 步骤2：加载训练/验证/测试数据
    # ============================================================
    logger.info("Read data complete!")

    # 读取CSV数据文件（格式：concept1_id, concept2_id, label）
    train_data_df = pd.read_csv(f'./data/{args.dataset}/train.csv', header=0)
    val_data_df = pd.read_csv(f'./data/{args.dataset}/val.csv', header=0)
    test_data_df = pd.read_csv(f'./data/{args.dataset}/test.csv', header=0)

    # 将DataFrame转换为元组列表，每条记录：(c1_id, c2_id, label)
    train_data = [tuple(x) for x in train_data_df.to_numpy()]
    val_data = [tuple(x) for x in val_data_df.to_numpy()]
    test_data = [tuple(x) for x in test_data_df.to_numpy()]

    # ============================================================
    # 步骤3：加载概念-资源超图 (CRHG)
    # ============================================================
    # 读取概念索引，获取概念总数
    concept_df = pd.read_csv(f'./data/{args.dataset}/concepts_index.csv', header=None)
    num_concepts = len(concept_df)

    # 从关联矩阵 H 生成超图拉普拉斯矩阵 G
    # G = D_v^(-1/2) · H · W · D_e^(-1) · H^T · D_v^(-1/2)
    adj = generate_G_from_H(
        pd.read_csv(f'./data/{args.dataset}/Hypergraph_H.csv', header=None)
    )
    G = adj.to(device)

    # ============================================================
    # 步骤4：加载学习行为图 (LBG)
    # ============================================================
    # 从点击流数据生成出度图和入度图的邻接矩阵
    # adj_out[i][j]: 从概念 i 到概念 j 的转移频次（归一化后）
    # adj_in[i][j]:  从概念 j 到概念 i 的转移频次（即逆方向，归一化后）
    adj_out, adj_in = generate_adj_matrices(
        f'./data/{args.dataset}/clickStreamLink_data_id.csv',
        num_concepts
    )
    adj_in = adj_in.to(device)
    adj_out = adj_out.to(device)

    # ============================================================
    # 步骤5：加载预训练的BERT概念嵌入
    # ============================================================
    # 从CSV读取BERT嵌入并转换为张量
    # 每个概念的嵌入为768维向量
    bert_embeddings_df = pd.read_csv(f'./data/{args.dataset}/bert_embeddings.csv')
    embeddings = np.stack(
        bert_embeddings_df['bert_embedding'].apply(
            lambda x: np.fromstring(x, sep=',')
        ).values
    )
    feature_matrix = torch.tensor(embeddings, dtype=torch.float32).to(device)

    # ============================================================
    # 步骤6：初始化模型、优化器和损失函数
    # ============================================================
    model = CPL(
        args.in_channels, args.out_channels1, args.out_channels2,
        G, adj_out, adj_in, feature_matrix,
        dropout_rate=args.dropout_rate
    ).to(device)

    # Adam优化器
    optimizer = optima.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 自定义损失函数：BCE + 知识蒸馏
    loss_func = LossFunc(device, T=args.T)

    # ============================================================
    # 步骤7：训练循环
    # ============================================================
    logger.info("Training!!!!")
    best_val_auc = 0.0  # 跟踪最佳验证集AUC

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        logger.info(f'epoch: {epoch + 1}, lr = {optimizer.param_groups[0]["lr"]}')

        # --- 7.1 数据打乱 ---
        # 每个epoch随机打乱训练数据，提高泛化能力
        X_train = np.array(shuffle(train_data, random_state=args.seed))
        sum_total_loss = 0.0

        model.train()  # 切换到训练模式（启用dropout等）

        # --- 7.2 按batch迭代 ---
        batch_idx = 0
        for i in range(X_train.shape[0] // args.batch_size):
            # 获取当前batch的数据
            x = X_train[
                batch_idx * args.batch_size :
                batch_idx * args.batch_size + args.batch_size
            ]
            batch_idx += 1

            # 解析batch：c1和c2是概念索引，target是标签（0或1）
            c1, c2 = x[:, 0], x[:, 1]
            target = x[:, -1]
            target = torch.tensor(target).to(device)

            # 清空梯度
            optimizer.zero_grad()

            # --- 7.3 前向传播 ---
            # 获取三个分支的预测logit：
            #   logit_H: 仅使用知识视角（CRHG）
            #   logit_N: 仅使用行为视角（LBG）
            #   logit_T: 使用融合视角（GKD蒸馏结果）
            logit_H, logit_N, logit_T = model(c1, c2)

            # --- 7.4 计算损失 ---
            # loss:     BCE损失（三个分支的二分类交叉熵之和）
            # loss_kd:  知识蒸馏损失（教师软标签与学生的L1距离）
            loss, loss_kd, prediction, ground_truth = loss_func(
                logit_H, logit_N, logit_T, target[:, None].float()
            )

            # 总损失 = BCE + λ * KD
            # λ (kd_loss_weight) 通常很小（1E-5），因为蒸馏起辅助作用
            total_loss = loss + args.kd_loss_weight * loss_kd
            sum_total_loss += total_loss

            # --- 7.5 反向传播与参数更新 ---
            total_loss.backward(retain_graph=True)
            optimizer.step()

        # --- 7.6 计算并记录平均损失 ---
        average_loss = (sum_total_loss / batch_idx).float()
        logger.info(
            f"Average train loss for epoch {epoch + 1}: {average_loss.item()}"
        )

        # --- 7.7 在训练集上评估 ---
        train_metrics = evaluate(model, train_data, args.batch_size, device)
        logger.info(
            f"Train metrics: ACC = {train_metrics['ACC']:.4f}, "
            f"F1 = {train_metrics['F1']:.4f}, "
            f"Precision = {train_metrics['Precision']:.4f}, "
            f"Recall = {train_metrics['Recall']:.4f}, "
            f"AUC = {train_metrics['AUC']:.4f}, "
            f"AP = {train_metrics['AP']:.4f}"
        )

        # --- 7.8 在验证集上评估 ---
        val_metrics = evaluate(model, val_data, args.batch_size, device)
        logger.info(
            f"Validation metrics: ACC = {val_metrics['ACC']:.4f}, "
            f"F1 = {val_metrics['F1']:.4f}, "
            f"Precision = {val_metrics['Precision']:.4f}, "
            f"Recall = {val_metrics['Recall']:.4f}, "
            f"AUC = {val_metrics['AUC']:.4f}, "
            f"AP = {val_metrics['AP']:.4f}"
        )

        # --- 7.9 保存最佳模型 ---
        # 以验证集AUC为选择标准
        if val_metrics['AUC'] > best_val_auc:
            best_val_auc = val_metrics['AUC']
            logger.info(
                f'Saved new best model at epoch {epoch + 1} '
                f'with val_auc: {best_val_auc:.4f}!!!'
            )
            model_name = (
                f"./best_model/{args.dataset}/"
                f"{args.dataset}-DGCPL_best_net.pth"
            )
            torch.save({'model': model.state_dict()}, model_name)
            logger.info(f"Model parameters saved to {model_name}!")

        # --- 7.10 记录epoch耗时 ---
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        logger.info(
            f'Epoch {epoch + 1} duration: {epoch_duration:.2f} seconds\n'
        )


if __name__ == '__main__':
    start_time = time.time()
    train(args)
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f'Train time: {total_time:.2f} seconds\n')
