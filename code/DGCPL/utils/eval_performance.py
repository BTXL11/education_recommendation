"""
模型评估指标计算
=================
提供训练/验证用的 evaluate 函数和测试用的 evaluate_test 函数。

评估指标（6个）：
    - ACC (Accuracy):           准确率 = (TP+TN) / Total
    - F1 (F1-Score):            F1值 = 2 × Precision × Recall / (Precision + Recall)
    - Precision (精确率):        TP / (TP + FP)
    - Recall (召回率):           TP / (TP + FN)
    - AUC (Area Under ROC):     ROC曲线下面积
    - AP (Average Precision):   平均精确率（PR曲线下面积）

预测融合策略：
    取三个分支（知识视角、行为视角、融合视角）的预测概率平均值
    p_mean = (sigmoid(logit_H) + sigmoid(logit_N) + sigmoid(logit_T)) / 3
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, average_precision_score
)


def evaluate(model, data, batch_size, device):
    """
    在给定数据集上评估模型性能（训练/验证用）

    参数：
        model:      CPL模型实例
        data:       数据列表，每条记录为 (c1_id, c2_id, label)
        batch_size: 批次大小
        device:     GPU/CPU设备

    返回：
        metrics: 包含6个评估指标的字典
    """
    model.eval()  # 切换到评估模式（禁用dropout）
    total_preds, total_targets = [], []

    with torch.no_grad():  # 禁用梯度计算，节省内存
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]

            # 解包batch数据
            c1, c2, target = zip(*batch)
            c1 = torch.tensor(c1).to(device)
            c2 = torch.tensor(c2).to(device)
            target = torch.tensor(target).to(device).float()

            # 前向传播：获取三个分支的logit
            logit_H, logit_N, logit_T = model(c1, c2)

            # 将logit转为概率（sigmoid）
            y_H = torch.sigmoid(logit_H)  # 知识视角预测概率
            y_N = torch.sigmoid(logit_N)  # 行为视角预测概率
            y_T = torch.sigmoid(logit_T)  # 融合视角预测概率

            # === 融合策略：三个分支取平均 ===
            # 集成学习的思想：多个视角的预测取平均
            # 通常比单一视角的预测更稳定、更准确
            p_mean = (y_H + y_N + y_T) / 3.0

            # 转为numpy数组用于sklearn指标计算
            predictions = p_mean.cpu().numpy()
            targets = target.cpu().numpy()

            total_preds.extend(predictions)
            total_targets.extend(targets)

    # 转换为numpy数组并展平
    total_preds = np.array(total_preds).flatten()
    total_targets = np.array(total_targets).flatten()

    # 二值化：概率 > 0.5 → 正类（存在先决条件关系）
    pred_label = (total_preds > 0.5).astype(int)
    target_label = total_targets.astype(int)

    # 计算6个评估指标
    metrics = {
        "ACC": accuracy_score(target_label, pred_label),
        "F1": f1_score(target_label, pred_label),
        "Precision": precision_score(target_label, pred_label, zero_division=1),
        "Recall": recall_score(target_label, pred_label),
        "AUC": roc_auc_score(target_label, total_preds),
        "AP": average_precision_score(target_label, total_preds)
    }

    return metrics


def evaluate_test(model, data, batch_size, device, save_path):
    """
    在测试集上评估并保存预测结果

    与 evaluate 函数的主要区别：
        1. 额外记录 c1 和 c2 的索引
        2. 将预测结果保存为CSV文件

    参数：
        model:      CPL模型实例
        data:       测试数据列表
        batch_size: 批次大小
        device:     GPU/CPU设备
        save_path:  预测结果保存路径（CSV）

    返回：
        metrics: 包含6个评估指标的字典
    """
    model.eval()
    total_preds, total_targets, total_c1, total_c2 = [], [], [], []

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]

            c1, c2, target = zip(*batch)
            c1 = torch.tensor(c1).to(device)
            c2 = torch.tensor(c2).to(device)
            target = torch.tensor(target).to(device).float()

            # 前向传播
            logit_H, logit_N, logit_T = model(c1, c2)

            # 转为概率
            y_H = torch.sigmoid(logit_H)
            y_N = torch.sigmoid(logit_N)
            y_T = torch.sigmoid(logit_T)

            # 三视角平均融合
            p_mean = (y_H + y_N + y_T) / 3.0

            # 收集预测结果、真实标签和概念索引
            predictions = p_mean.cpu().numpy()
            targets = target.cpu().numpy()

            total_preds.extend(predictions)
            total_targets.extend(targets)
            total_c1.extend(c1.cpu().numpy())
            total_c2.extend(c2.cpu().numpy())

    # 展平
    total_preds = np.array(total_preds).flatten()
    total_targets = np.array(total_targets).flatten()

    # 二值化
    pred_label = (total_preds > 0.5).astype(int)
    target_label = total_targets.astype(int)

    # === 保存预测结果 ===
    # 便于后续进行错误分析和案例研究
    if save_path:
        results_df = pd.DataFrame({
            'Concept_1': total_c1,      # 概念1的ID
            'Concept_2': total_c2,      # 概念2的ID
            'Predictions': pred_label,  # 模型预测标签
            'Targets': target_label     # 真实标签
        })
        results_df.to_csv(save_path, index=False)

    # 计算评估指标
    metrics = {
        "ACC": accuracy_score(target_label, pred_label),
        "F1": f1_score(target_label, pred_label),
        "Precision": precision_score(target_label, pred_label, zero_division=1),
        "Recall": recall_score(target_label, pred_label),
        "AUC": roc_auc_score(target_label, total_preds),
        "AP": average_precision_score(target_label, total_preds)
    }

    return metrics
