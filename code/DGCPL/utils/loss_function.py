"""
损失函数
========
DGCPL的损失函数由两部分组成：BCE损失 + 知识蒸馏损失

总损失公式：
    Loss_total = Loss_BCE + λ · Loss_KD

    其中：
    Loss_BCE = BCE(y_H, target) + BCE(y_N, target) + BCE(y_T, target)
    Loss_KD  = |p_T - p_H|₁ + |p_T - p_N|₁

BCE损失（主损失）：
    三个分支（H=知识视角, N=行为视角, T=融合视角）
    各自与真实标签计算二分类交叉熵

知识蒸馏损失（辅助损失）：
    教师网络（T=融合视角）拥有最丰富的信息，
    通过L1距离将软知识传递给学生网络（H和N）
    温度T控制软标签的平滑程度

参数λ（kd_loss_weight）：
    通常很小（1E-5量级），因为蒸馏起辅助作用
    太大会使学生网络过度模仿教师，失去单视角的独特优势
"""

import torch
import torch.nn as nn


class LossFunc(nn.Module):
    """
    DGCPL复合损失函数

    同时优化三个目标：
        1. 知识视角分支的BCE损失
        2. 行为视角分支的BCE损失
        3. 融合视角分支的BCE损失
        4. 教师→学生的知识蒸馏损失
    """

    def __init__(self, device, T=0.5):
        """
        参数：
            device: GPU/CPU设备
            T:      知识蒸馏温度系数
                    温度越低，软标签越接近硬标签（one-hot）
                    温度越高，软标签越平滑（类间差异缩小）
                    T=0.5 是中等偏低的温度，保留类别区分度的同时提供软信息
        """
        super(LossFunc, self).__init__()

        # 二分类交叉熵损失
        # 注意：BCELoss 期望输入已经过sigmoid的概率值
        self.crossEntropy = nn.BCELoss()

        # 均方误差损失（预留，当前未使用）
        self.mse = nn.MSELoss()

        # Sigmoid激活函数：将logit转为概率
        self.sig = nn.Sigmoid()

        # 蒸馏温度
        self.T = T

        # 设备
        self.device = device

    def forward(self, logit_h, logit_n, logit_t, target):
        """
        计算复合损失

        参数：
            logit_h: 知识视角（CRHG）的原始logit (batch × 1)
            logit_n: 行为视角（LBG）的原始logit (batch × 1)
            logit_t: 融合视角（Teacher）的原始logit (batch × 1)
            target:  真实标签 (batch × 1)，值为0或1

        返回：
            loss:          BCE损失（三个分支之和）
            loss_kd:       知识蒸馏损失（L1距离）
            prediction:    融合后的预测概率（三个分支平均）
            ground_truth:  真实标签
        """

        # === 步骤1：logit → 概率（sigmoid） ===
        y_H = self.sig(logit_h)  # 知识视角预测概率
        y_N = self.sig(logit_n)  # 行为视角预测概率
        y_T = self.sig(logit_t)  # 融合视角预测概率（教师）

        # === 步骤2：知识蒸馏（KD）损失 ===
        # 使用温度缩放的软标签：
        #   p_soft = sigmoid(logit / T)
        # T < 1 时，sigmoid输入被放大，输出更接近0或1（更"硬"）
        # T > 1 时，sigmoid输入被缩小，输出更接近0.5（更"软"）
        p0_c = self.sig(logit_h / self.T)    # 学生1的软标签
        p0_t = self.sig(logit_n / self.T)    # 学生2的软标签
        p0_enm = self.sig(logit_t / self.T)  # 教师的软标签

        # L1范数：教师软标签与学生软标签的绝对差异之和
        # 物理意义：让学生网络模仿教师网络的"置信度分布"
        loss_kd = (
            torch.sum(torch.abs(p0_enm - p0_c)) +
            torch.sum(torch.abs(p0_enm - p0_t))
        )

        # === 步骤3：BCE损失 ===
        # 初始化损失张量
        loss = torch.tensor([0.0], device=self.device)
        prediction = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([], device=self.device)

        # 三个分支各自的BCE损失：
        #   每个分支独立地与真实标签计算交叉熵
        #   这确保每个分支都能独立完成预测任务
        loss = (
            loss +
            self.crossEntropy(y_H, target) +
            self.crossEntropy(y_N, target) +
            self.crossEntropy(y_T, target)
        )

        # === 步骤4：最终预测（三视角平均） ===
        # 集成三个分支的预测，取平均作为最终输出
        p_mean = (y_H + y_N + y_T) / 3.0

        # 拼接预测值和真实标签（用于后续指标计算）
        prediction = torch.cat([prediction, p_mean])
        ground_truth = torch.cat([ground_truth, target])

        return loss, loss_kd, prediction, ground_truth
