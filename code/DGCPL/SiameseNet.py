"""
孪生网络 (Siamese Network)
===========================
用于预测一对概念之间是否存在先决条件关系。

网络结构：
    c1, c2 (各256维概念嵌入)
        │
        ├──→ fc_layer(c1) ──→ ReLU ──→ c1' (64维)
        ├──→ fc_layer(c2) ──→ ReLU ──→ c2' (64维)
        │
        ├──→ diff = c1' - c2'                  (64维，捕获不对称性)
        ├──→ multiply = c1' * c2'              (64维，捕获相关性)
        │
        └──→ concat(c1', c2', diff, multiply)  (256维)
                │
                └──→ Linear(256 → 1) ──→ logit

孪生网络的设计要点：
  - 共享权重：c1 和 c2 经过同一个 fc_layer，保证对称处理
  - diff 特征：捕获方向性（先决条件关系是不对称的：A是B的先决 ≠ B是A的先决）
  - multiply 特征：捕获两个概念嵌入的共性（关联强度）
  - 四合一拼接：充分利用嵌入对的语义关系
"""

import torch
import torch.nn as nn


class SiameseNet(nn.Module):
    """
    孪生网络：输入一对概念嵌入，输出先决条件关系的logit

    模型中有三个独立的 SiameseNet 实例：
        - siameseNet1: 知识视角 (X_H) → 学生网络1
        - siameseNet2: 行为视角 (X_N) → 学生网络2
        - siameseNet3: 融合视角 (X_T) → 教师网络
    """

    def __init__(self, input_dim):
        """
        参数：
            input_dim: 输入概念嵌入的维度（256）
        """
        super(SiameseNet, self).__init__()

        # === 共享的特征变换层 ===
        # 将256维概念嵌入压缩到64维
        # 两个输入概念共享此层权重，保证对称处理
        self.fc_layer = nn.Linear(input_dim, 64)

        # ReLU非线性激活
        self.relu_layer = nn.ReLU()

        # === 分类层 ===
        # 输入：4 × 64 = 256维（c1', c2', diff, multiply 的拼接）
        # 输出：1维 logit（未经过sigmoid，由损失函数内部处理）
        self.classificaton_layer = nn.Linear(64 * 4, 1)

        # 可选的 sigmoid 层（注释掉，因损失函数使用 BCELoss + sigmoid）
        # self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x1, x2):
        """
        前向传播

        参数：
            x1: 概念1的嵌入向量 (batch_size × 256)
            x2: 概念2的嵌入向量 (batch_size × 256)

        返回：
            logit: 原始预测值 (batch_size × 1)
                   正值 → 倾向于预测1（存在先决条件关系）
                   负值 → 倾向于预测0（不存在先决条件关系）
        """
        # === 步骤1：共享的嵌入变换 ===
        # 两个概念分别经过同一个 fc_layer + ReLU
        c1 = self.relu_layer(self.fc_layer(x1))  # (batch, 64)
        c2 = self.relu_layer(self.fc_layer(x2))  # (batch, 64)

        # === 步骤2：计算交互特征 ===
        # diff: 逐元素差值 → 捕获不对称性
        #   如果 A→B 存在先决条件关系，期望 c1 - c2 呈现特定模式
        #   如果 B→A 则 diff 符号相反，从而区分方向
        diff = torch.sub(c1, c2)

        # multiply: 逐元素乘积 → 捕获共性
        #   如果两个概念高度相关（无论方向），乘积值较大
        #   如果两个概念无关，乘积值接近0
        multiply = torch.mul(c1, c2)

        # === 步骤3：特征拼接 ===
        # 四部分合并：[原始c1 | 原始c2 | 差异 | 乘积]
        # 共 64+64+64+64 = 256 维
        v = torch.cat((c1, c2, diff, multiply), 1)

        # === 步骤4：分类 ===
        # 从256维拼接特征映射到1维logit
        # 不使用 sigmoid，因为 LossFunc 中使用 BCELoss(sigmoid(logit), target)
        # （注释掉的行：如果在此处加 sigmoid，则得到概率而非logit）
        # pred_prob = self.sigmoid_layer(self.classificaton_layer(v))
        logit = self.classificaton_layer(v)

        return logit
