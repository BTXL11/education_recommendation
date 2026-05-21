"""
随机种子设置模块
=================
固定所有随机源，确保实验可复现（reproducibility）。

固定的随机源包括：
  - Python内置 random 模块
  - NumPy 随机数生成器
  - PyTorch CPU 随机数生成器
  - PyTorch CUDA 随机数生成器（所有GPU）
  - cuDNN 后端行为
  - Python 哈希种子
"""

import torch
import numpy as np
import os
import random


def set_seed(seed):
    """
    固定所有随机种子

    参数：
        seed: 随机种子值（论文中不同数据集使用不同种子：
              MOOC=42, University_Course=25, LectureBank=25）
    """

    # Python内置random模块
    random.seed(seed)

    # NumPy随机数生成器
    np.random.seed(seed)

    # 设置可见GPU设备
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'

    # Python哈希种子（影响字典、集合等的迭代顺序）
    os.environ['PYTHONHASHSEED'] = str(seed)

    # PyTorch CPU随机数
    torch.manual_seed(seed)

    # PyTorch CUDA随机数（当前GPU）
    torch.cuda.manual_seed(seed)

    # PyTorch CUDA随机数（所有GPU）
    torch.cuda.manual_seed_all(seed)

    # cuDNN确定性模式：
    #   deterministic=True:  使用确定性算法（可复现但可能更慢）
    #   benchmark=False:     不自动寻找最优卷积算法（避免不确定性）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
