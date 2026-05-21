"""
日志配置模块
============
为训练过程设置日志系统，支持：
  - 自动创建日志和模型保存目录
  - 按时间戳命名日志文件，避免覆盖
  - 同时输出到文件和控制台
"""

import os
from datetime import datetime
import logging


def set_logger(dataset):
    """
    初始化日志系统

    参数：
        dataset: 数据集名称（如 'MOOC'），用于目录命名

    返回：
        logger: Python logging.Logger 实例

    日志文件命名格式：
        logs/{dataset}/training_YYYY-MM-DD_HH-MM-SS.log

    目录自动创建：
        logs/{dataset}/      — 存放训练日志
        best_model/{dataset}/ — 存放最佳模型权重
    """

    # 创建日志目录（如果不存在）
    if not os.path.exists(f"logs/{dataset}/"):
        os.makedirs(f"logs/{dataset}/")

    # 创建最佳模型保存目录（如果不存在）
    if not os.path.exists(f"best_model/{dataset}/"):
        os.makedirs(f"best_model/{dataset}/")

    # 生成带时间戳的日志文件名
    # 格式示例：training_2026-05-21_14-30-00.log
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(
        "logs", f"{dataset}/training_{current_time}.log"
    )

    # 配置日志
    # level=INFO: 记录INFO及以上级别的日志
    # format:     时间 - 级别 - 消息内容
    # handlers:
    #   FileHandler:     写入文件（mode='w'表示每次覆盖）
    #   StreamHandler:   输出到控制台
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="w"),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger()
