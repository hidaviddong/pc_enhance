# logger.py

import logging
import datetime
import os

def setup_logger():
    """
    设置并返回logger实例
    """
    # 创建logs目录
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # 获取当前时间作为文件名的一部分
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/training_log_{current_time}.txt'

    # 创建logger
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)

    # 清除现有的处理器（避免重复）
    if logger.handlers:
        logger.handlers.clear()

    # 创建文件处理器和控制台处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()

    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 创建全局logger实例
logger = setup_logger()
