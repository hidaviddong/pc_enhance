import os
import re
import time
import numpy as np
import pandas as pd
import torch
import open3d as o3d
import MinkowskiEngine as ME
from torch.utils.data import Dataset
from scipy.spatial import KDTree
from logger import logger
import matplotlib.pyplot as plt


def plot_loss_curve(pred_losses, baseline_losses, save_path='loss_curve.png', current_epoch=None):
    """
    绘制多条损失曲线
    
    Args:
        pred_losses: list, 预测重建的损失值列表
        baseline_losses: list, baseline的损失值列表
        save_path: str, 保存图像的路径
        current_epoch: int, 当前的epoch数
    """
    plt.figure(figsize=(12, 8))
    
    # 绘制预测重建损失曲线
    plt.plot(pred_losses, 'b-', label='Prediction Loss', linewidth=2)
    
    # 绘制baseline损失曲线
    plt.plot(baseline_losses, 'r--', label='Baseline Loss', linewidth=2)
    
    # 设置图表属性
    plt.title(f'Training Losses (Current Epoch: {current_epoch})' if current_epoch is not None else 'Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # 设置y轴范围
    if pred_losses and baseline_losses:
        min_loss = min(min(pred_losses), min(baseline_losses))
        max_loss = max(max(pred_losses), max(baseline_losses))
        plt.ylim([max(0, min_loss * 0.9), max_loss * 1.1])
    
    plt.savefig(save_path)
    plt.close()
    
# def position_loss(pred, origin, compress):
#     """
#     计算位置损失
    
#     Args:
#         pred: [N, 3] 预测的偏移量
#         origin: [N, 3] 原始点云
#         compress: [N, 3] 压缩点云
    
#     Returns:
#         tuple: (total_loss, pred_loss, baseline_loss)
#     """
#     # 计算预测的重建点云
#     reconstructed = pred + compress
    
#     # 计算预测的重建损失
#     pred_loss = torch.nn.functional.mse_loss(reconstructed, origin)
    
#     # 计算baseline损失（直接使用压缩点云）
#     baseline_loss = torch.nn.functional.mse_loss(compress, origin)
    
#     return pred_loss, baseline_loss

def position_loss(pred, origin, compress):
    """
    计算位置损失
    Args:
        pred: 网络预测的偏移量 [N, 3]
        origin: 目标点云坐标 [N, 3]
        compress: 压缩点云坐标 [N, 3]
    Returns:
        tuple: (pred_loss, baseline_loss)
    """
    # 计算目标偏移量
    target_offset = origin - compress
    
    # 计算预测偏移量的损失
    pred_loss = torch.nn.functional.mse_loss(pred, target_offset)
    
    # 计算baseline损失（压缩点云与原始点云的差异）
    baseline_loss = torch.nn.functional.mse_loss(compress, origin)
    
    # TODO:打印compress origin 值
    
    return pred_loss, baseline_loss

def remove_duplicates(points):
    """
    去除点云数据中的重复点。

    参数:
    points (numpy.ndarray): 点云数据，形状为 (N, D)。

    返回:
    numpy.ndarray: 去除重复点后的点云数据。
    """
    df = pd.DataFrame(points)
    df = df.drop_duplicates()
    return df.to_numpy()

def has_duplicates(points, tol=1e-9):
    """
    检查点云数据集中是否存在重复的点。

    参数:
    points (torch.Tensor or numpy.ndarray): 点云数据，形状为 (N, D)。
    tol (float): 判断重复点的容差，默认为 1e-9。

    返回:
    bool: 如果存在重复的点，则返回 True；否则返回 False。
    """
    if isinstance(points, torch.Tensor):
        # 如果是 GPU 张量，先移动到 CPU
        if points.is_cuda:
            points = points.cpu()
        # 转换为 NumPy 数组
        points = points.numpy()

    tree = KDTree(points)
    for i, point in enumerate(points):
        distances, indices = tree.query(point, k=2)
        if distances[1] < tol:
            return True
    return False


def has_duplicates_output(points, tol=1e-9):
    """
    检查点云数据集中是否存在重复的点，并输出重复的坐标。

    参数:
    points (torch.Tensor or numpy.ndarray): 点云数据，形状为 (N, D)。
    tol (float): 判断重复点的容差，默认为 1e-9。

    返回:
    tuple: (bool, list)，如果存在重复的点，则返回 (True, 重复点的列表)；否则返回 (False, 空列表)。
    """
    if isinstance(points, torch.Tensor):
        # 如果是 GPU 张量，先移动到 CPU
        if points.is_cuda:
            points = points.cpu()
        # 转换为 NumPy 数组
        points = points.numpy()

    tree = KDTree(points)
    duplicates = []
    for i, point in enumerate(points):
        distances, indices = tree.query(point, k=2)
        if distances[1] < tol:
            duplicates.append(point)

    has_dup = len(duplicates) > 0
    return has_dup, duplicates
    
def find_corresponding_original_points(compressed_points, original_points):
    """
    為壓縮點雲中的每個點找到原始點雲中未被使用的最近點
    
    參數:
    compressed_points: 壓縮後的點雲數據 (N, D)
    original_points: 原始點雲數據 (M, D)
    
    返回:
    numpy.ndarray: 與壓縮點雲相同shape的矩陣，包含來自原始點雲的未重複點
    
    異常:
    ValueError: 當無法找到足夠的唯一對應點時拋出
    """
    # 首先驗證基本條件
    if len(compressed_points) > len(original_points):
        raise ValueError(
            f"壓縮點數量({len(compressed_points)})不能大於原始點數量({len(original_points)})"
        )
    
    # 將原始點轉換為tuple以便使用set操作
    original_points_set = set(map(tuple, original_points))
    if len(original_points_set) < len(compressed_points):
        raise ValueError(
            f"原始點雲中的唯一點數量({len(original_points_set)})小於壓縮點數量({len(compressed_points)})"
        )
    
    tree = KDTree(original_points)
    result = np.zeros_like(compressed_points)
    used_original_indices = set()
    
    # 為每個壓縮點找對應的原始點
    for i, comp_point in enumerate(compressed_points):
        # 初始搜索範圍設為所有剩餘的原始點
        remaining_points = len(original_points) - len(used_original_indices)
        if remaining_points < (len(compressed_points) - i):
            raise ValueError(
                f"剩餘可用點數({remaining_points})小於待處理的壓縮點數({len(compressed_points) - i})"
            )
        
        # 查詢所有剩餘的點
        distances, indices = tree.query(comp_point, k=len(original_points))
        
        # 找到第一個未使用的點
        found = False
        for idx in indices:
            if idx not in used_original_indices:
                # 驗證這個點確實來自原始點雲
                point_tuple = tuple(original_points[idx])
                if point_tuple in original_points_set:
                    result[i] = original_points[idx]
                    used_original_indices.add(idx)
                    found = True
                    break
        
        if not found:
            raise ValueError(f"無法為壓縮點 {i} 找到未使用的對應點")
        
        # 打印進度
        if (i + 1) % 1000 == 0:
            logger.info(f"已處理: {i + 1}/{len(compressed_points)} 點")
    
    # 最終驗證
    result_points_set = set(map(tuple, result))
    
    # 驗證結果長度
    if len(result) != len(compressed_points):
        raise ValueError(
            f"結果點數({len(result)})與壓縮點數({len(compressed_points)})不匹配"
        )
    
    # 驗證沒有重複點
    if len(result_points_set) != len(result):
        raise ValueError(
            f"結果中存在重複點: 唯一點數({len(result_points_set)}) != 總點數({len(result)})"
        )
    
    # 驗證所有點都來自原始點雲
    if not result_points_set.issubset(original_points_set):
        raise ValueError("結果中包含不在原始點雲中的點")
    
    logger.info("驗證通過:")
    logger.info(f"- 結果點數: {len(result)}")
    logger.info(f"- 唯一點數: {len(result_points_set)}")
    logger.info(f"- 所有點都來自原始點雲")
    
    return result


def save_point_cloud_as_ply(coords, filename):
    """
    将点云数据保存为 PLY 文件，只保存坐标信息。
    
    Args:
        coords: (N, 3) 点云坐标
        filename: 要保存的 PLY 文件名
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)

    # 保存为 PLY 文件
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
    logger.info(f"保存成功: {filename}")


def match_points_o3d(points_A, points_B):
    # 将numpy数组转换为open3d的点云格式
    pcd_A = o3d.geometry.PointCloud()
    pcd_A.points = o3d.utility.Vector3dVector(points_A)
    
    # 构建KDTree
    kdtree = o3d.geometry.KDTreeFlann(pcd_A)
    
    # 初始化结果点云C，大小与B相同
    points_C = np.zeros_like(points_B)
    matched_points = set()
    
    # 对B中的每个点找A中最近的点
    for i, point in enumerate(points_B):
        k = 1  # 先找1个最近邻
        while True:
            # search_knn 返回 (查找成功与否, 最近邻索引列表, 距离列表)
            _, idx, _ = kdtree.search_knn_vector_3d(point, k)
            
            # 找到未被匹配的点
            for j in idx:
                if j not in matched_points:
                    matched_points.add(j)
                    points_C[i] = points_A[j]  # 直接存储到结果点云C中
                    break
            else:
                # 如果都已匹配，增加搜索范围
                k *= 2
                continue
            break
    
    return points_C

class PointCloudDataset(Dataset):
    def __init__(self, preprocessed_data):
        self.blocks = []
        for file_blocks in preprocessed_data:
            self.blocks.extend(file_blocks)
    
    def __len__(self):
        return len(self.blocks)
    
    def __getitem__(self, idx):
        block = self.blocks[idx]

        
        # 去掉多余的维度，确保数据格式正确
        return {
            'compress_coords': block['compress_coords'], 
            'compress_feats': block['compress_feats'],
            'new_origin_coords': block['new_origin_coords'],
            'new_origin_feats': block['new_origin_feats'],
            'original_compress_coords': block['original_compress_coords'],
            'original_new_origin_coords': block['original_new_origin_coords'],
            'block_index': block['block_index'],
            'file_index': block['file_index'],
            'total_blocks': block['total_blocks'],
            'points_in_block': block['points_in_block']
        }
