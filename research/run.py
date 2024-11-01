import subprocess
import logging
import datetime
import os
import re  # 用于从文件名中提取编号
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d  # 用于读取 PLY 文件
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from MinkowskiEngine import utils as ME_utils
from scipy.spatial import KDTree
import pandas as pd


# 获取当前时间作为文件名的一部分
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f'training_log_{current_time}.txt'

def setup_logger():
    # 创建日志记录器
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 设置日志记录器
logger = setup_logger()


def custom_collate_fn(batch):
    """
    自定义的collate函数，处理不同大小的点云数据
    """
    # 检查是否有无效样本
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None, None
        
    chunks_original_batch = []
    chunks_compress_batch = []
    
    for chunks_original, chunks_compress in batch:
        chunks_original_batch.append(chunks_original)
        chunks_compress_batch.append(chunks_compress)
    
    return chunks_original_batch, chunks_compress_batch

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


def chunk_point_cloud_fixed_size(points, block_size=256, cube_size=1024, min_points_ratio=0.1, device='cuda'):
    """
    将点云数据切分为固定大小的块，支持可选偏移，并确保块的数量和大小一致。
    """
    points = torch.tensor(points, device=device, dtype=torch.float32)  # 使用 float32
    coords, colors = points[:, :3], points[:, 3:]

    min_points_threshold = int(block_size ** 3 * min_points_ratio)

    x_range = torch.arange(0, cube_size, block_size, device=device)
    y_range = torch.arange(0, cube_size, block_size, device=device)
    z_range = torch.arange(0, cube_size, block_size, device=device)

    blocks = []

    for x in x_range:
        for y in y_range:
            for z in z_range:
                mask = (
                    (coords[:, 0] >= x) & (coords[:, 0] < x + block_size) &
                    (coords[:, 1] >= y) & (coords[:, 1] < y + block_size) &
                    (coords[:, 2] >= z) & (coords[:, 2] < z + block_size)
                )

                block_coords = coords[mask]
                block_colors = colors[mask]
                logger.info(f"阈值点数：{min_points_threshold} . block点数：{len(block_coords)}", )
                if len(block_coords) >= min_points_threshold:
                    block_points = torch.cat((block_coords, block_colors), dim=1)
                    blocks.append((block_points.cpu().numpy(), (x.item(), y.item(), z.item())))

                # 清理未使用的张量
                del block_coords, block_colors
                torch.cuda.empty_cache()

    return blocks


def adjust_points(points, target_num_points, perturbation_scale=0.01):
    """
    调整点的数量到目标数量。如果点的数量少于目标数量，则通过复制并扰动原始点来补齐。
    如果点的数量多于目标数量，则随机采样点。

    参数:
    points (numpy.ndarray): 原始点云数据，形状为 (N, D)。
    target_num_points (int): 目标点的数量。
    perturbation_scale (float): 扰动的尺度，默认为 0.01。

    返回:
    numpy.ndarray: 调整后的点云数据，形状为 (target_num_points, D)。
    """
    current_num_points = len(points)
    if current_num_points == target_num_points:
        return points
    elif current_num_points < target_num_points:
        num_missing_points = target_num_points - current_num_points
        existing_coords = set(map(tuple, points))

        new_points = []
        while len(new_points) < num_missing_points:
            indices = np.random.choice(current_num_points, num_missing_points, replace=True)
            missing_points = points[indices]

            # 对选中的点进行微小的扰动
            perturbations = np.random.normal(scale=perturbation_scale, size=missing_points.shape)
            perturbed_points = missing_points + perturbations

            # 检查是否有重复的坐标
            for point in perturbed_points:
                point_tuple = tuple(point)
                if point_tuple not in existing_coords:
                    existing_coords.add(point_tuple)
                    new_points.append(point)
                    if len(new_points) >= num_missing_points:
                        break

        return np.vstack((points, np.array(new_points)))
    else:
        indices = np.random.choice(current_num_points, target_num_points, replace=False)
        return points[indices]
    
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



class PointCloudDataset(Dataset):
    def __init__(self, folder_A, folder_B, num_blocks=20):
        """
        初始化数据集
        :param folder_A: 原始点云文件夹路径
        :param folder_B: 压缩点云文件夹路径
        :param num_blocks: 将点云分成多少块
        """
        self.files_A = self.match_files(folder_A, folder_B)
        self.num_blocks = num_blocks
        
        if not self.files_A:
            logger.info("没有找到匹配的文件对，请检查文件名和路径是否正确！")
        else:
            logger.info(f"共找到 {len(self.files_A)} 对文件。")
            logger.info("匹配的文件对：")
            for file_a, file_b in self.files_A:
                logger.info(f"A: {file_a} <-> B: {file_b}")

    def match_files(self, folder_A, folder_B):
        """
        匹配两个文件夹中的点云文件
        :return: 匹配的文件对列表 [(file_A, file_B), ...]
        """
        # 获取两个文件夹中的所有 .ply 文件
        files_A = sorted([f for f in os.listdir(folder_A) if f.endswith('.ply')])
        files_B = sorted([f for f in os.listdir(folder_B) if f.endswith('.ply')])

        logger.info(f"文件夹A中的文件数量: {len(files_A)}")
        logger.info(f"文件夹B中的文件数量: {len(files_B)}")

        # 从文件名中提取编号
        def extract_id(filename):
            match = re.search(r'(\d{3,4})(?=\.ply$)', filename)
            return match.group(1) if match else None

        # 创建文件路径映射
        files_A_dict = {extract_id(f): os.path.join(folder_A, f) for f in files_A}
        files_B_dict = {extract_id(f): os.path.join(folder_B, f) for f in files_B}

        # 找到共同的文件编号
        common_ids = set(files_A_dict.keys()) & set(files_B_dict.keys())
        logger.info(f"找到的匹配文件数量: {len(common_ids)}")

        # 返回匹配的文件对
        matched_files = [
            (files_A_dict[id_], files_B_dict[id_])
            for id_ in sorted(common_ids)
        ]

        return matched_files

    def split_points_by_coordinate(self, points, num_blocks=20):
        """
        按最长维度分割点云
        :param points: 点云数据 (N, 6)
        :param num_blocks: 分块数量
        :return: 分块列表
        """
        logger.info(f"开始分割点云，总点数：{len(points)}")
        
        # 1. 找到点云的边界
        min_coords = np.min(points[:, :3], axis=0)
        max_coords = np.max(points[:, :3], axis=0)
        ranges = max_coords - min_coords
        
        logger.info(f"点云范围: X轴: {ranges[0]:.2f}, Y轴: {ranges[1]:.2f}, Z轴: {ranges[2]:.2f}")
        
        # 2. 确定最长的维度
        split_dim = np.argmax(ranges)
        dim_names = ['X', 'Y', 'Z']
        logger.info(f"选择 {dim_names[split_dim]} 轴进行分割（最长维度）")
        
        # 3. 按照最长维度排序
        sorted_indices = np.argsort(points[:, split_dim])
        sorted_points = points[sorted_indices]
        
        # 4. 分割点云
        points_per_block = len(points) // num_blocks
        blocks = []
        for i in range(num_blocks):
            start_idx = i * points_per_block
            end_idx = start_idx + points_per_block if i < num_blocks-1 else len(points)
            block = sorted_points[start_idx:end_idx]
            blocks.append(block)
            logger.info(f"Block {i}: {len(block)} 个点")
        
        return blocks

    def __len__(self):
        return len(self.files_A)

    def __getitem__(self, idx):
        """
        获取数据集中的一对点云文件
        :param idx: 索引
        :return: (原始点云块列表, 压缩点云块列表) 或 None（如果处理失败）
        """
        try:
            file_A, file_B = self.files_A[idx]
            logger.info(f"\n处理第 {idx} 对文件:")
            logger.info(f"原始文件: {file_A}")
            logger.info(f"压缩文件: {file_B}")
            
            # 加载点云数据
            points_A = self.load_ply(file_A)
            points_B = self.load_ply(file_B)
            
            # 检查并移除重复点
            check_compress = has_duplicates(points_B)
            if check_compress:
                logger.info("检测到压缩点云中存在重复点，正在移除...")
                points_B = remove_duplicates(points_B)
                logger.info(f"移除重复点后的点数: {len(points_B)}")
            
            # 分割点云
            logger.info("\n开始分割原始点云...")
            chunks_A = self.split_points_by_coordinate(points_A, self.num_blocks)
            logger.info("\n开始分割压缩点云...")
            chunks_B = self.split_points_by_coordinate(points_B, self.num_blocks)
            
            return (chunks_A, chunks_B)
            
        except Exception as e:
            print(f"处理文件时发生错误: {str(e)}")
            return None

    def load_ply(self, file_path):
        """
        加载PLY文件
        :param file_path: 文件路径
        :return: 点云数组 (N, 6)，包含坐标和颜色
        """
        logger.info(f"\n加载文件：{file_path}")
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        logger.info(f"加载完成，点数量：{points.shape[0]}")
        return np.hstack((points, colors))


def save_point_cloud_as_ply(coords, feats, filename):
    """
    将点云数据保存为 PLY 文件。
    :param coords: (N, 3) 点云坐标
    :param feats: (N, 3) RGB 颜色
    :param filename: 要保存的 PLY 文件名
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(feats)

    # 保存为 PLY 文件
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
    logger.info(f"保存成功: {filename}")



class MyNet(ME.MinkowskiNetwork):
    def __init__(self, in_channels=3, out_channels=3, D=3):
        ME.MinkowskiNetwork.__init__(self, D)
        
        # 编码器第一层
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            dimension=D)
        self.norm1 = ME.MinkowskiBatchNorm(32)
        
        # 编码器第二层
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            dimension=D)
        self.norm2 = ME.MinkowskiBatchNorm(64)
        
        # 解码器层
        self.conv2_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            dimension=D)
        self.norm2_tr = ME.MinkowskiBatchNorm(32)
        
        # 最终输出层
        self.final = ME.MinkowskiConvolution(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dimension=D)

    def forward(self, x):
        # 编码器路径
        out1 = self.norm1(self.conv1(x))
        out1 = MEF.relu(out1)
        
        out2 = self.norm2(self.conv2(out1))
        out2 = MEF.relu(out2)
        
        # 解码器路径
        out = self.norm2_tr(self.conv2_tr(out2))
        out = MEF.relu(out)
        
        # skip connection
        out = out + out1
        
        # 最终输出
        out = self.final(out)
        
        return out


if __name__ == '__main__':
    dataset = PointCloudDataset(folder_A='./train/original',
                                folder_B='./train/compress',
                                num_blocks = 50
                                )
    data_loader = DataLoader(dataset, batch_size=8,collate_fn=custom_collate_fn)
    test_dataset = PointCloudDataset(folder_A='./test/original',folder_B='./test/compress',num_blocks = 50)

    def position_loss(pred, target):
        """
        改进的位置损失函数，包含归一化处理
        """
        if isinstance(pred, ME.SparseTensor) and isinstance(target, ME.SparseTensor):
            pred = pred.F
            target = target.F

        # 对预测值和目标值进行归一化
        pred_min, _ = pred.min(dim=0, keepdim=True)
        pred_max, _ = pred.max(dim=0, keepdim=True)
        target_min, _ = target.min(dim=0, keepdim=True)
        target_max, _ = target.max(dim=0, keepdim=True)

        # 使用相同的范围进行归一化
        min_vals = torch.min(pred_min, target_min)
        max_vals = torch.max(pred_max, target_max)

        # 避免除以零
        scale = max_vals - min_vals
        scale[scale == 0] = 1.0

        pred_normalized = (pred - min_vals) / scale
        target_normalized = (target - min_vals) / scale

        # 计算归一化后的MSE损失
        loss = torch.nn.functional.mse_loss(pred_normalized, target_normalized)

        # 添加调试信息
        if torch.isnan(loss) or torch.isinf(loss):
            logger.info(f"Warning: Loss is {loss}")
            logger.info(f"Pred range: {pred.min().item():.4f} to {pred.max().item():.4f}")
            logger.info(f"Target range: {target.min().item():.4f} to {target.max().item():.4f}")

        return loss

    def train_model(model, data_loader, optimizer, device='cuda', epochs=50):
        model = model.to(device)
        model.train()

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            logger.info(f"\n Start epoch {epoch}, total_loss= {total_loss}, num_batches= {num_batches}")
            for batch_idx, (batch_chunks_original, batch_chunks_compress) in enumerate(data_loader):
                batch_loss = 0  # 累积整个批次的损失

                # 遍历batch中的每个样本
                for sample_idx, (chunks_original, chunks_compress) in enumerate(zip(batch_chunks_original, batch_chunks_compress)):
                    logger.info(f"\nProcessing batch {batch_idx}, sample {sample_idx}")

                    # 遍历当前样本的所有块
                    for block_idx in range(len(chunks_original)):
                        # 获取当前块的数据
                        coords_original = chunks_original[block_idx][:, :3]  # [N, 3] 坐标
                        coords_compress = chunks_compress[block_idx][:, :3]  # [N, 3] 坐标

                        # 转换为torch.Tensor
                        coords_original = torch.from_numpy(coords_original).float()
                        coords_compress = torch.from_numpy(coords_compress).float()

                        # 转换为ME需要的格式
                        coords_original_tensor = ME_utils.batched_coordinates([coords_original], device=device)
                        coords_compress_tensor = ME_utils.batched_coordinates([coords_compress], device=device)
                       
                        coords_original_features = coords_original.to(device)
                        coords_compress_features = coords_compress.to(device)

                        # 构建稀疏张量
                        original_sparse_tensor = ME.SparseTensor(features=coords_original_features, coordinates=coords_original_tensor)
                        
                        compress_sparse_tensor = ME.SparseTensor(features=coords_compress_features
, coordinates=coords_compress_tensor)

                        logger.info(f'Origin: {original_sparse_tensor.shape}')
                        logger.info(f'Compress: {compress_sparse_tensor.shape}')
                      
                        coords_compress_dedup = compress_sparse_tensor.C[:, 1:].cpu().numpy()  # 去掉批次维度并转为numpy
                        coords_original_numpy = coords_original.cpu().numpy()

                        coords_new_original = find_corresponding_original_points(coords_compress_dedup, coords_original_numpy)
                        coords_new_original = torch.from_numpy(coords_new_original).float()
                        coords_new_original_features = coords_new_original.to(device)
                        coords_new_original_tensor = ME_utils.batched_coordinates([coords_new_original], device=device)
                        
                        new_original_sparse_tensor = ME.SparseTensor(features=coords_new_original_features, coordinates=coords_new_original_tensor)
                        logger.info(f"New Origin {new_original_sparse_tensor.shape}")

                        # 前向传播
                        output = model(compress_sparse_tensor)

                        # 计算损失
                        loss = position_loss(output.F.float(), new_original_sparse_tensor.F.float())
                        batch_loss += loss

                        logger.info(f'Sample {sample_idx}, Block {block_idx} Loss: {loss.item():.4f}')

                # 对整个batch进行反向传播和优化
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()
                num_batches += 1
                logger.info(f'Batch {batch_idx} total loss: {batch_loss.item():.4f}')

            # 每个epoch结束后打印平均损失
            avg_loss = total_loss / num_batches
            logger.info(f'\nEpoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')

            # 保存模型权重
            save_path = f'epoch_{epoch + 1}_model.pth'
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved at epoch {epoch + 1} to {save_path}")



    model = MyNet()
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, data_loader, optimizer)
    
    def evaluate_and_save(model_path, dataset, output_dir='./output'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model = MyNet()
        model.load_state_dict(torch.load(model_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        data_loader = DataLoader(dataset, batch_size=8, collate_fn=custom_collate_fn)

        with torch.no_grad():
            for batch_idx, (chunks_original, chunks_compress) in enumerate(data_loader):
                logger.info(f"\n处理第 {batch_idx + 1} 个点云")

                # 初始化存储所有block的数组
                all_original_coords = []
                all_original_feats = []
                all_compress_coords = []
                all_compress_feats = []
                all_predicted_coords = []
                all_predicted_feats = []

                # 处理每个块
                for block_idx in range(len(chunks_original[0])):
                    logger.info(f"\n处理块 {block_idx}")

                    # 获取原始点云数据
                    original_coords = chunks_original[0][block_idx][:, :3]
                    original_feats = chunks_original[0][block_idx][:, 3:]
                    all_original_coords.append(original_coords)
                    all_original_feats.append(original_feats)

                    # 处理压缩点云
                    compress_coords = chunks_compress[0][block_idx][:, :3]
                    compress_feats = chunks_compress[0][block_idx][:, 3:]
                    all_compress_coords.append(compress_coords)
                    all_compress_feats.append(compress_feats)

                    # 模型处理部分
                    compress_coords_tensor = torch.from_numpy(compress_coords).float()
                    compress_coords_tensor_batch = ME_utils.batched_coordinates([compress_coords_tensor], device=device)
                    compress_coords_features = compress_coords_tensor.to(device)
                    compress_sparse_tensor = ME.SparseTensor(
                        features=compress_coords_features,
                        coordinates=compress_coords_tensor_batch
                    )

                    output = model(compress_sparse_tensor)
                    output_coords = output.C[:, 1:].cpu().numpy()

                    # 找到匹配的点
                    matched_indices = []
                    compress_coords_numpy = compress_coords
                    for coord in output_coords:
                        matches = np.where((compress_coords_numpy == coord).all(axis=1))[0]
                        if len(matches) > 0:
                            matched_indices.append(matches[0])

                    C_coords = compress_coords_numpy[matched_indices]
                    C_feats = compress_feats[matched_indices]

                    # 计算预测坐标
                    predicted_offsets = output.F.cpu().numpy()
                    predicted_coords = C_coords + predicted_offsets

                    all_predicted_coords.append(predicted_coords)
                    all_predicted_feats.append(C_feats)

                # 合并所有block的点云
                merged_original_coords = np.concatenate(all_original_coords, axis=0)
                merged_original_feats = np.concatenate(all_original_feats, axis=0)
                merged_compress_coords = np.concatenate(all_compress_coords, axis=0)
                merged_compress_feats = np.concatenate(all_compress_feats, axis=0)
                merged_predicted_coords = np.concatenate(all_predicted_coords, axis=0)
                merged_predicted_feats = np.concatenate(all_predicted_feats, axis=0)

                # 保存合并后的点云
                original_file = f"{output_dir}/original_merged_batch_{batch_idx}.ply"
                compressed_file = f"{output_dir}/compressed_merged_batch_{batch_idx}.ply"
                result_file = f"{output_dir}/result_merged_batch_{batch_idx}.ply"

                save_point_cloud_as_ply(merged_original_coords, merged_original_feats, original_file)
                save_point_cloud_as_ply(merged_compress_coords, merged_compress_feats, compressed_file)
                save_point_cloud_as_ply(merged_predicted_coords, merged_predicted_feats, result_file)

                logger.info(f"已保存合并后的点云文件")
                logger.info(f"合并后原始点云点数: {len(merged_original_coords)}")
                logger.info(f"合并后压缩点云点数: {len(merged_compress_coords)}")
                logger.info(f"合并后预测点云点数: {len(merged_predicted_coords)}")

                # 计算PSNR
                cmd1 = f"../../mpeg-pcc-tmc2/bin/PccAppMetrics --uncompressedDataPath={original_file} --reconstructedDataPath={compressed_file} --resolution=1023 --frameCount=1"
                process1 = subprocess.Popen(cmd1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output1, error1 = process1.communicate()

                psnr_file1 = f"{output_dir}/psnr_compressed_merged_batch_{batch_idx}.txt"
                with open(psnr_file1, 'w') as f:
                    f.write(output1.decode())
                logger.info(f"压缩点云PSNR结果已保存到：{psnr_file1}")

                cmd2 = f"../../mpeg-pcc-tmc2/bin/PccAppMetrics --uncompressedDataPath={original_file} --reconstructedDataPath={result_file} --resolution=1023 --frameCount=1"
                process2 = subprocess.Popen(cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output2, error2 = process2.communicate()

                psnr_file2 = f"{output_dir}/psnr_result_merged_batch_{batch_idx}.txt"
                with open(psnr_file2, 'w') as f:
                    f.write(output2.decode())
                logger.info(f"处理结果PSNR已保存到：{psnr_file2}")

    logger.info("\n开始评估...")
    evaluate_and_save(model_path='epoch_50_model.pth',dataset=test_dataset)
    logger.info("评估完成！请查看 output 文件夹中的结果")
    
