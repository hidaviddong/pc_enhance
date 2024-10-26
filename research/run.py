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
                print(f"阈值点数：{min_points_threshold} . block点数：{len(block_coords)}", )
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
    """
    tree = KDTree(original_points)
    result = np.zeros_like(compressed_points)
    used_original_indices = set()  # 追踪已使用的原始點的索引
    
    print(f"壓縮點雲shape: {compressed_points.shape}")
    print(f"原始點雲shape: {original_points.shape}")
    
    # 為每個壓縮點找對應的原始點
    for i, comp_point in enumerate(compressed_points):
        # 查詢足夠多的近鄰點
        k = min(10000, len(original_points))  # 可以根據需要調整 k 的值
        distances, indices = tree.query(comp_point, k=k)
        
        # 找到第一個未使用的點
        found = False
        for idx in indices:
            if idx not in used_original_indices:
                result[i] = original_points[idx]
                used_original_indices.add(idx)
                found = True
                break
                
        if not found:
            print(f"警告：點 {i} 無法找到未使用的對應點")
            # 這裡可以根據需求決定如何處理找不到未使用點的情況
            # 比如使用距離最近的點，或者拋出異常
        
        # 打印進度
        if (i + 1) % 1000 == 0:
            print(f"已處理: {i + 1}/{len(compressed_points)} 點")
    
    # 驗證結果
    unique_points = set(map(tuple, result))
    print(f"結果點數: {len(result)}")
    print(f"唯一點數: {len(unique_points)}")
    
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
            print("没有找到匹配的文件对，请检查文件名和路径是否正确！")
        else:
            print(f"共找到 {len(self.files_A)} 对文件。")
            print("匹配的文件对：")
            for file_a, file_b in self.files_A:
                print(f"A: {file_a} <-> B: {file_b}")

    def match_files(self, folder_A, folder_B):
        """
        匹配两个文件夹中的点云文件
        :return: 匹配的文件对列表 [(file_A, file_B), ...]
        """
        # 获取两个文件夹中的所有 .ply 文件
        files_A = sorted([f for f in os.listdir(folder_A) if f.endswith('.ply')])
        files_B = sorted([f for f in os.listdir(folder_B) if f.endswith('.ply')])

        print(f"文件夹A中的文件数量: {len(files_A)}")
        print(f"文件夹B中的文件数量: {len(files_B)}")

        # 从文件名中提取编号
        def extract_id(filename):
            match = re.search(r'(\d{3,4})(?=\.ply$)', filename)
            return match.group(1) if match else None

        # 创建文件路径映射
        files_A_dict = {extract_id(f): os.path.join(folder_A, f) for f in files_A}
        files_B_dict = {extract_id(f): os.path.join(folder_B, f) for f in files_B}

        # 找到共同的文件编号
        common_ids = set(files_A_dict.keys()) & set(files_B_dict.keys())
        print(f"找到的匹配文件数量: {len(common_ids)}")

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
        print(f"开始分割点云，总点数：{len(points)}")
        
        # 1. 找到点云的边界
        min_coords = np.min(points[:, :3], axis=0)
        max_coords = np.max(points[:, :3], axis=0)
        ranges = max_coords - min_coords
        
        print(f"点云范围: X轴: {ranges[0]:.2f}, Y轴: {ranges[1]:.2f}, Z轴: {ranges[2]:.2f}")
        
        # 2. 确定最长的维度
        split_dim = np.argmax(ranges)
        dim_names = ['X', 'Y', 'Z']
        print(f"选择 {dim_names[split_dim]} 轴进行分割（最长维度）")
        
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
            print(f"Block {i}: {len(block)} 个点")
        
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
            print(f"\n处理第 {idx} 对文件:")
            print(f"原始文件: {file_A}")
            print(f"压缩文件: {file_B}")
            
            # 加载点云数据
            points_A = self.load_ply(file_A)
            points_B = self.load_ply(file_B)
            
            # 检查并移除重复点
            check_compress = has_duplicates(points_B)
            if check_compress:
                print("检测到压缩点云中存在重复点，正在移除...")
                points_B = remove_duplicates(points_B)
                print(f"移除重复点后的点数: {len(points_B)}")
            
            # 分割点云
            print("\n开始分割原始点云...")
            chunks_A = self.split_points_by_coordinate(points_A, self.num_blocks)
            print("\n开始分割压缩点云...")
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
        print(f"\n加载文件：{file_path}")
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        print(f"加载完成，点数量：{points.shape[0]}")
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
    pcd.colors = o3d.utility.Vector3dVector(feats / 255.0)  # 归一化颜色

    # 保存为 PLY 文件
    o3d.io.write_point_cloud(filename, pcd)
    print(f"保存成功: {filename}")



class MyNet(ME.MinkowskiNetwork):
    def __init__(self, in_channels=3, out_channels=3, D=3):
        ME.MinkowskiNetwork.__init__(self, D)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D
        )
        self.norm1 = ME.MinkowskiBatchNorm(out_channels)
        self.relu1 = ME.MinkowskiReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        return out

if __name__ == '__main__':
    dataset = PointCloudDataset(folder_A='./data/original',
                                folder_B='./data/compress',
                                num_blocks = 50
                                )
    data_loader = DataLoader(dataset, batch_size=1)


    def position_loss(pred, target):
        if isinstance(pred, ME.SparseTensor) and isinstance(target, ME.SparseTensor):
            # 使用稀疏张量的密集特征进行损失计算
            return torch.nn.functional.mse_loss(pred.F, target.F)
        else:
            # 假设 pred 和 target 都是普通张量
            return torch.nn.functional.mse_loss(pred, target)


    def train_model(model, data_loader, optimizer, device='cuda', epochs=10):
        model = model.to(device)
        model.train()

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for batch_idx, (chunks_original, chunks_compress) in enumerate(data_loader):
                # 遍历每一对块
                for block_idx in range(len(chunks_original[0])):
                    # 获取当前块的数据
                    coords_original = chunks_original[0][block_idx][:, :3]  # [N, 3] 坐标
                    feats_original = chunks_original[0][block_idx][:, 3:]   # [N, 3] 特征(RGB)
                    coords_compress = chunks_compress[0][block_idx][:, :3]  # [N, 3] 坐标
                    feats_compress = chunks_compress[0][block_idx][:, 3:]   # [N, 3] 特征(RGB)

                    print(f'\nProcessing block {block_idx}:')
                    print(f'Original points: {coords_original.shape[0]}, Compressed points: {coords_compress.shape[0]}')

                    # 转换为ME需要的格式
                    coords_original_tensor = ME_utils.batched_coordinates([coords_original.view(-1, 3)], device=device)
                    coords_compress_tensor = ME_utils.batched_coordinates([coords_compress.view(-1, 3)], device=device)
                    feats_original_tensor = feats_original.to(device).float()
                    feats_compress_tensor = feats_compress.to(device).float()
                 
                    assert feats_compress_tensor.shape[0] == coords_compress_tensor.shape[0], "Feature and coordinates row count must match!"
                    assert feats_original_tensor.shape[0] == coords_original_tensor.shape[0], "Feature and coordinates row count must match!"

                    # 构建稀疏张量
                    original_sparse_tensor = ME.SparseTensor(features=feats_original_tensor, coordinates=coords_original_tensor)
                    compress_sparse_tensor = ME.SparseTensor(features=feats_compress_tensor, coordinates=coords_compress_tensor)

                    print('Input shapes:')
                    print(f'Origin: {original_sparse_tensor.shape}')
                    print(f'Input x: {compress_sparse_tensor.shape}') #ME 会压缩让点数变少，因为VPCC 后的 点云有重复的。
                    
                    coords_compress_dedup = compress_sparse_tensor.C[:, 1:].cpu().numpy()  # 去掉批次维度并转为numpy
                    coords_original_numpy = coords_original.cpu().numpy() if torch.is_tensor(coords_original) else coords_original
             
                    coords_new_original = find_corresponding_original_points(coords_compress_dedup, coords_original_numpy)
                
                    coords_new_original = torch.from_numpy(coords_new_original).float()
                    coords_new_original_tensor = ME_utils.batched_coordinates([coords_new_original], device=device)
                    print("coords_new_original",coords_new_original)
                    print("coords_new_original_tensor",coords_new_original_tensor.shape)
                    
                    def generate_corresponding_features(coords_new_original, coords_original, feats_original):
                        # 确保所有输入都是 numpy 数组
                        if torch.is_tensor(coords_new_original):
                            coords_new_original = coords_new_original.cpu().numpy()
                        if torch.is_tensor(coords_original):
                            coords_original = coords_original.cpu().numpy()
                        if torch.is_tensor(feats_original):
                            feats_original = feats_original.cpu().numpy()

                        # 创建新的特征数组
                        feats_new = np.zeros((len(coords_new_original), feats_original.shape[1]))

                        # 遍历每个新坐标点
                        for i, target_point in enumerate(coords_new_original):
                            # 现在 target_point 和 search_points 都是 numpy array，可以正确计算距离
                            distances = np.linalg.norm(coords_original - target_point, axis=1)
                            nearest_idx = np.argmin(distances)
                            feats_new[i] = feats_original[nearest_idx]

                            if (i + 1) % 1000 == 0:
                                print(f"已处理: {i + 1}/{len(coords_new_original)} 点")

                        # 转换为tensor并移到指定设备
                        feats_new_original_tensor = torch.tensor(feats_new, device=device).float()

                        return feats_new_original_tensor


                    feats_new_original_tensor = generate_corresponding_features(
                        coords_new_original,
                        coords_original,
                        feats_original
                    )
                    print("feats_new_original_tensor",feats_new_original_tensor.shape) # [18645,3]
                    
                    new_original_sparse_tensor = ME.SparseTensor(features=feats_new_original_tensor, coordinates=coords_new_original_tensor)
                    
                    print("new_original_sparse_tensor",new_original_sparse_tensor.shape) # [18645,3]
                   
                    # 前向传播和优化
                    optimizer.zero_grad()
                    output = model(compress_sparse_tensor)
                    print("Shapes before residual calculation:")
                    print("output.F shape:", output.F.shape)
                    print("feats_compress_tensor shape:", feats_compress_tensor.shape)
                    print("compress_sparse_tensor.F shape:", compress_sparse_tensor.F.shape)
                    print("new_original_sparse_tensor.F shape:", new_original_sparse_tensor.F.shape)

                    # 计算损失
                    loss = position_loss(output.F.float(), new_original_sparse_tensor.F.float())
                    total_loss += loss.item()

                    # 反向传播
                    loss.backward()
                    optimizer.step()

                    num_batches += 1
                    print(f'Block {block_idx} Loss: {loss.item():.4f}')

            # 每个epoch结束后打印平均损失
            avg_loss = total_loss / num_batches
            print(f'\nEpoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')

            # 保存模型权重
            save_path = f'epoch_{epoch + 1}_model.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1} to {save_path}")

    model = MyNet()
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, data_loader, optimizer)
    
    def evaluate_and_save(model_path, dataset, output_dir='./output'):
        """
        加载模型，处理点云数据并保存结果

        参数:
        model_path: 训练好的模型路径（.pth文件）
        dataset: 数据集
        output_dir: 输出文件夹路径
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 加载模型
        model = MyNet()
        model.load_state_dict(torch.load(model_path))
        model = model.to('cuda')
        model.eval()

        # 创建数据加载器
        data_loader = DataLoader(dataset, batch_size=4)

        with torch.no_grad():
            for batch_idx, (chunks_original, chunks_compress) in enumerate(data_loader):
                print(f"\n处理第 {batch_idx + 1} 个点云")

                # 处理每个块
                for block_idx in range(len(chunks_original[0])):
                    print(f"\n处理块 {block_idx}")

                    # 保存原始点云
                    original_coords = chunks_original[0][block_idx][:, :3]
                    original_feats = chunks_original[0][block_idx][:, 3:]
                    save_point_cloud_as_ply(
                        original_coords, 
                        original_feats,
                        f"{output_dir}/original_batch_{batch_idx}_block_{block_idx}.ply"
                    )
                    print(f"已保存原始点云")

                    # 保存压缩点云
                    compress_coords = chunks_compress[0][block_idx][:, :3]
                    compress_feats = chunks_compress[0][block_idx][:, 3:]
                    save_point_cloud_as_ply(
                        compress_coords, 
                        compress_feats,
                        f"{output_dir}/compressed_batch_{batch_idx}_block_{block_idx}.ply"
                    )
                    print(f"已保存压缩点云")

                    # 处理压缩点云
                    coords_compress_tensor = ME_utils.batched_coordinates([compress_coords], device='cuda')
                    feats_compress_tensor = compress_feats.to('cuda').float()

                    # 创建稀疏张量
                    compress_sparse_tensor = ME.SparseTensor(
                        features=feats_compress_tensor,
                        coordinates=coords_compress_tensor
                    )

                    # 模型预测
                    output = model(compress_sparse_tensor)

                    # 获取预测结果
                    pred_coords = output.C[:, 1:].cpu().numpy()  # 去掉批次维度
                    pred_feats = output.F.cpu().numpy()

                    # 保存处理后的结果
                    save_point_cloud_as_ply(
                        pred_coords, 
                        pred_feats,
                        f"{output_dir}/result_batch_{batch_idx}_block_{block_idx}.ply"
                    )
                    print(f"已保存处理后的结果")

                    # 打印点数信息便于对比
                    print(f"原始点云点数: {len(original_coords)}")
                    print(f"压缩点云点数: {len(compress_coords)}")
                    print(f"处理后点数: {len(pred_coords)}")



    # 评估部分
#     print("\n开始评估...")
#     evaluate_and_save(
#         model_path='epoch_10_model.pth',  # 使用最后一个epoch的模型
#         dataset=dataset
#     )
#     print("评估完成！请查看 output 文件夹中的结果")
    
