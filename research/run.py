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
import numpy as np
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

class PointCloudDataset(Dataset):
    def __init__(self, folder_A, folder_B, block_size=256, cube_size=1024,min_points_ratio=0.1):
        """
        初始化数据集，并确保文件编号匹配。
        :param folder_A: 原始点云A的文件夹路径
        :param folder_B: 压缩后点云B的文件夹路径
        :param chunk_size: 每个块的大小
        :param stride: 切块的步长
        """
        self.files_A = self.match_files(folder_A, folder_B)
        self.block_size = block_size
        self.cube_size = cube_size
        self.min_points_ratio = min_points_ratio

        if not self.files_A:
            print("没有找到匹配的文件对，请检查文件名和路径是否正确！")
        else:
            print(f"共找到 {len(self.files_A)} 对文件。")

    def match_files(self, folder_A, folder_B):
        """根据编号匹配压缩前后的文件对。"""

        # 获取两个文件夹中的所有 .ply 文件
        files_A = sorted([f for f in os.listdir(folder_A) if f.endswith('.ply')])
        files_B = sorted([f for f in os.listdir(folder_B) if f.endswith('.ply')])

        # 正则表达式：匹配文件名末尾的 3-4 位数字编号
        def extract_id(filename):
            match = re.search(r'(\d{3,4})(?=\.ply$)', filename)
            return match.group(1) if match else None

        # 创建以编号为键的文件路径映射
        files_A_dict = {extract_id(f): os.path.join(folder_A, f) for f in files_A}
        files_B_dict = {extract_id(f): os.path.join(folder_B, f) for f in files_B}

        # 打印匹配的文件编号字典，供调试用
        print("files_A_dict:", files_A_dict)
        print("files_B_dict:", files_B_dict)

        # 匹配两组文件编号的交集
        matched_files = [
            (files_A_dict[id_], files_B_dict[id_])
            for id_ in files_A_dict.keys() & files_B_dict.keys()
        ]

        if not matched_files:
            print("没有找到匹配的文件对，请检查文件名编号是否一致。")
        return matched_files

    def __len__(self):
        return len(self.files_A)

    def __getitem__(self, idx):
        file_A, file_B = self.files_A[idx]
        points_A = self.load_ply(file_A)
        points_B = self.load_ply(file_B)
        check_compress = has_duplicates(points_B)
        if check_compress:
            points_B = remove_duplicates(points_B)
            # debug用
            # has_dup, duplicates = has_duplicates_output(points_B)
            # print(duplicates)
        chunks_A = chunk_point_cloud_fixed_size(points_A, self.block_size, self.cube_size, self.min_points_ratio)
        chunks_B = chunk_point_cloud_fixed_size(points_B, self.block_size, self.cube_size, self.min_points_ratio)
        adjusted_chunks_A = []
        adjusted_chunks_B = []


        for (chunk_A, index_A), (chunk_B, index_B) in zip(chunks_A, chunks_B):
            if index_A == index_B:
                print('未补齐之前，compress block是否有重复：',has_duplicates(chunk_B))
                adjusted_chunk_B = adjust_points(chunk_B, len(chunk_A))
                adjusted_chunks_A.append(chunk_A)
                adjusted_chunks_B.append(adjusted_chunk_B)
        if not adjusted_chunks_A or not adjusted_chunks_B:
            print(f"第 {idx} 对文件没有找到匹配的块，A: {len(adjusted_chunks_A)} 块, B: {len(adjusted_chunks_B)} 块.")
            return None
        print(f"第 {idx} 对文件切块完成，A: {len(adjusted_chunks_A)} 块, B: {len(adjusted_chunks_B)} 块. ",
              "其中一块的形状:", adjusted_chunks_A[0].shape)
        print('-------------------------------开始打印每块的点数------------------------')
        for i in range(len(adjusted_chunks_A)):
            print(f'第{i}块: ', adjusted_chunks_A[i].shape)

        return (adjusted_chunks_A, adjusted_chunks_B)

    def load_ply(self, file_path):
        """读取 PLY 文件并返回 (N, 6) 的点云数组"""
        print(f"正在加载文件：{file_path}")
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        print(f"加载点数量：{points.shape[0]}")
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



class ResNet(nn.Module):
    """
    Basic block: Residual
    """

    def __init__(self, channels):
        super(ResNet, self).__init__()
        # path_1
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv0(x))
        out = self.conv1(out)
        out += x

        return out


class MyInception_1(nn.Module):
    def __init__(self,
                 channels,
                 stride=1,
                 dilation=1,
                 bn_momentum=0.1,
                 dimension=3):
        super(MyInception_1, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            channels, channels // 4, kernel_size=1, stride=stride, dilation=dilation, bias=True, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(channels // 4, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            channels // 4, channels // 4, kernel_size=3, stride=stride, dilation=dilation, bias=True,
            dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(channels // 4, momentum=bn_momentum)
        self.conv3 = ME.MinkowskiConvolution(
            channels // 4, channels // 2, kernel_size=1, stride=stride, dilation=dilation, bias=True,
            dimension=dimension)
        self.norm3 = ME.MinkowskiBatchNorm(channels // 2, momentum=bn_momentum)

        self.conv4 = ME.MinkowskiConvolution(
            channels, channels // 4, kernel_size=3, stride=stride, dilation=dilation, bias=True, dimension=dimension)
        self.norm4 = ME.MinkowskiBatchNorm(channels // 4, momentum=bn_momentum)
        self.conv5 = ME.MinkowskiConvolution(
            channels // 4, channels // 2, kernel_size=3, stride=stride, dilation=dilation, bias=True,
            dimension=dimension)
        self.norm5 = ME.MinkowskiBatchNorm(channels // 2, momentum=bn_momentum)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        # 1
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.relu(out)

        # 2
        out1 = self.conv4(x)
        out1 = self.norm4(out1)
        out1 = self.relu(out1)

        out1 = self.conv5(out1)
        out1 = self.norm5(out1)
        out1 = self.relu(out1)

        # 3
        out2 = ME.cat(out, out1)
        out2 += x

        return out2


class Pyramid_1(nn.Module):
    def __init__(self,
                 channels,
                 bn_momentum=0.1,
                 dimension=3):
        super(Pyramid_1, self).__init__()
        assert dimension > 0

        self.aspp1 = ME.MinkowskiConvolution(
            channels, channels // 4, kernel_size=1, stride=1, dilation=1, bias=True, dimension=dimension)
        self.aspp2 = ME.MinkowskiConvolution(
            channels, channels // 4, kernel_size=3, stride=1, dilation=6, bias=True, dimension=dimension)
        self.aspp3 = ME.MinkowskiConvolution(
            channels, channels // 4, kernel_size=3, stride=1, dilation=12, bias=True, dimension=dimension)
        self.aspp4 = ME.MinkowskiConvolution(
            channels, channels // 4, kernel_size=3, stride=1, dilation=18, bias=True, dimension=dimension)
        self.aspp5 = ME.MinkowskiConvolution(
            channels, channels // 4, kernel_size=1, stride=1, dilation=1, bias=True, dimension=dimension)

        self.aspp1_bn = ME.MinkowskiBatchNorm(channels // 4, momentum=bn_momentum)
        self.aspp2_bn = ME.MinkowskiBatchNorm(channels // 4, momentum=bn_momentum)
        self.aspp3_bn = ME.MinkowskiBatchNorm(channels // 4, momentum=bn_momentum)
        self.aspp4_bn = ME.MinkowskiBatchNorm(channels // 4, momentum=bn_momentum)
        self.aspp5_bn = ME.MinkowskiBatchNorm(channels // 4, momentum=bn_momentum)

        self.conv2 = ME.MinkowskiConvolution(
            channels // 4 * 5, channels, kernel_size=1, stride=1, dilation=1, bias=True, dimension=dimension)
        self.bn2 = ME.MinkowskiBatchNorm(channels, momentum=bn_momentum)

        self.pooling = ME.MinkowskiGlobalPooling()
        self.broadcast = ME.MinkowskiBroadcast()
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)

        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)

        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)

        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)

        x5 = self.pooling(x)
        x5 = self.broadcast(x, x5)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)

        x6 = ME.cat(x1, x2, x3, x4, x5)
        x6 = self.conv2(x6)
        x6 = self.bn2(x6)
        x6 = self.relu(x6)

        x7 = x6 + x

        return x7


class MyNet(ME.MinkowskiNetwork):
    CHANNELS = [None, 32, 32, 64, 128, 256]
    TR_CHANNELS = [None, 32, 32, 64, 128, 256]
    BLOCK_1 = MyInception_1
    BLOCK_2 = Pyramid_1

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 bn_momentum=0.1,
                 last_kernel_size=5,
                 D=3):
        ME.MinkowskiNetwork.__init__(self, D)
        CHANNELS = self.CHANNELS
        TR_CHANNELS = self.TR_CHANNELS
        BLOCK_1 = self.BLOCK_1
        BLOCK_2 = self.BLOCK_2

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=CHANNELS[1],
            kernel_size=5,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm1 = ME.MinkowskiBatchNorm(CHANNELS[1], momentum=bn_momentum)
        self.block1 = self.make_layer(BLOCK_1, BLOCK_2, CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm2 = ME.MinkowskiBatchNorm(CHANNELS[2], momentum=bn_momentum)
        self.block2 = self.make_layer(BLOCK_1, BLOCK_2, CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm3 = ME.MinkowskiBatchNorm(CHANNELS[3], momentum=bn_momentum)
        self.block3 = self.make_layer(BLOCK_1, BLOCK_2, CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.conv4 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[3],
            out_channels=CHANNELS[4],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm4 = ME.MinkowskiBatchNorm(CHANNELS[4], momentum=bn_momentum)
        self.block4 = self.make_layer(BLOCK_1, BLOCK_2, CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.conv5 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[4],
            out_channels=CHANNELS[5],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm5 = ME.MinkowskiBatchNorm(CHANNELS[5], momentum=bn_momentum)
        self.block5 = self.make_layer(BLOCK_1, BLOCK_2, CHANNELS[5], bn_momentum=bn_momentum, D=D)

        self.conv5_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[5],
            out_channels=TR_CHANNELS[5],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm5_tr = ME.MinkowskiBatchNorm(TR_CHANNELS[5], momentum=bn_momentum)
        self.block5_tr = self.make_layer(BLOCK_1, BLOCK_2, TR_CHANNELS[5], bn_momentum=bn_momentum, D=D)

        self.conv4_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[4] + TR_CHANNELS[5],
            out_channels=TR_CHANNELS[4],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm4_tr = ME.MinkowskiBatchNorm(TR_CHANNELS[4], momentum=bn_momentum)
        self.block4_tr = self.make_layer(BLOCK_1, BLOCK_2, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.conv3_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[3] + TR_CHANNELS[4],
            out_channels=TR_CHANNELS[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm3_tr = ME.MinkowskiBatchNorm(TR_CHANNELS[3], momentum=bn_momentum)
        self.block3_tr = self.make_layer(BLOCK_1, BLOCK_2, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.conv2_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[2] + TR_CHANNELS[3],
            out_channels=TR_CHANNELS[2],
            kernel_size=last_kernel_size,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm2_tr = ME.MinkowskiBatchNorm(TR_CHANNELS[2], momentum=bn_momentum)
        self.block2_tr = self.make_layer(BLOCK_1, BLOCK_2, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.conv1_tr = ME.MinkowskiConvolution(
            in_channels=TR_CHANNELS[2],
            out_channels=TR_CHANNELS[1],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        # self.norm1_tr = ME.MinkowskiBatchNorm(TR_CHANNELS[1], momentum=bn_momentum)
        # self.block1_tr = self.make_layer(BLOCK_1, BLOCK_2, TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.final = ME.MinkowskiConvolution(
            in_channels=TR_CHANNELS[1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=D)

        self.pruning = ME.MinkowskiPruning()

    def make_layer(self, block_1, block_2, channels, bn_momentum, D):
        layers = []
        layers.append(block_1(channels=channels, bn_momentum=bn_momentum))
        layers.append(block_2(channels=channels, bn_momentum=bn_momentum))
        layers.append(block_1(channels=channels, bn_momentum=bn_momentum))

        return nn.Sequential(*layers)

    def forward(self, x):
        out_s1 = self.conv1(x)
        out_s1 = self.norm1(out_s1)
        out_s1 = self.block1(out_s1)
        out = MEF.relu(out_s1)

        out_s2 = self.conv2(out)
        out_s2 = self.norm2(out_s2)
        out_s2 = self.block2(out_s2)
        out = MEF.relu(out_s2)

        out_s4 = self.conv3(out)
        out_s4 = self.norm3(out_s4)
        out_s4 = self.block3(out_s4)
        out = MEF.relu(out_s4)

        out_s8 = self.conv4(out)
        out_s8 = self.norm4(out_s8)
        out_s8 = self.block4(out_s8)
        out = MEF.relu(out_s8)

        out_s16 = self.conv5(out)
        out_s16 = self.norm5(out_s16)
        out_s16 = self.block5(out_s16)
        out = MEF.relu(out_s16)

        out = self.conv5_tr(out)
        out = self.norm5_tr(out)
        out = self.block5_tr(out)
        out_s8_tr = MEF.relu(out)

        out = ME.cat(out_s8_tr, out_s8)

        out = self.conv4_tr(out)
        out = self.norm4_tr(out)
        out = self.block4_tr(out)
        out_s4_tr = MEF.relu(out)

        out = ME.cat(out_s4_tr, out_s4)

        out = self.conv3_tr(out)
        out = self.norm3_tr(out)
        out = self.block3_tr(out)
        out_s2_tr = MEF.relu(out)

        out = ME.cat(out_s2_tr, out_s2)

        out = self.conv2_tr(out)
        out = self.norm2_tr(out)
        out = self.block2_tr(out)
        out_s1_tr = MEF.relu(out)

        out = out_s1_tr + out_s1
        out = self.conv1_tr(out)
        out = MEF.relu(out)

        out_cls = self.final(out)

        return out_cls


class MyTestNet(ME.MinkowskiNetwork):
    def __init__(self, in_channels=3, out_channels=32, D=3):
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

def pad_sparse_tensor(x, origin):
    """
    给稀疏张量 x 添加数据，使其形状与 origin 保持一致。

    参数:
    x (ME.SparseTensor): 需要填补的稀疏张量。
    origin (ME.SparseTensor): 参考的稀疏张量，其形状为目标形状。

    返回:
    ME.SparseTensor: 填补后的稀疏张量 x。
    """
    # 获取需要填补的数量
    num_to_add = origin.shape[0] - x.shape[0]

    if num_to_add <= 0:
        return x  # 如果 x 的形状已经大于或等于 origin，则不需要填补

    # 获取 x 的设备
    device = x.F.device

    # 生成随机特征
    random_feats = torch.randn(num_to_add, x.F.shape[1], device=device)

    # 生成唯一的随机坐标
    existing_coords = set(tuple(coord.tolist()) for coord in x.C)
    random_coords = []
    while len(random_coords) < num_to_add:
        coord = torch.randint(0, 100, (x.C.shape[1],), device=device, dtype=torch.float32).tolist()
        if tuple(coord) not in existing_coords:
            random_coords.append(coord)
            existing_coords.add(tuple(coord))

    random_coords = torch.tensor(random_coords, device=device, dtype=torch.float32).int()

    # 将原始数据和随机数据拼接
    new_feats = torch.cat([x.F, random_feats], dim=0)
    new_coords = torch.cat([x.C, random_coords], dim=0)

    # 创建新的稀疏张量
    new_x = ME.SparseTensor(features=new_feats, coordinates=new_coords, coordinate_manager=x.coordinate_manager)

    # 调试信息
    print(f'Original shape: {x.shape}')
    print(f'Origin shape: {origin.shape}')
    print(f'New shape: {new_x.shape}')
    print(f'Number of features added: {num_to_add}')

    return new_x

if __name__ == '__main__':
    dataset = PointCloudDataset(folder_A='./data/original',
                                folder_B='./data/compress',
                                min_points_ratio=0.0001
                                )

    data_loader = DataLoader(dataset, batch_size=1)


    # -----------------------------run train test -------------------------------
    # -----------------------------要去服务器执行，本地Windows没装ME -------------------------------
    def position_loss(pred, target):
        """计算位置增强任务的 MSE 损失"""
        return torch.nn.functional.mse_loss(pred.F, target.F)


    def train_model(model, data_loader, optimizer, device='cuda', epochs=10, blocks_per_epoch=2):
        model = model.to(device)
        model.train()

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            block_buffer_A, block_buffer_B = [], []

            for (chunks_A, chunks_B) in data_loader:
                block_buffer_A.extend(chunks_A)
                block_buffer_B.extend(chunks_B)

                if len(block_buffer_A) >= blocks_per_epoch:
                    # 每N个块进行处理
                    for i in range(0, len(block_buffer_A), blocks_per_epoch):
                        print(block_buffer_A[i].shape,block_buffer_B[i].shape)
                        coords_A1, feats_A1 = block_buffer_A[i][0, :, :3], block_buffer_A[i][0, :, 3:]
                        coords_A2, feats_A2 = block_buffer_A[i + 1][0, :, :3], block_buffer_A[i + 1][0, :, 3:]
                        coords_B1, feats_B1 = block_buffer_B[i][0, :, :3], block_buffer_B[i][0, :, 3:]
                        coords_B2, feats_B2 = block_buffer_B[i + 1][0, :, :3], block_buffer_B[i + 1][0, :, 3:]
                        # ([1, 13824, 3])
                        print('coords_A1_shape:',coords_A1.shape, coords_A2.shape, coords_B1.shape, coords_B2.shape)

                        coords_A_tensor = ME_utils.batched_coordinates(
                            [coords_A1.view(-1, 3), coords_A2.view(-1, 3)], device=device
                        )
                        feats_A_tensor = torch.cat([feats_A1, feats_A2]).to(device)

                        coords_B_tensor = ME_utils.batched_coordinates(
                            [coords_B1.view(-1, 3), coords_B2.view(-1, 3)], device=device
                        )
                        feats_B_tensor = torch.cat([feats_B1, feats_B2]).to(device)

                        print('coords_AB_tensor_shape:', coords_A_tensor.shape, coords_B_tensor.shape)
                        print('feats_AB_tensor_shape:', feats_A_tensor.shape, feats_B_tensor.shape)

                        # 确保行数一致
                        assert feats_B_tensor.shape[0] == coords_B_tensor.shape[
                            0], "Feature and coordinates row count must match!"

                        # 构造SparseTensor
                        print(f"MODEL INPUT CHECK feats_B_tensor shape: {feats_B_tensor.shape}")
                        print(f"coords_B_tensor shape: {coords_B_tensor.shape}")
                        print(has_duplicates(feats_B_tensor))
                        print(has_duplicates(coords_B_tensor))



                        # 确保行数一致
                        assert feats_A_tensor.shape[0] == coords_A_tensor.shape[
                            0], "Feature and coordinates row count must match!"

                        origin = ME.SparseTensor(features=feats_A_tensor, coordinates=coords_A_tensor)
                        print('model_origin_shape:',origin.shape)


                        # 压缩后的数据重复导致x.shape减少
                        x = ME.SparseTensor(features=feats_B_tensor, coordinates=coords_B_tensor)
                        print('model_input_x_shape:',x.shape,type(x))
                        new_x = pad_sparse_tensor(x,origin)
                        print('after pad model_input_x_shape:',new_x.shape,type(new_x))

                        optimizer.zero_grad()
                        output = model(new_x)

                        # 计算残差
                        residual = output.F - feats_B_tensor
                        combined_output = output.F + residual

                        # 计算损失
                        loss = position_loss(combined_output, origin)
                        total_loss += loss.item()

                        loss.backward()
                        optimizer.step()

                    avg_loss = total_loss / (blocks_per_epoch // 2)
                    print(f"Epoch {epoch + 1}/{epochs}, Batch {num_batches + 1}, Average Loss: {avg_loss:.4f}")

                    # 清空缓冲区
                    block_buffer_A.clear()
                    block_buffer_B.clear()
                    total_loss = 0
                    num_batches += 1


    # 初始化模型和优化器
    model = MyTestNet()
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, data_loader, optimizer)

    # ----------------------------display ply-1 ---------------------------------


    # # 批量处理合并后的大块数据
    # for (coords_A, feats_A), (coords_B, feats_B) in data_loader:
    #     print(f"A 文件点数量: {len(coords_A)}, B 文件点数量: {len(coords_B)}")
    #     print(coords_A[0].shape)
    #     # 在每个批次中处理所有块
    #     for i in range(len(coords_A)):
    #         # 这里处理每个块而不是每四个
    #         coords = coords_A[i].numpy().astype(np.float64)[:, :3]  # 第四列是batch
    #         feats = feats_A[i].numpy().astype(np.float64)
    #
    #         print(f"coords shape: {coords.shape}, feats shape: {feats.shape}")
    #
    #         # 显示 A 或 B 的某个块
    #         display_point_cloud(coords, feats)

    # ----------------------------display block-1 ---------------------------------

    # # 想看每个小块用下面的
    # for idx, (chunks_A, chunks_B) in enumerate(data_loader):
    #     # 遍历每对文件的所有块
    #     for i, (coords_A, feats_A) in enumerate(chunks_A):
    #         print(f"A 文件的第 {i} 个块点数量: {len(coords_A)}")
    #         display_point_cloud(coords_A, feats_A)
    #         break

    # ----------------------------save---------------------------------
    # 创建保存文件夹
    # os.makedirs("output_chunks", exist_ok=True)
    #
    # # 遍历每对数据，并将每个小块保存为单独的 PLY 文件
    # for idx, (chunks_A, _) in enumerate(data_loader):
    #     # chunks_A 是一个包含多个块的元组
    #     print(f"第 {idx} 对文件的块信息类型: {type(chunks_A)}, 内容: {chunks_A}")
    #
    #     # 遍历每个小块
    #     for chunk_idx in range(chunks_A[0].size(0)):  # 获取块的数量
    #         coords = chunks_A[0][chunk_idx]  # 获取第 chunk_idx 个块的坐标
    #         feats = chunks_A[1][chunk_idx]  # 获取第 chunk_idx 个块的特征
    #
    #         coords_np = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else coords
    #         feats_np = feats.cpu().numpy() if isinstance(feats, torch.Tensor) else feats
    #
    #         # 检查每个块的形状
    #         print(f"块 {chunk_idx} 坐标形状: {coords_np.shape}, 特征形状: {feats_np.shape}")
    #
    #         # 提取前三列作为坐标，提取后面三列作为特征
    #         coords_np = coords_np[:, :3]  # 只取坐标
    #         feats_np = feats_np[:, :3]  # 只取特征
    #
    #         # 保存每一块到单独的 PLY 文件
    #         filename = f"output_chunks/chunk_{idx}_{chunk_idx}.ply"
    #         save_point_cloud_as_ply(coords_np, feats_np, filename)
    #
    #         print(f"保存文件: {filename}，坐标数量: {len(coords_np)}")
    #
