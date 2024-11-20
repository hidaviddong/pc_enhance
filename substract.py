import numpy as np
import open3d as o3d
from logger import logger

def calculate_point_cloud_difference(cloud1_path, cloud2_path, output_path=None):
    """
    计算两个点云文件的坐标差异
    
    Args:
        cloud1_path: 第一个点云文件路径
        cloud2_path: 第二个点云文件路径
        output_path: 差异点云保存路径（可选）
    
    Returns:
        diff_coords: 坐标差异的numpy数组 (N,3)
    """
    # 加载点云
    pcd1 = o3d.io.read_point_cloud(cloud1_path)
    pcd2 = o3d.io.read_point_cloud(cloud2_path)
    
    # 转换为numpy数组
    coords1 = np.asarray(pcd1.points)  # (N,3)
    coords2 = np.asarray(pcd2.points)  # (N,3)
    
    # 检查点数是否相同
    if len(coords1) != len(coords2):
        raise ValueError(f"点云点数不匹配: {len(coords1)} vs {len(coords2)}")
    
    # 计算差异
    diff_coords = coords1 - coords2
    
    # 如果指定了输出路径，保存差异点云
    if output_path:
        diff_pcd = o3d.geometry.PointCloud()
        diff_pcd.points = o3d.utility.Vector3dVector(diff_coords)
        o3d.io.write_point_cloud(output_path, diff_pcd, write_ascii=True)
        logger.info(f"差异点云已保存到: {output_path}")
    
    return diff_coords

diff_coords = calculate_point_cloud_difference(
    cloud1_path="./blocks_visualization/file_0/compress_block_0.ply",  # 第一个点云文件
    cloud2_path="./blocks_visualization/file_0/origin_block_0.ply",     # 第二个点云文件
    output_path="./blocks_visualization/file_0/difference_block_0.ply"          # 保存差异的文件
)

