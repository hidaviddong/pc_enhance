import numpy as np
import open3d as o3d
import os

def load_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)

def save_ply(points, file_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(file_path, pcd)

def split_points_by_coordinate(points, num_blocks=10):
    # 获取点云的边界框
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    # 选择最长的维度进行分割
    ranges = max_coords - min_coords
    split_dim = np.argmax(ranges)
    
    # 沿着最长维度排序
    sorted_indices = np.argsort(points[:, split_dim])
    sorted_points = points[sorted_indices]
    
    # 分割点云
    points_per_block = len(points) // num_blocks
    blocks = []
    for i in range(num_blocks):
        start_idx = i * points_per_block
        end_idx = start_idx + points_per_block if i < num_blocks-1 else len(points)
        block = sorted_points[start_idx:end_idx]
        blocks.append(block)
    
    return blocks

def main(origin_dir, compress_dir, output_dir, num_blocks=10):
    os.makedirs(output_dir, exist_ok=True)
    
    origin_files = sorted([f for f in os.listdir(origin_dir) if f.endswith('.ply')])
    compress_files = sorted([f for f in os.listdir(compress_dir) if f.endswith('.ply')])

    for origin_file, compress_file in zip(origin_files, compress_files):
        origin_points = load_ply(os.path.join(origin_dir, origin_file))
        compress_points = load_ply(os.path.join(compress_dir, compress_file))

        # 分别对原始点云和压缩点云进行空间分割
        origin_blocks = split_points_by_coordinate(origin_points, num_blocks)
        compress_blocks = split_points_by_coordinate(compress_points, num_blocks)

        for i, (origin_block, compress_block) in enumerate(zip(origin_blocks, compress_blocks)):
            origin_output_path = os.path.join(output_dir, f'origin_{origin_file[:-4]}_block{i}.ply')
            compress_output_path = os.path.join(output_dir, f'compress_{compress_file[:-4]}_block{i}.ply')
            save_ply(origin_block, origin_output_path)
            save_ply(compress_block, compress_output_path)

if __name__ == "__main__":
    origin_dir = 'data/original'
    compress_dir = 'data/compress'
    output_dir = 'data/output'
    main(origin_dir, compress_dir, output_dir, num_blocks=10)
