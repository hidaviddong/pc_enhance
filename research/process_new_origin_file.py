import os
import logging
import numpy as np
import open3d as o3d
import subprocess
import re
from logger import logger

def get_sequence_number(filename):
    """
    从文件名中提取四位数字序号
    """
    match = re.search(r'_(\d{4})\.ply$', filename)
    if match:
        return match.group(1)
    return None

def process_folder(origin_folder, compress_folder):
    """
    处理文件夹中的所有点云文件
    :param origin_folder: 原始点云文件夹路径
    :param compress_folder: 压缩点云文件夹路径
    """
    # 获取文件夹中的所有PLY文件
    origin_files = {get_sequence_number(f): f for f in os.listdir(origin_folder) 
                   if f.endswith('.ply') and get_sequence_number(f)}
    compress_files = {get_sequence_number(f): f for f in os.listdir(compress_folder) 
                     if f.endswith('.ply') and get_sequence_number(f)}

    # 找到共同的序号
    common_sequences = set(origin_files.keys()) & set(compress_files.keys())

    logger.info(f"找到 {len(common_sequences)} 对匹配的文件:")
    for seq in common_sequences:
        logger.info(f"\n序号 {seq}:")
        logger.info(f"原始文件: {origin_files[seq]}")
        logger.info(f"压缩文件: {compress_files[seq]}")

    # 处理每对匹配的文件
    for seq in common_sequences:
        origin_path = os.path.join(origin_folder, origin_files[seq])
        compress_path = os.path.join(compress_folder, compress_files[seq])
        
        logger.info(f"\n开始处理序号 {seq} 的文件对...")
        load_and_match_ply(origin_path, compress_path)

def load_and_match_ply(origin_path, compress_path):
    """
    加载PLY文件并进行点云匹配
    :param origin_path: 原始点云文件路径
    :param compress_path: 压缩点云文件路径
    """
    logger.info(f"\n加载文件：\n原始点云：{origin_path}\n压缩点云：{compress_path}")

    # 加载点云
    pcd_origin = o3d.io.read_point_cloud(origin_path)
    pcd_compress = o3d.io.read_point_cloud(compress_path)

    logger.info(f"原始点云点数：{len(pcd_origin.points)}")
    logger.info(f"压缩点云点数：{len(pcd_compress.points)}")
    

    # 转换为numpy数组
    points_origin = np.asarray(pcd_origin.points)
    points_compress = np.asarray(pcd_compress.points)
    
    unique_points_compress = np.unique(points_compress, axis=0)
    pcd_compress = o3d.cuda.pybind.geometry.PointCloud()
    pcd_compress.points = o3d.utility.Vector3dVector(unique_points_compress)
    logger.info(f"去重后压缩点云点数：{len(pcd_compress.points)}")
    

    logger.info("构建 kd-tree...")
    kdtree = o3d.geometry.KDTreeFlann(pcd_origin)
    logger.info("kd-tree 构建完成")

    # 初始化结果数组
    points_result = np.zeros_like(pcd_compress.points)
    matched_points = set()

    logger.info("开始点云匹配...")
    progress_interval = max(1, len(pcd_compress.points) // 20)

    for i, point in enumerate(pcd_compress.points):
        if (i + 1) % progress_interval == 0:
            progress = (i + 1) / len(pcd_compress.points) * 100
            logger.info(f"匹配进度: {progress:.1f}% ({i+1}/{len(pcd_compress.points)})")

        k = 1
        while True:
            _, idx, _ = kdtree.search_knn_vector_3d(point, k)
            
            if isinstance(idx, int):  # 处理单个索引的情况
                idx = [idx]
                
            for j in idx:
                if j not in matched_points:
                    matched_points.add(j)
                    points_result[i] = points_origin[j]
                    break
            else:
                k *= 2
                continue
            break

    logger.info("点云匹配完成")
    logger.info(f"最终匹配点数：{len(matched_points)}")

    # 获取输出目录
    
    # 保存结果点云 new origin
    result_pcd = o3d.geometry.PointCloud()
    result_pcd.points = o3d.utility.Vector3dVector(points_result)
    result_filename = f"new_origin_{os.path.basename(origin_path)}"
    result_path = os.path.join("./real_train/new_origin", result_filename)
    o3d.io.write_point_cloud(result_path, result_pcd, write_ascii=True)
    
    # 保存压缩点云 compress
    compress_pcd = o3d.geometry.PointCloud()
    compress_pcd.points = o3d.utility.Vector3dVector(pcd_compress.points)
    compress_filename = f"compress_{os.path.basename(origin_path)}"
    compress_path_new = os.path.join("./real_train/compress", compress_filename)
    o3d.io.write_point_cloud(compress_path_new, compress_pcd, write_ascii=True)
    
    # 保存原始点云 origin
    origin_pcd = o3d.geometry.PointCloud()
    origin_pcd.points = o3d.utility.Vector3dVector(points_origin)
    origin_filename = f"origin_{os.path.basename(origin_path)}"
    origin_path_new = os.path.join("./real_train/origin", origin_filename)
    o3d.io.write_point_cloud(origin_path_new, origin_pcd, write_ascii=True)
    
    
    #计算并保存PSNR结果
    calculate_and_save_psnr(origin_path_new, compress_path_new, result_path)
    
    logger.info(f"所有结果已保存")

def calculate_and_save_psnr(original_file, compressed_file, result_file):
    """计算并保存PSNR结果"""
    # 创建保存PSNR结果的目录
    psnr_dir = "./real_train/psnr"
    os.makedirs(psnr_dir, exist_ok=True)  # 添加这行来创建目录
    
    # 计算压缩点云的PSNR
    sequence_number = get_sequence_number(os.path.basename(original_file))
    
    cmd1 = f"../../mpeg-pcc-tmc2/bin/PccAppMetrics --uncompressedDataPath={original_file} --reconstructedDataPath={compressed_file} --resolution=1023 --frameCount=1"
    process1 = subprocess.Popen(cmd1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output1, _ = process1.communicate()

    psnr_file1 = os.path.join(psnr_dir, f"psnr_compressed_{sequence_number}.txt")
    with open(psnr_file1, 'w') as f:
        f.write(output1.decode())
    logger.info(f"压缩点云PSNR结果已保存到：{psnr_file1}")

    # 计算预测结果的PSNR
    cmd2 = f"../../mpeg-pcc-tmc2/bin/PccAppMetrics --uncompressedDataPath={original_file} --reconstructedDataPath={result_file} --resolution=1023 --frameCount=1"
    process2 = subprocess.Popen(cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output2, _ = process2.communicate()

    psnr_file2 = os.path.join(psnr_dir, f"psnr_new_origin_{sequence_number}.txt")
    with open(psnr_file2, 'w') as f:
        f.write(output2.decode())
    logger.info(f"处理结果PSNR已保存到：{psnr_file2}")


if __name__ == "__main__":
    # 设置文件夹路径
    origin_folder = "./train/original"
    compress_folder = "./train/compress"
    
    # 处理train文件夹中的所有匹配文件
    process_folder(origin_folder, compress_folder)
    
    origin_folder = "./test/original"
    compress_folder = "./test/compress"
    
    # 处理test文件夹中的所有匹配文件
    process_folder(origin_folder, compress_folder)
    
    
    
     
