import os
import numpy as np
import open3d as o3d
import re
import torch
import MinkowskiEngine as ME
from MinkowskiEngine import utils as ME_utils
from pathlib import Path
from logger import logger

class PointCloudProcessor:
    def __init__(self, compress_dir, new_origin_dir, num_blocks, device='cuda', save_path='preprocessed_blocks.pth'):
        """
        初始化点云处理器
        
        Args:
            compress_dir: compress文件夹路径
            new_origin_dir: new_origin文件夹路径
            num_blocks: 需要分割的块数
            device: 使用的设备 ('cuda' 或 'cpu')
            save_path: 预处理数据保存路径
        """
        self.compress_dir = compress_dir
        self.new_origin_dir = new_origin_dir
        self.num_blocks = num_blocks
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.save_path = save_path
        
        logger.info(f"初始化处理器:")
        logger.info(f"- 压缩点云目录: {compress_dir}")
        logger.info(f"- 原始点云目录: {new_origin_dir}")
        logger.info(f"- 分块数量: {num_blocks}")
        logger.info(f"- 使用设备: {self.device}")
        logger.info(f"- 保存路径: {save_path}")
        
    def match_files(self):
        """匹配compress和new_origin文件夹中的对应文件"""
        compress_files = sorted([f for f in os.listdir(self.compress_dir) if f.endswith('.ply')])
        new_origin_files = sorted([f for f in os.listdir(self.new_origin_dir) if f.endswith('.ply')])
        
        matched_pairs = []
        for comp_file in compress_files:
            comp_id = re.search(r'_(\d+)\.ply$', comp_file)
            if comp_id:
                file_index = comp_id.group(1)
                for orig_file in new_origin_files:
                    if f"_{file_index}.ply" in orig_file:
                        matched_pairs.append((
                            os.path.join(self.compress_dir, comp_file),
                            os.path.join(self.new_origin_dir, orig_file),
                            file_index
                        ))
                        break
        
        logger.info(f"找到 {len(matched_pairs)} 对匹配文件")
        return matched_pairs
    
    def load_and_check_ply(self, compress_path, new_origin_path):
        """
        加载PLY文件并进行检查
        
        Returns:
            tuple: (compress_points, new_origin_points) 如果检查通过
            None: 如果检查不通过
        """
        try:
            # 加载点云
            compress_pcd = o3d.io.read_point_cloud(compress_path)
            new_origin_pcd = o3d.io.read_point_cloud(new_origin_path)
            
            # 转换为numpy数组
            compress_points = np.asarray(compress_pcd.points)
            new_origin_points = np.asarray(new_origin_pcd.points)
            
            # 检查点数是否相等
            if len(compress_points) != len(new_origin_points):
                logger.warning(f"点数不相等: compress={len(compress_points)}, new_origin={len(new_origin_points)}")
                return None
            
            # 检查是否有重复点
            compress_unique = np.unique(compress_points, axis=0)
            new_origin_unique = np.unique(new_origin_points, axis=0)
            
            if len(compress_unique) != len(compress_points) or len(new_origin_unique) != len(new_origin_points):
                logger.warning("检测到重复点")
                return None
                
            return compress_points, new_origin_points
            
        except Exception as e:
            logger.error(f"加载文件时发生错误: {str(e)}")
            return None

           
    def split_points(self, points):
        """
        保持点的原始对应关系进行分块
        如果最后一个块点数不足，则丢弃

        Args:
            points: (N, 3) 点云数据

        Returns:
            list: 分割后的点云块列表
        """
        # 计算每块的点数
        points_per_block = len(points) // self.num_blocks

        # 分块
        blocks = []
        for i in range(self.num_blocks):
            start_idx = i * points_per_block
            end_idx = start_idx + points_per_block

            # 如果是最后一个块且点数不足，就跳过
            if end_idx > len(points):
                break

            block = points[start_idx:end_idx]
            blocks.append(block)

        return blocks

    
    def process_point_clouds(self, compress_points, new_origin_points, file_index):
        """处理点云数据并转换为所需格式"""
        try:
            # 分割点云
            compress_blocks = self.split_points(compress_points)
            new_origin_blocks = self.split_points(new_origin_points)
            
            sample_data = []
            
            for block_idx, (compress_block, new_origin_block) in enumerate(zip(compress_blocks, new_origin_blocks)):
                logger.info(f"处理块 {block_idx + 1}/{self.num_blocks}")
                
                # 转换为torch tensors
                compress_tensor = torch.from_numpy(compress_block).float()
                new_origin_tensor = torch.from_numpy(new_origin_block).float()
                
                # 转换为ME需要的格式
                compress_coords = ME_utils.batched_coordinates([compress_tensor], device=self.device)
                new_origin_coords = ME_utils.batched_coordinates([new_origin_tensor], device=self.device)
                
                # 创建特征张量（使用坐标作为特征）
                compress_feats = compress_tensor.to(self.device)
                new_origin_feats = new_origin_tensor.to(self.device)
                
                # 创建SparseTensor
                compress_sparse = ME.SparseTensor(
                    features=compress_feats,
                    coordinates=compress_coords,
                    device=self.device
                )
                
                new_origin_sparse = ME.SparseTensor(
                    features=new_origin_feats,
                    coordinates=new_origin_coords,
                    device=self.device
                )
                
                # 获取唯一的坐标和特征
                unique_compress_coords = compress_sparse.C
                unique_compress_feats = compress_sparse.F
                unique_new_origin_coords = new_origin_sparse.C
                unique_new_origin_feats = new_origin_sparse.F
                
                # 存储块数据
                block_data = {
                    'compress_coords': unique_compress_coords,
                    'compress_feats': unique_compress_feats,
                    'new_origin_coords': unique_new_origin_coords,
                    'new_origin_feats': unique_new_origin_feats,
                    'original_compress_coords': compress_tensor,  # 添加：保存compress原始坐标
                    'original_new_origin_coords': new_origin_tensor,  # 添加：保存new_origin原始坐标
                    'block_index': block_idx,
                    'file_index': file_index,
                    'total_blocks': self.num_blocks,
                    'points_in_block': len(compress_block)
                }
               
                sample_data.append(block_data)
                
                logger.info(f"Block {block_idx} shapes:")
                logger.info(f"Compress coords shape: {unique_compress_coords.shape}")
                logger.info(f"Compress feats shape: {unique_compress_feats.shape}")
                logger.info(f"New origin coords shape: {unique_new_origin_coords.shape}")
                logger.info(f"New origin feats shape: {unique_new_origin_feats.shape}")
                logger.info(f"File index: {file_index}")
            
            return sample_data
            
        except Exception as e:
            logger.error(f"处理点云时发生错误: {str(e)}")
            return None

    def process_and_save(self, force_preprocess=False):
            """处理并保存点云数据"""
     
            # 获取匹配的文件对
            matched_pairs = self.match_files()
            preprocessed_data = []

            for pair_idx, (compress_path, new_origin_path, file_index) in enumerate(matched_pairs):
                logger.info(f"处理文件对 {pair_idx + 1}/{len(matched_pairs)}, 文件索引: {file_index}")

                # 加载并检查点云
                result = self.load_and_check_ply(compress_path, new_origin_path)
                if result is None:
                    logger.warning("检查未通过，跳过该文件对")
                    continue

                compress_points, new_origin_points = result

                # 分割点云并转换为torch tensor
                sample_data = self.process_point_clouds(compress_points, new_origin_points, file_index)
                if sample_data:
                    preprocessed_data.append(sample_data)

            # 保存预处理数据
            logger.info(f"保存预处理数据到: {self.save_path}")
            try:
                torch.save(preprocessed_data, self.save_path)
                logger.info("预处理数据保存成功")
            except Exception as e:
                logger.error(f"保存预处理数据失败: {str(e)}")

            return preprocessed_data



def main():
    # 设置参数
    compress_dir = 'real_train/compress'
    new_origin_dir = 'real_train/new_origin'
    num_blocks = 150
    device = 'cuda'
    save_path = 'preprocessed_blocks.pth'
    
    # 创建处理器实例
    processor = PointCloudProcessor(
        compress_dir=compress_dir,
        new_origin_dir=new_origin_dir,
        num_blocks=num_blocks,
        device=device,
        save_path=save_path
    )

    # 执行处理并保存
    preprocessed_data = processor.process_and_save(force_preprocess=False)
    logger.info("数据预处理完成")

if __name__ == '__main__':
    main()
