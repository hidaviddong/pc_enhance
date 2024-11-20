#  1、KD Tree 找 new origin
#  2、看看生成的 new origin 是什么样
#  3、 比较 A = PSNR(new_origin, origin), B = PSNR(compress, origin) ，看看是否 A > B ?

import os
import torch
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from MinkowskiEngine import utils as ME_utils
from data import (
    PointCloudDataset,
    find_corresponding_original_points,
    save_point_cloud_as_ply,
    position_loss
)
from logger import logger

def block(data_loader, device='cuda'):
    """
    预处理数据，提前计算对应关系
    """
    preprocessed_data = []
    total_batches = len(data_loader)
    
    for batch_idx, (batch_chunks_original, batch_chunks_compress) in enumerate(data_loader):
        logger.info(f"处理批次: {batch_idx + 1}/{total_batches}")
        batch_data = []
        
        for sample_idx, (chunks_original, chunks_compress) in enumerate(zip(batch_chunks_original, batch_chunks_compress)):
            sample_data = []
            total_blocks = len(chunks_original)
            
            for block_idx in range(total_blocks):
                logger.info(f"处理批次 {batch_idx + 1}/{total_batches}, 样本 {sample_idx + 1}, 块 {block_idx + 1}/{total_blocks}")
                
                # 获取当前块的数据
                coords_original = chunks_original[block_idx][:, :3]
                coords_compress = chunks_compress[block_idx][:, :3]
                
                # 查看shape
                
                logger.info(f"coords_original: {coords_original.shape}")
                logger.info(f"coords_original: {coords_compress.shape}")
#                 # 转换为torch tensors
#                 coords_original = torch.from_numpy(coords_original).float()
#                 coords_compress = torch.from_numpy(coords_compress).float()
                
#                 # 转换为ME需要的格式
#                 coords_original_tensor = ME_utils.batched_coordinates([coords_original], device=device)
#                 coords_compress_tensor = ME_utils.batched_coordinates([coords_compress], device=device)
                
#                 coords_original_features = coords_original.to(device)
#                 coords_compress_features = coords_compress.to(device)
                
#                 # 创建临时SparseTensor来获取唯一的坐标
#                 temp_sparse_tensor = ME.SparseTensor(
#                     features=coords_compress_features,
#                     coordinates=coords_compress_tensor
#                 )

#                 # 获取唯一的坐标和特征
#                 unique_coords = temp_sparse_tensor.C
#                 unique_feats = temp_sparse_tensor.F

#                 # 使用这些唯一的坐标重新计算对应的原始点
#                 coords_compress_dedup = unique_coords[:, 1:].cpu().numpy()  # 去掉batch维度
#                 coords_original_numpy = coords_original.cpu().numpy()
#                 coords_new_original = find_corresponding_original_points(coords_compress_dedup, coords_original_numpy)
#                 coords_new_original = torch.from_numpy(coords_new_original).float()
#                 coords_new_original_features = coords_new_original.to(device)
#                 coords_new_original_tensor = ME_utils.batched_coordinates([coords_new_original], device=device)

#                 logger.info(f"Block {block_idx} shapes:")
#                 logger.info(f"Original compress_features shape: {coords_compress_features.shape}")
#                 logger.info(f"After dedup compress_features shape: {unique_feats.shape}")
#                 logger.info(f"compress_coordinates shape: {unique_coords.shape}")
#                 logger.info(f"new_original_features shape: {coords_new_original_features.shape}")
#                 logger.info(f"new_original_coordinates shape: {coords_new_original_tensor.shape}")


                 
#                 # 存储预处理后的数据
#                 block_data = {
#                 'compress_features': unique_feats,  # 使用去重后的特征
#                 'compress_coordinates': unique_coords,  # 使用去重后的坐标
#                 'new_original_features': coords_new_original_features,
#                 'new_original_coordinates': coords_new_original_tensor
#                 }
                
#                 sample_data.append(block_data)
            
            batch_data.append(sample_data)
        
        preprocessed_data.append(batch_data)

if __name__ == '__main__':
    # 创建数据集和数据加载器
    dataset = PointCloudDataset(
        folder_A='./test_part/original',
        folder_B='./test_part/compress',
        num_blocks= 10
    )
    
    data_loader = DataLoader(
        dataset, 
        batch_size=8
    )

    # 预处理数据
    logger.info("开始预处理数据...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    block(
        data_loader, 
        device=device
    )
    logger.info("数据预处理完成")