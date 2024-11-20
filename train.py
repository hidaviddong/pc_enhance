import os
import torch
from torch.utils.data import DataLoader, Dataset

import MinkowskiEngine as ME
from MinkowskiEngine import utils as ME_utils
from data import (
    PointCloudDataset,
    find_corresponding_original_points,
    save_point_cloud_as_ply,
    position_loss,
    plot_loss_curve
    
)
from network import MyNet
from logger import logger

def train_model(model, dataloader, optimizer, scheduler, epochs):        
    model = model.to('cuda')
    model.train()
    
    pred_loss_history = []
    baseline_loss_history = []
    
    for epoch in range(epochs):
        total_pred_loss = 0
        total_baseline_loss = 0
        num_batches = 0
        logger.info(f"\nStart epoch {epoch}")
        
        for batch_idx, batch_data in enumerate(dataloader):
            compress_feats = batch_data['compress_feats'].reshape(-1, 3).float().to('cuda')
            compress_coords = batch_data['compress_coords'].reshape(-1, 4).int().to('cuda')
            
            original_new_origin_coords = batch_data['original_new_origin_coords'].reshape(-1, 3).float().to('cuda')
            
            original_compress_coords = batch_data['original_compress_coords'].reshape(-1, 3).float().to('cuda')
            
            
            # 构建稀疏张量
            compress_sparse_tensor = ME.SparseTensor(
                features=compress_feats,
                coordinates=compress_coords
            )

            # 前向传播 - 得到预测的偏移量
            pred_offset = model(compress_sparse_tensor)
            
            # 计算损失
            pred_loss, baseline_loss = position_loss(
                pred_offset.F.float(), 
                original_new_origin_coords,
                original_compress_coords
            )

            # 反向传播和优化
            optimizer.zero_grad()
            pred_loss.backward()
            optimizer.step()

            # 记录损失
            total_pred_loss += pred_loss.item()
            total_baseline_loss += baseline_loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}, Pred Loss: {pred_loss.item():.4f}, Baseline Loss: {baseline_loss.item():.4f}')

        # 计算平均损失
        avg_pred_loss = total_pred_loss / num_batches
        avg_baseline_loss = total_baseline_loss / num_batches
        
        pred_loss_history.append(avg_pred_loss)
        baseline_loss_history.append(avg_baseline_loss)
        
        logger.info(f'\nEpoch {epoch + 1}/{epochs}, Pred Loss: {avg_pred_loss:.4f}, Baseline Loss: {avg_baseline_loss:.4f}')

        # 学习率调整
        if (epoch + 1) % 50 == 0:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'Learning rate updated to: {current_lr}')
        
        # 保存检查点和绘制损失曲线
        if (epoch + 1) % 10 == 0:
            plot_loss_curve(
                pred_loss_history,
                baseline_loss_history,
                save_path='loss_curve.png',
                current_epoch=epoch+1
            )
            
        if (epoch + 1) % 100 == 0:
            save_path = f'models/epoch_{epoch + 1}_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'pred_loss': avg_pred_loss,
                'baseline_loss': avg_baseline_loss
            }, save_path)

    return pred_loss_history, baseline_loss_history


if __name__ == '__main__':

    preprocessed_data = torch.load("preprocessed_blocks.pth")

    dataset = PointCloudDataset(preprocessed_data)
    dataloader = DataLoader(
        dataset, 
        batch_size=5
    )
    
    # 初始化模型和优化器
    model = MyNet()
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, 
        gamma=0.95
    )

    # 训练模型
    train_model(model, dataloader, optimizer, scheduler, epochs=3000)
    
 