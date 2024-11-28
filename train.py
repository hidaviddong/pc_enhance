import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import MinkowskiEngine as ME
from MinkowskiEngine import utils as ME_utils
from data import (
    PointCloudDataset,
    position_loss,
    plot_loss_curve
    
)
from network import MyNet
from logger import logger


def validate_model(model, val_dataloader):
    model.eval()  # 设置为评估模式
    total_pred_loss = 0
    total_baseline_loss = 0
    num_batches = 0
    
    with torch.no_grad():  # 不计算梯度
        for batch_data in val_dataloader:
            compress_feats = batch_data['compress_feats'].reshape(-1, 3).float().to('cuda')
            compress_coords = batch_data['compress_coords'].reshape(-1, 4).int().to('cuda')
            original_new_origin_coords = batch_data['original_new_origin_coords'].reshape(-1, 3).float().to('cuda')
            original_compress_coords = batch_data['original_compress_coords'].reshape(-1, 3).float().to('cuda')
            
            # 构建稀疏张量
            compress_sparse_tensor = ME.SparseTensor(
                features=compress_feats,
                coordinates=compress_coords
            )

            # 前向传播
            pred_offset = model(compress_sparse_tensor)
            
            # 计算损失
            pred_loss, baseline_loss = position_loss(
                pred_offset.F.float(), 
                original_new_origin_coords,
                original_compress_coords
            )
            
            total_pred_loss += pred_loss.item()
            total_baseline_loss += baseline_loss.item()
            num_batches += 1
            
    avg_pred_loss = total_pred_loss / num_batches
    avg_baseline_loss = total_baseline_loss / num_batches
    
    model.train()  # 切回训练模式
    return avg_pred_loss, avg_baseline_loss


def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs):        
    model = model.to('cuda')
    model.train()
    
    train_pred_loss_history = []
    train_baseline_loss_history = []
    val_pred_loss_history = []
    val_baseline_loss_history = []
    
    for epoch in range(epochs):
        total_pred_loss = 0
        total_baseline_loss = 0
        num_batches = 0
        logger.info(f"\nStart epoch {epoch}")
        
        for batch_idx, batch_data in enumerate(train_dataloader):
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
        
        train_pred_loss_history.append(avg_pred_loss)
        train_baseline_loss_history.append(avg_baseline_loss)
        
        logger.info(f'\nEpoch {epoch + 1}/{epochs}, Pred Loss: {avg_pred_loss:.4f}, Baseline Loss: {avg_baseline_loss:.4f}')

        # 学习率调整
        if (epoch + 1) % 50 == 0:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'Learning rate updated to: {current_lr}')
        
        # 验证阶段 - 每5个epoch验证一次
        if (epoch + 1) % 5 == 0:
            val_pred_loss, val_baseline_loss = validate_model(model, val_dataloader)
            val_pred_loss_history.append(val_pred_loss)
            val_baseline_loss_history.append(val_baseline_loss)
            
            logger.info(f'\nEpoch {epoch + 1}/{epochs}')
            logger.info(f'Val - Pred Loss: {val_pred_loss:.4f}, Baseline Loss: {val_baseline_loss:.4f}')
            
            plot_loss_curve(
                pred_losses=val_pred_loss_history,
                baseline_losses=val_baseline_loss_history,
                save_path='val_loss_curve.png',
                current_epoch=epoch+1
            )

        # 定期保存检查点和绘制损失曲线 - 每10个epoch
        if (epoch + 1) % 10 == 0:
            # 绘制损失曲线
            plot_loss_curve(
                pred_losses=train_pred_loss_history,
                baseline_losses=train_baseline_loss_history,
                save_path='train_loss_curve.png',
                current_epoch=epoch+1
            )
            
            # 保存定期检查点
            save_path = f'models/epoch_{epoch + 1}_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_pred_loss': avg_train_pred_loss,
                'train_baseline_loss': avg_train_baseline_loss,
                'train_pred_loss_history': train_pred_loss_history,
                'train_baseline_loss_history': train_baseline_loss_history,
                'val_pred_loss_history': val_pred_loss_history,
                'val_baseline_loss_history': val_baseline_loss_history
            }, save_path)

    return (train_pred_loss_history, train_baseline_loss_history, 
            val_pred_loss_history, val_baseline_loss_history)


if __name__ == '__main__':

    preprocessed_data = torch.load("preprocessed_blocks.pth")
    
       
    # 随机选择部分文件作为验证集
    num_files = len(preprocessed_data)
    num_train_files = int(0.8 * num_files)
    
    # 使用固定的随机种子
    np.random.seed(42)
    file_indices = np.random.permutation(num_files)
    
    # 创建训练集和验证集的数据
    train_data = [preprocessed_data[i] for i in file_indices[:num_train_files]]
    val_data = [preprocessed_data[i] for i in file_indices[num_train_files:]]
    
    logger.info(f"训练集文件数: {len(train_data)}")
    logger.info(f"验证集文件数: {len(val_data)}")
    
    # 直接将文件列表传给数据集类
    train_dataset = PointCloudDataset(train_data)
    val_dataset = PointCloudDataset(val_data)
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=5,
        shuffle=False
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=5,
        shuffle=False
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
    train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs=3000)
    
 
