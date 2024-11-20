import os
import subprocess
import torch
import numpy as np
import MinkowskiEngine as ME
from MinkowskiEngine import utils as ME_utils
from network import MyNet
from logger import logger
from data import save_point_cloud_as_ply

def predict(model_path, preprocessed_data_path, output_dir='./output'):
    """
    使用训练好的模型进行预测
    """
    logger.info("开始预测...")

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载模型
    model = MyNet()
    checkpoint = torch.load(model_path)
    # 修改这里：加载完整的checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 加载预处理数据
    preprocessed_data = torch.load(preprocessed_data_path)
    first_sample = preprocessed_data[0]  # 移除一个[0]，因为结构变了
    logger.info("预处理数据加载成功")

    with torch.no_grad():
        # 初始化存储所有block的数组
        all_original_coords = []
        all_compress_coords = []
        all_predicted_coords = []
        all_output_coords = []

        # 处理第一张图片的所有块
        for block_idx, block_data in enumerate(first_sample):
            # 构建稀疏张量 - 修改为与训练时相同的格式
            compress_sparse_tensor = ME.SparseTensor(
                features=block_data['compress_feats'].reshape(-1, 3).to(device),  # 使用feats而不是features
                coordinates=block_data['compress_coords'].reshape(-1, 4).to(device)  # 使用coords而不是coordinates
            )
            
            # 获取原始点云坐标
            original_coords = block_data['original_new_origin_coords'].reshape(-1, 3).cpu()  # 使用feats而不是coordinates
            compress_coords = block_data['original_compress_coords'].reshape(-1, 3).cpu()  # 使用feats而不是coordinates
            
            # 获取模型输出
            output = model(compress_sparse_tensor)
            output_coords = output.F.cpu()
            
            # 计算预测坐标 - 与训练时相同的方式
            predicted_coords = output.F.cpu() + compress_coords 
            
            # 记录坐标 - 确保都是numpy数组
            all_original_coords.append(original_coords.numpy())
            all_compress_coords.append(compress_coords.numpy())
            all_predicted_coords.append(predicted_coords.numpy())
            all_output_coords.append(output_coords.numpy())

            logger.info(f"处理完block {block_idx}, 形状: {predicted_coords.shape}")

        # 合并所有block的点云
        merged_original_coords = np.concatenate(all_original_coords, axis=0)
        merged_compress_coords = np.concatenate(all_compress_coords, axis=0)
        merged_predicted_coords = np.concatenate(all_predicted_coords, axis=0)
        merged_output_coords = np.concatenate(all_output_coords, axis=0)

        logger.info(f"合并后的点云形状:")
        logger.info(f"- 原始点云: {merged_original_coords.shape}")
        logger.info(f"- 压缩点云: {merged_compress_coords.shape}")
        logger.info(f"- 预测点云: {merged_predicted_coords.shape}")
        file_index = block_data['file_index']
        # 保存点云文件
        original_file = f"{output_dir}/new_origin_merged_{file_index}.ply"
        compressed_file = f"{output_dir}/compress_merged_{file_index}.ply"
        predict_file = f"{output_dir}/predict_merged_{file_index}.ply"
        output_file = f"{output_dir}/output_merged_{file_index}.ply"
        # 保存点云文件
        save_point_cloud_as_ply(merged_original_coords, original_file)
        save_point_cloud_as_ply(merged_compress_coords, compressed_file)
        save_point_cloud_as_ply(merged_predicted_coords, predict_file)
        save_point_cloud_as_ply(merged_output_coords, output_file)

        # 计算和保存PSNR
#        original file 应该用最原始的,我们这里用 1051
        origin_file = "./real_train/new_origin/new_origin_longdress_vox10_1051.ply"
        calculate_and_save_psnr(origin_file, compressed_file, predict_file, output_dir, logger)

        
def calculate_and_save_psnr(original_file, compressed_file, result_file, output_dir, logger):
    """计算并保存PSNR结果"""
    # 计算压缩点云的PSNR
    cmd1 = f"../../mpeg-pcc-tmc2/bin/PccAppMetrics --uncompressedDataPath={original_file} --reconstructedDataPath={compressed_file} --resolution=1023 --frameCount=1"
    process1 = subprocess.Popen(cmd1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output1, _ = process1.communicate()

    psnr_file1 = f"{output_dir}/psnr_compress.txt"
    with open(psnr_file1, 'w') as f:
        f.write(output1.decode())
    logger.info(f"压缩点云PSNR结果已保存到：{psnr_file1}")

    # 计算预测结果的PSNR
    cmd2 = f"../../mpeg-pcc-tmc2/bin/PccAppMetrics --uncompressedDataPath={original_file} --reconstructedDataPath={result_file} --resolution=1023 --frameCount=1"
    process2 = subprocess.Popen(cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output2, _ = process2.communicate()

    psnr_file2 = f"{output_dir}/psnr_predict.txt"
    with open(psnr_file2, 'w') as f:
        f.write(output2.decode())
    logger.info(f"处理结果PSNR已保存到：{psnr_file2}")

if __name__ == '__main__':
    # 设置参数
    MODEL_PATH = 'models/epoch_3000_model.pth'  # 模型权重路径
    PREPROCESSED_DATA_PATH = 'preprocessed_blocks.pth'  # 预处理数据路径
    OUTPUT_DIR = './output'  # 输出目录

    # 执行预测
    predict(
        model_path=MODEL_PATH,
        preprocessed_data_path=PREPROCESSED_DATA_PATH,
        output_dir=OUTPUT_DIR
    )
