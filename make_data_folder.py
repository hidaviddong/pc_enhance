import os
import shutil
import glob

def extract_first_frames(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有.ply文件并排序
    ply_files = glob.glob(os.path.join(input_folder, "*.ply"))
    ply_files.sort()
    
    # 确保找到了文件
    if not ply_files:
        print(f"在 {input_folder} 中未找到.ply文件！")
        return
    
    # 每10帧选择一帧（第一帧）
    for i in range(0, len(ply_files), 10):
        source_file = ply_files[i]
        file_name = os.path.basename(source_file)
        destination_file = os.path.join(output_folder, file_name)
        
        # 复制文件
        shutil.copy2(source_file, destination_file)
        print(f"已复制: {file_name}")

def process_all_folders():
    # 定义所有需要处理的路径对
    train_pairs = [
        # 处理 original 数据
        ("/home/jupyter-haoyu/Data/8i_test/orig/longdress/Ply", "/home/jupyter-haoyu/pc_enhance/research/train/original"),
        ("/home/jupyter-haoyu/Data/8i_test/orig/loot/Ply", "/home/jupyter-haoyu/pc_enhance/research/train/original"),
        ("/home/jupyter-haoyu/Data/8i_test/orig/redandblack/Ply", "/home/jupyter-haoyu/pc_enhance/research/train/original"),
        
        # 处理 compress 数据
        ("/home/jupyter-haoyu/Data/8i_test/8x/longdress/Ply", "/home/jupyter-haoyu/pc_enhance/research/train/compress"),
        ("/home/jupyter-haoyu/Data/8i_test/8x/loot/Ply", "/home/jupyter-haoyu/pc_enhance/research/train/compress"),
        ("/home/jupyter-haoyu/Data/8i_test/8x/redandblack/Ply", "/home/jupyter-haoyu/pc_enhance/research/train/compress")
    ]
    
    test_pairs = [
        # 处理测试数据
        ("/home/jupyter-haoyu/Data/8i_test/orig/soldier/Ply", "/home/jupyter-haoyu/pc_enhance/research/test/original"),
        ("/home/jupyter-haoyu/Data/8i_test/8x/soldier/Ply", "/home/jupyter-haoyu/pc_enhance/research/test/compress")
    ]
    
    # 处理训练数据
    print("开始处理训练数据...")
    for input_folder, output_folder in train_pairs:
        print(f"\n处理文件夹: {input_folder}")
        extract_first_frames(input_folder, output_folder)
    
    # 处理测试数据
    print("\n开始处理测试数据...")
    for input_folder, output_folder in test_pairs:
        print(f"\n处理文件夹: {input_folder}")
        extract_first_frames(input_folder, output_folder)
    
    print("\n所有文件处理完成！")

if __name__ == "__main__":
    process_all_folders()
