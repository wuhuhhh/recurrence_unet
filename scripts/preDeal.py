import os
import cv2
import numpy as np
from tqdm import tqdm

# --------------------------
# 在这里修改你的路径
# --------------------------
INPUT_DIR = r"D:\all_need_mixture_splitted\bin_labels\test"    # 原始标签文件夹
OUTPUT_DIR = r"D:\all_need_mixture_splitted\true_bin_labels\test"  # 处理后标签保存文件夹
INVERT_MASK = True                            # 你的情况设为True（黑滑坡→1，白背景→0）
THRESHOLD = 127                               # 二值化阈值，一般不用改
# --------------------------

def create_directory(dir_path):
    """创建目录（如果不存在）"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"创建输出目录: {dir_path}")

def process_mask(mask_path):
    """处理单个标签（不改变原始尺寸）"""
    # 以灰度模式读取（确保单通道）
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        raise ValueError(f"无法读取图像: {mask_path}")
    
    # 二值化处理（保持原始尺寸）
    _, mask_binary = cv2.threshold(mask, THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # 归一化到[0,1]
    mask_normalized = mask_binary.astype(np.float32) / 255.0
    
    # 反转标签（如果需要）
    if INVERT_MASK:
        mask_normalized = 1 - mask_normalized
    
    return mask_normalized

def main():
    create_directory(OUTPUT_DIR)
    
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # 获取所有标签文件
    all_files = os.listdir(INPUT_DIR)
    mask_files = [
        f for f in all_files 
        if os.path.splitext(f)[1].lower() in image_extensions 
        and not f.startswith('.') 
        and '.ipynb_checkpoints' not in f
    ]
    
    if not mask_files:
        print("未找到任何标签文件！")
        return
    
    # 批量处理
    for filename in tqdm(mask_files, desc="处理进度"):
        try:
            input_path = os.path.join(INPUT_DIR, filename)
            if not os.path.isfile(input_path):
                continue
            
            # 处理并保存（保持原始尺寸）
            processed_mask = process_mask(input_path)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(OUTPUT_DIR, f"{base_name}.png")
            cv2.imwrite(output_path, (processed_mask * 255).astype(np.uint8))
            
        except Exception as e:
            print(f"处理 {filename} 出错: {str(e)}")
    
    print(f"处理完成，共处理 {len(mask_files)} 个文件，保留原始尺寸")

if __name__ == "__main__":
    main()
