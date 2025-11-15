import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from models import UNet, BCEDiceLoss,ResidualUNet
from training.metrics import SegmentationMetrics, MetricTracker
import matplotlib.pyplot as plt
from pathlib import Path
import time
from data.dataset import LandslideDataset  # 添加这行导入

class LandslideValidator:
    """滑坡分割验证器"""

    def __init__(self, model_path, test_loader, device='cpu', image_size=512, save_dir='val_results'):
        # 加载模型
        # self.model = UNet(n_channels=3, n_classes=1)
        self.model = ResidualUNet(n_channels=3, n_classes=1)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        self.test_loader = test_loader
        self.device = device
        self.image_size = image_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 指标计算
        self.metrics_calculator = SegmentationMetrics()
        self.metric_tracker = MetricTracker()
        self.criterion = nn.BCEWithLogitsLoss()

    def validate(self, save_predictions=True, save_samples=10):
        """执行验证"""
        self.metric_tracker.reset()
        sample_count = 0
        
        print(f"开始验证，测试集大小: {len(self.test_loader.dataset)}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                # 修改：根据实际返回的数据解包
                images, masks = batch  # 只解包两个值
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # 计算指标
                metrics = self.metrics_calculator.calculate_all_metrics(outputs, masks, threshold=0.5)
                self.metric_tracker.update(metrics, loss.item())
                
                # 保存样本可视化
                if save_samples > 0 and sample_count < save_samples:
                    self._save_sample_visualization(images, masks, outputs, sample_count)
                    sample_count += len(images)
                
                # 打印进度
                if (batch_idx + 1) % 5 == 0:
                    current_metrics = self.metric_tracker.average()
                    print(f'批次 [{batch_idx+1}/{len(self.test_loader)}] - '
                          f'Loss: {current_metrics["loss"]:.4f} - '
                          f'Acc: {current_metrics["accuracy"]:.4f} - '
                          f'mIoU: {current_metrics["miou"]:.4f}')

        # 计算最终指标
        final_metrics = self.metric_tracker.average()
        
        # 保存结果
        self._save_results(final_metrics)
        
        return final_metrics

    def _save_sample_visualization(self, images, masks, outputs, start_idx):
        """保存样本可视化对比图"""
        samples_dir = self.save_dir / 'samples'
        samples_dir.mkdir(exist_ok=True)
        
        for i in range(min(2, len(images))):  # 每个批次保存2个样本
            idx = start_idx + i
            
            # 获取图像、真实标签和预测
            image = images[i].cpu().numpy().transpose(1, 2, 0)
            mask = masks[i].squeeze().cpu().numpy()
            prediction = torch.sigmoid(outputs[i]).squeeze().cpu().numpy()
            prediction_binary = (prediction > 0.5).astype(np.uint8)
            
            # 反归一化图像
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image * std + mean
            image = np.clip(image, 0, 1)
            
            # 创建可视化图像
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # 原始图像
            axes[0].imshow(image)
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            # 真实标签
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # 预测概率图
            axes[2].imshow(prediction, cmap='hot')
            axes[2].set_title('Prediction Probability')
            axes[2].axis('off')
            
            # 二值预测
            axes[3].imshow(prediction_binary, cmap='gray')
            axes[3].set_title('Binary Prediction')
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(samples_dir / f'sample_{idx:03d}.png', dpi=120, bbox_inches='tight')
            plt.close()

    def _save_results(self, final_metrics):
        """保存验证结果"""
        # 保存指标到文本文件
        with open(self.save_dir / 'validation_metrics.txt', 'w') as f:
            f.write("滑坡分割验证结果\n")
            f.write("=" * 50 + "\n")
            f.write(f"测试集大小: {len(self.test_loader.dataset)}\n")
            f.write(f"图像尺寸: {self.image_size}\n")
            f.write(f"设备: {self.device}\n")
            f.write(f"日期: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("性能指标:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Loss:       {final_metrics['loss']:.6f}\n")
            f.write(f"Accuracy:   {final_metrics['accuracy']:.6f}\n")
            f.write(f"mIoU:       {final_metrics['miou']:.6f}\n")
            f.write(f"Dice:       {final_metrics['dice']:.6f}\n")
            f.write(f"F1 Score:   {final_metrics['f1']:.6f}\n")
            f.write(f"Precision:  {final_metrics['precision']:.6f}\n")
            f.write(f"Recall:     {final_metrics['recall']:.6f}\n")
            f.write(f"IoU:        {final_metrics['iou']:.6f}\n")

    def print_results(self, final_metrics):
        """打印验证结果"""
        print("\n" + "="*70)
        print("滑坡分割验证结果")
        print("="*70)
        print(f"测试集大小: {len(self.test_loader.dataset)}")
        print(f"图像尺寸: {self.image_size}")
        print(f"设备: {self.device}")
        print(f"日期: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*70)
        
        print("性能指标:")
        print(f"  Loss:      {final_metrics['loss']:.6f}")
        print(f"  Accuracy:  {final_metrics['accuracy']:.6f}")
        print(f"  mIoU:      {final_metrics['miou']:.6f}")
        print(f"  Dice:      {final_metrics['dice']:.6f}")
        print(f"  F1 Score:  {final_metrics['f1']:.6f}")
        print(f"  Precision: {final_metrics['precision']:.6f}")
        print(f"  Recall:    {final_metrics['recall']:.6f}")
        print(f"  IoU:       {final_metrics['iou']:.6f}")
        print("="*70)


def validate_landslide(model_path, test_loader, device='cpu', image_size=512, save_dir='val_results'):
    """
    滑坡分割验证主函数
    
    参数:
        model_path: 模型权重路径
        test_loader: 测试数据加载器
        device: 设备
        image_size: 图像尺寸
        save_dir: 结果保存目录
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # 创建验证器
    validator = LandslideValidator(
        model_path=model_path,
        test_loader=test_loader,
        device=device,
        image_size=image_size,
        save_dir=save_dir
    )
    
    # 执行验证
    print("开始验证滑坡分割模型...")
    start_time = time.time()
    
    final_metrics = validator.validate(
        save_predictions=True,
        save_samples=10
    )
    
    end_time = time.time()
    
    # 打印结果
    validator.print_results(final_metrics)
    print(f"验证完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"结果保存至: {validator.save_dir}")
    
    return final_metrics


# 使用示例
if __name__ == "__main__":
    # 创建测试数据集和数据加载器
    test_dataset = LandslideDataset(
        image_dir=r'/root/autodl-tmp/dataset/BFA_splitted/test/images',
        mask_dir=r'/root/autodl-tmp/dataset/BFA_splitted/test/masks'
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # 执行验证
    metrics = validate_landslide(
        model_path="/root/autodl-tmp/recurrence_unet/scripts/run/ResidualUNet_adamw_2/best_landslide_unet.pth",
        test_loader=test_loader,
        device='cuda',
        image_size=512,
        save_dir='validation/residualUnet_adaw'
    )