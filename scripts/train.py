# scripts/train.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import pandas as pd
import matplotlib.pyplot as plt
from models import UNet, BCEDiceLoss
from data.dataset import get_data_loaders
from training.trainer import UNetTrainer, get_optimizer, get_scheduler

def plot_training_history(history, save_path='training_history.png'):
    """绘制训练历史图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(history['epoch'], history['train_loss'], label='Train Loss')
    if 'val_loss' in history and history['val_loss'][0] is not None:
        axes[0, 0].plot(history['epoch'], history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 准确率和mIoU
    axes[0, 1].plot(history['epoch'], history['train_accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history['epoch'], history['train_miou'], label='Train mIoU')
    if 'val_accuracy' in history and history['val_accuracy'][0] is not None:
        axes[0, 1].plot(history['epoch'], history['val_accuracy'], '--', label='Val Accuracy')
        axes[0, 1].plot(history['epoch'], history['val_miou'], '--', label='Val mIoU')
    axes[0, 1].set_title('Accuracy & mIoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Dice和F1分数
    axes[1, 0].plot(history['epoch'], history['train_dice'], label='Train Dice')
    axes[1, 0].plot(history['epoch'], history['train_f1'], label='Train F1')
    if 'val_dice' in history and history['val_dice'][0] is not None:
        axes[1, 0].plot(history['epoch'], history['val_dice'], '--', label='Val Dice')
        axes[1, 0].plot(history['epoch'], history['val_f1'], '--', label='Val F1')
    axes[1, 0].set_title('Dice & F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 精确率和召回率
    axes[1, 1].plot(history['epoch'], history['train_precision'], label='Train Precision')
    axes[1, 1].plot(history['epoch'], history['train_recall'], label='Train Recall')
    if 'val_precision' in history and history['val_precision'][0] is not None:
        axes[1, 1].plot(history['epoch'], history['val_precision'], '--', label='Val Precision')
        axes[1, 1].plot(history['epoch'], history['val_recall'], '--', label='Val Recall')
    axes[1, 1].set_title('Precision & Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练历史图表已保存至: {save_path}")

def save_metrics_to_csv(history, save_path='training_metrics.csv'):
    """保存指标到CSV文件"""
    df = pd.DataFrame(history)
    df.to_csv(save_path, index=False)
    print(f"训练指标已保存至: {save_path}")
    
    # 打印最终指标
    print("\n🎯 最终训练结果:")
    final_metrics = {}
    for key in history:
        if key != 'epoch' and history[key]:
            final_metrics[key] = history[key][-1]
    
    # 格式化输出最终指标
    metric_groups = [
        ['train_loss', 'val_loss'],
        ['train_accuracy', 'val_accuracy'],
        ['train_miou', 'val_miou'],
        ['train_dice', 'val_dice'],
        ['train_f1', 'val_f1'],
        ['train_precision', 'val_precision'],
        ['train_recall', 'val_recall']
    ]
    
    for group in metric_groups:
        for metric in group:
            if metric in final_metrics and final_metrics[metric] is not None:
                print(f"{metric:.<20} {final_metrics[metric]:.4f}")

def create_dummy_data_loader(batch_size=4, image_size=512, num_samples=32):
    """创建虚拟数据加载器用于测试"""
    from torch.utils.data import Dataset, DataLoader
    import torch
    
    class DummyDataset(Dataset):
        def __len__(self):
            return num_samples
        
        def __getitem__(self, idx):
            # 创建随机图像和掩码
            image = torch.randn(3, image_size, image_size)
            # 创建偏向背景的掩码（模拟真实数据分布）
            mask = torch.bernoulli(torch.full((1, image_size, image_size), 0.1)).float()
            return image, mask
    
    return DataLoader(DummyDataset(), batch_size=batch_size, shuffle=True)

def main():
    # 配置参数
    config = {
        'image_size': 512,
        'batch_size': 8,
        'epochs': 5,
        'learning_rate': 1e-4,
        'optimizer': 'adam',
        'loss_function': 'bce_dice',
        'scheduler': 'plateau',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # 数据路径 - 请根据实际情况修改
        'train_image_dir': r'/root/autodl-tmp/dataset/all_need_dataset/images/train',
        'train_mask_dir': r'/root/autodl-tmp/dataset/all_need_dataset/mask/train',
        'val_image_dir': r'/root/autodl-tmp/dataset/all_need_dataset/images/val',
        'val_mask_dir': r'/root/autodl-tmp/dataset/all_need_dataset/mask/val',
        
        # 使用虚拟数据（如果没有真实数据）
        'use_dummy_data': False
    }
    
    print("🚀 开始训练滑坡分割UNet模型")
    print("=" * 50)
    print(f"设备: {config['device']}")
    print(f"图像尺寸: {config['image_size']}x{config['image_size']}")
    print(f"批大小: {config['batch_size']}")
    print(f"训练轮数: {config['epochs']}")
    print(f"学习率: {config['learning_rate']}")
    print("=" * 50)
    
    # 创建模型
    model = UNet(n_channels=3, n_classes=1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 获取数据加载器
    if config.get('use_dummy_data', False):
        print("使用虚拟数据进行测试...")
        train_loader = create_dummy_data_loader(config['batch_size'], config['image_size'])
        val_loader = create_dummy_data_loader(config['batch_size'], config['image_size'])
        print(f"训练样本数: {len(train_loader.dataset)}")
        print(f"验证样本数: {len(val_loader.dataset)}")
    else:
        try:
            train_loader, val_loader = get_data_loaders(
                train_image_dir=config['train_image_dir'],
                train_mask_dir=config['train_mask_dir'],
                val_image_dir=config['val_image_dir'],
                val_mask_dir=config['val_mask_dir'],
                batch_size=config['batch_size'],
                image_size=config['image_size']
            )
            print(f"训练样本数: {len(train_loader.dataset)}")
            if val_loader:
                print(f"验证样本数: {len(val_loader.dataset)}")
        except Exception as e:
            print(f"数据加载失败: {e}")
            print("切换到虚拟数据模式...")
            train_loader = create_dummy_data_loader(config['batch_size'], config['image_size'])
            val_loader = create_dummy_data_loader(config['batch_size'], config['image_size'])
    
    # 设置损失函数 - 修正这里！
    if config['loss_function'] == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif config['loss_function'] == 'dice':
        from models.losses import DiceLoss
        criterion = DiceLoss()
    else:  # bce_dice
        criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)  # 修正参数名
    
    # 设置优化器、调度器
    optimizer = get_optimizer(model, config['optimizer'], config['learning_rate'])
    scheduler = get_scheduler(optimizer, config['scheduler'])
    
    # 创建训练器并开始训练
    trainer = UNetTrainer(model, config['device'], train_loader, val_loader)
    
    print("\n开始训练...")
    history = trainer.train(
        epochs=config['epochs'],
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        save_path='run/bcd_dice_5epoch/best_landslide_unet.pth'
    )
    
    # 保存结果
    plot_training_history(history)
    save_metrics_to_csv(history)
    
    print("\n✅ 训练完成!")
    return model, history

if __name__ == "__main__":
    model, history = main()