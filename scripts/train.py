import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import matplotlib.pyplot as plt
import pandas as pd
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


def main():
    # 配置参数
    config = {
        'image_size': 512,
        'batch_size': 4,
        'epochs': 5,
        'learning_rate': 1e-4,
        'optimizer': 'adam',
        'loss_function': 'bce_dice',
        'scheduler': 'plateau',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        # 数据路径 - 请根据实际情况修改
        'train_image_dir': r'D:\all_need_mixture_splitted\images\train',
        'train_mask_dir': r'D:\all_need_mixture_splitted\true_bin_labels\train',
        'val_image_dir': r'D:\all_need_mixture_splitted\images\val',
        'val_mask_dir': r'D:\all_need_mixture_splitted\true_bin_labels\val',
    }

    print("🚀 开始训练滑坡分割模型")
    print("=" * 50)
    print(f"设备: {config['device']}")
    print(f"图像尺寸: {config['image_size']}x{config['image_size']}")
    print(f"批大小: {config['batch_size']}")
    print(f"训练轮数: {config['epochs']}")
    print(f"学习率: {config['learning_rate']}")
    print("=" * 50)

    # 创建模型
    model = UNet(in_channels=3, out_channels=1)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 获取数据加载器
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

    # 设置损失函数、优化器、调度器
    criterion = BCEDiceLoss(alpha=0.5)
    optimizer = get_optimizer(model, config['optimizer'], config['learning_rate'])
    scheduler = get_scheduler(optimizer, config['scheduler'])

    # 创建训练器并开始训练
    trainer = UNetTrainer(model, config['device'], train_loader, val_loader)

    print("\n📈 开始训练...")
    history = trainer.train(
        epochs=config['epochs'],
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        save_path='best_landslide_unet.pth'
    )

    # 保存结果
    plot_training_history(history)
    save_metrics_to_csv(history)

    print("\n✅ 训练完成!")
    return model, history


if __name__ == "__main__":
    model, history = main()