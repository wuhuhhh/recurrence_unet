# scripts/train.py
import sys
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import wandb  # 导入wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models import UNet, BCEDiceLoss,ResidualUNet
from data.dataset import get_data_loaders
from training.trainer import UNetTrainer, get_optimizer, get_scheduler


def plot_training_history(history, save_path='training_history.png'):
    # 保持原有实现不变
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
    # 保持原有实现不变
    df = pd.DataFrame(history)
    df.to_csv(save_path, index=False)
    print(f"训练指标已保存至: {save_path}")

    print("\n🎯 最终训练结果:")
    final_metrics = {}
    for key in history:
        if key != 'epoch' and history[key]:
            final_metrics[key] = history[key][-1]

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
    # 保持原有实现不变
    from torch.utils.data import Dataset, DataLoader

    class DummyDataset(Dataset):
        def __len__(self):
            return num_samples

        def __getitem__(self, idx):
            image = torch.randn(3, image_size, image_size)
            mask = torch.bernoulli(torch.full((1, image_size, image_size), 0.1)).float()
            return image, mask

    return DataLoader(DummyDataset(), batch_size=batch_size, shuffle=True)


def create_save_directory(base_dir='run', model_name=None):
    """
    创建保存目录，如果目录已存在则创建新目录
    
    Args:
        base_dir: 基础目录
        model_name: 模型名称，如果为None则自动生成
    
    Returns:
        save_dir: 创建的保存目录路径
    """
    if model_name is None:
        # 自动生成模型名称，包含时间戳
        from datetime import datetime
        model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    save_dir = os.path.join(base_dir, model_name)
    
    # 如果目录已存在，创建带序号的新目录
    counter = 1
    original_save_dir = save_dir
    while os.path.exists(save_dir):
        save_dir = f"{original_save_dir}_{counter}"
        counter += 1
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 模型保存目录: {save_dir}")
    return save_dir


def main():
    # 配置参数
    config = {
        'image_size': 512,
        'batch_size': 8,
        'epochs': 5,
        'learning_rate': 1e-4,
        'optimizer': 'adamw',
        'loss_function': 'bce_dice',
        'scheduler': 'plateau',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        # 'train_image_dir': r'/root/autodl-tmp/dataset/all_need_dataset/images/train',
        # 'train_mask_dir': r'/root/autodl-tmp/dataset/all_need_dataset/mask/train',
        # 'val_image_dir': r'/root/autodl-tmp/dataset/all_need_dataset/images/val',
        # 'val_mask_dir': r'/root/autodl-tmp/dataset/all_need_dataset/mask/val',
        'train_image_dir': r'/root/autodl-tmp/dataset/BFA_splitted/train/images',
        'train_mask_dir': r'/root/autodl-tmp/dataset/BFA_splitted/train/masks',
        'val_image_dir': r'/root/autodl-tmp/dataset/BFA_splitted/val/images',
        'val_mask_dir': r'/root/autodl-tmp/dataset/BFA_splitted/val/masks',
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

    # 初始化wandb (新增)
    experiment = wandb.init(
        project='U-Net',
        resume='allow',
        anonymous='must',
        config=config  # 直接传入完整配置
    )

    # 创建模型
    # model = UNet(n_channels=3, n_classes=1)
    model = ResidualUNet(n_channels=3, n_classes=1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    experiment.log({"total_parameters": total_params})  # 记录参数量

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

    # 设置损失函数
    if config['loss_function'] == 'bce':
        import torch.nn as nn
        criterion = nn.BCEWithLogitsLoss()
    elif config['loss_function'] == 'dice':
        from models.losses import DiceLoss
        criterion = DiceLoss()
    else:
        criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    # 设置优化器、调度器
    optimizer = get_optimizer(model, config['optimizer'], config['learning_rate'])
    scheduler = get_scheduler(optimizer, config['scheduler'])

    # 创建保存目录
    model_type = "ResidualUNet_adamw"  # 根据实际使用的模型修改
    save_dir = create_save_directory(base_dir='run', model_name=model_type)
    
    # 设置保存路径
    best_model_path = os.path.join(save_dir, 'best_landslide_unet.pth')
    final_model_path = os.path.join(save_dir, 'final_landslide_unet.pth')

    # 创建训练器并开始训练 (传入wandb实验对象)
    trainer = UNetTrainer(
        model, 
        config['device'], 
        train_loader, 
        val_loader, 
        experiment,
        save_dir=save_dir  # 传递保存目录给训练器
    )

    print("\n开始训练...")
    history = trainer.train(
        epochs=config['epochs'],
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        save_path=best_model_path
    )

    # 保存最终模型
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'history': history,
        'config': config
    }, final_model_path)
    print(f"💾 最终模型已保存至: {final_model_path}")

    # 保存训练图表和指标到保存目录
    plot_save_path = os.path.join(save_dir, 'training_history.png')
    csv_save_path = os.path.join(save_dir, 'training_metrics.csv')
    
    plot_training_history(history, save_path=plot_save_path)
    save_metrics_to_csv(history, save_path=csv_save_path)
    
    # 记录到wandb
    experiment.log({
        "training_history": wandb.Image(plot_save_path),
        "best_model": wandb.File(best_model_path),
        "final_model": wandb.File(final_model_path)
    })
    
    experiment.finish()  # 结束wandb实验

    print("\n✅ 训练完成!")
    print(f"📁 所有文件保存在: {save_dir}")
    print(f"🏆 最佳模型: {best_model_path}")
    print(f"🔚 最终模型: {final_model_path}")
    
    return model, history, save_dir


if __name__ == "__main__":
    model, history, save_dir = main()