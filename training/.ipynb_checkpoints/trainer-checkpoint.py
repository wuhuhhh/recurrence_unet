import torch
import torch.optim as optim
import wandb  # 导入wandb
from .metrics import SegmentationMetrics, MetricTracker


class UNetTrainer:
    def __init__(self, model, device, train_loader, val_loader=None, experiment=None, save_dir='run'):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metrics_calculator = SegmentationMetrics()
        self.model.to(self.device)
        self.experiment = experiment  # WandB实验对象
        self.global_step = 0  # 全局步数计数
        self.save_dir = save_dir  # 新增保存目录
    
    def train_epoch(self, optimizer, criterion, cur_epoch, scheduler=None):
        """训练一个epoch"""
        self.model.train()
        tracker = MetricTracker()
        epoch_loss = 0

        # 定期记录图像示例（每10个batch）
        log_images = True

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images, masks = images.to(self.device), masks.to(self.device)
            self.global_step += 1

            # 前向传播
            optimizer.zero_grad()
            outputs = self.model(images)

            # 计算损失
            loss = criterion(outputs, masks)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 计算所有指标
            batch_metrics = self.metrics_calculator.calculate_all_metrics(outputs, masks)
            tracker.update(batch_metrics, loss.item())

            epoch_loss += loss.item()

            # 记录训练指标到WandB
            if self.experiment:
                # 记录每个batch的损失和学习率
                self.experiment.log({
                    'train/loss': loss.item(),
                    'train/accuracy': batch_metrics['accuracy'],
                    'train/dice': batch_metrics['dice'],
                    'train/mIoU': batch_metrics['miou'],
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'step': self.global_step,
                    'epoch': cur_epoch
                })
                

                # 定期记录图像示例
                # if log_images and batch_idx % 10 == 0:
                #     # 处理输出以可视化（如果是 sigmoid 输出，转换为概率）
                #     preds = torch.sigmoid(outputs) if outputs.shape[1] == 1 else torch.softmax(outputs, dim=1)
                #     pred_masks = (preds > 0.5).float()  # 二值化

                #     # 记录图像、真实掩码和预测掩码
                #     self.experiment.log({
                #         'train/sample_images': wandb.Image(
                #             images[0].cpu(),
                #             masks={
                #                 "true": wandb.Image(masks[0].cpu()),
                #                 "pred": wandb.Image(pred_masks[0].cpu())
                #             },
                #             caption=f"Train Epoch {cur_epoch}, Batch {batch_idx}"
                #         )
                #     })
                #     log_images = False  # 每个epoch只记录一次图像

            if batch_idx % 10 == 0:
                current_metrics = tracker.average()
                print(f'Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {current_metrics["loss"]:.4f}, '
                      f'Acc: {current_metrics["accuracy"]:.4f}')

        avg_metrics = tracker.average()
        return avg_metrics

    def validate(self, criterion, epoch):
        """验证并记录指标到WandB"""
        if self.val_loader is None:
            return None

        self.model.eval()
        tracker = MetricTracker()
        log_images = True  # 记录验证图像

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(self.val_loader):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)

                loss = criterion(outputs, masks)
                batch_metrics = self.metrics_calculator.calculate_all_metrics(outputs, masks)
                tracker.update(batch_metrics, loss.item())

                # 记录验证图像
                # if self.experiment and log_images:
                #     preds = torch.sigmoid(outputs) if outputs.shape[1] == 1 else torch.softmax(outputs, dim=1)
                #     pred_masks = (preds > 0.5).float()

                #     self.experiment.log({
                #         'val/sample_images': wandb.Image(
                #             images[0].cpu(),
                #             masks={
                #                 "true": wandb.Image(masks[0].cpu()),
                #                 "pred": wandb.Image(pred_masks[0].cpu())
                #             },
                #             caption=f"Val Epoch {epoch}"
                #         )
                #     })
                #     log_images = False

        val_metrics = tracker.average()

        # 记录验证指标到WandB
        if self.experiment:
            self.experiment.log({
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/dice': val_metrics['dice'],
                'val/mIoU': val_metrics['miou'],
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall'],
                'val/f1': val_metrics['f1'],
                'epoch': epoch
            })

        return val_metrics

    def train(self, epochs, optimizer, criterion, scheduler=None, save_path='best_model.pth'):
        """完整训练过程"""
        best_val_loss = float('inf')
        train_history = {
            'epoch': [],
            'train_loss': [], 'train_iou': [], 'train_dice': [], 'train_precision': [],
            'train_recall': [], 'train_f1': [], 'train_accuracy': [], 'train_miou': [],
            'val_loss': [], 'val_iou': [], 'val_dice': [], 'val_precision': [],
            'val_recall': [], 'val_f1': [], 'val_accuracy': [], 'val_miou': []
        }

        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print('-' * 60)

            # 训练
            train_metrics = self.train_epoch(optimizer, criterion, epoch, scheduler)

            # 记录训练指标
            train_history['epoch'].append(epoch + 1)
            for key in train_metrics:
                train_history[f'train_{key}'].append(train_metrics[key])

            print(f"Train - {self._format_metrics(train_metrics)}")

            # 记录 epoch 级别的训练指标
            if self.experiment:
                self.experiment.log({
                    'train/epoch_loss': train_metrics['loss'],
                    'train/epoch_accuracy': train_metrics['accuracy'],
                    'train/epoch_mIoU': train_metrics['miou'],
                    'train/epoch_dice': train_metrics['dice'],
                    'epoch': epoch + 1
                })

            # 验证
            if self.val_loader is not None:
                val_metrics = self.validate(criterion, epoch + 1)

                # 记录验证指标
                for key in val_metrics:
                    train_history[f'val_{key}'].append(val_metrics[key])

                print(f"Val   - {self._format_metrics(val_metrics)}")

                # 保存最佳模型
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    torch.save(self.model.state_dict(), save_path)
                    print(f"💾 保存最佳模型: {save_path} (Loss: {best_val_loss:.4f})")
                    if self.experiment:
                        self.experiment.log({"best_val_loss": best_val_loss})

            # 学习率调度
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) and self.val_loader is not None:
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()

            # 每10个epoch打印详细指标
            if (epoch + 1) % 10 == 0:
                self._print_detailed_metrics(train_metrics, val_metrics if self.val_loader else None)

        return train_history

    def _format_metrics(self, metrics: dict) -> str:
        """格式化指标输出"""
        return (f"Loss: {metrics['loss']:.4f} | "
                f"Acc: {metrics['accuracy']:.4f} | "
                f"mIoU: {metrics['miou']:.4f} | "
                f"Dice: {metrics['dice']:.4f} | "
                f"F1: {metrics['f1']:.4f}")

    def _print_detailed_metrics(self, train_metrics: dict, val_metrics: dict = None):
        """打印详细指标"""
        print("\n" + "=" * 80)
        print("📊 详细指标报告")
        print("=" * 80)

        headers = ["Metric", "Train"]
        if val_metrics:
            headers.append("Val")

        print(f"{'Metric':<12} {'Train':<10} {'Val':<10}" if val_metrics else f"{'Metric':<12} {'Train':<10}")
        print("-" * (35 if val_metrics else 25))

        metrics_list = [
            ('Loss', 'loss'), ('Accuracy', 'accuracy'), ('mIoU', 'miou'),
            ('Dice', 'dice'), ('IoU', 'iou'), ('F1', 'f1'),
            ('Precision', 'precision'), ('Recall', 'recall')
        ]

        for name, key in metrics_list:
            if val_metrics:
                print(f"{name:<12} {train_metrics[key]:.4f}     {val_metrics[key]:.4f}")
            else:
                print(f"{name:<12} {train_metrics[key]:.4f}")

        print("=" * 80)


def get_optimizer(model, optimizer_name='adam', learning_rate=1e-4, weight_decay=1e-5):
    """获取优化器"""
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name='step', **kwargs):
    """获取学习率调度器"""
    if scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=kwargs.get('step_size', 30),
                                         gamma=kwargs.get('gamma', 0.1))
    elif scheduler_name == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=kwargs.get('patience', 10),
                                                    factor=kwargs.get('factor', 0.5))
    elif scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs.get('T_max', 50))
    else:
        return None