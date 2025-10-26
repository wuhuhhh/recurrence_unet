import torch
import numpy as np
from typing import Dict, Tuple


class SegmentationMetrics:
    """分割任务评估指标类"""

    def __init__(self, num_classes: int = 2, smooth: float = 1e-6):
        """
        参数:
            num_classes: 类别数 (2 for binary segmentation)
            smooth: 平滑因子，避免除零
        """
        self.num_classes = num_classes
        self.smooth = smooth

    def calculate_confusion_matrix(self, predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> \
    Tuple[torch.Tensor, torch.Tensor]:
        """计算二分类的混淆矩阵"""
        # 将预测转换为二值
        pred_binary = (torch.sigmoid(predictions) > threshold).float()
        targets_binary = targets.float()

        # 计算TP, FP, FN, TN
        tp = (pred_binary * targets_binary).sum()
        fp = (pred_binary * (1 - targets_binary)).sum()
        fn = ((1 - pred_binary) * targets_binary).sum()
        tn = ((1 - pred_binary) * (1 - targets_binary)).sum()

        return torch.tensor([tp, fp, fn, tn])

    def calculate_all_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[
        str, float]:
        """计算所有评估指标"""

        # 计算混淆矩阵
        confusion_matrix = self.calculate_confusion_matrix(predictions, targets, threshold)
        tp, fp, fn, tn = confusion_matrix

        # 1. IoU (交并比)
        iou = self.iou_score(predictions, targets, threshold)

        # 2. Dice系数
        dice = self.dice_score(predictions, targets, threshold)

        # 3. 精确率 (Precision)
        precision = self.precision_score(tp, fp, fn, tn)

        # 4. 召回率 (Recall)
        recall = self.recall_score(tp, fp, fn, tn)

        # 5. F1分数
        f1 = self.f1_score(precision, recall)

        # 6. 总体准确率 (Overall Accuracy)
        accuracy = self.accuracy_score(tp, fp, fn, tn)

        # 7. mIoU (对于二分类，mIoU就是正负类的平均IoU)
        miou = self.mean_iou_score(predictions, targets, threshold)

        return {
            'iou': iou,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'miou': miou
        }

    def iou_score(self, predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
        """计算IoU (Jaccard指数) - 正类"""
        predictions_binary = (torch.sigmoid(predictions) > threshold).float()
        targets_binary = targets.float()

        intersection = (predictions_binary * targets_binary).sum()
        union = predictions_binary.sum() + targets_binary.sum() - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou.item()

    def mean_iou_score(self, predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
        """计算平均IoU (正类和负类的平均)"""
        predictions_binary = (torch.sigmoid(predictions) > threshold).float()
        targets_binary = targets.float()

        # 正类IoU
        intersection_pos = (predictions_binary * targets_binary).sum()
        union_pos = predictions_binary.sum() + targets_binary.sum() - intersection_pos
        iou_pos = (intersection_pos + self.smooth) / (union_pos + self.smooth)

        # 负类IoU
        predictions_neg = 1 - predictions_binary
        targets_neg = 1 - targets_binary
        intersection_neg = (predictions_neg * targets_neg).sum()
        union_neg = predictions_neg.sum() + targets_neg.sum() - intersection_neg
        iou_neg = (intersection_neg + self.smooth) / (union_neg + self.smooth)

        # 平均IoU
        miou = (iou_pos + iou_neg) / 2
        return miou.item()

    def dice_score(self, predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
        """计算Dice系数"""
        predictions_binary = (torch.sigmoid(predictions) > threshold).float()
        targets_binary = targets.float()

        intersection = (predictions_binary * targets_binary).sum()
        dice = (2. * intersection + self.smooth) / (predictions_binary.sum() + targets_binary.sum() + self.smooth)
        return dice.item()

    def precision_score(self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tn: torch.Tensor) -> float:
        """计算精确率"""
        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        return precision.item()

    def recall_score(self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tn: torch.Tensor) -> float:
        """计算召回率"""
        recall = (tp + self.smooth) / (tp + fn + self.smooth)
        return recall.item()

    def f1_score(self, precision: float, recall: float) -> float:
        """计算F1分数"""
        f1 = (2 * precision * recall) / (precision + recall + self.smooth)
        return f1

    def accuracy_score(self, tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tn: torch.Tensor) -> float:
        """计算总体准确率"""
        accuracy = (tp + tn) / (tp + fp + fn + tn + self.smooth)
        return accuracy.item()


class MetricTracker:
    """用于跟踪训练过程中的指标"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有统计"""
        self.metrics_sum = {
            'loss': 0.0, 'iou': 0.0, 'dice': 0.0, 'precision': 0.0,
            'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'miou': 0.0
        }
        self.num_batches = 0

    def update(self, metrics: Dict[str, float], loss: float = None):
        """更新指标统计"""
        if loss is not None:
            self.metrics_sum['loss'] += loss

        for key in ['iou', 'dice', 'precision', 'recall', 'f1', 'accuracy', 'miou']:
            if key in metrics:
                self.metrics_sum[key] += metrics[key]

        self.num_batches += 1

    def average(self) -> Dict[str, float]:
        """计算平均指标"""
        if self.num_batches == 0:
            return {key: 0.0 for key in self.metrics_sum.keys()}

        averages = {}
        for key, total in self.metrics_sum.items():
            averages[key] = total / self.num_batches

        return averages

    def get_summary_string(self) -> str:
        """获取指标摘要字符串"""
        avg_metrics = self.average()
        return (f"Loss: {avg_metrics['loss']:.4f} | "
                f"Acc: {avg_metrics['accuracy']:.4f} | "
                f"mIoU: {avg_metrics['miou']:.4f} | "
                f"Dice: {avg_metrics['dice']:.4f} | "
                f"F1: {avg_metrics['f1']:.4f}")