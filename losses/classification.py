"""分类损失"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """交叉熵分类损失

    Args:
        label_smoothing: 标签平滑系数
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) 分类logits
            labels: (B,) 真实标签

        Returns:
            loss: 标量损失
        """
        loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        return loss


class FocalLoss(nn.Module):
    """Focal Loss（适用于类别不平衡）

    Args:
        alpha: 类别权重
        gamma: 聚焦参数
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C)
            labels: (B,)

        Returns:
            loss: 标量
        """
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()
