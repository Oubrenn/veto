"""反事实损失"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CounterfactualLoss(nn.Module):
    """反事实阶段路径损失

    要求真实路径的得分高于反事实路径

    Args:
        margin: 间隔参数
        use_hinge: 是否使用hinge loss
    """

    def __init__(self, margin: float = 1.0, use_hinge: bool = True):
        super().__init__()
        self.margin = margin
        self.use_hinge = use_hinge

    def forward(self,
                real_logits: torch.Tensor,
                cf_logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            real_logits: (B, C) 真实样本的分类logits
            cf_logits: (B, C) 反事实样本的分类logits
            labels: (B,) 真实标签

        Returns:
            loss: 标量损失
        """
        B = real_logits.shape[0]

        # 提取真实类别的得分
        real_scores = real_logits[torch.arange(B), labels]  # (B,)
        cf_scores = cf_logits[torch.arange(B), labels]  # (B,)

        # Margin loss: 要求 real_score > cf_score + margin
        if self.use_hinge:
            # Hinge loss: max(0, margin - (real_score - cf_score))
            loss = F.relu(self.margin - (real_scores - cf_scores))
        else:
            # 指数损失: exp(cf_score - real_score + margin)
            loss = torch.exp(cf_scores - real_scores + self.margin)

        return loss.mean()


class ContrastiveCounterfactualLoss(nn.Module):
    """对比式反事实损失

    使用对比学习的思想，拉近真实样本与正类，推远反事实样本
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self,
                real_logits: torch.Tensor,
                cf_logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            real_logits: (B, C)
            cf_logits: (B, C)
            labels: (B,)

        Returns:
            loss: 标量
        """
        B, C = real_logits.shape

        # 对于每个样本，真实标签是正样本，反事实是负样本
        real_scores = real_logits[torch.arange(B), labels]  # (B,)
        cf_scores = cf_logits[torch.arange(B), labels]  # (B,)

        # 对比损失: -log(exp(real) / (exp(real) + exp(cf)))
        real_exp = torch.exp(real_scores / self.temperature)
        cf_exp = torch.exp(cf_scores / self.temperature)

        loss = -torch.log(real_exp / (real_exp + cf_exp + 1e-8))

        return loss.mean()
