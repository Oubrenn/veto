"""不确定性估计模块"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyEstimator(nn.Module):
    """阶段分配和转移的不确定性估计

    通过以下指标评估不确定性：
    1. 分配熵：阶段分配的熵
    2. 转移残差：相邻窗口阶段转移是否符合类别转移矩阵
    """

    def __init__(self, entropy_weight: float = 1.0, transition_weight: float = 1.0):
        """
        Args:
            entropy_weight: 熵权重
            transition_weight: 转移残差权重
        """
        super().__init__()
        self.entropy_weight = entropy_weight
        self.transition_weight = transition_weight

    def compute_assignment_entropy(self, phase_assignment: torch.Tensor) -> torch.Tensor:
        """计算阶段分配熵

        Args:
            phase_assignment: (B, N, C, K) 阶段分配概率

        Returns:
            entropy: (B, N, C) 每个窗口在每个类别下的分配熵
        """
        # H(q) = -sum(q * log(q))
        log_q = torch.log(phase_assignment + 1e-10)
        entropy = -(phase_assignment * log_q).sum(dim=-1)  # (B, N, C)

        return entropy

    def compute_transition_residual(self,
                                   phase_assignment: torch.Tensor,
                                   transition_matrices: torch.Tensor) -> torch.Tensor:
        """计算转移残差

        衡量相邻窗口的阶段转移是否符合类别转移矩阵

        Args:
            phase_assignment: (B, N, C, K) 阶段分配
            transition_matrices: (C, K, K) 类别转移矩阵

        Returns:
            residual: (B, N-1, C) 转移残差
        """
        B, N, C, K = phase_assignment.shape

        if N < 2:
            return torch.zeros(B, 0, C, device=phase_assignment.device)

        # 相邻时刻的阶段分配
        q_prev = phase_assignment[:, :-1, :, :]  # (B, N-1, C, K)
        q_curr = phase_assignment[:, 1:, :, :]   # (B, N-1, C, K)

        # 期望转移：q_{t-1} @ A_y
        # (B, N-1, C, K) @ (C, K, K) -> (B, N-1, C, K)
        expected_transition = torch.einsum('bncj,cjk->bnck', q_prev, transition_matrices)

        # 转移残差：||q_t - q_{t-1} @ A_y||
        residual = torch.norm(q_curr - expected_transition, dim=-1)  # (B, N-1, C)

        return residual

    def forward(
        self,
        phase_assignment: torch.Tensor,
        transition_matrices: torch.Tensor,
        window_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """前向传播

        Args:
            phase_assignment: (B, N, C, K)
            transition_matrices: (C, K, K)

        Returns:
            output: {
                'assignment_entropy': (B, N, C),
                'transition_residual': (B, N-1, C),
                'uncertainty_score': (B, C) 总体不确定性
            }
        """
        # 计算分配熵
        entropy = self.compute_assignment_entropy(phase_assignment)  # (B, N, C)

        # 计算转移残差
        trans_residual = self.compute_transition_residual(
            phase_assignment, transition_matrices
        )  # (B, N-1, C)

        # 总体不确定性（平均）
        if window_mask is None:
            window_mask = torch.ones(
                phase_assignment.shape[:2],
                dtype=torch.bool,
                device=phase_assignment.device,
            )
        elif window_mask.shape != phase_assignment.shape[:2]:
            raise ValueError("window_mask must match phase_assignment[:2]")
        else:
            window_mask = window_mask.to(
                device=phase_assignment.device,
                dtype=torch.bool,
            )

        entropy_weights = window_mask.to(entropy.dtype).unsqueeze(-1)
        mean_entropy = (entropy * entropy_weights).sum(dim=1) / entropy_weights.sum(
            dim=1
        ).clamp_min(1.0)

        if trans_residual.shape[1] > 0:
            transition_mask = window_mask[:, :-1] & window_mask[:, 1:]
            transition_weights = transition_mask.to(trans_residual.dtype).unsqueeze(-1)
            mean_trans_residual = (
                trans_residual * transition_weights
            ).sum(dim=1) / transition_weights.sum(dim=1).clamp_min(1.0)
        else:
            mean_trans_residual = torch.zeros_like(mean_entropy)

        uncertainty_score = (
            self.entropy_weight * mean_entropy +
            self.transition_weight * mean_trans_residual
        )  # (B, C)

        return {
            'assignment_entropy': entropy,
            'transition_residual': trans_residual,
            'uncertainty_score': uncertainty_score
        }


class ConfidenceEstimator(nn.Module):
    """置信度估计（不确定性的补集）"""

    def __init__(self):
        super().__init__()
        self.uncertainty_estimator = UncertaintyEstimator()

    def forward(self,
                phase_assignment: torch.Tensor,
                transition_matrices: torch.Tensor) -> dict:
        """计算置信度

        Returns:
            output: {
                'confidence_score': (B, C) 置信度分数
            }
        """
        uncertainty_output = self.uncertainty_estimator(
            phase_assignment, transition_matrices
        )

        # 置信度 = 1 / (1 + uncertainty)
        confidence_score = 1.0 / (1.0 + uncertainty_output['uncertainty_score'])

        return {
            'confidence_score': confidence_score,
            **uncertainty_output
        }
