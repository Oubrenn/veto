"""转移损失"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransitionLoss(nn.Module):
    """转移一致性损失

    鼓励相邻窗口的阶段转移符合类别转移矩阵

    Args:
        loss_type: 'mse' 或 'kl'
    """

    def __init__(self, loss_type: str = 'mse'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self,
                phase_assignment: torch.Tensor,
                transition_matrices: torch.Tensor,
                labels: torch.Tensor,
                window_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            phase_assignment: (B, N, C, K) 阶段分配
            transition_matrices: (C, K, K) 转移矩阵
            labels: (B,) 真实标签

        Returns:
            loss: 标量损失
        """
        B, N, C, K = phase_assignment.shape

        if N < 2:
            return torch.tensor(0.0, device=phase_assignment.device)

        if window_mask is None:
            window_mask = torch.ones(
                B,
                N,
                dtype=torch.bool,
                device=phase_assignment.device,
            )
        elif window_mask.shape != (B, N):
            raise ValueError(
                f"window_mask must have shape {(B, N)}, got {tuple(window_mask.shape)}"
            )
        else:
            window_mask = window_mask.to(
                device=phase_assignment.device,
                dtype=torch.bool,
            )

        # 只计算真实类别的转移损失
        loss = phase_assignment.new_zeros(())
        for b in range(B):
            y = labels[b].item()

            # 当前样本的阶段分配
            q = phase_assignment[b, :, y, :]  # (N, K)

            # 类别转移矩阵
            A = transition_matrices[y]  # (K, K)

            # 相邻时刻
            valid_transitions = window_mask[b, :-1] & window_mask[b, 1:]
            if not bool(valid_transitions.any()):
                continue
            q_prev = q[:-1][valid_transitions]  # (N_valid-1, K)
            q_curr = q[1:][valid_transitions]   # (N_valid-1, K)

            # 期望转移: q_{t-1} @ A
            expected_transition = q_prev @ A  # (N-1, K)

            # 损失
            if self.loss_type == 'mse':
                loss += F.mse_loss(q_curr, expected_transition)
            elif self.loss_type == 'kl':
                # KL散度
                loss += F.kl_div(
                    torch.log(q_curr + 1e-10),
                    expected_transition,
                    reduction='batchmean'
                )
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss / B


class TransitionRegularizationLoss(nn.Module):
    """转移矩阵正则化损失

    鼓励转移矩阵具有特定结构（如稀疏性、平滑性）
    """

    def __init__(self, sparsity: float = 0.1, smoothness: float = 0.1):
        """
        Args:
            sparsity: 稀疏性系数（L1正则）
            smoothness: 平滑性系数（鼓励对角占优）
        """
        super().__init__()
        self.sparsity = sparsity
        self.smoothness = smoothness

    def forward(self, transition_matrices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            transition_matrices: (C, K, K)

        Returns:
            loss: 标量
        """
        loss = 0.0

        # 稀疏性：L1正则
        if self.sparsity > 0:
            sparsity_loss = torch.mean(torch.abs(transition_matrices))
            loss += self.sparsity * sparsity_loss

        # 平滑性：鼓励对角元素大
        if self.smoothness > 0:
            C, K, _ = transition_matrices.shape
            diag_elements = torch.diagonal(transition_matrices, dim1=1, dim2=2)  # (C, K)
            smoothness_loss = -torch.mean(diag_elements)  # 对角越大越好
            loss += self.smoothness * smoothness_loss

        return loss
