"""记忆损失"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryLoss(nn.Module):
    """记忆一致性损失

    鼓励高可靠性样本的阶段嵌入接近确认记忆

    Args:
        margin: 可靠性阈值
    """

    def __init__(self, margin: float = 0.8):
        super().__init__()
        self.margin = margin

    def forward(self,
                embeddings: torch.Tensor,
                phase_assignment: torch.Tensor,
                confirmed_memory: torch.Tensor,
                reliability: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, N, d) 局部嵌入
            phase_assignment: (B, N, C, K) 阶段分配
            confirmed_memory: (C, K, d) 确认记忆
            reliability: (B, N, C, K) 可靠性
            labels: (B,) 真实标签

        Returns:
            loss: 标量损失
        """
        B, N, d = embeddings.shape
        _, _, C, K = phase_assignment.shape

        loss = 0.0
        count = 0

        for b in range(B):
            y = labels[b].item()

            for n in range(N):
                # 当前嵌入
                h = embeddings[b, n]  # (d,)

                # 阶段分配和可靠性
                q = phase_assignment[b, n, y, :]  # (K,)
                rel = reliability[b, n, y, :]  # (K,)

                # 找到高可靠性的阶段
                high_rel_mask = rel > self.margin

                if high_rel_mask.any():
                    # 加权记忆
                    weights = q * high_rel_mask.float()
                    weights = weights / (weights.sum() + 1e-8)

                    # 期望记忆
                    memory_y = confirmed_memory[y]  # (K, d)
                    expected_memory = (weights.unsqueeze(-1) * memory_y).sum(dim=0)  # (d,)

                    # MSE损失
                    loss += F.mse_loss(h, expected_memory)
                    count += 1

        if count > 0:
            return loss / count
        else:
            return torch.tensor(0.0, device=embeddings.device)


class MemoryDriftRegularization(nn.Module):
    """记忆漂移正则化

    惩罚记忆的过度变化
    """

    def __init__(self):
        super().__init__()
        self.prev_memory = None

    def forward(self, confirmed_memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            confirmed_memory: (C, K, d) 当前记忆

        Returns:
            loss: 标量
        """
        if self.prev_memory is None:
            self.prev_memory = confirmed_memory.detach().clone()
            return torch.tensor(0.0, device=confirmed_memory.device)

        # 计算变化量
        drift = torch.norm(confirmed_memory - self.prev_memory, dim=-1)  # (C, K)
        loss = drift.mean()

        # 更新记录
        self.prev_memory = confirmed_memory.detach().clone()

        return loss
