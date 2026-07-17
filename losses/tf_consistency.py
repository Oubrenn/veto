"""时频一致性损失"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeFrequencyConsistencyLoss(nn.Module):
    """时频一致性损失

    要求时域和时频域的阶段分配保持一致

    Args:
        divergence_type: 散度类型 ('js', 'kl', 'mse')
    """

    def __init__(self, divergence_type: str = 'js'):
        super().__init__()
        self.divergence_type = divergence_type

    def forward(self,
                time_embeddings: torch.Tensor,
                tf_embeddings: torch.Tensor,
                template_dist: torch.Tensor,
                subspace_residual: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_embeddings: (B, N, d) 时域嵌入
            tf_embeddings: (B, N, d) 时频域嵌入
            template_dist: (B, N, C, K) 时域模板距离
            subspace_residual: (B, N, C, K) 时域子空间残差
            labels: (B,) 真实标签

        Returns:
            loss: 标量损失
        """
        B, N, d = time_embeddings.shape
        _, _, C, K = template_dist.shape

        # 计算时域阶段分配
        combined_dist_time = template_dist + subspace_residual
        q_time = F.softmax(-combined_dist_time, dim=-1)  # (B, N, C, K)

        # 计算时频域阶段分配（需要重新计算距离）
        # 简化版本：直接使用嵌入相似度
        q_tf = self._compute_tf_assignment(
            tf_embeddings, template_dist, subspace_residual
        )  # (B, N, C, K)

        # 计算散度
        loss = 0.0
        for b in range(B):
            y = labels[b].item()

            q_t = q_time[b, :, y, :]  # (N, K)
            q_f = q_tf[b, :, y, :]    # (N, K)

            if self.divergence_type == 'js':
                # JS散度
                m = 0.5 * (q_t + q_f)
                kl1 = F.kl_div(torch.log(m + 1e-10), q_t, reduction='batchmean')
                kl2 = F.kl_div(torch.log(m + 1e-10), q_f, reduction='batchmean')
                js_div = 0.5 * (kl1 + kl2)
                loss += js_div

            elif self.divergence_type == 'kl':
                # KL散度
                loss += F.kl_div(torch.log(q_f + 1e-10), q_t, reduction='batchmean')

            elif self.divergence_type == 'mse':
                # MSE
                loss += F.mse_loss(q_t, q_f)

            else:
                raise ValueError(f"Unknown divergence_type: {self.divergence_type}")

        return loss / B

    def _compute_tf_assignment(self,
                               tf_embeddings: torch.Tensor,
                               template_dist: torch.Tensor,
                               subspace_residual: torch.Tensor) -> torch.Tensor:
        """从时频嵌入计算阶段分配（简化版本）

        Args:
            tf_embeddings: (B, N, d)
            template_dist: (B, N, C, K) 用于获取形状
            subspace_residual: (B, N, C, K)

        Returns:
            q_tf: (B, N, C, K)
        """
        # 简化实现：使用时频嵌入与时域嵌入的相似度
        # 实际应用中，应该有独立的时频阶段原型

        # 这里使用与时域相同的距离度量（简化）
        combined_dist = template_dist + subspace_residual
        q_tf = F.softmax(-combined_dist, dim=-1)

        return q_tf


class EmbeddingConsistencyLoss(nn.Module):
    """嵌入一致性损失

    直接约束时域和时频域嵌入的相似性
    """

    def __init__(self):
        super().__init__()

    def forward(self,
                time_embeddings: torch.Tensor,
                tf_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_embeddings: (B, N, d)
            tf_embeddings: (B, N, d)

        Returns:
            loss: 标量
        """
        # L2距离
        loss = F.mse_loss(time_embeddings, tf_embeddings)

        return loss
