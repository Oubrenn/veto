"""阶段分配模块"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseAssignment(nn.Module):
    """类别条件阶段分配

    基于模板距离和子空间残差，计算每个窗口属于不同阶段的概率

    Args:
        temperature: softmax温度参数
        distance_weight: 模板距离权重
        residual_weight: 子空间残差权重
    """

    def __init__(self,
                 temperature: float = 1.0,
                 distance_weight: float = 1.0,
                 residual_weight: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.distance_weight = distance_weight
        self.residual_weight = residual_weight

    def compute_assignment(self,
                          template_dist: torch.Tensor,
                          subspace_residual: torch.Tensor) -> torch.Tensor:
        """计算阶段分配概率

        Args:
            template_dist: (B, N, C, K) 模板距离
            subspace_residual: (B, N, C, K) 子空间残差

        Returns:
            q: (B, N, C, K) 阶段分配概率
        """
        # 综合距离度量（距离越小越好）
        combined_dist = (
            self.distance_weight * template_dist +
            self.residual_weight * subspace_residual
        )

        # 转换为相似度（取负）
        similarity = -combined_dist

        # Softmax归一化
        q = F.softmax(similarity / self.temperature, dim=-1)  # (B, N, C, K)

        return q

    def forward(self,
                template_dist: torch.Tensor,
                subspace_residual: torch.Tensor) -> dict:
        """前向传播

        Args:
            template_dist: (B, N, C, K)
            subspace_residual: (B, N, C, K)

        Returns:
            output: {
                'phase_assignment': (B, N, C, K) 阶段分配概率
            }
        """
        q = self.compute_assignment(template_dist, subspace_residual)

        return {
            'phase_assignment': q
        }


class ContextAwarePhaseAssignment(PhaseAssignment):
    """上下文感知阶段分配

    考虑相邻窗口的阶段分配，增强时序一致性
    """

    def __init__(self,
                 embed_dim: int,
                 n_phases: int,
                 temperature: float = 1.0,
                 distance_weight: float = 1.0,
                 residual_weight: float = 1.0,
                 context_weight: float = 0.1):
        super().__init__(temperature, distance_weight, residual_weight)
        self.context_weight = context_weight

        # 上下文调制网络
        self.context_net = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),  # 前-当前-后
            nn.ReLU(),
            nn.Linear(embed_dim, n_phases)
        )

    def compute_assignment_with_context(self,
                                       embeddings: torch.Tensor,
                                       template_dist: torch.Tensor,
                                       subspace_residual: torch.Tensor) -> torch.Tensor:
        """基于上下文的阶段分配

        Args:
            embeddings: (B, N, d)
            template_dist: (B, N, C, K)
            subspace_residual: (B, N, C, K)

        Returns:
            q: (B, N, C, K)
        """
        B, N, d = embeddings.shape

        # 基础分配
        q_base = self.compute_assignment(template_dist, subspace_residual)  # (B, N, C, K)

        # 构造上下文（前一个、当前、后一个）
        embeddings_prev = F.pad(embeddings[:, :-1], (0, 0, 1, 0))  # (B, N, d)
        embeddings_next = F.pad(embeddings[:, 1:], (0, 0, 0, 1))  # (B, N, d)

        context = torch.cat([embeddings_prev, embeddings, embeddings_next], dim=-1)  # (B, N, 3d)

        # 上下文调制
        context_logits = self.context_net(context)  # (B, N, K)
        context_prob = F.softmax(context_logits, dim=-1)  # (B, N, K)

        # 结合上下文（对所有类别使用相同的上下文先验）
        context_prob_exp = context_prob.unsqueeze(2)  # (B, N, 1, K)

        q = (1 - self.context_weight) * q_base + self.context_weight * context_prob_exp

        return q

    def forward(self,
                embeddings: torch.Tensor,
                template_dist: torch.Tensor,
                subspace_residual: torch.Tensor) -> dict:
        """前向传播

        Args:
            embeddings: (B, N, d)
            template_dist: (B, N, C, K)
            subspace_residual: (B, N, C, K)

        Returns:
            output: {
                'phase_assignment': (B, N, C, K)
            }
        """
        q = self.compute_assignment_with_context(embeddings, template_dist, subspace_residual)

        return {
            'phase_assignment': q
        }
