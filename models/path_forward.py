"""路径前向传播模块（类似HMM forward算法）"""
from typing import Optional

import torch
import torch.nn as nn


class PathForward(nn.Module):
    """可微路径前向传播

    类似HMM的forward算法，计算整条阶段序列在某个类别路径下的概率

    给定:
    - 初始分布 pi_y: (K,)
    - 转移矩阵 A_y: (K, K)
    - 阶段分配 q_t: (N, K)

    计算路径概率
    """

    def __init__(self, log_space: bool = True):
        """
        Args:
            log_space: 是否在log空间计算（数值稳定）
        """
        super().__init__()
        self.log_space = log_space

    def forward(
        self,
        init_dist: torch.Tensor,
        transition_matrix: torch.Tensor,
        phase_assignment: torch.Tensor,
        window_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播计算路径概率

        Args:
            init_dist: (B, K) 或 (K,) 初始阶段分布
            transition_matrix: (B, K, K) 或 (K, K) 转移矩阵
            phase_assignment: (B, N, K) 阶段分配概率

        Returns:
            path_prob: (B,) 路径概率
        """
        B, N, K = phase_assignment.shape
        if N == 0:
            raise ValueError("phase_assignment must contain at least one window")

        if window_mask is None:
            window_mask = torch.ones(
                B,
                N,
                dtype=torch.bool,
                device=phase_assignment.device,
            )
        else:
            if window_mask.shape != (B, N):
                raise ValueError(
                    f"window_mask must have shape {(B, N)}, got {tuple(window_mask.shape)}"
                )
            window_mask = window_mask.to(device=phase_assignment.device, dtype=torch.bool)
            if not bool(window_mask[:, 0].all()):
                raise ValueError("Each sample must contain at least one valid window")
            if N > 1 and bool(((~window_mask[:, :-1]) & window_mask[:, 1:]).any()):
                raise ValueError("window_mask must be a contiguous prefix for each sample")

        # 确保维度匹配
        if init_dist.ndim == 1:
            init_dist = init_dist.unsqueeze(0).expand(B, -1)  # (B, K)

        if transition_matrix.ndim == 2:
            transition_matrix = transition_matrix.unsqueeze(0).expand(B, -1, -1)  # (B, K, K)

        if self.log_space:
            return self._forward_log(
                init_dist,
                transition_matrix,
                phase_assignment,
                window_mask,
            )
        else:
            return self._forward_normal(
                init_dist,
                transition_matrix,
                phase_assignment,
                window_mask,
            )

    def _forward_normal(
        self,
        init_dist: torch.Tensor,
        transition_matrix: torch.Tensor,
        phase_assignment: torch.Tensor,
        window_mask: torch.Tensor,
    ) -> torch.Tensor:
        """正常空间的forward算法

        Args:
            init_dist: (B, K)
            transition_matrix: (B, K, K)
            phase_assignment: (B, N, K)

        Returns:
            path_prob: (B,)
        """
        B, N, K = phase_assignment.shape

        # 初始化 alpha_0 = pi * q_0
        alpha = init_dist * phase_assignment[:, 0, :]  # (B, K)

        # 递推 alpha_t = (alpha_{t-1} @ A) * q_t
        for t in range(1, N):
            # alpha_{t-1} @ A: (B, K) @ (B, K, K) -> (B, K)
            alpha_next = torch.bmm(alpha.unsqueeze(1), transition_matrix).squeeze(1)

            # 乘以当前观测概率
            alpha_next = alpha_next * phase_assignment[:, t, :]
            alpha = torch.where(window_mask[:, t].unsqueeze(-1), alpha_next, alpha)

        # 总概率：sum over all final states
        path_prob = alpha.sum(dim=-1)  # (B,)

        return path_prob

    def _forward_log(
        self,
        init_dist: torch.Tensor,
        transition_matrix: torch.Tensor,
        phase_assignment: torch.Tensor,
        window_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Log空间的forward算法（数值稳定）

        Args:
            init_dist: (B, K)
            transition_matrix: (B, K, K)
            phase_assignment: (B, N, K)

        Returns:
            log_path_prob: (B,)
        """
        B, N, K = phase_assignment.shape

        # 转换到log空间
        log_init = torch.log(init_dist + 1e-10)  # (B, K)
        log_trans = torch.log(transition_matrix + 1e-10)  # (B, K, K)
        log_emission = torch.log(phase_assignment + 1e-10)  # (B, N, K)

        # 初始化 log_alpha_0 = log(pi) + log(q_0)
        log_alpha = log_init + log_emission[:, 0, :]  # (B, K)

        # 递推
        for t in range(1, N):
            # log_alpha_{t-1} + log(A): (B, K, 1) + (B, K, K) -> (B, K, K)
            log_alpha_trans = log_alpha.unsqueeze(2) + log_trans  # (B, K, K)

            # logsumexp over previous states
            log_alpha_next = torch.logsumexp(log_alpha_trans, dim=1)

            # 加上当前观测
            log_alpha_next = log_alpha_next + log_emission[:, t, :]
            log_alpha = torch.where(
                window_mask[:, t].unsqueeze(-1),
                log_alpha_next,
                log_alpha,
            )

        # 总概率：logsumexp over final states
        log_path_prob = torch.logsumexp(log_alpha, dim=-1)  # (B,)

        return log_path_prob


class PathForwardWithViterbi(PathForward):
    """扩展版本：同时支持forward和Viterbi解码"""

    def viterbi_decode(self,
                       init_dist: torch.Tensor,
                       transition_matrix: torch.Tensor,
                       phase_assignment: torch.Tensor) -> torch.Tensor:
        """Viterbi解码：找到最优阶段路径

        Args:
            init_dist: (B, K) 或 (K,)
            transition_matrix: (B, K, K) 或 (K, K)
            phase_assignment: (B, N, K)

        Returns:
            best_path: (B, N) 最优阶段序列
        """
        B, N, K = phase_assignment.shape

        # 确保维度
        if init_dist.ndim == 1:
            init_dist = init_dist.unsqueeze(0).expand(B, -1)

        if transition_matrix.ndim == 2:
            transition_matrix = transition_matrix.unsqueeze(0).expand(B, -1, -1)

        # Log空间
        log_init = torch.log(init_dist + 1e-10)
        log_trans = torch.log(transition_matrix + 1e-10)
        log_emission = torch.log(phase_assignment + 1e-10)

        # 初始化
        log_delta = log_init + log_emission[:, 0, :]  # (B, K)
        backpointers = []

        # 递推
        for t in range(1, N):
            # log_delta + log_trans: (B, K, 1) + (B, K, K) -> (B, K, K)
            log_delta_trans = log_delta.unsqueeze(2) + log_trans  # (B, K, K)

            # 取最大
            log_delta, bp = torch.max(log_delta_trans, dim=1)  # (B, K), (B, K)
            log_delta = log_delta + log_emission[:, t, :]

            backpointers.append(bp)

        # 回溯
        best_path = []
        last_state = torch.argmax(log_delta, dim=-1)  # (B,)
        best_path.append(last_state)

        for bp in reversed(backpointers):
            last_state = bp[torch.arange(B), last_state]
            best_path.append(last_state)

        best_path = torch.stack(list(reversed(best_path)), dim=1)  # (B, N)

        return best_path
