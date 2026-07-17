"""FCN编码器"""
import torch
import torch.nn as nn


class FCN(nn.Module):
    """全卷积网络编码器

    Args:
        in_channels: 输入通道数
        embed_dim: 输出嵌入维度
    """

    def __init__(self, in_channels: int, embed_dim: int = 128):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=8, padding=4)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool1d(1)

        # 投影到目标维度
        self.projection = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, C) 或 (B, N, L, C)

        Returns:
            embeddings: (B, d) 或 (B, N, d)
        """
        if x.ndim == 4:
            B, N, L, C = x.shape
            x = x.reshape(B * N, L, C)
            batch_mode = True
        else:
            batch_mode = False

        # (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)

        # 卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # 全局池化
        x = self.gap(x).squeeze(-1)  # (B, 128)

        # 投影
        embeddings = self.projection(x)  # (B, d)

        if batch_mode:
            embeddings = embeddings.reshape(B, N, -1)

        return embeddings
