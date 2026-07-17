"""ResNet1D编码器"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """1D残差块"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """1D ResNet编码器

    Args:
        in_channels: 输入通道数
        embed_dim: 输出嵌入维度
        n_blocks: 残差块数量
        base_channels: 基础通道数
    """

    def __init__(self,
                 in_channels: int,
                 embed_dim: int = 128,
                 n_blocks: int = 3,
                 base_channels: int = 64):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # 初始卷积
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 残差块
        blocks = []
        current_channels = base_channels

        for i in range(n_blocks):
            out_channels = base_channels * (2 ** i)
            stride = 2 if i > 0 else 1

            blocks.append(
                ResidualBlock(current_channels, out_channels, stride=stride)
            )
            current_channels = out_channels

        self.res_blocks = nn.Sequential(*blocks)

        # 全局平均池化 + 投影
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(current_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, C) 或 (B, N, L, C)

        Returns:
            embeddings: (B, d) 或 (B, N, d)
        """
        original_shape = x.shape
        if x.ndim == 4:
            B, N, L, C = x.shape
            x = x.reshape(B * N, L, C)
            batch_mode = True
        else:
            batch_mode = False

        # (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)

        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 残差块
        x = self.res_blocks(x)

        # 全局池化
        x = self.gap(x).squeeze(-1)

        # 投影
        embeddings = self.projection(x)

        if batch_mode:
            embeddings = embeddings.reshape(B, N, -1)

        return embeddings
