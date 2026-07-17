"""InceptionTime编码器"""
import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    """Inception模块

    多尺度卷积核并行处理
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes=[9, 19, 39], bottleneck: int = 32):
        super().__init__()

        self.bottleneck = nn.Conv1d(in_channels, bottleneck, kernel_size=1) if bottleneck else None

        # 多尺度卷积分支
        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck if bottleneck else in_channels,
                     out_channels,
                     kernel_size=k,
                     padding=k // 2)
            for k in kernel_sizes
        ])

        # MaxPooling分支
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (B, C, L)

        Returns:
            out: (B, C*4, L)
        """
        if self.bottleneck:
            x_bottleneck = self.bottleneck(x)
        else:
            x_bottleneck = x

        # 多尺度卷积
        conv_outputs = [conv(x_bottleneck) for conv in self.convs]

        # MaxPooling分支
        maxpool_output = self.maxpool_conv(x)

        # 拼接
        out = torch.cat(conv_outputs + [maxpool_output], dim=1)

        out = self.bn(out)
        out = self.relu(out)

        return out


class InceptionTime(nn.Module):
    """InceptionTime编码器

    Args:
        in_channels: 输入通道数
        embed_dim: 输出嵌入维度
        n_inception_modules: Inception模块数量
        inception_channels: Inception模块输出通道数
    """

    def __init__(self,
                 in_channels: int,
                 embed_dim: int = 128,
                 n_inception_modules: int = 3,
                 inception_channels: int = 32):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Inception模块堆叠
        modules = []
        current_channels = in_channels

        for i in range(n_inception_modules):
            modules.append(
                InceptionModule(
                    in_channels=current_channels,
                    out_channels=inception_channels,
                    bottleneck=32 if i > 0 else None
                )
            )
            current_channels = inception_channels * 4

        self.inception_blocks = nn.Sequential(*modules)

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
            # (B, N, L, C) -> (B*N, L, C)
            B, N, L, C = x.shape
            x = x.reshape(B * N, L, C)
            batch_mode = True
        else:
            batch_mode = False

        # (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)

        # Inception blocks
        x = self.inception_blocks(x)  # (B, C', L)

        # 全局池化
        x = self.gap(x).squeeze(-1)  # (B, C')

        # 投影
        embeddings = self.projection(x)  # (B, d)

        if batch_mode:
            embeddings = embeddings.reshape(B, N, -1)

        return embeddings
