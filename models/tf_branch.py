"""时频分支模块"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeFrequencyBranch(nn.Module):
    """辅助时频一致性分支

    在时频域计算阶段分配，与时域分配保持一致

    Args:
        in_channels: 输入通道数
        embed_dim: 嵌入维度
        n_fft: FFT窗口大小
    """

    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 n_fft: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.n_fft = n_fft

        # 时频特征提取器
        self.tf_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, embed_dim)
        )

    def compute_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """计算时频谱

        Args:
            x: (B, N, L, C) 窗口序列

        Returns:
            spec: (B, N, C, F, T) 时频谱
        """
        B, N, L, C = x.shape

        # 重塑为 (B*N*C, L)
        x_flat = x.permute(0, 1, 3, 2).reshape(-1, L)  # (B*N*C, L)

        # 计算STFT
        spec = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.n_fft // 4,
            win_length=self.n_fft,
            return_complex=True,
            normalized=True
        )  # (B*N*C, F, T)

        # 取幅度谱
        spec = torch.abs(spec)  # (B*N*C, F, T)

        # 重塑回 (B, N, C, F, T)
        F, T = spec.shape[1], spec.shape[2]
        spec = spec.reshape(B, N, C, F, T)

        return spec

    def encode_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        """编码时频谱

        Args:
            spec: (B, N, C, F, T)

        Returns:
            tf_embeddings: (B, N, d)
        """
        B, N, C, F, T = spec.shape

        # 重塑为 (B*N, C, F, T)
        spec_flat = spec.reshape(B * N, C, F, T)

        # 编码
        tf_embeddings = self.tf_encoder(spec_flat)  # (B*N, d)

        # 重塑为 (B, N, d)
        tf_embeddings = tf_embeddings.reshape(B, N, self.embed_dim)

        return tf_embeddings

    def forward(self, windows: torch.Tensor) -> dict:
        """前向传播

        Args:
            windows: (B, N, L, C) 窗口序列

        Returns:
            output: {
                'tf_embeddings': (B, N, d),
                'spectrogram': (B, N, C, F, T)
            }
        """
        # 计算时频谱
        spec = self.compute_spectrogram(windows)  # (B, N, C, F, T)

        # 编码
        tf_embeddings = self.encode_spectrogram(spec)  # (B, N, d)

        return {
            'tf_embeddings': tf_embeddings,
            'spectrogram': spec
        }


class SimpleTimeFrequencyBranch(nn.Module):
    """简化版时频分支（不使用STFT，使用卷积模拟）"""

    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, windows: torch.Tensor) -> dict:
        """
        Args:
            windows: (B, N, L, C)

        Returns:
            output: {'tf_embeddings': (B, N, d)}
        """
        B, N, L, C = windows.shape

        # 重塑为 (B*N, C, L)
        windows_flat = windows.reshape(B * N, L, C).permute(0, 2, 1)

        # 编码
        tf_embeddings = self.encoder(windows_flat)  # (B*N, d)

        # 重塑
        tf_embeddings = tf_embeddings.reshape(B, N, -1)

        return {
            'tf_embeddings': tf_embeddings
        }
