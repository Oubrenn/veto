"""Trainable baseline classifiers used for UEA comparison experiments.

These classes provide lightweight in-repo implementations with the same
``{"logits": tensor}`` output contract as :class:`PhasePathNet`. They are meant
for reproducible local baselines when external reported numbers are unavailable.
They are not drop-in copies of the original papers' codebases.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import InceptionTime


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalCNNClassifier(nn.Module):
    """Compact CNN baseline for generic multivariate time-series classification."""

    def __init__(self, n_channels: int, n_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(n_channels, hidden_dim, 7),
            ConvBlock(hidden_dim, hidden_dim, 5, dilation=2),
            ConvBlock(hidden_dim, hidden_dim, 3, dilation=4),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        x = x.transpose(1, 2)
        h = self.encoder(x).mean(dim=-1)
        return {"logits": self.classifier(h)}


class InceptionTimeClassifier(nn.Module):
    """In-repository InceptionTime classifier wrapper."""

    def __init__(self, n_channels: int, n_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = InceptionTime(n_channels, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        h = self.encoder(x)
        return {"logits": self.classifier(h)}


class RandomConvClassifier(nn.Module):
    """ROCKET/MultiRocket-style fixed random convolution feature classifier."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        hidden_dim: int = 128,
        n_kernels: int = 96,
        multiscale: bool = False,
    ):
        super().__init__()
        kernel_sizes = [7, 9, 11] if multiscale else [9]
        dilations = [1, 2, 4, 8] if multiscale else [1, 2, 4]
        specs = []
        for idx in range(n_kernels):
            specs.append((kernel_sizes[idx % len(kernel_sizes)], dilations[idx % len(dilations)]))
        self.specs = specs
        for idx, (kernel_size, _dilation) in enumerate(specs):
            weight = torch.randn(1, n_channels, kernel_size) / math.sqrt(n_channels * kernel_size)
            bias = torch.empty(1).uniform_(-1.0, 1.0)
            self.register_buffer(f"weight_{idx}", weight)
            self.register_buffer(f"bias_{idx}", bias)
        feature_dim = n_kernels * 4
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, n_classes),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        x = x.transpose(1, 2)
        features = []
        for idx, (kernel_size, dilation) in enumerate(self.specs):
            weight = getattr(self, f"weight_{idx}")
            bias = getattr(self, f"bias_{idx}")
            padding = dilation * (kernel_size // 2)
            y = F.conv1d(x, weight, bias=bias, padding=padding, dilation=dilation)
            features.extend(
                [
                    y.mean(dim=-1),
                    y.std(dim=-1),
                    y.amax(dim=-1),
                    (y > 0).float().mean(dim=-1),
                ]
            )
        return {"logits": self.classifier(torch.cat(features, dim=-1))}


class TimesBlock(nn.Module):
    """Small TimesNet-style 2D temporal-variation block.

    The block folds a sequence into a period grid, applies 2D convolutions, and
    unfolds it back to the original temporal length.
    """

    def __init__(self, channels: int, period: int):
        super().__init__()
        self.period = period
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, length = x.shape
        pad = (self.period - length % self.period) % self.period
        if pad:
            x_pad = F.pad(x, (0, pad), mode="replicate")
        else:
            x_pad = x
        folded = x_pad.reshape(bsz, channels, -1, self.period)
        y = self.conv(folded).reshape(bsz, channels, -1)[..., :length]
        return F.gelu(x + y)


class TimesNetClassifier(nn.Module):
    """Lightweight TimesNet-inspired classifier."""

    def __init__(self, n_channels: int, n_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.input_proj = nn.Conv1d(n_channels, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([TimesBlock(hidden_dim, period) for period in (2, 4, 8)])
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        h = self.input_proj(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        h = self.norm(h).mean(dim=-1)
        return {"logits": self.classifier(h)}


class TapNetClassifier(nn.Module):
    """Prototype-distance baseline inspired by TapNet-style class anchors."""

    def __init__(self, n_channels: int, n_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(n_channels, hidden_dim, 5),
            ConvBlock(hidden_dim, hidden_dim, 3),
            ConvBlock(hidden_dim, hidden_dim, 3),
        )
        self.prototypes = nn.Parameter(torch.randn(n_classes, hidden_dim) * 0.02)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        h = self.encoder(x.transpose(1, 2)).mean(dim=-1)
        h = F.normalize(h, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        logits = self.scale.exp().clamp(max=100.0) * h @ prototypes.t()
        return {"logits": logits}


class PDFTimeClassifier(nn.Module):
    """Prototype-distribution baseline inspired by PDFTime."""

    def __init__(self, n_channels: int, n_classes: int, hidden_dim: int = 128, n_prototypes: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(n_channels, hidden_dim, 5),
            ConvBlock(hidden_dim, hidden_dim, 3),
        )
        self.prototypes = nn.Parameter(torch.randn(n_classes, n_prototypes, hidden_dim) * 0.02)
        self.local_head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        h_seq = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        pooled = torch.stack(
            [
                h_seq.mean(dim=1),
                F.adaptive_avg_pool1d(h_seq.transpose(1, 2), 4).transpose(1, 2).mean(dim=1),
                F.adaptive_max_pool1d(h_seq.transpose(1, 2), 4).transpose(1, 2).mean(dim=1),
            ],
            dim=1,
        ).mean(dim=1)
        local_logits = self.local_head(pooled)
        dist = torch.cdist(pooled.unsqueeze(1), self.prototypes.reshape(-1, pooled.shape[-1]))
        proto_score = -dist.reshape(pooled.shape[0], -1, self.prototypes.shape[1]).mean(dim=-1)
        return {"logits": local_logits + proto_score}


class GraphTemporalClassifier(nn.Module):
    """Channel-graph baseline for MTS2Graph/SimTSC-style comparisons."""

    def __init__(self, n_channels: int, n_classes: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.temporal = nn.Sequential(
            ConvBlock(n_channels, hidden_dim, 7),
            ConvBlock(hidden_dim, hidden_dim, 5),
        )
        self.adj_logits = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        h = self.temporal(x.transpose(1, 2)).mean(dim=-1)
        adj = torch.softmax(self.adj_logits, dim=-1)
        for layer in self.layers:
            h = F.gelu(layer(h @ adj))
        return {"logits": self.classifier(h)}


class MotifGraphAttentionClassifier(nn.Module):
    """Motif-aware graph-attention baseline for TMA-GAT-style experiments."""

    def __init__(self, n_channels: int, n_classes: int, hidden_dim: int = 128, n_heads: int = 4):
        super().__init__()
        self.motif_encoder = nn.Sequential(
            ConvBlock(n_channels, hidden_dim, 3),
            ConvBlock(hidden_dim, hidden_dim, 5),
            ConvBlock(hidden_dim, hidden_dim, 7),
        )
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.max_tokens = 512

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        motifs = self.motif_encoder(x.transpose(1, 2)).transpose(1, 2)
        if motifs.shape[1] > self.max_tokens:
            motifs = F.adaptive_avg_pool1d(motifs.transpose(1, 2), self.max_tokens).transpose(1, 2)
        attn_out, _ = self.attn(motifs, motifs, motifs, need_weights=False)
        h = self.norm(motifs + attn_out).mean(dim=1)
        return {"logits": self.classifier(h)}


class SimTSCClassifier(nn.Module):
    """Similarity-aware temporal classifier with learned anchors."""

    def __init__(self, n_channels: int, n_classes: int, hidden_dim: int = 128, n_anchors: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(n_channels, hidden_dim, 7),
            ConvBlock(hidden_dim, hidden_dim, 5),
        )
        self.anchors = nn.Parameter(torch.randn(n_anchors, hidden_dim) / math.sqrt(hidden_dim))
        self.classifier = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        h = self.encoder(x.transpose(1, 2)).mean(dim=-1)
        sim = torch.softmax(h @ self.anchors.t() / math.sqrt(h.shape[-1]), dim=-1)
        context = sim @ self.anchors
        return {"logits": self.classifier(torch.cat([h, context], dim=-1))}


class HC2LiteClassifier(nn.Module):
    """Small neural ensemble proxy used only when original HC2 values are unavailable."""

    def __init__(self, n_channels: int, n_classes: int, hidden_dim: int = 128):
        super().__init__()
        branch_dim = max(32, hidden_dim // 2)
        self.cnn = TemporalCNNClassifier(n_channels, n_classes, branch_dim)
        self.inception = InceptionTimeClassifier(n_channels, n_classes, branch_dim)
        self.prototype = TapNetClassifier(n_channels, n_classes, branch_dim)
        self.mixer = nn.Linear(n_classes * 3, n_classes)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        logits = torch.cat(
            [
                self.cnn(x)["logits"],
                self.inception(x)["logits"],
                self.prototype(x)["logits"],
            ],
            dim=-1,
        )
        return {"logits": self.mixer(logits)}


BASELINE_MODELS = {
    "rocket": lambda n_channels, n_classes, hidden_dim: RandomConvClassifier(
        n_channels, n_classes, hidden_dim, n_kernels=64, multiscale=False
    ),
    "multirocket": lambda n_channels, n_classes, hidden_dim: RandomConvClassifier(
        n_channels, n_classes, hidden_dim, n_kernels=128, multiscale=True
    ),
    "inceptiontime": InceptionTimeClassifier,
    "timesnet": TimesNetClassifier,
    "tapnet": TapNetClassifier,
    "pdftime": PDFTimeClassifier,
    "hc2_lite": HC2LiteClassifier,
    "mts2graph": GraphTemporalClassifier,
    "simtsc": SimTSCClassifier,
    "tma_gat": MotifGraphAttentionClassifier,
    "temporal_cnn": TemporalCNNClassifier,
}


def build_baseline_model(name: str, n_channels: int, n_classes: int, hidden_dim: int = 128):
    try:
        cls = BASELINE_MODELS[name]
    except KeyError as exc:
        available = ", ".join(sorted(BASELINE_MODELS))
        raise ValueError(f"Unknown baseline model '{name}'. Available: {available}") from exc
    return cls(n_channels=n_channels, n_classes=n_classes, hidden_dim=hidden_dim)
