"""Confirmed memory module for phase prototypes."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfirmedMemory(nn.Module):
    """Class-phase memory with either gated confirmation or direct EMA writes."""

    def __init__(
        self,
        n_classes: int,
        n_phases: int,
        embed_dim: int,
        evidence_threshold: int = 5,
        residual_threshold: float = 0.5,
        entropy_threshold: float = 0.3,
        momentum: float = 0.9,
        update_mode: str = "confirmed",
    ):
        super().__init__()
        if update_mode not in {"confirmed", "direct_ema"}:
            raise ValueError("update_mode must be one of: confirmed, direct_ema")
        self.n_classes = n_classes
        self.n_phases = n_phases
        self.embed_dim = embed_dim
        self.evidence_threshold = evidence_threshold
        self.residual_threshold = residual_threshold
        self.entropy_threshold = entropy_threshold
        self.momentum = momentum
        self.update_mode = update_mode

        self.register_buffer("confirmed_memory", torch.randn(n_classes, n_phases, embed_dim))
        self.register_buffer("candidate_buffer", torch.zeros(n_classes, n_phases, embed_dim))
        self.register_buffer("evidence_counter", torch.zeros(n_classes, n_phases, dtype=torch.long))
        self.freeze_memory = False

    def compute_reliability(
        self,
        subspace_residual: torch.Tensor,
        assignment_entropy: torch.Tensor,
        transition_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual_reliability = torch.sigmoid(-subspace_residual + self.residual_threshold)
        entropy_expanded = assignment_entropy.unsqueeze(-1).expand_as(subspace_residual)
        entropy_reliability = torch.sigmoid(-entropy_expanded + self.entropy_threshold)

        if transition_residual is not None and transition_residual.shape[1] > 0:
            trans_padded = F.pad(transition_residual, (0, 0, 1, 0), value=0.0)
            trans_expanded = trans_padded.unsqueeze(-1).expand_as(subspace_residual)
            trans_reliability = torch.sigmoid(-trans_expanded + 0.5)
            return (residual_reliability + entropy_reliability + trans_reliability) / 3.0
        return (residual_reliability + entropy_reliability) / 2.0

    def update_memory(
        self,
        embeddings: torch.Tensor,
        phase_assignment: torch.Tensor,
        reliability: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        window_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if self.freeze_memory or not self.training or class_labels is None:
            return

        _, n_windows, _ = embeddings.shape
        _, _, _, n_phases = phase_assignment.shape
        if window_mask is None:
            window_mask = torch.ones(
                embeddings.shape[:2],
                dtype=torch.bool,
                device=embeddings.device,
            )
        elif window_mask.shape != embeddings.shape[:2]:
            raise ValueError("window_mask must match embeddings[:2]")
        else:
            window_mask = window_mask.to(device=embeddings.device, dtype=torch.bool)

        if self.update_mode == "direct_ema":
            with torch.no_grad():
                for y_item in torch.unique(class_labels).tolist():
                    y = int(y_item)
                    mask = class_labels == y
                    q = phase_assignment[mask, :, y, :]
                    q = q * window_mask[mask].to(q.dtype).unsqueeze(-1)
                    emb = embeddings[mask]
                    denom = q.sum(dim=(0, 1)).clamp_min(1e-8)
                    update = torch.einsum("bnk,bnd->kd", q, emb) / denom.unsqueeze(-1)
                    self.confirmed_memory[y] = (
                        self.momentum * self.confirmed_memory[y]
                        + (1 - self.momentum) * update
                    )
            return

        with torch.no_grad():
            for b in range(embeddings.shape[0]):
                y = int(class_labels[b].item())
                for n in range(n_windows):
                    if not bool(window_mask[b, n]):
                        continue
                    for k in range(n_phases):
                        rel = float(reliability[b, n, y, k].item())
                        if rel > 0.8:
                            self.evidence_counter[y, k] += 1
                            self.candidate_buffer[y, k] = (
                                0.8 * self.candidate_buffer[y, k]
                                + 0.2 * embeddings[b, n]
                            )
                            if self.evidence_counter[y, k] >= self.evidence_threshold:
                                self.confirmed_memory[y, k] = (
                                    self.momentum * self.confirmed_memory[y, k]
                                    + (1 - self.momentum) * self.candidate_buffer[y, k]
                                )
                                self.evidence_counter[y, k] = 0
                        else:
                            self.evidence_counter[y, k] = torch.clamp(
                                self.evidence_counter[y, k] - 1,
                                min=0,
                            )

    def get_memory(self, class_idx: int) -> torch.Tensor:
        return self.confirmed_memory[class_idx]

    def reset_evidence(self, class_idx: Optional[int] = None, phase_idx: Optional[int] = None) -> None:
        if class_idx is None:
            self.evidence_counter.zero_()
        elif phase_idx is None:
            self.evidence_counter[class_idx] = 0
        else:
            self.evidence_counter[class_idx, phase_idx] = 0

    def forward(
        self,
        embeddings: torch.Tensor,
        phase_assignment: torch.Tensor,
        subspace_residual: torch.Tensor,
        assignment_entropy: torch.Tensor,
        transition_residual: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        window_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        reliability = self.compute_reliability(
            subspace_residual,
            assignment_entropy,
            transition_residual,
        )
        if window_mask is not None:
            if window_mask.shape != embeddings.shape[:2]:
                raise ValueError("window_mask must match embeddings[:2]")
            reliability = reliability * window_mask.to(
                device=reliability.device,
                dtype=reliability.dtype,
            ).unsqueeze(-1).unsqueeze(-1)
        if self.training:
            self.update_memory(
                embeddings,
                phase_assignment,
                reliability,
                class_labels,
                window_mask=window_mask,
            )
        return {
            "reliability": reliability,
            "confirmed_memory": self.confirmed_memory,
            "evidence_counter": self.evidence_counter,
        }
