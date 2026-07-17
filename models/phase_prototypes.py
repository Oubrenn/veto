"""Low-rank phase prototype modules."""
from typing import Tuple

import torch
import torch.nn as nn


class LowRankPhasePrototypes(nn.Module):
    """Class-conditional or shared low-rank phase prototypes.

    Each phase prototype is represented by ``U @ V.T``. With ``shared=True`` a
    single dictionary is expanded over all classes, supporting the "shared
    dictionary only" ablation without changing downstream code.
    """

    def __init__(
        self,
        n_classes: int,
        n_phases: int,
        embed_dim: int,
        rank: int = 32,
        shared: bool = False,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_phases = n_phases
        self.embed_dim = embed_dim
        self.rank = min(rank, embed_dim)
        self.shared = shared
        self.prototype_classes = 1 if shared else n_classes

        self.U = nn.Parameter(
            torch.randn(self.prototype_classes, n_phases, embed_dim, self.rank)
        )
        self.V = nn.Parameter(
            torch.randn(self.prototype_classes, n_phases, embed_dim, self.rank)
        )

        nn.init.orthogonal_(self.U.view(-1, embed_dim, self.rank))
        nn.init.orthogonal_(self.V.view(-1, embed_dim, self.rank))

    def _parameter_class(self, class_idx: int) -> int:
        return 0 if self.shared else class_idx

    def get_prototypes(self, class_idx: int) -> torch.Tensor:
        param_idx = self._parameter_class(class_idx)
        U_y = self.U[param_idx]
        V_y = self.V[param_idx]
        return torch.bmm(U_y, V_y.transpose(1, 2))

    def compute_distance(
        self,
        embeddings: torch.Tensor,
        class_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = embeddings.shape
        param_idx = self._parameter_class(class_idx)
        U_y = self.U[param_idx]
        V_y = self.V[param_idx]

        embeddings_exp = embeddings.unsqueeze(2)

        projected = []
        for k in range(self.n_phases):
            proj = embeddings @ V_y[k]
            projected.append(proj @ U_y[k].T)
        projected = torch.stack(projected, dim=2)
        template_dist = torch.norm(embeddings_exp - projected, dim=-1)

        subspace_projected = []
        for k in range(self.n_phases):
            proj = embeddings @ U_y[k]
            subspace_projected.append(proj @ U_y[k].T)
        subspace_projected = torch.stack(subspace_projected, dim=2)
        subspace_residual = torch.norm(embeddings_exp - subspace_projected, dim=-1)

        return template_dist.view(B, N, self.n_phases), subspace_residual.view(
            B, N, self.n_phases
        )

    def compute_all_distances(
        self,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.shared:
            template_dist, subspace_residual = self.compute_distance(embeddings, 0)
            template_dist = template_dist.unsqueeze(2).expand(
                -1, -1, self.n_classes, -1
            )
            subspace_residual = subspace_residual.unsqueeze(2).expand(
                -1, -1, self.n_classes, -1
            )
            return template_dist, subspace_residual

        template_dists = []
        subspace_residuals = []
        for class_idx in range(self.n_classes):
            template_dist, subspace_residual = self.compute_distance(
                embeddings,
                class_idx,
            )
            template_dists.append(template_dist)
            subspace_residuals.append(subspace_residual)

        return torch.stack(template_dists, dim=2), torch.stack(
            subspace_residuals,
            dim=2,
        )

    def forward(self, embeddings: torch.Tensor) -> dict:
        template_dist, subspace_residual = self.compute_all_distances(embeddings)
        return {
            "template_dist": template_dist,
            "subspace_residual": subspace_residual,
        }
