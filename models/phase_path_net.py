"""Phase-Path network for multivariate time-series classification."""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import WindowPartitioner
from .backbones import FCN, InceptionTime, ResNet1D
from .confirmed_memory import ConfirmedMemory
from .path_forward import PathForward, PathForwardWithViterbi
from .phase_assignment import PhaseAssignment
from .phase_graph import ClassPhaseGraph
from .phase_prototypes import LowRankPhasePrototypes
from .tf_branch import SimpleTimeFrequencyBranch
from .uncertainty import UncertaintyEstimator


class PhasePathNet(nn.Module):
    """Composable Phase-Path classifier.

    The default configuration is the full VETO-style head. Ablation settings are
    intentionally exposed as constructor arguments so experiment scripts can run
    controlled comparisons without editing model code.
    """

    def __init__(
        self,
        n_classes: int,
        n_channels: int,
        seq_length: int,
        n_phases: int = 5,
        embed_dim: int = 128,
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
        backbone: str = "inception",
        use_tf_branch: bool = False,
        use_memory: bool = True,
        transition_mode: str = "free",
        prototype_mode: str = "class",
        head_mode: str = "veto",
        path_score_mode: str = "gain",
        path_weight_override: Optional[float] = None,
        use_uncertainty: bool = True,
        memory_update_mode: str = "confirmed",
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.seq_length = seq_length
        self.n_phases = n_phases
        self.embed_dim = embed_dim
        self.use_tf_branch = use_tf_branch
        self.use_memory = use_memory
        self.transition_mode = transition_mode
        self.prototype_mode = prototype_mode
        self.head_mode = head_mode
        self.path_score_mode = path_score_mode
        self.path_weight_override = path_weight_override
        self.use_uncertainty = use_uncertainty
        self.memory_update_mode = memory_update_mode
        if path_score_mode not in {"gain", "raw"}:
            raise ValueError("path_score_mode must be one of: gain, raw")

        if window_size is None:
            window_size = max(10, seq_length // 10)
        if stride is None:
            stride = max(1, window_size // 2)
        self.window_partitioner = WindowPartitioner(window_size, stride, padding=True)

        if backbone == "inception":
            self.encoder = InceptionTime(n_channels, embed_dim)
        elif backbone == "resnet":
            self.encoder = ResNet1D(n_channels, embed_dim)
        elif backbone == "fcn":
            self.encoder = FCN(n_channels, embed_dim)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone_classifier = nn.Linear(embed_dim, n_classes)
        self.orderless_classifier = nn.Linear(embed_dim, n_classes)

        shared_prototypes = prototype_mode == "shared"
        rank = embed_dim if prototype_mode == "full" else min(32, max(1, embed_dim // 4))
        self.phase_prototypes = LowRankPhasePrototypes(
            n_classes=n_classes,
            n_phases=n_phases,
            embed_dim=embed_dim,
            rank=rank,
            shared=shared_prototypes,
        )
        self.phase_assignment = PhaseAssignment(temperature=1.0)
        self.phase_graph = ClassPhaseGraph(
            n_classes=n_classes,
            n_phases=n_phases,
            transition_mode=transition_mode,
        )
        self.path_forward = PathForward(log_space=True)
        self.uncertainty_estimator = UncertaintyEstimator()

        if use_memory:
            self.confirmed_memory = ConfirmedMemory(
                n_classes=n_classes,
                n_phases=n_phases,
                embed_dim=embed_dim,
                update_mode=memory_update_mode,
            )
        if use_tf_branch:
            self.tf_branch = SimpleTimeFrequencyBranch(n_channels, embed_dim)

        self.score_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

    @staticmethod
    def _masked_mean(
        values: torch.Tensor,
        window_mask: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        """Mean over windows while excluding right-padding entries."""
        weights = window_mask.to(dtype=values.dtype)
        while weights.ndim < values.ndim:
            weights = weights.unsqueeze(-1)
        numerator = (values * weights).sum(dim=dim)
        denominator = weights.sum(dim=dim).clamp_min(1.0)
        return numerator / denominator

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        valid_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        windows = self.window_partitioner.partition(x)
        window_mask = None
        if valid_lengths is not None:
            lengths = torch.as_tensor(valid_lengths, device=x.device, dtype=torch.long)
            if lengths.ndim == 0:
                lengths = lengths.unsqueeze(0)
            if lengths.shape != (x.shape[0],):
                raise ValueError(
                    f"valid_lengths must have shape {(x.shape[0],)}, got {tuple(lengths.shape)}"
                )
            if bool((lengths > x.shape[1]).any()):
                raise ValueError("valid_lengths cannot exceed the padded input length")
            window_mask = self.window_partitioner.get_window_mask(
                lengths,
                n_windows=windows.shape[1],
            )

        if window_mask is None:
            embeddings = self.encoder(windows)
        else:
            # Backbones flatten B x N before BatchNorm. Encoding every padded
            # window would therefore let the masked tail alter valid-window
            # embeddings and running statistics during training. Encode only
            # fully observed windows, then scatter them back for the masked
            # downstream computations.
            valid_embeddings = self.encoder(windows[window_mask])
            embeddings = valid_embeddings.new_zeros(
                (*windows.shape[:2], valid_embeddings.shape[-1])
            )
            embeddings[window_mask] = valid_embeddings
        return self.forward_from_embeddings(
            embeddings,
            labels=labels,
            windows=windows,
            window_mask=window_mask,
        )

    def forward_from_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        windows: Optional[torch.Tensor] = None,
        window_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Score an already encoded window sequence.

        This keeps order-corruption diagnostics honest: latent shuffles can
        permute the same local embeddings instead of reconstructing raw series
        and re-encoding slightly different windows.
        """
        if window_mask is None:
            window_mask = torch.ones(
                embeddings.shape[:2],
                dtype=torch.bool,
                device=embeddings.device,
            )
        else:
            if window_mask.shape != embeddings.shape[:2]:
                raise ValueError(
                    "window_mask must match the embedding batch/window dimensions"
                )
            window_mask = window_mask.to(device=embeddings.device, dtype=torch.bool)
            if not bool(window_mask[:, 0].all()):
                raise ValueError("Each sample must contain at least one valid window")

        sequence_embedding = self._masked_mean(embeddings, window_mask, dim=1)

        proto_output = self.phase_prototypes(embeddings)
        template_dist = proto_output["template_dist"]
        subspace_residual = proto_output["subspace_residual"]

        assign_output = self.phase_assignment(template_dist, subspace_residual)
        phase_assignment = assign_output["phase_assignment"]

        graph_output = self.phase_graph()
        transition_matrices = graph_output["transition_matrices"]
        init_distributions = graph_output["init_distributions"]

        path_log_probs = []
        iid_log_probs = []
        for y in range(self.n_classes):
            class_assignment = phase_assignment[:, :, y, :]
            log_prob = self.path_forward(
                init_dist=init_distributions[y],
                transition_matrix=transition_matrices[y],
                phase_assignment=class_assignment,
                window_mask=window_mask,
            )
            marginal = self._masked_mean(
                class_assignment,
                window_mask,
                dim=1,
            ).clamp_min(1e-8)
            marginal = marginal / marginal.sum(dim=-1, keepdim=True)
            iid_transition = marginal.unsqueeze(1).expand(-1, self.n_phases, -1)
            iid_init = marginal
            iid_log_prob = self.path_forward(
                init_dist=iid_init,
                transition_matrix=iid_transition,
                phase_assignment=class_assignment,
                window_mask=window_mask,
            )
            path_log_probs.append(log_prob)
            iid_log_probs.append(iid_log_prob)
        path_log_probs = torch.stack(path_log_probs, dim=1)
        iid_log_probs = torch.stack(iid_log_probs, dim=1)
        transition_gain = path_log_probs - iid_log_probs

        unc_output = self.uncertainty_estimator(
            phase_assignment,
            transition_matrices,
            window_mask=window_mask,
        )
        uncertainty_score = unc_output["uncertainty_score"]

        memory_output = None
        if self.use_memory:
            memory_output = self.confirmed_memory(
                embeddings=embeddings,
                phase_assignment=phase_assignment,
                subspace_residual=subspace_residual,
                assignment_entropy=unc_output["assignment_entropy"],
                transition_residual=unc_output["transition_residual"],
                class_labels=labels,
                window_mask=window_mask,
            )

        tf_output = None
        if self.use_tf_branch and windows is not None:
            tf_output = self.tf_branch(windows)

        proto_score = -self._masked_mean(
            template_dist,
            window_mask,
            dim=1,
        ).mean(dim=-1)
        path_score = transition_gain if self.path_score_mode == "gain" else path_log_probs
        unc_penalty = uncertainty_score if self.use_uncertainty else torch.zeros_like(
            uncertainty_score
        )

        if self.head_mode == "backbone":
            logits = self.backbone_classifier(sequence_embedding)
        elif self.head_mode == "orderless":
            logits = self.orderless_classifier(sequence_embedding)
        elif self.head_mode == "prototype":
            logits = proto_score
        elif self.head_mode == "hmm":
            logits = path_score
        elif self.head_mode == "veto":
            w_proto, w_path, w_unc = F.softplus(self.score_weights)
            if self.path_weight_override is not None:
                w_path = path_score.new_tensor(float(self.path_weight_override))
            logits = w_proto * proto_score + w_path * path_score - w_unc * unc_penalty
        else:
            raise ValueError(
                "head_mode must be one of: backbone, orderless, prototype, hmm, veto"
            )

        output = {
            "logits": logits,
            "embeddings": embeddings,
            "sequence_embedding": sequence_embedding,
            "phase_assignment": phase_assignment,
            "template_dist": template_dist,
            "subspace_residual": subspace_residual,
            "path_log_probs": path_log_probs,
            "iid_log_probs": iid_log_probs,
            "transition_gain": transition_gain,
            "path_score": path_score,
            "proto_score": proto_score,
            "uncertainty_score": uncertainty_score,
            "transition_matrices": transition_matrices,
            "init_distributions": init_distributions,
            "score_weights": F.softplus(self.score_weights),
            "window_mask": window_mask,
        }
        if memory_output is not None:
            output.update(memory_output)
        if tf_output is not None:
            output["tf_embeddings"] = tf_output["tf_embeddings"]
        return output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.forward(x)["logits"], dim=-1)

    def get_phase_path(self, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        x = x.unsqueeze(0)
        output = self.forward(x)
        phase_assignment = output["phase_assignment"][0, :, class_idx, :]
        transition_matrix = output["transition_matrices"][class_idx]
        init_dist = output["init_distributions"][class_idx]

        viterbi = PathForwardWithViterbi()
        return viterbi.viterbi_decode(
            init_dist=init_dist,
            transition_matrix=transition_matrix,
            phase_assignment=phase_assignment.unsqueeze(0),
        ).squeeze(0)
