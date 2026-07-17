"""Class-conditional phase transition graphs."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassPhaseGraph(nn.Module):
    """Generate class-conditional initial distributions and transition matrices.

    Supported transition modes:
    - ``uniform``: fixed uniform HMM head.
    - ``free``: one free transition matrix and initial distribution per class.
    - ``class_independent``: one learnable transition model shared by all classes.
    - ``neural``: shared neural generator conditioned on class embeddings.
    - ``attention``: class-conditioned phase query/key attention head.
    """

    def __init__(
        self,
        n_classes: int,
        n_phases: int,
        learnable_transition: bool = True,
        transition_mode: str = None,
        class_embed_dim: int = 16,
        generator_hidden: int = 64,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_phases = n_phases
        if transition_mode is None:
            transition_mode = "free" if learnable_transition else "uniform"
        self.transition_mode = transition_mode
        self.learnable_transition = transition_mode != "uniform"

        if transition_mode == "free":
            self.transition_logits = nn.Parameter(torch.randn(n_classes, n_phases, n_phases))
            self.init_logits = nn.Parameter(torch.randn(n_classes, n_phases))
        elif transition_mode == "uniform":
            self.register_buffer(
                "transition_matrix",
                torch.ones(n_classes, n_phases, n_phases) / n_phases,
            )
            self.register_buffer(
                "init_distribution",
                torch.ones(n_classes, n_phases) / n_phases,
            )
        elif transition_mode == "class_independent":
            self.transition_logits = nn.Parameter(torch.randn(n_phases, n_phases))
            self.init_logits = nn.Parameter(torch.randn(n_phases))
        elif transition_mode == "neural":
            self.class_embedding = nn.Embedding(n_classes, class_embed_dim)
            self.transition_generator = nn.Sequential(
                nn.Linear(class_embed_dim, generator_hidden),
                nn.ReLU(),
                nn.Linear(generator_hidden, n_phases * n_phases),
            )
            self.init_generator = nn.Sequential(
                nn.Linear(class_embed_dim, generator_hidden),
                nn.ReLU(),
                nn.Linear(generator_hidden, n_phases),
            )
        elif transition_mode == "attention":
            self.class_embedding = nn.Embedding(n_classes, class_embed_dim)
            self.phase_query = nn.Parameter(torch.randn(n_phases, class_embed_dim))
            self.phase_key = nn.Parameter(torch.randn(n_phases, class_embed_dim))
            self.init_generator = nn.Linear(class_embed_dim, n_phases)
        else:
            raise ValueError(
                "transition_mode must be one of: uniform, free, class_independent, neural, attention"
            )

    def get_transition_matrix(self, class_idx: int) -> torch.Tensor:
        return self.get_all_transition_matrices()[class_idx]

    def get_init_distribution(self, class_idx: int) -> torch.Tensor:
        return self.get_all_init_distributions()[class_idx]

    def get_all_transition_matrices(self) -> torch.Tensor:
        if self.transition_mode == "free":
            return F.softmax(self.transition_logits, dim=-1)
        if self.transition_mode == "class_independent":
            transition = F.softmax(self.transition_logits, dim=-1)
            return transition.unsqueeze(0).expand(self.n_classes, -1, -1)
        if self.transition_mode == "uniform":
            return self.transition_matrix
        if self.transition_mode == "neural":
            class_ids = torch.arange(self.n_classes, device=self.class_embedding.weight.device)
            class_emb = self.class_embedding(class_ids)
            logits = self.transition_generator(class_emb)
            return F.softmax(
                logits.view(self.n_classes, self.n_phases, self.n_phases),
                dim=-1,
            )
        if self.transition_mode == "attention":
            class_ids = torch.arange(self.n_classes, device=self.class_embedding.weight.device)
            class_emb = self.class_embedding(class_ids)
            queries = self.phase_query.unsqueeze(0) + class_emb.unsqueeze(1)
            keys = self.phase_key.unsqueeze(0) + class_emb.unsqueeze(1)
            logits = torch.matmul(queries, keys.transpose(1, 2)) / (queries.shape[-1] ** 0.5)
            return F.softmax(logits, dim=-1)
        raise RuntimeError(f"Unhandled transition mode: {self.transition_mode}")

    def get_all_init_distributions(self) -> torch.Tensor:
        if self.transition_mode == "free":
            return F.softmax(self.init_logits, dim=-1)
        if self.transition_mode == "class_independent":
            init = F.softmax(self.init_logits, dim=-1)
            return init.unsqueeze(0).expand(self.n_classes, -1)
        if self.transition_mode == "uniform":
            return self.init_distribution
        class_ids = torch.arange(self.n_classes, device=self.class_embedding.weight.device)
        class_emb = self.class_embedding(class_ids)
        return F.softmax(self.init_generator(class_emb), dim=-1)

    def forward(self) -> dict:
        return {
            "transition_matrices": self.get_all_transition_matrices(),
            "init_distributions": self.get_all_init_distributions(),
        }


class StructuredPhaseGraph(ClassPhaseGraph):
    """Structured transition graph with optional monotonic masking."""

    def __init__(
        self,
        n_classes: int,
        n_phases: int,
        sparsity: float = 0.0,
        monotonic: bool = False,
    ):
        super().__init__(n_classes, n_phases, transition_mode="free")
        self.sparsity = sparsity
        self.monotonic = monotonic

    def get_all_transition_matrices(self) -> torch.Tensor:
        logits = self.transition_logits
        if self.monotonic:
            mask = torch.triu(torch.ones_like(logits))
            logits = logits * mask + (1 - mask) * (-1e9)
        return F.softmax(logits, dim=-1)

    def get_sparsity_loss(self) -> torch.Tensor:
        if self.sparsity == 0:
            return torch.tensor(0.0, device=self.transition_logits.device)
        return self.sparsity * torch.mean(torch.abs(self.get_all_transition_matrices()))
