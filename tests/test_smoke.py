import numpy as np
import torch

from data.dataloader import PhasePathDataset
from models import PhasePathNet


def test_to_numpy_stacks_channel_first_cache_samples():
    samples = [
        torch.ones(3, 5),
        torch.zeros(3, 5),
    ]

    array = PhasePathDataset._to_numpy(samples)

    assert array.shape == (2, 3, 5)
    assert array.dtype == np.float32


def test_phase_path_net_forward_shapes():
    model = PhasePathNet(
        n_classes=4,
        n_channels=3,
        seq_length=32,
        n_phases=3,
        embed_dim=16,
        backbone="fcn",
        use_tf_branch=False,
        use_memory=False,
    )

    x = torch.randn(2, 32, 3)
    output = model(x)

    assert output["logits"].shape == (2, 4)
    assert output["phase_assignment"].shape[-2:] == (4, 3)
    assert output["path_log_probs"].shape == (2, 4)


def test_phase_path_net_ablation_modes_forward():
    for transition_mode in ["uniform", "free", "neural", "attention"]:
        model = PhasePathNet(
            n_classes=3,
            n_channels=2,
            seq_length=24,
            n_phases=3,
            embed_dim=12,
            backbone="fcn",
            use_memory=False,
            transition_mode=transition_mode,
        )
        output = model(torch.randn(2, 24, 2))
        assert output["transition_matrices"].shape == (3, 3, 3)
        assert output["logits"].shape == (2, 3)

    for prototype_mode in ["class", "shared", "full"]:
        model = PhasePathNet(
            n_classes=3,
            n_channels=2,
            seq_length=24,
            n_phases=3,
            embed_dim=12,
            backbone="fcn",
            use_memory=False,
            prototype_mode=prototype_mode,
        )
        output = model(torch.randn(2, 24, 2))
        assert output["template_dist"].shape[-2:] == (3, 3)

    for head_mode in ["backbone", "orderless", "prototype", "hmm", "veto"]:
        model = PhasePathNet(
            n_classes=3,
            n_channels=2,
            seq_length=24,
            n_phases=3,
            embed_dim=12,
            backbone="fcn",
            use_memory=False,
            head_mode=head_mode,
        )
        output = model(torch.randn(2, 24, 2))
        assert output["logits"].shape == (2, 3)
