import copy

import torch
import numpy as np

from experiments.generate_figure5 import (
    SampleEvidence,
    infer_effective_lengths,
    select_median_gain_sample,
)
from models import PhasePathNet
from models.path_forward import PathForward
from losses import TransitionLoss
from data.windowing import WindowPartitioner


def test_path_forward_mask_matches_explicitly_cropped_sequence():
    torch.manual_seed(7)
    assignment = torch.softmax(torch.randn(2, 6, 3), dim=-1)
    transition = torch.softmax(torch.randn(3, 3), dim=-1)
    initial = torch.softmax(torch.randn(3), dim=-1)
    mask = torch.tensor(
        [[True, True, True, False, False, False], [True] * 6],
        dtype=torch.bool,
    )

    for log_space in (True, False):
        scorer = PathForward(log_space=log_space)
        masked = scorer(initial, transition, assignment, window_mask=mask)
        cropped_first = scorer(initial, transition, assignment[:1, :3])
        cropped_second = scorer(initial, transition, assignment[1:, :])
        expected = torch.cat([cropped_first, cropped_second])
        torch.testing.assert_close(masked, expected, rtol=1e-5, atol=1e-6)


def test_phase_path_scores_ignore_masked_embedding_tail():
    torch.manual_seed(11)
    model = PhasePathNet(
        n_classes=3,
        n_channels=2,
        seq_length=32,
        n_phases=3,
        embed_dim=12,
        backbone="fcn",
        use_memory=False,
    ).eval()

    prefix = torch.randn(1, 3, 12)
    embeddings_a = torch.cat([prefix, torch.randn(1, 3, 12)], dim=1)
    embeddings_b = torch.cat([prefix, 10.0 * torch.randn(1, 3, 12)], dim=1)
    mask = torch.tensor([[True, True, True, False, False, False]])

    with torch.no_grad():
        out_a = model.forward_from_embeddings(embeddings_a, window_mask=mask)
        out_b = model.forward_from_embeddings(embeddings_b, window_mask=mask)

    for key in (
        "sequence_embedding",
        "proto_score",
        "path_log_probs",
        "iid_log_probs",
        "transition_gain",
        "uncertainty_score",
        "logits",
    ):
        torch.testing.assert_close(out_a[key], out_b[key], rtol=1e-5, atol=1e-5)


def test_train_mode_encoder_excludes_masked_tail_from_batchnorm():
    torch.manual_seed(13)
    base_model = PhasePathNet(
        n_classes=3,
        n_channels=2,
        seq_length=32,
        n_phases=3,
        embed_dim=12,
        window_size=8,
        stride=4,
        backbone="fcn",
        use_memory=False,
    )
    model_a = copy.deepcopy(base_model).train()
    model_b = copy.deepcopy(base_model).train()

    lengths = torch.tensor([16, 20])
    x_a = torch.randn(2, 32, 2)
    x_b = x_a.clone()
    for sample_idx, length in enumerate(lengths.tolist()):
        x_b[sample_idx, length:] = 100.0 * torch.randn_like(
            x_b[sample_idx, length:]
        )

    out_a = model_a(x_a, valid_lengths=lengths)
    out_b = model_b(x_b, valid_lengths=lengths)

    for key in ("embeddings", "transition_gain", "logits"):
        torch.testing.assert_close(out_a[key], out_b[key], rtol=0.0, atol=0.0)
    buffers_b = dict(model_b.named_buffers())
    for name, buffer_a in model_a.named_buffers():
        if name.endswith(("running_mean", "running_var", "num_batches_tracked")):
            torch.testing.assert_close(
                buffer_a,
                buffers_b[name],
                rtol=0.0,
                atol=0.0,
            )


def test_transition_loss_ignores_masked_assignment_tail():
    torch.manual_seed(17)
    assignment_a = torch.softmax(torch.randn(2, 6, 3, 4), dim=-1)
    assignment_b = assignment_a.clone()
    assignment_b[:, 3:] = torch.softmax(
        20.0 * torch.randn_like(assignment_b[:, 3:]),
        dim=-1,
    )
    transition = torch.softmax(torch.randn(3, 4, 4), dim=-1)
    labels = torch.tensor([0, 2])
    mask = torch.tensor([[True, True, True, False, False, False]] * 2)

    criterion = TransitionLoss(loss_type="mse")
    loss_a = criterion(assignment_a, transition, labels, window_mask=mask)
    loss_b = criterion(assignment_b, transition, labels, window_mask=mask)

    torch.testing.assert_close(loss_a, loss_b, rtol=0.0, atol=0.0)


def test_valid_lengths_create_full_window_prefix_mask():
    torch.manual_seed(19)
    model = PhasePathNet(
        n_classes=2,
        n_channels=2,
        seq_length=32,
        n_phases=3,
        embed_dim=8,
        backbone="fcn",
        use_memory=False,
    ).eval()

    with torch.no_grad():
        output = model(
            torch.randn(2, 32, 2),
            valid_lengths=torch.tensor([32, 20]),
        )

    assert output["window_mask"].sum(dim=1).tolist() == [5, 3]
    assert output["window_mask"].dtype == torch.bool


def test_exact_stride_fit_does_not_add_an_extra_padded_window():
    partitioner = WindowPartitioner(window_size=10, stride=5, padding=True)
    windows = partitioner.partition(torch.randn(2, 30, 1))
    positions = partitioner.get_window_positions(30)

    assert windows.shape[:2] == (2, 5)
    assert positions.shape == (5, 2)
    assert positions[-1].tolist() == [20, 30]
    assert partitioner.get_window_mask(torch.tensor([30, 20]), 5).sum(1).tolist() == [5, 3]


def test_effective_length_uses_cross_channel_final_constant_run():
    values = np.zeros((2, 10, 2), dtype=np.float32)
    values[0, :6] = np.arange(12, dtype=np.float32).reshape(6, 2)
    values[0, 6:] = np.array([3.0, -2.0], dtype=np.float32)
    values[1] = np.arange(20, dtype=np.float32).reshape(10, 2)
    values[1, -2:] = values[1, -1]

    lengths = infer_effective_lengths(values, min_padding_run=3, atol=0.0)

    assert lengths.tolist() == [7, 10]


def test_median_sample_selection_uses_per_transition_gain():
    def record(sample_idx: int, gain: float) -> SampleEvidence:
        return SampleEvidence(
            sample_idx=sample_idx,
            class_idx=0,
            class_label="1",
            gain_per_valid_transition=gain,
            effective_length=20,
            n_valid_windows=3,
            x=np.zeros((20, 1), dtype=np.float32),
            responsibility=np.full((3, 2), 0.5, dtype=np.float32),
        )

    selected = select_median_gain_sample(
        [record(9, 0.80), record(3, 0.20), record(5, 0.50)]
    )

    assert selected.sample_idx == 5
