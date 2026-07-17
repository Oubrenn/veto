import numpy as np
import pandas as pd
import torch
from pathlib import Path
from types import SimpleNamespace

from experiments.memory_pollution_stress import (
    cosine_prototype_drift,
    heldout_query_accuracy,
)
from experiments.run_counterfactual_order_experiment import (
    _resume_protocol_matches,
    compute_transition_gain,
    score_embeddings_with_shuffles,
    signed_normalized_order_gap,
)
from experiments.plot_figure6_fix import (
    load_manuscript_variant_data,
    paired_variant_tests,
)


class _DummyGainModel:
    def forward_from_embeddings(self, embeddings, window_mask=None):
        del embeddings, window_mask
        return {
            "transition_gain": torch.tensor(
                [[8.0, -4.0], [3.0, -6.0]],
                dtype=torch.float32,
            )
        }


class _OrderAwareDummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward_from_embeddings(self, embeddings, window_mask=None):
        del window_mask
        positions = torch.arange(
            1,
            embeddings.shape[1] + 1,
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        score = (embeddings[..., 0] * positions).sum(dim=1)
        return {"transition_gain": torch.stack([score, -score], dim=1)}


def test_order_gain_uses_model_native_signed_score_and_valid_transition_count():
    embeddings = torch.zeros(2, 5, 3)
    labels = torch.tensor([0, 1])
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
        ]
    )

    gain = compute_transition_gain(
        _DummyGainModel(),
        embeddings,
        labels,
        window_mask=mask,
    )

    torch.testing.assert_close(gain, torch.tensor([4.0, -1.5]))


def test_signed_normalized_order_gap_preserves_sign_and_is_bounded():
    result = signed_normalized_order_gap(
        torch.tensor([2.0, -2.0, 0.0]),
        torch.tensor([1.0, -1.0, 0.0]),
    )
    torch.testing.assert_close(result[:2], torch.tensor([1.0 / 3.0, -1.0 / 3.0]))
    assert result[2].item() == 0.0
    assert bool((result.abs() <= 1.0).all())


def test_chunked_shuffle_scoring_matches_scalar_loop():
    embeddings = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    labels = torch.tensor([0, 1])
    scalar = score_embeddings_with_shuffles(
        _OrderAwareDummyModel(),
        embeddings,
        labels,
        n_shuffles=7,
        shuffle_seed=123,
        shuffle_chunk_size=1,
    )
    chunked = score_embeddings_with_shuffles(
        _OrderAwareDummyModel(),
        embeddings,
        labels,
        n_shuffles=7,
        shuffle_seed=123,
        shuffle_chunk_size=4,
    )
    for key in scalar:
        np.testing.assert_allclose(chunked[key], scalar[key])


def test_prototype_drift_and_query_accuracy_are_independent_computations():
    clean = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
    noisy = np.array([[[0.98, 0.20], [0.0, 1.0]]], dtype=np.float64)
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    labels = np.array([0, 1])

    drift = cosine_prototype_drift(clean, noisy)
    clean_acc = heldout_query_accuracy(clean, queries, labels)
    noisy_acc = heldout_query_accuracy(noisy, queries, labels)

    assert drift > 0.0
    assert clean_acc - noisy_acc == 0.0
    assert drift != clean_acc - noisy_acc


def test_resume_accepts_only_complete_masked_protocol_rows():
    args = SimpleNamespace(
        epochs=10,
        n_shuffles=100,
        diagnostic_batches=0,
        cf_start_epoch=5,
    )
    row = pd.Series(
        {
            "status": "ok",
            "epochs": 10,
            "n_shuffles": 100,
            "diagnostic_batches": 0,
            "counterfactual_active": True,
            "padding_mask": "valid-prefix mask from loader lengths",
            "n_samples": 2,
        }
    )
    samples = pd.DataFrame(
        {
            "n_shuffles": [100, 100],
            "padding_mask": [
                "valid-prefix mask from loader lengths",
                "valid-prefix mask from loader lengths",
            ],
        }
    )

    assert _resume_protocol_matches(row, samples, args, "VETO full")
    legacy = samples.copy()
    legacy.loc[0, "padding_mask"] = "none (legacy protocol)"
    assert not _resume_protocol_matches(row, legacy, args, "VETO full")


def test_variant_pairing_uses_one_six_dataset_mean_per_seed():
    rows = []
    for seed, full, raw, no_cf in [
        (42, 0.30, 0.20, 0.10),
        (43, 0.40, 0.20, 0.21),
        (44, 0.50, 0.20, 0.32),
        (45, 0.60, 0.20, 0.43),
        (46, 0.70, 0.20, 0.54),
    ]:
        rows.extend(
            [
                {"seed": seed, "variant": "VETO full", "delta_ord": full},
                {"seed": seed, "variant": "Raw transition", "delta_ord": raw},
                {"seed": seed, "variant": "w/o counterfactual", "delta_ord": no_cf},
            ]
        )

    tests = paired_variant_tests(pd.DataFrame(rows))

    np.testing.assert_allclose(
        tests["VETO full vs Raw transition"]["mean_difference"], 0.30
    )
    np.testing.assert_allclose(
        tests["VETO full vs w/o counterfactual"]["mean_difference"], 0.18
    )


def test_figure6_manuscript_order_gaps_match_design_ablation_table():
    frame = load_manuscript_variant_data(
        Path("diagnostics/paper_figures/source_data/fig6_manuscript_order_gaps.csv")
    )
    observed = dict(zip(frame["variant"], frame["delta_ord"]))
    assert observed == {
        "Class-independent transition": 0.0176,
        "Raw transition": 0.0364,
        "w/o counterfactual": 0.0288,
        "VETO": 0.0472,
    }
    assert set(frame["n_datasets"]) == {6}
    assert set(frame["n_seeds"]) == {5}
