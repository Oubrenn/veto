"""Generate Figure 5 from the real Handwriting checkpoint and cached data.

The figure defends one specific claim: local phase marginals can be similar
while class-conditioned phase-transition paths remain different. No synthetic
or placeholder values are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import PhasePathDataset
from models import PhasePathNet


DEFAULT_DATA_DIR = ROOT / "diagnostics" / "uea_target_cache"
DEFAULT_CHECKPOINT = ROOT / "checkpoints_official" / "Handwriting_best.pth"
DEFAULT_OUTPUT_DIR = ROOT / "diagnostics" / "paper_figures"

PHASE_COLORS = ["#7FA6C9", "#E7A45B", "#78B786", "#5FA8A3", "#A88AC2"]
CHANNEL_COLORS = ["#326A9B", "#D97826", "#3C8C63"]


@dataclass
class SampleEvidence:
    sample_idx: int
    class_idx: int
    class_label: str
    gain_per_valid_transition: float
    effective_length: int
    n_valid_windows: int
    x: np.ndarray
    responsibility: np.ndarray

    @property
    def phase_marginal(self) -> np.ndarray:
        return self.responsibility.mean(axis=0)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def infer_effective_lengths(
    x: np.ndarray,
    *,
    min_padding_run: int = 3,
    atol: float = 1e-7,
) -> np.ndarray:
    """Infer lengths from the final cross-channel constant run.

    The Handwriting cache declares a uniform length of 152 but stores a long,
    sample-specific constant tail. For each sample, the effective segment keeps
    the first endpoint vector in that final run: if the run begins at zero-based
    index ``L``, the effective length is ``L + 1``. Equivalently, it is the
    last cross-time change index plus two. A tail is removed only when the final
    constant run contains at least ``min_padding_run`` time points.
    """
    values = np.asarray(x)
    if values.ndim != 3:
        raise ValueError(f"Expected (samples, time, channels), got {values.shape}")
    if min_padding_run < 2:
        raise ValueError("min_padding_run must be at least 2")

    n_samples, seq_length, _ = values.shape
    lengths = np.full(n_samples, seq_length, dtype=np.int64)
    for sample_idx, sample in enumerate(values):
        matches_final = np.all(
            np.isclose(sample, sample[-1], rtol=0.0, atol=atol),
            axis=1,
        )
        run_start = seq_length - 1
        while run_start > 0 and bool(matches_final[run_start - 1]):
            run_start -= 1
        run_length = seq_length - run_start
        if run_length >= min_padding_run and run_start > 0:
            lengths[sample_idx] = run_start + 1
    return lengths


def length_summary(lengths: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(lengths, dtype=float)
    return {
        "min": int(values.min()),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": int(values.max()),
    }


def class_label(dataset: PhasePathDataset, class_idx: int) -> str:
    labels = getattr(dataset, "class_labels", None)
    if labels and class_idx < len(labels):
        raw = str(labels[class_idx])
        try:
            return str(int(float(raw)))
        except ValueError:
            return raw
    return str(class_idx)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_handwriting_model(
    dataset: PhasePathDataset,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[PhasePathNet, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict")

    model = PhasePathNet(
        n_classes=dataset.n_classes,
        n_channels=dataset.n_channels,
        seq_length=dataset.seq_length,
        n_phases=5,
        embed_dim=128,
        transition_mode="free",
        prototype_mode="class",
        head_mode="veto",
        path_score_mode="gain",
        use_memory=True,
        use_uncertainty=True,
    )
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    allowed_missing = {
        "backbone_classifier.weight",
        "backbone_classifier.bias",
        "orderless_classifier.weight",
        "orderless_classifier.bias",
    }
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    model.to(device).eval()
    checkpoint_info = {
        "epoch": checkpoint.get("epoch"),
        "best_acc": checkpoint.get("best_acc"),
        "sha256": file_sha256(checkpoint_path),
        "allowed_unused_missing_keys": sorted(incompatible.missing_keys),
        "unexpected_keys": sorted(incompatible.unexpected_keys),
    }
    return model, checkpoint_info


def collect_evidence(
    dataset: PhasePathDataset,
    model: PhasePathNet,
    effective_lengths: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[
    dict[int, list[SampleEvidence]],
    np.ndarray,
    np.ndarray,
    dict[str, float],
]:
    n_classes = dataset.n_classes
    n_phases = model.n_phases
    phase_marginal_sum = np.zeros((n_classes, n_phases), dtype=np.float64)
    class_sample_counts = np.zeros(n_classes, dtype=np.int64)
    records: dict[int, list[SampleEvidence]] = {idx: [] for idx in range(n_classes)}
    masked_correct = 0
    legacy_correct = 0

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            stop = min(len(dataset), start + batch_size)
            x_cpu = torch.from_numpy(dataset.X[start:stop]).float()
            y_cpu = torch.from_numpy(np.asarray(dataset.y[start:stop])).long()
            lengths_cpu = torch.from_numpy(effective_lengths[start:stop]).long()

            x = x_cpu.to(device)
            y = y_cpu.to(device)
            lengths = lengths_cpu.to(device)
            output = model(x, labels=y, valid_lengths=lengths)
            masked_pred = output["logits"].argmax(dim=1)

            # Audit the legacy checkpoint under the original all-window path.
            # Re-run the actual legacy all-window path. Masked forward now
            # encodes only valid windows and scatters zeros into the tail, so
            # reusing those embeddings would not reproduce the historical
            # unmasked checkpoint behavior.
            legacy_output = model(x, labels=y)
            legacy_pred = legacy_output["logits"].argmax(dim=1)

            masked_correct += int((masked_pred == y).sum().item())
            legacy_correct += int((legacy_pred == y).sum().item())
            assignment = output["phase_assignment"].detach().cpu().numpy()
            gains = output["transition_gain"].detach().cpu().numpy()
            valid_counts = output["window_mask"].sum(dim=1).detach().cpu().numpy()
            masked_pred_cpu = masked_pred.detach().cpu().numpy()

            for local_idx in range(stop - start):
                sample_idx = start + local_idx
                true_class = int(y_cpu[local_idx].item())
                n_valid = int(valid_counts[local_idx])
                responsibility = assignment[local_idx, :n_valid, true_class, :]
                phase_marginal_sum[true_class] += responsibility.mean(axis=0)
                class_sample_counts[true_class] += 1

                if int(masked_pred_cpu[local_idx]) != true_class:
                    continue
                records[true_class].append(
                    SampleEvidence(
                        sample_idx=sample_idx,
                        class_idx=true_class,
                        class_label=class_label(dataset, true_class),
                        gain_per_valid_transition=(
                            float(gains[local_idx, true_class]) / max(n_valid - 1, 1)
                        ),
                        effective_length=int(effective_lengths[sample_idx]),
                        n_valid_windows=n_valid,
                        x=x_cpu[local_idx].numpy().copy(),
                        responsibility=responsibility.copy(),
                    )
                )

    if bool((class_sample_counts == 0).any()):
        missing = np.flatnonzero(class_sample_counts == 0).tolist()
        raise RuntimeError(f"No test samples for class indices: {missing}")
    phase_marginals = phase_marginal_sum / class_sample_counts[:, None]
    phase_marginals /= np.clip(phase_marginals.sum(axis=1, keepdims=True), 1e-12, None)
    transition_matrices = (
        model.phase_graph()["transition_matrices"].detach().cpu().numpy()
    )
    transition_matrices /= np.clip(
        transition_matrices.sum(axis=2, keepdims=True),
        1e-12,
        None,
    )
    metrics = {
        "masked_accuracy": masked_correct / len(dataset),
        "legacy_unmasked_accuracy": legacy_correct / len(dataset),
    }
    return records, phase_marginals, transition_matrices, metrics


def normalized_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / np.clip(p.sum(), 1e-12, None)
    q = q / np.clip(q.sum(), 1e-12, None)
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log(left[mask] / right[mask])))

    return (0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)) / math.log(2.0)


def transition_divergence(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    marginal_a: np.ndarray,
    marginal_b: np.ndarray,
) -> float:
    """Phase-occupancy-weighted row-wise Jensen-Shannon divergence."""
    phase_weights = 0.5 * (marginal_a + marginal_b)
    phase_weights /= np.clip(phase_weights.sum(), 1e-12, None)
    row_divergence = np.asarray(
        [
            normalized_js_divergence(matrix_a[row], matrix_b[row])
            for row in range(matrix_a.shape[0])
        ]
    )
    return float(np.sum(phase_weights * row_divergence))


def select_class_pair(
    records: dict[int, list[SampleEvidence]],
    phase_marginals: np.ndarray,
    transition_matrices: np.ndarray,
    *,
    candidate_fraction: float = 0.10,
) -> tuple[tuple[int, int], list[dict[str, Any]], int]:
    if not 0 < candidate_fraction <= 1:
        raise ValueError("candidate_fraction must be in (0, 1]")
    eligible = sorted(class_idx for class_idx, samples in records.items() if samples)
    if len(eligible) < 2:
        raise RuntimeError("Need correctly classified samples from at least two classes")

    pair_rows: list[dict[str, Any]] = []
    for left_pos, left in enumerate(eligible):
        for right in eligible[left_pos + 1 :]:
            marginal_js = normalized_js_divergence(
                phase_marginals[left],
                phase_marginals[right],
            )
            pair_rows.append(
                {
                    "class_idx_a": left,
                    "class_idx_b": right,
                    "marginal_js": marginal_js,
                    "phase_similarity": 1.0 - marginal_js,
                    "transition_divergence": transition_divergence(
                        transition_matrices[left],
                        transition_matrices[right],
                        phase_marginals[left],
                        phase_marginals[right],
                    ),
                }
            )

    pair_rows.sort(
        key=lambda row: (
            row["marginal_js"],
            row["class_idx_a"],
            row["class_idx_b"],
        )
    )
    n_candidates = max(1, int(math.ceil(candidate_fraction * len(pair_rows))))
    for rank, row in enumerate(pair_rows, start=1):
        row["phase_similarity_rank"] = rank
        row["high_similarity_candidate"] = rank <= n_candidates
        row["selected"] = False

    candidates = pair_rows[:n_candidates]
    selected_row = min(
        candidates,
        key=lambda row: (
            -row["transition_divergence"],
            row["marginal_js"],
            row["class_idx_a"],
            row["class_idx_b"],
        ),
    )
    selected_row["selected"] = True
    selected_pair = (
        int(selected_row["class_idx_a"]),
        int(selected_row["class_idx_b"]),
    )
    return selected_pair, pair_rows, n_candidates


def select_median_gain_sample(samples: Sequence[SampleEvidence]) -> SampleEvidence:
    if not samples:
        raise ValueError("Cannot select a median sample from an empty class")
    median_gain = float(
        np.median([sample.gain_per_valid_transition for sample in samples])
    )
    return min(
        samples,
        key=lambda sample: (
            abs(sample.gain_per_valid_transition - median_gain),
            sample.sample_idx,
        ),
    )


def highest_variance_channels(
    dataset: PhasePathDataset,
    effective_lengths: np.ndarray,
    n_channels: int = 3,
) -> list[int]:
    channel_sum = np.zeros(dataset.n_channels, dtype=np.float64)
    channel_sum_sq = np.zeros(dataset.n_channels, dtype=np.float64)
    count = 0
    for sample, length in zip(dataset.X, effective_lengths):
        observed = np.asarray(sample[: int(length)], dtype=np.float64)
        channel_sum += observed.sum(axis=0)
        channel_sum_sq += np.square(observed).sum(axis=0)
        count += observed.shape[0]
    mean = channel_sum / max(count, 1)
    variance = channel_sum_sq / max(count, 1) - np.square(mean)
    return np.argsort(variance)[::-1][:n_channels].astype(int).tolist()


def run_bounds(values: np.ndarray) -> list[tuple[int, int]]:
    if len(values) == 0:
        return []
    bounds: list[tuple[int, int]] = []
    start = 0
    for idx in range(1, len(values)):
        if int(values[idx]) != int(values[start]):
            bounds.append((start, idx - 1))
            start = idx
    bounds.append((start, len(values) - 1))
    return bounds


def compressed_path(top1: np.ndarray) -> list[int]:
    return [int(top1[start]) for start, _ in run_bounds(top1)]


def compact_path_label(top1: np.ndarray, max_states: int = 4) -> str:
    path = compressed_path(top1)
    labels = [f"P{phase}" for phase in path]
    if len(labels) > max_states:
        return ""
    return r"$" + r"\rightarrow".join(labels) + r"$"


def add_phase_background(
    ax: mpl.axes.Axes,
    top1: np.ndarray,
    positions: np.ndarray,
    effective_length: int,
) -> None:
    denominator = max(effective_length - 1, 1)
    for start_idx, end_idx in run_bounds(top1):
        phase = int(top1[start_idx])
        start = positions[start_idx, 0] / denominator
        end = (positions[end_idx, 1] - 1) / denominator
        ax.axvspan(start, min(1.0, end), color=PHASE_COLORS[phase], alpha=0.16, lw=0)


def plot_trace(
    ax: mpl.axes.Axes,
    sample: SampleEvidence,
    positions: np.ndarray,
    channel_indices: Sequence[int],
    panel_letter: str,
) -> None:
    observed = sample.x[: sample.effective_length]
    top1 = sample.responsibility.argmax(axis=1)
    add_phase_background(ax, top1, positions, sample.effective_length)
    time = np.linspace(0.0, 1.0, sample.effective_length)
    offsets: list[float] = []
    labels: list[str] = []
    for plot_idx, channel_idx in enumerate(channel_indices[:3]):
        signal = observed[:, channel_idx].astype(np.float64)
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        offset = 2.25 * plot_idx
        offsets.append(offset)
        labels.append(f"Ch{channel_idx + 1}")
        ax.plot(
            time,
            signal + offset,
            color=CHANNEL_COLORS[plot_idx],
            lw=0.85,
            solid_capstyle="round",
        )
    ax.set_title(
        f"({panel_letter}) Class {sample.class_label} · sample {sample.sample_idx}",
        loc="left",
        fontweight="bold",
        pad=3,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Normalized valid time")
    ax.set_ylabel("Channel (z-score)")
    ax.set_yticks(offsets)
    ax.set_yticklabels(labels)
    ax.grid(axis="x", color="#E8E8E8", lw=0.5)


def plot_responsibility(
    ax: mpl.axes.Axes,
    sample: SampleEvidence,
    panel_letter: str,
) -> mpl.collections.QuadMesh:
    responsibility = sample.responsibility
    top1 = responsibility.argmax(axis=1)
    path_text = compact_path_label(top1)
    image = ax.pcolormesh(
        np.arange(responsibility.shape[0] + 1),
        np.arange(responsibility.shape[1] + 1),
        responsibility.T,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        rasterized=False,
    )
    ax.set_xlim(0, responsibility.shape[0])
    ax.set_ylim(responsibility.shape[1], 0)
    title = f"({panel_letter}) Responsibilities"
    if path_text:
        title += f" · {path_text}"
    ax.set_title(title, loc="left", fontweight="bold", pad=7)
    ax.set_xlabel("Valid temporal windows")
    ax.set_ylabel("Phase")
    ax.set_yticks(np.arange(responsibility.shape[1]) + 0.5)
    ax.set_yticklabels([f"P{phase}" for phase in range(responsibility.shape[1])])
    if responsibility.shape[0] > 1:
        tick_positions = np.unique(
            np.rint(np.linspace(0, responsibility.shape[0] - 1, 3)).astype(int)
        )
        ax.set_xticks(tick_positions + 0.5)
        ax.set_xticklabels([str(position + 1) for position in tick_positions])

    strip = ax.inset_axes([0.0, 1.012, 1.0, 0.035])
    strip.pcolormesh(
        np.arange(top1.shape[0] + 1),
        np.array([0.0, 1.0]),
        top1[np.newaxis, :],
        cmap=ListedColormap(PHASE_COLORS),
        vmin=0,
        vmax=len(PHASE_COLORS) - 1,
        shading="flat",
        rasterized=False,
    )
    strip.set_xlim(0, top1.shape[0])
    strip.set_ylim(0, 1)
    strip.set_axis_off()
    return image


def plot_transition_matrix(
    ax: mpl.axes.Axes,
    matrix: np.ndarray,
    class_name: str,
    panel_letter: str,
) -> mpl.collections.QuadMesh:
    n_phases = matrix.shape[0]
    image = ax.pcolormesh(
        np.arange(n_phases + 1),
        np.arange(n_phases + 1),
        matrix,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        rasterized=False,
    )
    ax.set_xlim(0, n_phases)
    ax.set_ylim(n_phases, 0)
    ax.set_aspect("equal")
    ax.set_title(
        f"({panel_letter}) $A_y$ · Class {class_name}",
        loc="left",
        fontweight="bold",
        pad=3,
    )
    ax.set_xticks(np.arange(n_phases) + 0.5)
    ax.set_yticks(np.arange(n_phases) + 0.5)
    ax.set_xticklabels([f"P{phase}" for phase in range(n_phases)])
    ax.set_yticklabels([f"P{phase}" for phase in range(n_phases)])
    ax.set_xlabel("To phase")
    ax.set_ylabel("From phase")
    for row in range(n_phases):
        for col in range(n_phases):
            value = float(matrix[row, col])
            if value > 0.5:
                ax.text(
                    col + 0.5,
                    row + 0.5,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color="white" if value >= 0.65 else "#1F1F1F",
                )
    return image


def make_figure(
    samples: Sequence[SampleEvidence],
    transition_matrices: np.ndarray,
    model: PhasePathNet,
    channel_indices: Sequence[int],
) -> mpl.figure.Figure:
    fig = plt.figure(figsize=(7.2, 3.85))
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=[1.25, 1.15, 0.035, 0.76, 0.035],
        wspace=0.40,
        hspace=0.57,
    )
    heat_image = None
    transition_image = None
    panel_letters = (("a", "b", "c"), ("d", "e", "f"))

    for row, sample in enumerate(samples):
        positions = model.window_partitioner.get_window_positions(sample.x.shape[0])[
            : sample.n_valid_windows
        ]
        ax_trace = fig.add_subplot(grid[row, 0])
        plot_trace(
            ax_trace,
            sample,
            positions,
            channel_indices,
            panel_letters[row][0],
        )

        ax_heat = fig.add_subplot(grid[row, 1])
        heat_image = plot_responsibility(
            ax_heat,
            sample,
            panel_letters[row][1],
        )

        ax_transition = fig.add_subplot(grid[row, 3])
        transition_image = plot_transition_matrix(
            ax_transition,
            transition_matrices[sample.class_idx],
            sample.class_label,
            panel_letters[row][2],
        )

    if heat_image is None or transition_image is None:
        raise RuntimeError("Figure requires exactly two selected samples")
    heat_cax = fig.add_subplot(grid[:, 2])
    heat_colorbar = fig.colorbar(heat_image, cax=heat_cax)
    if heat_colorbar.solids is not None:
        heat_colorbar.solids.set_rasterized(False)
    heat_colorbar.set_ticks([0.0, 0.5, 1.0])
    heat_colorbar.ax.set_title(r"$q_t(k)$", fontsize=6.2, pad=4)
    heat_colorbar.ax.tick_params(labelsize=6.0)

    transition_cax = fig.add_subplot(grid[:, 4])
    transition_colorbar = fig.colorbar(transition_image, cax=transition_cax)
    if transition_colorbar.solids is not None:
        transition_colorbar.solids.set_rasterized(False)
    transition_colorbar.set_ticks([0.0, 0.5, 1.0])
    transition_colorbar.ax.set_title(r"$A_y$", fontsize=6.2, pad=4)
    transition_colorbar.ax.tick_params(labelsize=6.0)

    phase_handles = [
        Patch(facecolor=color, edgecolor="none", label=f"P{phase}")
        for phase, color in enumerate(PHASE_COLORS)
    ]
    fig.legend(
        handles=phase_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.985),
        ncol=len(phase_handles),
        frameon=False,
        handlelength=0.9,
        handleheight=0.7,
        columnspacing=0.8,
        borderaxespad=0.0,
        fontsize=6.2,
    )

    fig.subplots_adjust(left=0.075, right=0.968, bottom=0.105, top=0.90)
    return fig


def save_figure(fig: mpl.figure.Figure, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "fig5_phase_path_evidence"
    outputs: dict[str, str] = {}
    for suffix in ("pdf", "svg", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
        outputs[suffix] = path.name
    plt.close(fig)
    return outputs


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_source_data(
    output_dir: Path,
    dataset: PhasePathDataset,
    model: PhasePathNet,
    records: dict[int, list[SampleEvidence]],
    samples: Sequence[SampleEvidence],
    phase_marginals: np.ndarray,
    transition_matrices: np.ndarray,
    pair_rows: list[dict[str, Any]],
    n_candidates: int,
    train_lengths: np.ndarray,
    test_lengths: np.ndarray,
    metrics: dict[str, float],
    checkpoint_info: dict[str, Any],
    candidate_fraction: float,
    min_padding_run: int,
    padding_atol: float,
    figure_outputs: dict[str, str],
    channel_indices: Sequence[int],
) -> Path:
    source_dir = output_dir / "fig5_phase_path_evidence_source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    marginal_rows = []
    for class_idx in range(dataset.n_classes):
        row: dict[str, Any] = {
            "class_idx": class_idx,
            "class_label": class_label(dataset, class_idx),
            "test_samples": int(np.sum(np.asarray(dataset.y) == class_idx)),
            "correct_samples_masked": len(records[class_idx]),
        }
        row.update(
            {
                f"pi_P{phase}": float(phase_marginals[class_idx, phase])
                for phase in range(model.n_phases)
            }
        )
        marginal_rows.append(row)
    marginal_fields = [
        "class_idx",
        "class_label",
        "test_samples",
        "correct_samples_masked",
        *[f"pi_P{phase}" for phase in range(model.n_phases)],
    ]
    write_csv(source_dir / "class_phase_marginals.csv", marginal_rows, marginal_fields)

    pair_output_rows = []
    for row in pair_rows:
        enriched = dict(row)
        enriched["class_label_a"] = class_label(dataset, int(row["class_idx_a"]))
        enriched["class_label_b"] = class_label(dataset, int(row["class_idx_b"]))
        pair_output_rows.append(enriched)
    pair_fields = [
        "class_idx_a",
        "class_label_a",
        "class_idx_b",
        "class_label_b",
        "marginal_js",
        "phase_similarity",
        "transition_divergence",
        "phase_similarity_rank",
        "high_similarity_candidate",
        "selected",
    ]
    write_csv(source_dir / "class_pair_selection.csv", pair_output_rows, pair_fields)

    selected_ids = {sample.sample_idx for sample in samples}
    catalog_rows = []
    for class_idx, class_samples in records.items():
        if not class_samples:
            continue
        median_gain = float(
            np.median(
                [sample.gain_per_valid_transition for sample in class_samples]
            )
        )
        for sample in class_samples:
            catalog_rows.append(
                {
                    "sample_idx": sample.sample_idx,
                    "class_idx": class_idx,
                    "class_label": sample.class_label,
                    "transition_gain_per_valid_transition": sample.gain_per_valid_transition,
                    "class_median_gain_per_valid_transition": median_gain,
                    "absolute_distance_to_class_median": abs(
                        sample.gain_per_valid_transition - median_gain
                    ),
                    "effective_length": sample.effective_length,
                    "valid_windows": sample.n_valid_windows,
                    "selected": sample.sample_idx in selected_ids,
                }
            )
    catalog_rows.sort(key=lambda row: int(row["sample_idx"]))
    write_csv(
        source_dir / "correct_sample_catalog.csv",
        catalog_rows,
        [
            "sample_idx",
            "class_idx",
            "class_label",
            "transition_gain_per_valid_transition",
            "class_median_gain_per_valid_transition",
            "absolute_distance_to_class_median",
            "effective_length",
            "valid_windows",
            "selected",
        ],
    )

    selected_metadata = []
    for sample in samples:
        prefix = f"class_{sample.class_label}_sample_{sample.sample_idx}"
        observed = sample.x[: sample.effective_length]
        trace_rows = []
        for time_idx in range(sample.effective_length):
            row = {
                "sample_idx": sample.sample_idx,
                "class_idx": sample.class_idx,
                "class_label": sample.class_label,
                "time_idx": time_idx,
                "normalized_valid_time": time_idx / max(sample.effective_length - 1, 1),
            }
            row.update(
                {
                    f"channel_{channel + 1}": float(observed[time_idx, channel])
                    for channel in range(observed.shape[1])
                }
            )
            trace_rows.append(row)
        trace_fields = [
            "sample_idx",
            "class_idx",
            "class_label",
            "time_idx",
            "normalized_valid_time",
            *[f"channel_{channel + 1}" for channel in range(observed.shape[1])],
        ]
        write_csv(source_dir / f"{prefix}_trace.csv", trace_rows, trace_fields)

        positions = model.window_partitioner.get_window_positions(sample.x.shape[0])[
            : sample.n_valid_windows
        ]
        top1 = sample.responsibility.argmax(axis=1)
        responsibility_rows = []
        for window_idx in range(sample.n_valid_windows):
            row = {
                "sample_idx": sample.sample_idx,
                "class_idx": sample.class_idx,
                "class_label": sample.class_label,
                "valid_window_idx": window_idx,
                "window_start": int(positions[window_idx, 0]),
                "window_end_exclusive": int(positions[window_idx, 1]),
                "top1_phase": int(top1[window_idx]),
            }
            row.update(
                {
                    f"q_P{phase}": float(sample.responsibility[window_idx, phase])
                    for phase in range(model.n_phases)
                }
            )
            responsibility_rows.append(row)
        responsibility_fields = [
            "sample_idx",
            "class_idx",
            "class_label",
            "valid_window_idx",
            "window_start",
            "window_end_exclusive",
            "top1_phase",
            *[f"q_P{phase}" for phase in range(model.n_phases)],
        ]
        write_csv(
            source_dir / f"{prefix}_responsibilities.csv",
            responsibility_rows,
            responsibility_fields,
        )

        matrix_rows = []
        matrix = transition_matrices[sample.class_idx]
        for from_phase in range(model.n_phases):
            for to_phase in range(model.n_phases):
                probability = float(matrix[from_phase, to_phase])
                matrix_rows.append(
                    {
                        "class_idx": sample.class_idx,
                        "class_label": sample.class_label,
                        "from_phase": from_phase,
                        "to_phase": to_phase,
                        "probability": probability,
                        "annotated_gt_0_5": probability > 0.5,
                    }
                )
        write_csv(
            source_dir / f"class_{sample.class_label}_transition_matrix.csv",
            matrix_rows,
            [
                "class_idx",
                "class_label",
                "from_phase",
                "to_phase",
                "probability",
                "annotated_gt_0_5",
            ],
        )
        selected_metadata.append(
            {
                "sample_idx": sample.sample_idx,
                "class_idx": sample.class_idx,
                "class_label": sample.class_label,
                "transition_gain_per_valid_transition": sample.gain_per_valid_transition,
                "effective_length": sample.effective_length,
                "valid_windows": sample.n_valid_windows,
                "full_compressed_top1_path": compressed_path(top1),
                "displayed_path": compact_path_label(top1),
            }
        )

    selected_pair_row = next(row for row in pair_output_rows if bool(row["selected"]))
    metadata = {
        "figure": "Figure 5 | Learned phase-path evidence",
        "dataset": "Handwriting",
        "claim": (
            "Classes reuse partially shared phase prototypes while exhibiting different "
            "class-conditioned phase-transition paths."
        ),
        "data_are_placeholder": False,
        "dataset_padded_length": int(dataset.seq_length),
        "effective_length_rule": (
            "Across all channels, find the final cross-time change x[t] != x[t+1]; "
            "effective_length is t + 2, retaining the first endpoint vector and removing "
            "only its subsequent repetitions. Apply this rule only when the final constant "
            f"run contains at least {min_padding_run} points "
            f"(absolute tolerance {padding_atol:g}, relative tolerance 0)."
        ),
        "valid_window_rule": (
            "A window is included only when window_end_exclusive <= effective_length; "
            "partially or fully padded windows are excluded from phase marginals, path score, "
            "IID baseline, transition gain, prototype score, sequence pooling, uncertainty, "
            "and the plotted responsibility map."
        ),
        "representative_gain_definition": (
            "(transition_gain = log p_A - log p_IID) / max(N_valid_windows - 1, 1); "
            "this is the signed, per-valid-transition G_y used by the order-gap experiment."
        ),
        "length_audit": {
            "train": length_summary(train_lengths),
            "test": length_summary(test_lengths),
        },
        "window_size": int(model.window_partitioner.window_size),
        "stride": int(model.window_partitioner.stride),
        "displayed_channels_one_based": [int(index) + 1 for index in channel_indices],
        "channel_selection_rule": "Three channels with the highest training-set variance.",
        "phase_color_key": {
            f"P{phase}": color for phase, color in enumerate(PHASE_COLORS)
        },
        "class_pair_rule": {
            "phase_marginal": "Mean per-sample valid-window responsibility for the true class.",
            "phase_similarity": "1 - normalized Jensen-Shannon divergence of phase marginals.",
            "high_similarity_candidates": (
                f"Top {candidate_fraction:.0%} of eligible class pairs by phase similarity "
                f"({n_candidates} pairs)."
            ),
            "transition_divergence": (
                "Phase-occupancy-weighted mean normalized Jensen-Shannon divergence "
                "between corresponding rows of the two transition matrices."
            ),
            "final_choice": "Maximum transition divergence among high-similarity candidates.",
        },
        "selected_pair": selected_pair_row,
        "sample_rule": (
            "Within each selected class, use the correctly classified sample whose masked, "
            "per-valid-transition gain is closest to the class median; ties use the lower "
            "sample index."
        ),
        "selected_samples": selected_metadata,
        "checkpoint": checkpoint_info,
        "selection_device": "cpu",
        "deterministic_selection": True,
        "checkpoint_training_padding_masked": False,
        "figure_inference_padding_masked": True,
        "retraining_required_for_mask_consistent_model_claims": True,
        "accuracy_audit": metrics,
        "exported_figure_files": figure_outputs,
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "matplotlib": mpl.__version__,
        },
    }
    metadata_path = source_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    caption = (
        "Figure 5 | Learned phase-path evidence on Handwriting. The displayed classes "
        "reuse partially shared phase prototypes but have different class-conditioned "
        "transition paths. The displayed pair was selected by a fixed two-stage rule: "
        f"retain the top {candidate_fraction:.0%} of "
        "eligible class pairs by phase-marginal similarity, then maximize class-conditioned "
        "transition-matrix divergence. For each class, the correctly classified test sample "
        "with masked gain per valid transition closest to the class median is shown. "
        "The gain is (log p_A - log p_IID) / max(N_valid - 1, 1), matching the order-gap "
        "metric. Multichannel "
        "traces and phase responsibilities are restricted to the inferred effective length; "
        "only fully observed windows enter the heatmap and transition-gain calculation. "
        "Matrix values are annotated only when the transition probability exceeds 0.5. "
        "The selected pair has phase-marginal similarity "
        f"{float(selected_pair_row['phase_similarity']):.3f} "
        "(1 - normalized Jensen-Shannon divergence) and transition divergence "
        f"{float(selected_pair_row['transition_divergence']):.3f}. The displayed channels "
        f"are Ch{channel_indices[0] + 1}, Ch{channel_indices[1] + 1}, and "
        f"Ch{channel_indices[2] + 1}, selected by highest training-set variance. "
        "Background bands in the traces and the strips above the heatmaps show the "
        "top-1 phase for each valid temporal region. The class-conditioned matrices "
        "$A_y$ are row-normalized; annotated values are transition probabilities above "
        "0.5. The phase color key is P0 blue, P1 orange, P2 green, P3 teal, and P4 "
        "purple. Responsibility and transition color scales are shared across rows."
    )
    (source_dir / "caption.txt").write_text(caption + "\n", encoding="utf-8")

    audit_note = (
        "AUDIT NOTE: The available Handwriting checkpoint was trained before valid-window "
        "masking was implemented. This figure applies a single effective-length mask during "
        "inference and all derived calculations, but the checkpoint must be retrained with "
        "the same mask before making a training-time padding-invariance claim.\n"
    )
    (source_dir / "checkpoint_mask_audit.txt").write_text(audit_note, encoding="utf-8")
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate real-data Figure 5 phase-path evidence from Handwriting."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--candidate-fraction", type=float, default=0.10)
    parser.add_argument("--min-padding-run", type=int, default=3)
    parser.add_argument("--padding-atol", type=float, default=1e-7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    device = torch.device(args.device)
    if device.type != "cpu":
        raise ValueError(
            "Figure 5 selection is CPU-only so median-sample selection is exactly "
            "reproducible across canonical redraws"
        )
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    train_set = PhasePathDataset(
        str(args.data_dir),
        "Handwriting",
        split="train",
        normalize=True,
    )
    test_set = PhasePathDataset(
        str(args.data_dir),
        "Handwriting",
        split="test",
        normalize=True,
    )
    train_lengths = infer_effective_lengths(
        train_set.X,
        min_padding_run=args.min_padding_run,
        atol=args.padding_atol,
    )
    test_lengths = infer_effective_lengths(
        test_set.X,
        min_padding_run=args.min_padding_run,
        atol=args.padding_atol,
    )
    model, checkpoint_info = load_handwriting_model(test_set, args.checkpoint, device)
    if int(test_lengths.min()) < model.window_partitioner.window_size:
        raise RuntimeError("At least one effective sequence is shorter than a full window")

    records, phase_marginals, transition_matrices, metrics = collect_evidence(
        test_set,
        model,
        test_lengths,
        batch_size=args.batch_size,
        device=device,
    )
    selected_pair, pair_rows, n_candidates = select_class_pair(
        records,
        phase_marginals,
        transition_matrices,
        candidate_fraction=args.candidate_fraction,
    )
    samples = [select_median_gain_sample(records[class_idx]) for class_idx in selected_pair]
    selected_pair_metrics = next(row for row in pair_rows if bool(row["selected"]))
    channel_indices = highest_variance_channels(train_set, train_lengths, n_channels=3)

    figure = make_figure(
        samples,
        transition_matrices,
        model,
        channel_indices,
    )
    figure_outputs = save_figure(figure, args.output_dir)
    metadata_path = write_source_data(
        args.output_dir,
        test_set,
        model,
        records,
        samples,
        phase_marginals,
        transition_matrices,
        pair_rows,
        n_candidates,
        train_lengths,
        test_lengths,
        metrics,
        checkpoint_info,
        args.candidate_fraction,
        args.min_padding_run,
        args.padding_atol,
        figure_outputs,
        channel_indices,
    )

    print("Figure 5 generated from real Handwriting data.")
    print(
        "Selected classes:",
        ", ".join(f"Class {sample.class_label}" for sample in samples),
    )
    print(
        "Selected samples:",
        ", ".join(
            f"{sample.sample_idx} (gain/transition="
            f"{sample.gain_per_valid_transition:.4f}, "
            f"length={sample.effective_length}, windows={sample.n_valid_windows})"
            for sample in samples
        ),
    )
    print(
        f"Pair evidence: phase similarity={selected_pair_metrics['phase_similarity']:.4f}, "
        f"transition divergence={selected_pair_metrics['transition_divergence']:.4f}"
    )
    print(
        f"Accuracy audit: masked={metrics['masked_accuracy']:.4f}, "
        f"legacy-unmasked={metrics['legacy_unmasked_accuracy']:.4f}"
    )
    print(f"Source metadata: {metadata_path}")
    print(
        "WARNING: the available checkpoint was trained without valid-window masking; "
        "retraining is required for mask-consistent training claims."
    )


if __name__ == "__main__":
    main()
