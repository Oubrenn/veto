"""Strict counterfactual order verification for Figure 6.

The diagnostic keeps the encoded windows fixed and changes only their temporal
order.  For a sample with true label ``y`` and ``N`` encoded windows, the score
used everywhere in this file is

    G_y = (log p_A(path | y) - log p_IID(path | y)) / max(N - 1, 1),

where the numerator is the model's own ``transition_gain``.  The signed order
gap is ``G_real - mean(G_shuffle)``.  It is never converted to a magnitude.

The default command trains the full model for ten epochs on six representative
UEA datasets and five seeds. Both sample-level and seed-level source data are
written incrementally so the plotted means and error bars remain auditable and
long runs can resume without repeating completed combinations.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.official_benchmark import make_losses, make_model
from data import CounterfactualGenerator, get_dataloader
from training.train import evaluate, train_epoch
from utils.common import set_seed


DEFAULT_DATASETS = [
    "DuckDuckGeese",
    "Handwriting",
    "LSST",
    "MotorImagery",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
VARIANT_CONFIGS = {
    "Local-only": {
        "no_cf": True,
        "no_trans": True,
        "no_memory": True,
        "path_score_mode": "gain",
        "path_weight_override": 0.0,
    },
    "Raw transition": {
        "no_cf": False,
        "no_trans": False,
        "no_memory": False,
        "path_score_mode": "raw",
        "path_weight_override": None,
    },
    "Phase path only": {
        # P only: retain the class-conditional path scorer, but remove the
        # matched-IID scoring reference and both training auxiliaries M/C.
        "no_cf": True,
        "no_trans": False,
        "no_memory": True,
        "path_score_mode": "raw",
        "path_weight_override": None,
    },
    "Full w/o confirmed memory": {
        # P+R+C: keep matched-IID gain scoring and counterfactual reordering,
        # while disabling the confirmed-memory branch and its loss.
        "no_cf": False,
        "no_trans": False,
        "no_memory": True,
        "path_score_mode": "gain",
        "path_weight_override": None,
    },
    "w/o counterfactual": {
        "no_cf": True,
        "no_trans": False,
        "no_memory": False,
        "path_score_mode": "gain",
        "path_weight_override": None,
    },
    "VETO full": {
        "no_cf": False,
        "no_trans": False,
        "no_memory": False,
        "path_score_mode": "gain",
        "path_weight_override": None,
    },
}
METRIC_DEFINITION = (
    "mean_sample[(transition_gain_real - mean_shuffle(transition_gain_shuffle)) "
    "/ max(n_windows-1,1)]"
)
NORMALIZED_GAP_EPS = 1e-8
NORMALIZED_METRIC_DEFINITION = (
    "mean_sample[(G_real-G_shuffle)/(|G_real|+|G_shuffle|+1e-8)]"
)


def signed_normalized_order_gap(
    real: torch.Tensor,
    shuffled: torch.Tensor,
    eps: float = NORMALIZED_GAP_EPS,
) -> torch.Tensor:
    """Scale a signed order gap without discarding negative evidence."""
    return (real - shuffled) / (real.abs() + shuffled.abs() + eps)


def compute_transition_gain(
    model: torch.nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    window_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the model-native correct-class transition gain per transition.

    This deliberately delegates the path and IID-reference computations to
    ``model.forward_from_embeddings``.  Reconstructing an occurrence score as
    ``mean(q).sum()`` is invalid because phase responsibilities sum to one and
    would therefore create a constant baseline.
    """

    output = model.forward_from_embeddings(embeddings, window_mask=window_mask)
    if "transition_gain" not in output:
        raise KeyError("Model output does not contain 'transition_gain'.")
    gain = output["transition_gain"]
    if gain.ndim != 2:
        raise ValueError(f"Expected transition_gain [B,C], got {tuple(gain.shape)}")
    if labels.ndim != 1 or labels.shape[0] != gain.shape[0]:
        raise ValueError("labels must have shape [B] and match transition_gain")
    row = torch.arange(labels.shape[0], device=labels.device)
    if window_mask is None:
        n_transitions = torch.full(
            (embeddings.shape[0],),
            max(int(embeddings.shape[1]) - 1, 1),
            dtype=gain.dtype,
            device=gain.device,
        )
    else:
        if window_mask.shape != embeddings.shape[:2]:
            raise ValueError("window_mask must match embeddings[:2]")
        n_transitions = (window_mask.sum(dim=1) - 1).clamp_min(1).to(gain.dtype)
    return gain[row, labels] / n_transitions


def _permuted_embeddings(
    embeddings: torch.Tensor,
    generator: torch.Generator,
    window_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Permute only the valid prefix while preserving local embeddings."""

    shuffled = embeddings.clone()
    for sample_idx in range(embeddings.shape[0]):
        n_valid = (
            int(window_mask[sample_idx].sum().item())
            if window_mask is not None
            else int(embeddings.shape[1])
        )
        order_cpu = torch.randperm(n_valid, generator=generator, device="cpu")
        shuffled[sample_idx, :n_valid] = embeddings[
            sample_idx, order_cpu.to(embeddings.device)
        ]
    return shuffled


def score_embeddings_with_shuffles(
    model: torch.nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    n_shuffles: int,
    shuffle_seed: int,
    window_mask: torch.Tensor | None = None,
    shuffle_chunk_size: int = 10,
) -> dict[str, np.ndarray]:
    """Score real and shuffled orders and retain the signed components."""

    if n_shuffles < 1:
        raise ValueError("n_shuffles must be >= 1")
    if shuffle_chunk_size < 1:
        raise ValueError("shuffle_chunk_size must be >= 1")
    model.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(shuffle_seed))
    with torch.no_grad():
        real = compute_transition_gain(model, embeddings, labels, window_mask=window_mask)
        shuffled_scores = []
        completed = 0
        while completed < n_shuffles:
            chunk_size = min(shuffle_chunk_size, n_shuffles - completed)
            chunk = [
                _permuted_embeddings(
                    embeddings,
                    generator,
                    window_mask=window_mask,
                )
                for _ in range(chunk_size)
            ]
            flat_embeddings = torch.cat(chunk, dim=0)
            flat_labels = labels.repeat(chunk_size)
            flat_mask = (
                window_mask.repeat(chunk_size, 1)
                if window_mask is not None
                else None
            )
            chunk_scores = compute_transition_gain(
                model,
                flat_embeddings,
                flat_labels,
                window_mask=flat_mask,
            ).reshape(chunk_size, embeddings.shape[0])
            shuffled_scores.extend(chunk_scores.unbind(dim=0))
            completed += chunk_size
        shuffle_stack = torch.stack(shuffled_scores, dim=0)
        shuffle_mean = shuffle_stack.mean(dim=0)
        shuffle_std = shuffle_stack.std(dim=0, unbiased=n_shuffles > 1)
        signed_gap = real - shuffle_mean
        signed_normalized_gap = signed_normalized_order_gap(real, shuffle_mean)
    return {
        "g_real": real.detach().cpu().numpy(),
        "g_shuffle": shuffle_mean.detach().cpu().numpy(),
        "g_shuffle_std": shuffle_std.detach().cpu().numpy(),
        "delta_ord": signed_gap.detach().cpu().numpy(),
        "signed_normalized_order_gap": signed_normalized_gap.detach().cpu().numpy(),
    }


def evaluate_order_components(
    model: torch.nn.Module,
    loader: Iterable,
    device: str,
    n_shuffles: int,
    max_batches: int,
    shuffle_seed: int,
    dataset: str,
    seed: int,
    shuffle_chunk_size: int,
) -> tuple[dict, list[dict]]:
    """Evaluate true-label gains on a fixed test subset."""

    model.eval()
    sample_rows: list[dict] = []
    sample_offset = 0
    n_windows_seen: set[int] = set()
    with torch.no_grad():
        padding_mask_mode = "none (legacy protocol)"
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            if len(batch) == 2:
                x, labels = batch
                valid_lengths = None
            elif len(batch) == 3:
                x, labels, valid_lengths = batch
            else:
                raise ValueError("Expected loader batches of (x,label) or (x,label,length)")
            x = x.to(device)
            labels = labels.to(device)
            windows = model.window_partitioner.partition(x)
            embeddings = model.encoder(windows)
            window_mask = None
            if valid_lengths is not None:
                valid_lengths = torch.as_tensor(
                    valid_lengths,
                    dtype=torch.long,
                    device=embeddings.device,
                )
                window_mask = model.window_partitioner.get_window_mask(
                    valid_lengths,
                    n_windows=embeddings.shape[1],
                )
                padding_mask_mode = "valid-prefix mask from loader lengths"
            effective_windows = (
                window_mask.sum(dim=1).detach().cpu().numpy()
                if window_mask is not None
                else np.full(labels.shape[0], embeddings.shape[1], dtype=np.int64)
            )
            n_windows_seen.update(int(value) for value in effective_windows)
            components = score_embeddings_with_shuffles(
                model,
                embeddings,
                labels,
                n_shuffles=n_shuffles,
                shuffle_seed=shuffle_seed + batch_idx * 100_003,
                window_mask=window_mask,
                shuffle_chunk_size=shuffle_chunk_size,
            )
            labels_cpu = labels.detach().cpu().numpy()
            for local_idx in range(labels.shape[0]):
                n_windows = int(effective_windows[local_idx])
                sample_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "sample_idx": sample_offset + local_idx,
                        "label": int(labels_cpu[local_idx]),
                        "n_windows": n_windows,
                        "normalization_transitions": max(n_windows - 1, 1),
                        "g_real": float(components["g_real"][local_idx]),
                        "g_shuffle": float(components["g_shuffle"][local_idx]),
                        "g_shuffle_std": float(components["g_shuffle_std"][local_idx]),
                        "delta_ord": float(components["delta_ord"][local_idx]),
                        "signed_normalized_order_gap": float(
                            components["signed_normalized_order_gap"][local_idx]
                        ),
                        "n_shuffles": int(n_shuffles),
                        "padding_mask": padding_mask_mode,
                        "metric_definition": METRIC_DEFINITION,
                    }
                )
            sample_offset += int(labels.shape[0])

    if not sample_rows:
        raise RuntimeError("No test samples were evaluated for the order diagnostic.")
    frame = pd.DataFrame(sample_rows)
    seed_row = {
        "dataset": dataset,
        "seed": int(seed),
        "status": "ok",
        "n_samples": int(len(frame)),
        "n_windows_min": int(min(n_windows_seen)),
        "n_windows_max": int(max(n_windows_seen)),
        "n_shuffles": int(n_shuffles),
        "diagnostic_batches": int(max_batches),
        "g_real": float(frame["g_real"].mean()),
        "g_real_sample_std": float(frame["g_real"].std(ddof=1)),
        "g_shuffle": float(frame["g_shuffle"].mean()),
        "g_shuffle_sample_std": float(frame["g_shuffle"].std(ddof=1)),
        "delta_ord": float(frame["delta_ord"].mean()),
        "delta_ord_sample_std": float(frame["delta_ord"].std(ddof=1)),
        "signed_normalized_order_gap": float(
            frame["signed_normalized_order_gap"].mean()
        ),
        "signed_normalized_order_gap_sample_std": float(
            frame["signed_normalized_order_gap"].std(ddof=1)
        ),
        "positive_sample_ratio": float((frame["delta_ord"] > 0).mean()),
        "normalization": "per_transition=max(n_windows-1,1)",
        "padding_mask": padding_mask_mode,
        "metric_definition": METRIC_DEFINITION,
        "normalized_metric_definition": NORMALIZED_METRIC_DEFINITION,
    }
    # Numerical traceability invariant: mean(real - shuffled) must equal the
    # difference between the two reported component means.
    if not np.isclose(
        seed_row["delta_ord"],
        seed_row["g_real"] - seed_row["g_shuffle"],
        rtol=1e-6,
        atol=1e-9,
    ):
        raise AssertionError("Signed order-gap components are not internally consistent.")
    return seed_row, sample_rows


def _training_namespace(
    args: argparse.Namespace,
    variant: str,
) -> SimpleNamespace:
    """Expose the fields expected by the project's benchmark helpers."""

    config = VARIANT_CONFIGS[variant]
    return SimpleNamespace(
        model="veto",
        label_smoothing=args.label_smoothing,
        no_cf=config["no_cf"],
        no_trans=config["no_trans"],
        no_memory=config["no_memory"],
        n_phases=args.n_phases,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        stride=args.stride,
        backbone=args.backbone,
        transition_mode=args.transition_mode,
        prototype_mode="class",
        head_mode="veto",
        path_score_mode=config["path_score_mode"],
        path_weight_override=config["path_weight_override"],
        no_uncertainty=False,
        memory_update_mode="confirmed",
        device=args.device,
    )


def train_fresh_model(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    variant: str,
) -> tuple[torch.nn.Module, Iterable, dict]:
    """Train a one-epoch full model using the same inexpensive run-fast setup."""

    set_seed(seed)
    helper_args = _training_namespace(args, variant)
    train_loader = get_dataloader(
        args.data_path,
        dataset,
        batch_size=args.batch_size,
        split="train",
        normalize=not args.no_normalize,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        return_lengths=True,
    )
    test_loader = get_dataloader(
        args.data_path,
        dataset,
        batch_size=args.batch_size,
        split="test",
        normalize=not args.no_normalize,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        return_lengths=True,
    )
    model = make_model(helper_args, train_loader.dataset)
    losses = make_losses(helper_args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    cf_generator = (
        None
        if helper_args.no_cf
        else CounterfactualGenerator(min_phase_length=3)
    )
    loss_weights = {
        "cls": 1.0,
        "cf": 0.0 if helper_args.no_cf else args.cf_weight,
        "trans": 0.0 if helper_args.no_trans else args.trans_weight,
        "mem": 0.0 if helper_args.no_memory else args.mem_weight,
        "tf": 0.0,
    }
    started = time.perf_counter()
    final_loss = np.nan
    final_train_acc = np.nan
    for epoch in range(1, args.epochs + 1):
        final_loss, final_train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            losses,
            cf_generator,
            args.device,
            epoch,
            loss_weights=loss_weights,
            cf_start_epoch=args.cf_start_epoch,
        )
        scheduler.step()
        print(
            f"[{variant} | {dataset} seed={seed}] epoch {epoch}/{args.epochs}: "
            f"loss={final_loss:.4f}, train_acc={final_train_acc:.4f}",
            flush=True,
        )
    test_acc = np.nan
    if args.evaluate_test_accuracy:
        test_acc, _, _ = evaluate(model, test_loader, args.device)
    metadata = {
        "variant": variant,
        "epochs": int(args.epochs),
        "cf_start_epoch": int(args.cf_start_epoch) if not helper_args.no_cf else np.nan,
        "counterfactual_active": bool(
            not helper_args.no_cf and args.cf_start_epoch <= args.epochs
        ),
        "final_train_loss": float(final_loss),
        "final_train_acc": float(final_train_acc),
        "test_acc": float(test_acc),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return model, test_loader, metadata


def _load_serialized_model(path: str, device: str) -> torch.nn.Module:
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, torch.nn.Module):
        model = payload
    elif isinstance(payload, dict) and isinstance(payload.get("model"), torch.nn.Module):
        model = payload["model"]
    else:
        raise ValueError(
            "Checkpoint mode requires a serialized model object (or {'model': model}); "
            "a bare state_dict is insufficient without its architecture metadata."
        )
    return model.to(device)


def _write_outputs(
    seed_rows: list[dict],
    sample_rows: list[dict],
    output: Path,
    sample_output: Path,
    json_output: Path,
    args: argparse.Namespace,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(seed_rows).to_csv(output, index=False)
    pd.DataFrame(sample_rows).to_csv(sample_output, index=False)
    payload = {
        "metric_definition": METRIC_DEFINITION,
        "normalization": "per_transition=max(n_windows-1,1)",
        "signed_normalized_gap": NORMALIZED_METRIC_DEFINITION,
        "padding_mask": "valid lengths inferred by the dataset cache and used by training/evaluation",
        "datasets": list(args.datasets),
        "seeds": list(args.seeds),
        "n_shuffles": int(args.n_shuffles),
        "shuffle_chunk_size": int(args.shuffle_chunk_size),
        "diagnostic_batches": int(args.diagnostic_batches),
        "training_epochs": int(args.epochs),
        "task": args.task,
        "variants": [
            variant
            for variant in VARIANT_CONFIGS
            if any(row.get("variant") == variant for row in seed_rows)
        ],
        "mechanism_protocol": (
            f"{args.epochs} training epochs; counterfactual-enabled variants use "
            f"cf_start_epoch={args.cf_start_epoch}"
        ),
        "rows": seed_rows,
    }
    json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _resume_protocol_matches(
    row: pd.Series,
    samples: pd.DataFrame,
    args: argparse.Namespace,
    variant: str,
) -> bool:
    """Return whether an existing combination is complete for this protocol."""

    try:
        expected_cf = bool(
            not VARIANT_CONFIGS[variant]["no_cf"]
            and args.cf_start_epoch <= args.epochs
        )
        if str(row.get("status", "")) != "ok":
            return False
        if int(float(row.get("epochs", -1))) != args.epochs:
            return False
        if int(float(row.get("n_shuffles", -1))) != args.n_shuffles:
            return False
        if int(float(row.get("diagnostic_batches", -1))) != args.diagnostic_batches:
            return False
        if _truthy(row.get("counterfactual_active", False)) != expected_cf:
            return False
        if "valid-prefix" not in str(row.get("padding_mask", "")).lower():
            return False
        expected_samples = int(float(row.get("n_samples", -1)))
    except (TypeError, ValueError):
        return False

    if expected_samples < 1 or len(samples) != expected_samples:
        return False
    if samples.empty:
        return False
    sample_shuffles = pd.to_numeric(samples["n_shuffles"], errors="coerce")
    if bool(sample_shuffles.isna().any()) or not bool(
        (sample_shuffles.astype(int) == args.n_shuffles).all()
    ):
        return False
    return not bool(
        samples["padding_mask"].astype(str).str.contains("legacy", case=False).any()
    )


def _load_resume_outputs(
    args: argparse.Namespace,
    variants: list[str],
) -> tuple[list[dict], list[dict], set[tuple[str, str, int]]]:
    """Load valid completed rows and remove stale rows for pending targets."""

    output = Path(args.output)
    sample_output = Path(args.sample_output)
    if not args.resume or not output.is_file() or not sample_output.is_file():
        return [], [], set()

    seed_frame = pd.read_csv(output)
    sample_frame = pd.read_csv(sample_output)
    required_seed = {"variant", "dataset", "seed", "status"}
    required_sample = {"variant", "dataset", "seed", "n_shuffles", "padding_mask"}
    if not required_seed.issubset(seed_frame.columns):
        raise ValueError("Cannot resume: seed output lacks required identity columns")
    if not required_sample.issubset(sample_frame.columns):
        raise ValueError("Cannot resume: sample output lacks required identity columns")

    seed_frame["seed"] = pd.to_numeric(seed_frame["seed"], errors="raise").astype(int)
    sample_frame["seed"] = pd.to_numeric(sample_frame["seed"], errors="raise").astype(int)
    target_keys = {
        (variant, dataset, int(seed))
        for variant in variants
        for dataset in args.datasets
        for seed in args.seeds
    }
    completed: set[tuple[str, str, int]] = set()
    stale: set[tuple[str, str, int]] = set()
    for key in target_keys:
        variant, dataset, seed = key
        seed_subset = seed_frame[
            seed_frame["variant"].eq(variant)
            & seed_frame["dataset"].eq(dataset)
            & seed_frame["seed"].eq(seed)
        ]
        sample_subset = sample_frame[
            sample_frame["variant"].eq(variant)
            & sample_frame["dataset"].eq(dataset)
            & sample_frame["seed"].eq(seed)
        ]
        if len(seed_subset) == 1 and _resume_protocol_matches(
            seed_subset.iloc[0], sample_subset, args, variant
        ):
            completed.add(key)
        else:
            stale.add(key)

    if stale:
        seed_keys = list(zip(seed_frame["variant"], seed_frame["dataset"], seed_frame["seed"]))
        sample_keys = list(
            zip(sample_frame["variant"], sample_frame["dataset"], sample_frame["seed"])
        )
        seed_frame = seed_frame[[key not in stale for key in seed_keys]].copy()
        sample_frame = sample_frame[[key not in stale for key in sample_keys]].copy()

    return (
        seed_frame.to_dict(orient="records"),
        sample_frame.to_dict(orient="records"),
        completed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["order", "variants"], default="order")
    parser.add_argument("--data_path", default="diagnostics/uea_target_cache")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANT_CONFIGS),
        default=None,
        help="Optional subset for --task variants; quote names containing spaces",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only complete rows matching the requested protocol",
    )
    parser.add_argument("--model_path", default=None, help="Optional serialized model; single-dataset mode only")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--cf_weight", type=float, default=0.5)
    parser.add_argument("--trans_weight", type=float, default=0.1)
    parser.add_argument("--mem_weight", type=float, default=0.1)
    parser.add_argument("--cf_start_epoch", type=int, default=5)
    parser.add_argument("--n_phases", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--backbone", choices=["inception", "resnet", "fcn"], default="inception")
    parser.add_argument(
        "--transition_mode",
        choices=["uniform", "free", "class_independent", "neural", "attention"],
        default="neural",
    )
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_shuffles", type=int, default=100)
    parser.add_argument("--shuffle_chunk_size", type=int, default=10)
    parser.add_argument(
        "--diagnostic_batches",
        type=int,
        default=0,
        help="Maximum test batches; 0 evaluates the complete test split",
    )
    parser.add_argument("--evaluate_test_accuracy", action="store_true")
    parser.add_argument(
        "--output",
        default=None,
    )
    parser.add_argument(
        "--sample_output",
        default=None,
    )
    parser.add_argument(
        "--json_output",
        default=None,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if args.n_shuffles < 1:
        raise ValueError("--n_shuffles must be >= 1")
    if args.shuffle_chunk_size < 1:
        raise ValueError("--shuffle_chunk_size must be >= 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.model_path and (len(args.datasets) != 1 or len(args.seeds) != 1):
        raise ValueError("--model_path supports exactly one dataset and one evaluation seed")
    if args.model_path and args.task != "order":
        raise ValueError("--model_path is only supported for --task order")
    if args.output is None:
        stem = "fig6_order" if args.task == "order" else "fig6_variant_order"
        args.output = f"diagnostics/paper_figures/source_data/{stem}_seed_metrics.csv"
    if args.sample_output is None:
        stem = "fig6_order" if args.task == "order" else "fig6_variant_order"
        args.sample_output = f"diagnostics/paper_figures/source_data/{stem}_sample_metrics.csv"
    if args.json_output is None:
        stem = "fig6_order" if args.task == "order" else "fig6_variant_order"
        args.json_output = f"diagnostics/paper_figures/source_data/{stem}_seed_metrics.json"

    if args.task == "order":
        if args.variants and args.variants != ["VETO full"]:
            raise ValueError("--task order only supports the VETO full variant")
        variants = ["VETO full"]
    else:
        variants = list(args.variants or VARIANT_CONFIGS)
    args.resolved_variants = variants
    seed_rows, sample_rows, completed = _load_resume_outputs(args, variants)
    failures = 0
    for variant in variants:
        variant_index = list(VARIANT_CONFIGS).index(variant)
        for dataset_index, dataset in enumerate(args.datasets):
            for seed in args.seeds:
                key = (variant, dataset, int(seed))
                if key in completed:
                    print(f"SKIP complete: {variant} | {dataset} | seed {seed}", flush=True)
                    continue
                print(f"\n=== {variant} | {dataset} | seed {seed} ===", flush=True)
                model = None
                try:
                    if args.model_path:
                        model = _load_serialized_model(args.model_path, args.device)
                        test_loader = get_dataloader(
                            args.data_path,
                            dataset,
                            batch_size=args.batch_size,
                            split="test",
                            normalize=not args.no_normalize,
                            shuffle=False,
                            num_workers=args.num_workers,
                            drop_last=False,
                        )
                        train_meta = {
                            "variant": variant,
                            "epochs": 0,
                            "test_acc": np.nan,
                            "elapsed_seconds": 0.0,
                        }
                    else:
                        model, test_loader, train_meta = train_fresh_model(
                            args, dataset, seed, variant
                        )
                    seed_row, current_samples = evaluate_order_components(
                        model,
                        test_loader,
                        args.device,
                        n_shuffles=args.n_shuffles,
                        max_batches=args.diagnostic_batches,
                        shuffle_seed=(
                            9_000_001
                            + variant_index * 10_000_019
                            + dataset_index * 1_000_003
                            + seed * 10_007
                        ),
                        dataset=dataset,
                        seed=seed,
                        shuffle_chunk_size=args.shuffle_chunk_size,
                    )
                    seed_row.update(train_meta)
                    seed_rows.append(seed_row)
                    for row in current_samples:
                        row["variant"] = variant
                    sample_rows.extend(current_samples)
                    print(
                        f"G_real={seed_row['g_real']:.6g}, "
                        f"G_shuffle={seed_row['g_shuffle']:.6g}, "
                        f"signed delta={seed_row['delta_ord']:.6g}",
                        flush=True,
                    )
                except Exception as exc:  # preserve partial source data for diagnosis
                    failures += 1
                    error = "".join(
                        traceback.format_exception_only(type(exc), exc)
                    ).strip()
                    print(f"FAILED: {error}", flush=True)
                    seed_rows.append(
                        {
                            "variant": variant,
                            "dataset": dataset,
                            "seed": int(seed),
                            "status": "failed",
                            "error": error,
                            "n_shuffles": int(args.n_shuffles),
                            "diagnostic_batches": int(args.diagnostic_batches),
                            "normalization": "per_transition=max(n_windows-1,1)",
                            "padding_mask": "none (legacy protocol)",
                            "metric_definition": METRIC_DEFINITION,
                            "normalized_metric_definition": NORMALIZED_METRIC_DEFINITION,
                        }
                    )
                finally:
                    if model is not None:
                        del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                _write_outputs(
                    seed_rows,
                    sample_rows,
                    Path(args.output),
                    Path(args.sample_output),
                    Path(args.json_output),
                    args,
                )

    print(f"\nWrote {args.output}")
    print(f"Wrote {args.sample_output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
