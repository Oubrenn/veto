"""Controlled synthetic phase-order tables for the manuscript.

This script builds two diagnostics:

1. Exact matched-occurrence: every sample has the same hard phase-count vector.
   Labels differ only through class-specific transition preferences.
2. Marginal-frequency shift: training and matched test data follow the same
   transition skeletons, while mild/severe test sets apply a class-independent
   duration/occupancy multiplier.

The VETO rows are deliberately transparent synthetic variants:
- local-only: class-conditional IID occurrence score.
- raw transition: class-conditional Markov path likelihood.
- full model: Markov path likelihood minus a class-conditional IID reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.backbones import InceptionTime
from utils.common import set_seed


METHODS = [
    "Chance level",
    "Inception-style backbone",
    "Transformer backbone",
    "VETO local-only",
    "VETO raw transition",
    "VETO full model",
]

VETO_METHODS = [
    ("VETO local-only", "local_only"),
    ("VETO raw transition", "raw_transition"),
    ("VETO full model", "full"),
]


class InceptionClassifier(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, embed_dim: int = 64):
        super().__init__()
        self.encoder = InceptionTime(
            in_channels=n_channels,
            embed_dim=embed_dim,
            n_inception_modules=2,
            inception_channels=16,
        )
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        seq_length: int,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, seq_length + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.position[:, : h.shape[1], :]
        h = self.encoder(h)
        return self.classifier(self.norm(h[:, 0]))


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.maximum(matrix, 1e-12)
    return matrix / matrix.sum(axis=-1, keepdims=True)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.maximum(vector, 1e-12)
    return vector / vector.sum()


def class_steps(n_classes: int) -> list[int]:
    base = [1, -1, 2, -2, 3, -3]
    return [base[i % len(base)] for i in range(n_classes)]


def make_transition_bank(
    occurrence: np.ndarray,
    steps: list[int],
    order_strength: float,
    stay_strength: float,
    secondary_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build factorized transition matrices with class-specific order bias."""
    n_classes, n_phases = occurrence.shape
    init = occurrence.copy()
    transitions = np.zeros((n_classes, n_phases, n_phases), dtype=np.float64)
    for class_id in range(n_classes):
        step = steps[class_id]
        for phase in range(n_phases):
            logits = np.log(occurrence[class_id] + 1e-12)
            logits[(phase + step) % n_phases] += order_strength
            logits[(phase + 2 * step) % n_phases] += secondary_strength
            logits[phase] += stay_strength
            logits -= logits.max()
            transitions[class_id, phase] = np.exp(logits)
    return init.astype(np.float32), normalize_rows(transitions).astype(np.float32)


def make_uniform_occurrence(n_classes: int, n_phases: int) -> np.ndarray:
    return np.full((n_classes, n_phases), 1.0 / n_phases, dtype=np.float64)


def make_class_occurrence(n_classes: int, n_phases: int, strength: float) -> np.ndarray:
    priors = np.ones((n_classes, n_phases), dtype=np.float64)
    for class_id in range(n_classes):
        first = (2 * class_id) % n_phases
        second = (2 * class_id + 1) % n_phases
        priors[class_id, first] += strength
        priors[class_id, second] += strength * 0.55
        priors[class_id, (first + 3) % n_phases] += strength * 0.20
    priors /= priors.sum(axis=-1, keepdims=True)
    return priors


def apply_duration_multiplier(occurrence: np.ndarray, multiplier: np.ndarray) -> np.ndarray:
    shifted = occurrence * multiplier.reshape(1, -1)
    shifted /= shifted.sum(axis=-1, keepdims=True)
    return shifted


def sample_markov_paths(
    init: np.ndarray,
    transitions: np.ndarray,
    samples_per_class: int,
    path_length: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    n_classes, n_phases = init.shape
    paths: list[np.ndarray] = []
    labels: list[int] = []
    for class_id in range(n_classes):
        for _ in range(samples_per_class):
            path = np.empty(path_length, dtype=np.int64)
            path[0] = int(rng.choice(n_phases, p=init[class_id]))
            for t in range(1, path_length):
                path[t] = int(rng.choice(n_phases, p=transitions[class_id, path[t - 1]]))
            paths.append(path)
            labels.append(class_id)
    return paths, np.asarray(labels, dtype=np.int64)


def sample_fixed_count_paths(
    transitions: np.ndarray,
    counts: np.ndarray,
    samples_per_class: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    n_classes, n_phases, _ = transitions.shape
    paths: list[np.ndarray] = []
    labels: list[int] = []
    init_probs = counts / counts.sum()
    for class_id in range(n_classes):
        for _ in range(samples_per_class):
            remaining = counts.astype(np.int64).copy()
            path = np.empty(int(counts.sum()), dtype=np.int64)
            first = int(rng.choice(n_phases, p=init_probs))
            path[0] = first
            remaining[first] -= 1
            for t in range(1, len(path)):
                weights = transitions[class_id, path[t - 1]].astype(np.float64).copy()
                weights *= (remaining > 0)
                weights *= np.maximum(remaining, 0) ** 0.75
                if weights.sum() <= 0:
                    weights = (remaining > 0).astype(np.float64)
                weights = normalize_vector(weights)
                phase = int(rng.choice(n_phases, p=weights))
                path[t] = phase
                remaining[phase] -= 1
            paths.append(path)
            labels.append(class_id)
    return paths, np.asarray(labels, dtype=np.int64)


def paths_to_observations(
    paths: list[np.ndarray],
    prototypes: np.ndarray,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    observations = []
    for path in paths:
        x = prototypes[path] + rng.normal(
            0.0,
            noise_std,
            size=(len(path), prototypes.shape[1]),
        )
        observations.append(x.astype(np.float32))
    return np.stack(observations, axis=0)


def paths_to_assignments(
    paths: list[np.ndarray],
    n_phases: int,
    assignment_noise: float,
    jitter: float,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    assignments = []
    off_value = assignment_noise / max(n_phases - 1, 1)
    for path in paths:
        q = np.full((len(path), n_phases), off_value, dtype=np.float32)
        q[np.arange(len(path)), path] = 1.0 - assignment_noise
        if jitter > 0:
            q += rng.normal(0.0, jitter, size=q.shape).astype(np.float32)
        q = np.maximum(q, 1e-7)
        q /= q.sum(axis=-1, keepdims=True)
        assignments.append(q.astype(np.float32))
    return assignments


def make_prototypes(n_phases: int, n_channels: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    prototypes = rng.normal(0.0, 1.0, size=(n_phases, n_channels))
    trend = np.linspace(-0.8, 0.8, n_channels, dtype=np.float64)
    for phase in range(n_phases):
        prototypes[phase] += math.sin(phase * 0.7) * trend
    return prototypes.astype(np.float32)


def build_exact_dataset(args, seed: int, samples_per_class: int, split_offset: int) -> dict:
    occurrence = make_uniform_occurrence(args.n_classes, args.n_phases)
    _, transitions = make_transition_bank(
        occurrence,
        class_steps(args.n_classes),
        args.exact_order_strength,
        args.exact_stay_strength,
        args.exact_secondary_strength,
    )
    counts = np.full(args.n_phases, args.exact_count_per_phase, dtype=np.int64)
    paths, labels = sample_fixed_count_paths(
        transitions,
        counts,
        samples_per_class,
        seed + split_offset,
    )
    q = paths_to_assignments(
        paths,
        args.n_phases,
        args.assignment_noise,
        args.assignment_jitter,
        seed + split_offset + 101,
    )
    return {
        "paths": paths,
        "labels": labels,
        "q": q,
        "true_occurrence": occurrence.astype(np.float32),
        "true_transitions": transitions.astype(np.float32),
    }


def build_shift_dataset(
    args,
    seed: int,
    samples_per_class: int,
    split_offset: int,
    shift_level: str,
) -> dict:
    base_occurrence = make_class_occurrence(args.n_classes, args.n_phases, args.shift_occurrence_strength)
    if shift_level == "matched":
        multiplier = np.ones(args.n_phases, dtype=np.float64)
    elif shift_level == "mild":
        multiplier = np.asarray(args.mild_duration, dtype=np.float64)
    elif shift_level == "severe":
        multiplier = np.asarray(args.severe_duration, dtype=np.float64)
    else:
        raise ValueError(f"Unknown shift level: {shift_level}")
    occurrence = apply_duration_multiplier(base_occurrence, multiplier)
    init, transitions = make_transition_bank(
        occurrence,
        class_steps(args.n_classes),
        args.shift_order_strength,
        args.shift_stay_strength,
        args.shift_secondary_strength,
    )
    paths, labels = sample_markov_paths(
        init,
        transitions,
        samples_per_class,
        args.shift_path_length,
        seed + split_offset,
    )
    q = paths_to_assignments(
        paths,
        args.n_phases,
        args.assignment_noise,
        args.assignment_jitter,
        seed + split_offset + 101,
    )
    return {
        "paths": paths,
        "labels": labels,
        "q": q,
        "true_occurrence": occurrence.astype(np.float32),
        "true_transitions": transitions.astype(np.float32),
    }


def estimate_veto_parameters(train_data: dict, args) -> dict:
    labels = train_data["labels"]
    smooth = args.estimation_smooth
    n_classes = args.n_classes
    n_phases = args.n_phases
    init_counts = np.full((n_classes, n_phases), smooth, dtype=np.float64)
    transition_counts = np.full((n_classes, n_phases, n_phases), smooth, dtype=np.float64)
    occurrence_counts = np.full((n_classes, n_phases), smooth, dtype=np.float64)

    for q_np, label in zip(train_data["q"], labels):
        q = q_np.astype(np.float64)
        init_counts[label] += q[0]
        occurrence_counts[label] += q.sum(axis=0)
        transition_counts[label] += q[:-1].T @ q[1:]

    return {
        "init": (init_counts / init_counts.sum(axis=1, keepdims=True)).astype(np.float32),
        "transitions": (
            transition_counts / transition_counts.sum(axis=2, keepdims=True)
        ).astype(np.float32),
        "occurrence": (
            occurrence_counts / occurrence_counts.sum(axis=1, keepdims=True)
        ).astype(np.float32),
    }


def stack_q(q_list: list[np.ndarray], device: torch.device) -> torch.Tensor:
    return torch.tensor(np.stack(q_list, axis=0), dtype=torch.float32, device=device)


def score_components(q: torch.Tensor, params: dict, device: torch.device) -> dict[str, torch.Tensor]:
    init = torch.tensor(params["init"], dtype=torch.float32, device=device)
    transitions = torch.tensor(params["transitions"], dtype=torch.float32, device=device)
    occurrence = torch.tensor(params["occurrence"], dtype=torch.float32, device=device)

    log_q = torch.log(q + 1e-10)
    log_init = torch.log(init + 1e-10)
    log_trans = torch.log(transitions + 1e-10)
    log_occ = torch.log(occurrence + 1e-10)

    log_alpha = log_init.unsqueeze(0) + log_q[:, 0, :].unsqueeze(1)
    for t in range(1, q.shape[1]):
        log_alpha = torch.logsumexp(log_alpha.unsqueeze(3) + log_trans.unsqueeze(0), dim=2)
        log_alpha = log_alpha + log_q[:, t, :].unsqueeze(1)
    path = torch.logsumexp(log_alpha, dim=-1)

    iid = torch.einsum("mnk,ck->mc", q, log_occ)
    local = iid
    gain = path - iid
    return {"local_only": local, "raw_transition": path, "full": gain}


def score_veto(data: dict, params: dict, mode: str, device: torch.device) -> np.ndarray:
    q = stack_q(data["q"], device)
    with torch.no_grad():
        scores = score_components(q, params, device)[mode]
    return scores.detach().cpu().numpy()


def predict_with_random_ties(scores: np.ndarray, seed: int, tol: float = 1e-8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    preds = np.empty(scores.shape[0], dtype=np.int64)
    for idx, row in enumerate(scores):
        max_value = row.max()
        candidates = np.flatnonzero(np.abs(row - max_value) <= tol)
        preds[idx] = int(rng.choice(candidates))
    return preds


def evaluate_veto_method(
    data: dict,
    params: dict,
    name: str,
    mode: str,
    device: torch.device,
    tie_seed: int,
) -> dict:
    labels = data["labels"]
    scores = score_veto(data, params, mode, device)
    preds = predict_with_random_ties(scores, tie_seed)
    return {
        "method": name,
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def compute_order_gap(
    data: dict,
    params: dict,
    mode: str,
    n_permutations: int,
    seed: int,
    device: torch.device,
) -> float:
    labels = data["labels"]
    q = stack_q(data["q"], device)
    with torch.no_grad():
        base = score_components(q, params, device)[mode]
        base_true = base[torch.arange(len(labels), device=device), torch.tensor(labels, device=device)]

    rng = np.random.default_rng(seed)
    shuffled_true = []
    for _ in range(n_permutations):
        shuffled = [item[rng.permutation(item.shape[0])] for item in data["q"]]
        q_shuf = stack_q(shuffled, device)
        with torch.no_grad():
            scores = score_components(q_shuf, params, device)[mode]
            true_scores = scores[
                torch.arange(len(labels), device=device),
                torch.tensor(labels, device=device),
            ]
        shuffled_true.append(true_scores)
    shuffled_mean = torch.stack(shuffled_true, dim=0).mean(dim=0)
    gap = (base_true - shuffled_mean).mean().item() / max(q.shape[1] - 1, 1)
    return float(gap)


def train_backbone(
    name: str,
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    args,
    device: torch.device,
) -> dict[str, dict]:
    model = model.to(device)
    train_ds = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.long),
    )
    generator = torch.Generator()
    generator.manual_seed(args.loader_seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for _ in range(args.epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

    out: dict[str, dict] = {}
    model.eval()
    with torch.no_grad():
        for split, (x_np, y_np) in eval_sets.items():
            preds_all = []
            for start in range(0, len(x_np), args.eval_batch_size):
                x = torch.tensor(
                    x_np[start : start + args.eval_batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                preds_all.append(model(x).argmax(dim=1).cpu().numpy())
            preds = np.concatenate(preds_all, axis=0)
            out[split] = {
                "method": name,
                "accuracy": float(accuracy_score(y_np, preds)),
                "macro_f1": float(f1_score(y_np, preds, average="macro", zero_division=0)),
            }
    return out


def aggregate_metric(rows: list[dict], key: str) -> tuple[float | None, float | None]:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None, None
    values_np = np.asarray(values, dtype=np.float64)
    return float(values_np.mean()), float(values_np.std(ddof=0))


def fmt_metric(mean: float | None, std: float | None, show_std: bool) -> str:
    if mean is None:
        return "--"
    if abs(mean) < 0.0005:
        mean = 0.0
    if show_std:
        return f"{mean:.3f} $\\pm$ {std:.3f}"
    return f"{mean:.3f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def make_exact_rows(seed_rows: list[list[dict]], args) -> list[dict]:
    summary = []
    for method in METHODS:
        matched = [row for rows in seed_rows for row in rows if row["method"] == method]
        item = {"method": method}
        for key in ["accuracy", "macro_f1", "order_gap"]:
            mean, std = aggregate_metric(matched, key)
            item[key] = mean
            item[f"{key}_std"] = std
        summary.append(item)
    return summary


def make_shift_rows(seed_rows: list[dict], args) -> list[dict]:
    summary = []
    methods = METHODS[1:]
    for method in methods:
        matched = [row for row in seed_rows if row["method"] == method]
        item = {"method": method}
        for key in ["matched", "mild", "severe", "degradation"]:
            mean, std = aggregate_metric(matched, key)
            item[key] = mean
            item[f"{key}_std"] = std
        summary.append(item)
    return summary


def write_exact_tex(rows: list[dict], args) -> None:
    rows_path = Path(args.exact_tex_rows)
    table_path = Path(args.exact_tex_table)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in rows:
        lines.append(
            f"{row['method']} & {fmt_metric(row['accuracy'], row['accuracy_std'], args.show_std)} "
            f"& {fmt_metric(row['macro_f1'], row['macro_f1_std'], args.show_std)} "
            f"& {fmt_metric(row['order_gap'], row['order_gap_std'], args.show_std)} \\\\"
        )
    rows_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    table = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & Accuracy $\\uparrow$ & Macro-F1 $\\uparrow$ & $\\Delta_{\\mathrm{ord}}\\uparrow$ \\\\",
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    table_path.write_text("\n".join(table), encoding="utf-8")


def write_shift_tex(rows: list[dict], args) -> None:
    rows_path = Path(args.shift_tex_rows)
    table_path = Path(args.shift_tex_table)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in rows:
        lines.append(
            f"{row['method']} & {fmt_metric(row['matched'], row['matched_std'], args.show_std)} "
            f"& {fmt_metric(row['mild'], row['mild_std'], args.show_std)} "
            f"& {fmt_metric(row['severe'], row['severe_std'], args.show_std)} "
            f"& {fmt_metric(row['degradation'], row['degradation_std'], args.show_std)} \\\\"
        )
    rows_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    table = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Matched & Mild shift & Severe shift & Degradation $\\downarrow$ \\\\",
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    table_path.write_text("\n".join(table), encoding="utf-8")


def run_exact_seed(seed: int, args, device: torch.device) -> list[dict]:
    set_seed(seed)
    args.loader_seed = seed + 7000
    train_data = build_exact_dataset(args, seed, args.train_samples_per_class, 0)
    test_data = build_exact_dataset(args, seed, args.test_samples_per_class, 1000)
    prototypes = make_prototypes(args.n_phases, args.n_channels, seed + 2000)
    train_x = paths_to_observations(train_data["paths"], prototypes, args.raw_noise, seed + 3000)
    test_x = paths_to_observations(test_data["paths"], prototypes, args.raw_noise, seed + 4000)
    train_y = train_data["labels"]
    test_y = test_data["labels"]

    eval_sets = {"test": (test_x, test_y)}
    rows: list[dict] = [
        {
            "method": "Chance level",
            "accuracy": 1.0 / args.n_classes,
            "macro_f1": 1.0 / args.n_classes,
            "order_gap": None,
        }
    ]
    inception = train_backbone(
        "Inception-style backbone",
        InceptionClassifier(args.n_channels, args.n_classes, args.embed_dim),
        train_x,
        train_y,
        eval_sets,
        args,
        device,
    )["test"]
    inception["order_gap"] = None
    rows.append(inception)

    transformer = train_backbone(
        "Transformer backbone",
        TransformerClassifier(args.n_channels, args.n_classes, train_x.shape[1], args.embed_dim),
        train_x,
        train_y,
        eval_sets,
        args,
        device,
    )["test"]
    transformer["order_gap"] = None
    rows.append(transformer)

    params = estimate_veto_parameters(train_data, args)
    for idx, (method, mode) in enumerate(VETO_METHODS):
        row = evaluate_veto_method(test_data, params, method, mode, device, seed + 9000 + idx)
        row["order_gap"] = compute_order_gap(
            test_data,
            params,
            mode,
            args.order_gap_permutations,
            seed + 5000,
            device,
        )
        rows.append(row)
    return rows


def run_shift_seed(seed: int, args, device: torch.device) -> list[dict]:
    set_seed(seed)
    args.loader_seed = seed + 8000
    train_data = build_shift_dataset(args, seed, args.train_samples_per_class, 0, "matched")
    test_sets = {
        "matched": build_shift_dataset(args, seed, args.test_samples_per_class, 1000, "matched"),
        "mild": build_shift_dataset(args, seed, args.test_samples_per_class, 2000, "mild"),
        "severe": build_shift_dataset(args, seed, args.test_samples_per_class, 3000, "severe"),
    }
    prototypes = make_prototypes(args.n_phases, args.n_channels, seed + 6000)
    train_x = paths_to_observations(train_data["paths"], prototypes, args.raw_noise, seed + 7000)
    train_y = train_data["labels"]
    eval_sets = {
        name: (
            paths_to_observations(data["paths"], prototypes, args.raw_noise, seed + 7100 + idx * 100),
            data["labels"],
        )
        for idx, (name, data) in enumerate(test_sets.items())
    }

    rows: list[dict] = []
    for name, model in [
        ("Inception-style backbone", InceptionClassifier(args.n_channels, args.n_classes, args.embed_dim)),
        (
            "Transformer backbone",
            TransformerClassifier(args.n_channels, args.n_classes, args.shift_path_length, args.embed_dim),
        ),
    ]:
        metrics = train_backbone(name, model, train_x, train_y, eval_sets, args, device)
        row = {
            "method": name,
            "matched": metrics["matched"]["accuracy"],
            "mild": metrics["mild"]["accuracy"],
            "severe": metrics["severe"]["accuracy"],
        }
        row["degradation"] = row["matched"] - row["severe"]
        rows.append(row)

    params = estimate_veto_parameters(train_data, args)
    for method, mode in VETO_METHODS:
        row = {"method": method}
        for split, data in test_sets.items():
            labels = data["labels"]
            scores = score_veto(data, params, mode, device)
            preds = predict_with_random_ties(scores, seed + 9100 + len(row))
            row[split] = float(accuracy_score(labels, preds))
        row["degradation"] = row["matched"] - row["severe"]
        rows.append(row)
    return rows


def print_summary(title: str, rows: list[dict], keys: list[str], show_std: bool) -> None:
    print(title)
    for row in rows:
        values = []
        for key in keys:
            values.append(fmt_metric(row[key], row[f"{key}_std"], show_std))
        print("  " + row["method"] + ": " + ", ".join(f"{k}={v}" for k, v in zip(keys, values)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build controlled synthetic phase-order tables")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--n_phases", type=int, default=8)
    parser.add_argument("--n_channels", type=int, default=14)
    parser.add_argument("--train_samples_per_class", type=int, default=96)
    parser.add_argument("--test_samples_per_class", type=int, default=240)
    parser.add_argument("--exact_count_per_phase", type=int, default=3)
    parser.add_argument("--shift_path_length", type=int, default=30)
    parser.add_argument("--shift_occurrence_strength", type=float, default=2.6)
    parser.add_argument("--exact_order_strength", type=float, default=2.2)
    parser.add_argument("--exact_secondary_strength", type=float, default=0.45)
    parser.add_argument("--exact_stay_strength", type=float, default=-0.25)
    parser.add_argument("--shift_order_strength", type=float, default=1.75)
    parser.add_argument("--shift_secondary_strength", type=float, default=0.25)
    parser.add_argument("--shift_stay_strength", type=float, default=0.25)
    parser.add_argument("--mild_duration", nargs="*", type=float, default=[1.8, 1.45, 1.2, 1.0, 0.85, 0.75, 0.65, 0.55])
    parser.add_argument("--severe_duration", nargs="*", type=float, default=[4.2, 3.1, 2.2, 1.3, 0.75, 0.45, 0.30, 0.22])
    parser.add_argument("--assignment_noise", type=float, default=0.28)
    parser.add_argument("--assignment_jitter", type=float, default=0.0)
    parser.add_argument("--raw_noise", type=float, default=0.90)
    parser.add_argument("--estimation_smooth", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--order_gap_permutations", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--show_std", action="store_true")
    parser.add_argument("--skip_backbones", action="store_true")
    parser.add_argument("--exact_csv", default="diagnostics/synthetic_phase_order_exact.csv")
    parser.add_argument("--exact_json", default="diagnostics/synthetic_phase_order_exact.json")
    parser.add_argument("--exact_tex_rows", default="diagnostics/synthetic_phase_order_exact_rows.tex")
    parser.add_argument("--exact_tex_table", default="diagnostics/synthetic_phase_order_exact_table.tex")
    parser.add_argument("--shift_csv", default="diagnostics/synthetic_phase_order_shift.csv")
    parser.add_argument("--shift_json", default="diagnostics/synthetic_phase_order_shift.json")
    parser.add_argument("--shift_tex_rows", default="diagnostics/synthetic_phase_order_shift_rows.tex")
    parser.add_argument("--shift_tex_table", default="diagnostics/synthetic_phase_order_shift_table.tex")
    args = parser.parse_args()

    if len(args.mild_duration) != args.n_phases or len(args.severe_duration) != args.n_phases:
        raise ValueError("--mild_duration and --severe_duration must have n_phases values")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    exact_seed_rows = []
    shift_seed_rows = []
    for seed in args.seeds:
        print(f"seed={seed} exact")
        exact_rows = run_exact_seed(seed, args, device)
        exact_seed_rows.append(exact_rows)
        for row in exact_rows:
            gap_text = "--" if row.get("order_gap") is None else f"{row['order_gap']:.3f}"
            print(
                f"  {row['method']}: acc={row['accuracy']:.3f}, "
                f"f1={row['macro_f1']:.3f}, "
                f"gap={gap_text}"
            )

        print(f"seed={seed} shift")
        shift_rows = run_shift_seed(seed, args, device)
        shift_seed_rows.extend(shift_rows)
        for row in shift_rows:
            print(
                f"  {row['method']}: matched={row['matched']:.3f}, "
                f"mild={row['mild']:.3f}, severe={row['severe']:.3f}, "
                f"D={row['degradation']:.3f}"
            )

    exact_summary = make_exact_rows(exact_seed_rows, args)
    shift_summary = make_shift_rows(shift_seed_rows, args)

    write_csv(Path(args.exact_csv), exact_summary)
    write_json(Path(args.exact_json), {"summary": exact_summary, "per_seed": exact_seed_rows, "args": vars(args)})
    write_exact_tex(exact_summary, args)
    write_csv(Path(args.shift_csv), shift_summary)
    write_json(Path(args.shift_json), {"summary": shift_summary, "per_seed": shift_seed_rows, "args": vars(args)})
    write_shift_tex(shift_summary, args)

    print_summary("exact summary", exact_summary, ["accuracy", "macro_f1", "order_gap"], args.show_std)
    print_summary("shift summary", shift_summary, ["matched", "mild", "severe", "degradation"], args.show_std)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
