"""Synthetic occurrence/order path-control experiments for VETO.

The generated labels can depend on phase occurrence only, phase order only, or
both. This isolates whether transition gain captures ordered evidence beyond
local phase frequencies.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.path_forward import PathForward
from utils.common import set_seed


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.maximum(matrix, 1e-8)
    return matrix / matrix.sum(axis=-1, keepdims=True)


def make_order_paths(n_classes: int, n_phases: int, path_length: int) -> np.ndarray:
    paths = []
    for class_id in range(n_classes):
        base = np.arange(n_phases)
        if class_id % 2:
            base = base[::-1]
        base = np.roll(base, class_id % n_phases)
        if path_length > n_phases:
            base = np.tile(base, int(np.ceil(path_length / n_phases)))
        paths.append(base[:path_length])
    return np.asarray(paths, dtype=np.int64)


def make_occurrence_probs(
    task: str,
    n_classes: int,
    n_phases: int,
    occurrence_strength: float,
) -> np.ndarray:
    probs = np.ones((n_classes, n_phases), dtype=np.float64)
    if task in {"occurrence_only", "mixed"}:
        for class_id in range(n_classes):
            dominant = class_id % n_phases
            secondary = (class_id + 1) % n_phases
            probs[class_id, dominant] += occurrence_strength
            probs[class_id, secondary] += occurrence_strength * 0.5
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs


def transition_from_path(
    path: np.ndarray,
    n_phases: int,
    transition_overlap: float,
    smooth: float,
) -> np.ndarray:
    transition = np.full((n_phases, n_phases), smooth, dtype=np.float64)
    for prev_phase, next_phase in zip(path[:-1], path[1:]):
        transition[prev_phase, next_phase] += 1.0
        transition[prev_phase, prev_phase] += 0.25
    transition = normalize_rows(transition)
    uniform = np.ones_like(transition) / n_phases
    transition = (1.0 - transition_overlap) * transition + transition_overlap * uniform
    return normalize_rows(transition)


def make_transition_bank(args, paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transitions = []
    init = []
    for path in paths:
        transitions.append(
            transition_from_path(
                path,
                args.n_phases,
                args.transition_overlap,
                args.transition_smooth,
            )
        )
        init_dist = np.full(args.n_phases, args.transition_smooth, dtype=np.float64)
        init_dist[path[0]] += 1.0
        init.append(init_dist / init_dist.sum())
    return np.asarray(init), np.asarray(transitions)


def generate_phase_path(
    label: int,
    task: str,
    occurrence_probs: np.ndarray,
    base_paths: np.ndarray,
    args,
    rng: np.random.Generator,
) -> np.ndarray:
    if task == "occurrence_only":
        return rng.choice(args.n_phases, size=args.path_length, p=occurrence_probs[label])

    path = base_paths[label].copy()
    if task == "mixed":
        keep = rng.random(args.path_length) > args.order_strength
        sampled = rng.choice(args.n_phases, size=args.path_length, p=occurrence_probs[label])
        path[keep] = sampled[keep]
    return path


def path_to_q(path: np.ndarray, args, rng: np.random.Generator) -> np.ndarray:
    q = np.full(
        (len(path), args.n_phases),
        args.observation_noise / max(args.n_phases - 1, 1),
        dtype=np.float32,
    )
    q[np.arange(len(path)), path] = 1.0 - args.observation_noise
    if args.assignment_jitter > 0:
        q += rng.normal(0.0, args.assignment_jitter, size=q.shape).astype(np.float32)
    q = np.maximum(q, 1e-6)
    q /= q.sum(axis=-1, keepdims=True)
    return q


def shuffle_q(q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(q.shape[0])
    return q[order]


def make_dataset(task: str, args):
    rng = np.random.default_rng(args.seed)
    base_paths = make_order_paths(args.n_classes, args.n_phases, args.path_length)
    occurrence_probs = make_occurrence_probs(
        task,
        args.n_classes,
        args.n_phases,
        args.occurrence_strength,
    )
    init, transitions = make_transition_bank(args, base_paths)

    q_list = []
    shuffled_q_list = []
    labels = []
    hard_paths = []
    for class_id in range(args.n_classes):
        for _ in range(args.samples_per_class):
            path = generate_phase_path(
                class_id,
                task,
                occurrence_probs,
                base_paths,
                args,
                rng,
            )
            q = path_to_q(path, args, rng)
            q_list.append(q)
            shuffled_q_list.append(shuffle_q(q, rng))
            labels.append(class_id)
            hard_paths.append(path)

    return {
        "task": task,
        "q": q_list,
        "shuffled_q": shuffled_q_list,
        "labels": np.asarray(labels, dtype=np.int64),
        "paths": hard_paths,
        "occurrence_probs": occurrence_probs,
        "init": init,
        "transitions": transitions,
    }


def iid_transition_from_q(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    marginal = q.mean(dim=1).clamp_min(1e-8)
    marginal = marginal / marginal.sum(dim=-1, keepdim=True)
    transition = marginal.unsqueeze(1).expand(-1, q.shape[-1], -1)
    return marginal, transition


def score_all_classes(data: dict, mode: str, args, shuffled: bool = False) -> np.ndarray:
    path_forward = PathForward(log_space=True)
    q_source = data["shuffled_q"] if shuffled else data["q"]
    scores = []
    init = torch.tensor(data["init"], dtype=torch.float32)
    transitions = torch.tensor(data["transitions"], dtype=torch.float32)
    occurrence_probs = torch.tensor(data["occurrence_probs"], dtype=torch.float32)
    uniform_transition = torch.ones(args.n_phases, args.n_phases) / args.n_phases
    uniform_init = torch.ones(args.n_phases) / args.n_phases

    for q_np in q_source:
        q = torch.tensor(q_np, dtype=torch.float32).unsqueeze(0)
        row_scores = []
        for class_id in range(args.n_classes):
            path_score = path_forward(init[class_id], transitions[class_id], q).item()
            iid_init, iid_transition = iid_transition_from_q(q)
            iid_score = path_forward(iid_init, iid_transition, q).item()
            raw_transition = path_score
            transition_gain = path_score - iid_score
            local_score = torch.sum(q.squeeze(0) * torch.log(occurrence_probs[class_id] + 1e-8)).item()
            uniform_score = path_forward(uniform_init, uniform_transition, q).item()

            if mode == "local_only":
                score = local_score
            elif mode == "uniform_path":
                score = uniform_score
            elif mode == "raw_transition":
                score = raw_transition
            elif mode == "transition_gain":
                score = local_score + args.lambda_g * transition_gain
            elif mode == "gain_only":
                score = transition_gain
            else:
                raise ValueError(f"Unknown mode: {mode}")
            row_scores.append(score)
        scores.append(row_scores)
    return np.asarray(scores)


def evaluate_mode(data: dict, mode: str, args) -> dict:
    labels = data["labels"]
    scores = score_all_classes(data, mode, args, shuffled=False)
    shuffled_scores = score_all_classes(data, mode, args, shuffled=True)
    preds = scores.argmax(axis=1)
    true_score = scores[np.arange(len(labels)), labels]
    wrong_scores = scores.copy()
    wrong_scores[np.arange(len(labels)), labels] = -np.inf
    best_wrong = wrong_scores.max(axis=1)

    valid_gain = score_all_classes(data, "gain_only", args, shuffled=False)[
        np.arange(len(labels)), labels
    ]
    shuffled_gain = score_all_classes(data, "gain_only", args, shuffled=True)[
        np.arange(len(labels)), labels
    ]
    auroc_labels = np.concatenate([np.ones_like(valid_gain), np.zeros_like(shuffled_gain)])
    auroc_scores = np.concatenate([valid_gain, shuffled_gain])

    return {
        "task": data["task"],
        "mode": mode,
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "g_true_minus_g_wrong": float(np.mean(true_score - best_wrong)),
        "delta_g": float(np.mean(valid_gain - shuffled_gain)),
        "valid_path_vs_shuffled_auroc": float(roc_auc_score(auroc_labels, auroc_scores)),
        "valid_gain_mean": float(np.mean(valid_gain)),
        "shuffled_gain_mean": float(np.mean(shuffled_gain)),
        "n_classes": args.n_classes,
        "n_phases": args.n_phases,
        "path_length": args.path_length,
        "samples_per_class": args.samples_per_class,
        "transition_overlap": args.transition_overlap,
        "observation_noise": args.observation_noise,
        "occurrence_strength": args.occurrence_strength,
        "lambda_g": args.lambda_g,
        "seed": args.seed,
    }


def write_rows(rows, output: str, json_output: str) -> None:
    output_path = Path(output)
    json_path = Path(json_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic occurrence/order path-control experiments")
    parser.add_argument("--tasks", nargs="*", default=["occurrence_only", "order_only", "mixed"])
    parser.add_argument("--modes", nargs="*", default=[
        "local_only",
        "uniform_path",
        "raw_transition",
        "transition_gain",
    ])
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--n_phases", type=int, default=5)
    parser.add_argument("--path_length", type=int, default=10)
    parser.add_argument("--samples_per_class", type=int, default=200)
    parser.add_argument("--transition_overlap", type=float, default=0.15)
    parser.add_argument("--transition_smooth", type=float, default=0.03)
    parser.add_argument("--observation_noise", type=float, default=0.12)
    parser.add_argument("--assignment_jitter", type=float, default=0.01)
    parser.add_argument("--occurrence_strength", type=float, default=4.0)
    parser.add_argument("--order_strength", type=float, default=0.8)
    parser.add_argument("--lambda_g", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="diagnostics/synthetic_path_control.csv")
    parser.add_argument("--json_output", default="diagnostics/synthetic_path_control.json")
    args = parser.parse_args()

    set_seed(args.seed)
    rows = []
    for task in args.tasks:
        data = make_dataset(task, args)
        for mode in args.modes:
            row = evaluate_mode(data, mode, args)
            rows.append(row)
            print(
                f"{task}/{mode}: acc={row['accuracy']:.3f}, "
                f"dG={row['delta_g']:.3f}, AUROC={row['valid_path_vs_shuffled_auroc']:.3f}"
            )
    write_rows(rows, args.output, args.json_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
