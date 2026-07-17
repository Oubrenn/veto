"""Synthetic phase-path mechanism diagnostics.

This experiment directly tests whether a transition model scores valid phase
orders above corrupted orders. It is intentionally lightweight so it can be run
before expensive UEA sweeps.
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

from models.phase_graph import ClassPhaseGraph
from models.path_forward import PathForward, PathForwardWithViterbi
from utils.common import set_seed


def make_transition(paths: np.ndarray, n_phases: int, smooth: float = 0.05) -> np.ndarray:
    n_classes = paths.shape[0]
    transitions = np.full((n_classes, n_phases, n_phases), smooth, dtype=np.float32)
    init = np.full((n_classes, n_phases), smooth, dtype=np.float32)
    for y, path in enumerate(paths):
        init[y, path[0]] += 1.0
        for prev, curr in zip(path[:-1], path[1:]):
            transitions[y, prev, curr] += 1.0
            transitions[y, prev, prev] += 0.4
    transitions /= transitions.sum(axis=-1, keepdims=True)
    init /= init.sum(axis=-1, keepdims=True)
    return init, transitions


def corrupt_path(path: np.ndarray, corruption: str, rng: np.random.Generator) -> np.ndarray:
    path = path.copy()
    n = len(path)
    if corruption == "swap" and n >= 4:
        i, j = sorted(rng.choice(np.arange(1, n - 1), size=2, replace=False))
        path[i], path[j] = path[j], path[i]
    elif corruption == "reverse" and n >= 4:
        i, j = sorted(rng.choice(np.arange(1, n - 1), size=2, replace=False))
        path[i : j + 1] = path[i : j + 1][::-1]
    elif corruption == "missing" and n >= 3:
        i = int(rng.integers(1, n - 1))
        path[i:-1] = path[i + 1 :]
        path[-1] = path[-2]
    elif corruption == "repeat" and n >= 3:
        i = int(rng.integers(1, n - 1))
        path[i:] = np.roll(path[i:], 1)
        path[i] = path[i - 1]
    else:
        rng.shuffle(path)
    return path


def path_to_assignment(
    path: np.ndarray,
    n_phases: int,
    noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    q = np.full((len(path), n_phases), noise / max(n_phases - 1, 1), dtype=np.float32)
    for t, phase in enumerate(path):
        q[t, phase] = 1.0 - noise
    jitter = rng.normal(0.0, noise * 0.05, size=q.shape).astype(np.float32)
    q = np.maximum(q + jitter, 1e-5)
    q /= q.sum(axis=-1, keepdims=True)
    return q


def generate_dataset(args):
    rng = np.random.default_rng(args.seed)
    base_paths = []
    for y in range(args.n_classes):
        path = np.arange(args.n_phases)
        if y % 2 == 1:
            path = path[::-1]
        path = np.roll(path, y % args.n_phases)
        if args.path_length > args.n_phases:
            repeats = int(np.ceil(args.path_length / args.n_phases))
            path = np.tile(path, repeats)[: args.path_length]
        base_paths.append(path)
    base_paths = np.asarray(base_paths)
    init, transitions = make_transition(base_paths, args.n_phases)

    valid_q = []
    corrupt_q = []
    labels = []
    valid_paths = []
    corrupt_paths = []
    for _ in range(args.samples_per_class):
        for y, path in enumerate(base_paths):
            if args.duration_jitter > 0:
                keep = rng.random(len(path)) > args.duration_jitter
                if keep.sum() >= 2:
                    path_used = path[keep]
                else:
                    path_used = path
            else:
                path_used = path
            corrupted = corrupt_path(path_used, args.corruption, rng)
            valid_q.append(path_to_assignment(path_used, args.n_phases, args.noise, rng))
            corrupt_q.append(path_to_assignment(corrupted, args.n_phases, args.noise, rng))
            labels.append(y)
            valid_paths.append(path_used)
            corrupt_paths.append(corrupted)

    return {
        "init": init,
        "transitions": transitions,
        "valid_q": valid_q,
        "corrupt_q": corrupt_q,
        "labels": np.asarray(labels, dtype=np.int64),
        "valid_paths": valid_paths,
        "corrupt_paths": corrupt_paths,
    }


def instantiate_graph(mode: str, init: np.ndarray, transitions: np.ndarray, args):
    graph = ClassPhaseGraph(
        n_classes=args.n_classes,
        n_phases=args.n_phases,
        transition_mode=mode,
    )
    if mode == "uniform":
        pass
    elif mode == "free":
        with torch.no_grad():
            graph.transition_logits.copy_(torch.log(torch.tensor(transitions) + 1e-8))
            graph.init_logits.copy_(torch.log(torch.tensor(init) + 1e-8))
    elif mode in {"neural", "attention"}:
        # Fit the generated matrix by optimizing only the tiny generator/head.
        opt = torch.optim.Adam(graph.parameters(), lr=0.05)
        target_A = torch.tensor(transitions)
        target_pi = torch.tensor(init)
        for _ in range(args.generator_fit_steps):
            opt.zero_grad()
            out = graph()
            loss = torch.nn.functional.mse_loss(out["transition_matrices"], target_A)
            loss = loss + torch.nn.functional.mse_loss(out["init_distributions"], target_pi)
            loss.backward()
            opt.step()
    return graph


def score_q(path_forward, init, transitions, q_list, labels):
    scores = []
    preds = []
    for q, y in zip(q_list, labels):
        q_t = torch.tensor(q).unsqueeze(0)
        class_scores = []
        for c in range(init.shape[0]):
            score = path_forward(
                init_dist=init[c],
                transition_matrix=transitions[c],
                phase_assignment=q_t,
            )
            class_scores.append(float(score.item()))
        scores.append(class_scores)
        preds.append(int(np.argmax(class_scores)))
    return np.asarray(scores), np.asarray(preds)


def transition_f1(paths_true, paths_pred):
    y_true = []
    y_pred = []
    for true_path, pred_path in zip(paths_true, paths_pred):
        for prev, curr in zip(true_path[:-1], true_path[1:]):
            y_true.append(f"{prev}->{curr}")
        for prev, curr in zip(pred_path[:-1], pred_path[1:]):
            y_pred.append(f"{prev}->{curr}")
    labels = sorted(set(y_true) | set(y_pred))
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def evaluate_mode(mode: str, data: dict, args) -> dict:
    graph = instantiate_graph(mode, data["init"], data["transitions"], args)
    graph.eval()
    with torch.no_grad():
        graph_out = graph()
        init = graph_out["init_distributions"]
        transitions = graph_out["transition_matrices"]

    path_forward = PathForward(log_space=True)
    valid_scores, preds = score_q(path_forward, init, transitions, data["valid_q"], data["labels"])
    corrupt_scores, _ = score_q(path_forward, init, transitions, data["corrupt_q"], data["labels"])

    true_valid = valid_scores[np.arange(len(data["labels"])), data["labels"]]
    true_corrupt = corrupt_scores[np.arange(len(data["labels"])), data["labels"]]
    delta_g = true_valid - true_corrupt
    ppa = float(np.mean(delta_g > 0))
    margin = float(np.mean(delta_g))

    auroc_labels = np.concatenate([np.ones_like(true_valid), np.zeros_like(true_corrupt)])
    auroc_scores = np.concatenate([true_valid, true_corrupt])
    auroc = roc_auc_score(auroc_labels, auroc_scores)

    viterbi = PathForwardWithViterbi()
    decoded = []
    for q, y in zip(data["valid_q"], data["labels"]):
        path = viterbi.viterbi_decode(
            init_dist=init[y],
            transition_matrix=transitions[y],
            phase_assignment=torch.tensor(q).unsqueeze(0),
        )
        decoded.append(path.squeeze(0).cpu().numpy())

    path_acc = np.mean(
        [
            np.array_equal(decoded_path[: len(true_path)], true_path[: len(decoded_path)])
            for decoded_path, true_path in zip(decoded, data["valid_paths"])
        ]
    )
    phase_acc = np.mean(
        [
            accuracy_score(true_path[: len(decoded_path)], decoded_path[: len(true_path)])
            for decoded_path, true_path in zip(decoded, data["valid_paths"])
        ]
    )

    return {
        "mode": mode,
        "accuracy": float(accuracy_score(data["labels"], preds)),
        "macro_f1": float(f1_score(data["labels"], preds, average="macro", zero_division=0)),
        "phase_accuracy": float(phase_acc),
        "transition_f1": float(transition_f1(data["valid_paths"], decoded)),
        "path_accuracy": float(path_acc),
        "ppa": ppa,
        "delta_g_mean": margin,
        "score_margin": margin,
        "valid_corrupt_auroc": float(auroc),
        "params": int(sum(p.numel() for p in graph.parameters() if p.requires_grad)),
        "noise": args.noise,
        "corruption": args.corruption,
        "duration_jitter": args.duration_jitter,
    }


def write_outputs(rows, output: Path, json_output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with json_output.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Synthetic phase-path mechanism diagnostics")
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--n_phases", type=int, default=5)
    parser.add_argument("--path_length", type=int, default=8)
    parser.add_argument("--samples_per_class", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--duration_jitter", type=float, default=0.0)
    parser.add_argument("--corruption", choices=["swap", "reverse", "missing", "repeat", "shuffle"], default="swap")
    parser.add_argument("--modes", nargs="*", default=["uniform", "free", "neural", "attention"])
    parser.add_argument("--generator_fit_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="diagnostics/synthetic_phase_path.csv")
    parser.add_argument("--json_output", type=str, default="diagnostics/synthetic_phase_path.json")
    args = parser.parse_args()

    set_seed(args.seed)
    data = generate_dataset(args)
    rows = [evaluate_mode(mode, data, args) for mode in args.modes]
    write_outputs(rows, Path(args.output), Path(args.json_output))
    for row in rows:
        print(
            f"{row['mode']}: acc={row['accuracy']:.3f}, "
            f"PPA={row['ppa']:.3f}, AUROC={row['valid_corrupt_auroc']:.3f}, "
            f"margin={row['score_margin']:.3f}"
        )


if __name__ == "__main__":
    main()
