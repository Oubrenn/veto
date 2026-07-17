"""Generate the matched-occurrence synthetic table used in the manuscript.

All classes have the same stationary phase marginal distribution. Labels differ
only through class-specific Markov transition laws. This avoids the earlier
degenerate setup where each class had a nearly deterministic path template and
generic sequence backbones saturated at 100% accuracy.
"""
import argparse
import csv
import json
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
from models.path_forward import PathForward
from utils.common import set_seed


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
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, seq_length + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
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
        h = self.norm(h[:, 0])
        return self.classifier(h)


def make_circulant_transitions(
    n_classes: int,
    n_phases: int,
    preferred_strength: float,
    secondary_strength: float,
    stay_strength: float,
    smooth: float,
) -> np.ndarray:
    steps = [1, 2, -1, -2, 3, -3]
    transitions = []
    for class_id in range(n_classes):
        step = steps[class_id % len(steps)]
        base_row = np.full(n_phases, smooth, dtype=np.float64)
        base_row[0] += stay_strength
        base_row[step % n_phases] += preferred_strength
        base_row[(2 * step) % n_phases] += secondary_strength
        base_row /= base_row.sum()

        matrix = np.zeros((n_phases, n_phases), dtype=np.float64)
        for phase in range(n_phases):
            matrix[phase] = np.roll(base_row, phase)
        transitions.append(matrix)
    return np.asarray(transitions, dtype=np.float32)


def sample_markov_paths(
    transitions: np.ndarray,
    samples_per_class: int,
    path_length: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    n_classes, n_phases, _ = transitions.shape
    paths = []
    labels = []
    for class_id in range(n_classes):
        for _ in range(samples_per_class):
            path = np.empty(path_length, dtype=np.int64)
            path[0] = int(rng.integers(n_phases))
            for t in range(1, path_length):
                path[t] = int(rng.choice(n_phases, p=transitions[class_id, path[t - 1]]))
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
    samples = []
    for path in paths:
        x = prototypes[path] + rng.normal(0.0, noise_std, size=(len(path), prototypes.shape[1]))
        samples.append(x.astype(np.float32))
    return np.stack(samples, axis=0)


def path_to_assignment(
    path: np.ndarray,
    n_phases: int,
    observation_noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    q = np.full(
        (len(path), n_phases),
        observation_noise / max(n_phases - 1, 1),
        dtype=np.float32,
    )
    q[np.arange(len(path)), path] = 1.0 - observation_noise
    jitter = rng.normal(0.0, observation_noise * 0.05, size=q.shape).astype(np.float32)
    q = np.maximum(q + jitter, 1e-6)
    q /= q.sum(axis=-1, keepdims=True)
    return q


def make_dataset(args, seed: int, samples_per_class: int) -> dict:
    transitions = make_circulant_transitions(
        args.n_classes,
        args.n_phases,
        args.preferred_strength,
        args.secondary_strength,
        args.stay_strength,
        args.transition_smooth,
    )
    paths, labels = sample_markov_paths(transitions, samples_per_class, args.path_length, seed)
    rng = np.random.default_rng(seed + 17)
    q = [path_to_assignment(path, args.n_phases, args.assignment_noise, rng) for path in paths]
    shuffled_q = [item[rng.permutation(item.shape[0])] for item in q]
    init = np.full((args.n_classes, args.n_phases), 1.0 / args.n_phases, dtype=np.float32)
    occurrence = np.full((args.n_classes, args.n_phases), 1.0 / args.n_phases, dtype=np.float32)
    return {
        "paths": paths,
        "labels": labels,
        "q": q,
        "shuffled_q": shuffled_q,
        "init": init,
        "transitions": transitions,
        "occurrence": occurrence,
    }


def make_train_test(args, seed: int) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    train_data = make_dataset(args, seed, args.train_samples_per_class)
    test_data = make_dataset(args, seed + 1000, args.test_samples_per_class)

    rng = np.random.default_rng(seed + 2000)
    prototypes = rng.normal(0.0, 1.0, size=(args.n_phases, args.n_channels)).astype(np.float32)
    train_x = paths_to_observations(train_data["paths"], prototypes, args.raw_noise, seed + 3000)
    test_x = paths_to_observations(test_data["paths"], prototypes, args.raw_noise, seed + 4000)
    return train_data, test_data, train_x, test_x


def estimate_veto_parameters(train_data: dict, args) -> dict:
    """Estimate phase priors and transitions from training responsibilities.

    This keeps the synthetic diagnostic from using the generator's true Markov
    matrices at test time. The estimate is deliberately simple: it uses soft
    phase responsibilities and labelled training sequences, then applies
    Dirichlet-style smoothing before row normalization.
    """
    labels = train_data["labels"]
    n_classes = args.n_classes
    n_phases = args.n_phases
    smooth = args.estimation_smooth

    init_counts = np.full((n_classes, n_phases), smooth, dtype=np.float64)
    transition_counts = np.full((n_classes, n_phases, n_phases), smooth, dtype=np.float64)
    occurrence_counts = np.full((n_classes, n_phases), smooth, dtype=np.float64)

    for q_np, label in zip(train_data["q"], labels):
        q = q_np.astype(np.float64)
        init_counts[label] += q[0]
        occurrence_counts[label] += q.sum(axis=0)
        transition_counts[label] += q[:-1].T @ q[1:]

    init = init_counts / init_counts.sum(axis=1, keepdims=True)
    transitions = transition_counts / transition_counts.sum(axis=2, keepdims=True)
    occurrence = occurrence_counts / occurrence_counts.sum(axis=1, keepdims=True)

    return {
        "init": init.astype(np.float32),
        "transitions": transitions.astype(np.float32),
        "occurrence": occurrence.astype(np.float32),
    }


def score_all_classes(
    data: dict,
    params: dict,
    mode: str,
    shuffled: bool = False,
) -> np.ndarray:
    q_source = data["shuffled_q"] if shuffled else data["q"]
    q = torch.tensor(np.stack(q_source, axis=0), dtype=torch.float32)
    init = torch.tensor(params["init"], dtype=torch.float32)
    transitions = torch.tensor(params["transitions"], dtype=torch.float32)
    occurrence = torch.tensor(params["occurrence"], dtype=torch.float32)

    log_q = torch.log(q + 1e-10)
    log_init = torch.log(init + 1e-10)
    log_trans = torch.log(transitions + 1e-10)

    log_alpha = log_init.unsqueeze(0) + log_q[:, 0, :].unsqueeze(1)
    for t in range(1, q.shape[1]):
        log_alpha = torch.logsumexp(
            log_alpha.unsqueeze(3) + log_trans.unsqueeze(0),
            dim=2,
        )
        log_alpha = log_alpha + log_q[:, t, :].unsqueeze(1)
    path_score = torch.logsumexp(log_alpha, dim=-1)

    marginal = q.mean(dim=1).clamp_min(1e-8)
    marginal = marginal / marginal.sum(dim=-1, keepdim=True)
    log_marginal = torch.log(marginal + 1e-10)
    iid_alpha = log_marginal + log_q[:, 0, :]
    for t in range(1, q.shape[1]):
        iid_alpha = torch.logsumexp(
            iid_alpha.unsqueeze(2) + log_marginal.unsqueeze(1),
            dim=1,
        )
        iid_alpha = iid_alpha + log_q[:, t, :]
    iid_score = torch.logsumexp(iid_alpha, dim=-1).unsqueeze(1)

    local_score = torch.einsum("mnk,ck->mc", q, torch.log(occurrence + 1e-8))

    if mode == "local_only":
        scores = local_score
    elif mode == "raw_transition":
        scores = path_score
    elif mode == "transition_gain":
        scores = local_score + path_score - iid_score
    elif mode == "gain_only":
        scores = path_score - iid_score
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return scores.numpy().astype(np.float64)


def evaluate_predictions(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "method": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "gain_drop": None,
    }


def train_backbone(name: str, model: nn.Module, train_x, train_y, test_x, test_y, args) -> dict:
    device = torch.device(args.device)
    model = model.to(device)
    train_ds = TensorDataset(torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for _ in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(test_x, dtype=torch.float32, device=device))
        preds = logits.argmax(dim=1).cpu().numpy()
    return evaluate_predictions(name, test_y, preds)


def evaluate_veto_rows(train_data: dict, test_data: dict, args) -> list[dict]:
    labels = test_data["labels"]
    params = estimate_veto_parameters(train_data, args)
    rows = []
    modes = [
        ("VETO local-only", "local_only", "local_only"),
        ("VETO raw transition", "raw_transition", "raw_transition"),
        ("VETO full model", "transition_gain", "gain_only"),
    ]
    for name, mode, drop_mode in modes:
        scores = score_all_classes(test_data, params, mode, shuffled=False)
        preds = scores.argmax(axis=1)
        valid_scores = score_all_classes(test_data, params, drop_mode, shuffled=False)
        shuffled_scores = score_all_classes(test_data, params, drop_mode, shuffled=True)
        drop = valid_scores[np.arange(len(labels)), labels] - shuffled_scores[np.arange(len(labels)), labels]
        rows.append(
            {
                "method": name,
                "accuracy": float(accuracy_score(labels, preds)),
                "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
                "gain_drop": float(np.mean(drop)),
            }
        )
    return rows


def aggregate_rows(seed_rows: list[list[dict]]) -> list[dict]:
    methods = [row["method"] for row in seed_rows[0]]
    aggregated = []
    for method in methods:
        matched = [row for rows in seed_rows for row in rows if row["method"] == method]
        out = {"method": method}
        for key in ["accuracy", "macro_f1", "gain_drop"]:
            values = [row[key] for row in matched if row[key] is not None]
            if values:
                out[key] = float(np.mean(values))
                out[f"{key}_std"] = float(np.std(values, ddof=0))
            else:
                out[key] = None
                out[f"{key}_std"] = None
        aggregated.append(out)
    return aggregated


def format_mean(row: dict, key: str, show_std: bool) -> str:
    value = row[key]
    if value is None:
        return "--"
    if abs(value) < 0.0005:
        value = 0.0
    if show_std:
        return f"{value:.3f} $\\pm$ {row[f'{key}_std']:.3f}"
    return f"{value:.3f}"


def write_outputs(rows: list[dict], seed_rows: list[list[dict]], args) -> None:
    output = Path(args.output)
    json_output = Path(args.json_output)
    tex_output = Path(args.tex_output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "method",
        "accuracy",
        "accuracy_std",
        "macro_f1",
        "macro_f1_std",
        "gain_drop",
        "gain_drop_std",
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with json_output.open("w", encoding="utf-8") as file:
        json.dump({"summary": rows, "per_seed": seed_rows}, file, indent=2)

    lines = []
    for row in rows:
        lines.append(
            f"{row['method']} & {format_mean(row, 'accuracy', args.show_std)} & "
            f"{format_mean(row, 'macro_f1', args.show_std)} & "
            f"{format_mean(row, 'gain_drop', args.show_std)} \\\\"
        )
    tex_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one_seed(seed: int, args) -> list[dict]:
    set_seed(seed)
    train_data, test_data, train_x, test_x = make_train_test(args, seed)
    train_y = train_data["labels"]
    test_y = test_data["labels"]
    rows = [
        train_backbone(
            "Inception-style backbone",
            InceptionClassifier(args.n_channels, args.n_classes),
            train_x,
            train_y,
            test_x,
            test_y,
            args,
        ),
        train_backbone(
            "Transformer backbone",
            TransformerClassifier(args.n_channels, args.n_classes, seq_length=train_x.shape[1]),
            train_x,
            train_y,
            test_x,
            test_y,
            args,
        ),
    ]
    rows.extend(evaluate_veto_rows(train_data, test_data, args))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build matched-occurrence synthetic table rows")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--n_phases", type=int, default=8)
    parser.add_argument("--path_length", type=int, default=24)
    parser.add_argument("--train_samples_per_class", type=int, default=80)
    parser.add_argument("--test_samples_per_class", type=int, default=300)
    parser.add_argument("--n_channels", type=int, default=16)
    parser.add_argument("--preferred_strength", type=float, default=0.60)
    parser.add_argument("--secondary_strength", type=float, default=0.10)
    parser.add_argument("--stay_strength", type=float, default=0.10)
    parser.add_argument("--transition_smooth", type=float, default=0.25)
    parser.add_argument("--assignment_noise", type=float, default=0.35)
    parser.add_argument("--raw_noise", type=float, default=0.85)
    parser.add_argument("--estimation_smooth", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--show_std", action="store_true")
    parser.add_argument("--output", default="diagnostics/synthetic_matched_occurrence_table.csv")
    parser.add_argument("--json_output", default="diagnostics/synthetic_matched_occurrence_table.json")
    parser.add_argument("--tex_output", default="diagnostics/synthetic_matched_occurrence_table_rows.tex")
    args = parser.parse_args()

    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    seed_rows = []
    for seed in args.seeds:
        rows = run_one_seed(seed, args)
        seed_rows.append(rows)
        print(f"seed={seed}")
        for row in rows:
            gain = "--" if row["gain_drop"] is None else f"{row['gain_drop']:.3f}"
            print(f"  {row['method']}: acc={row['accuracy']:.3f}, f1={row['macro_f1']:.3f}, gain_drop={gain}")

    summary = aggregate_rows(seed_rows)
    write_outputs(summary, seed_rows, args)
    print("summary")
    for row in summary:
        gain = "--" if row["gain_drop"] is None else f"{row['gain_drop']:.3f}"
        print(f"  {row['method']}: acc={row['accuracy']:.3f}, f1={row['macro_f1']:.3f}, gain_drop={gain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
