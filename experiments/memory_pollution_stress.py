"""Controlled memory-pollution stress test for Figure 6(c,d).

Two memory policies receive paired streams that differ only at corrupted
candidate writes.  Prototype drift and predictive degradation are computed
independently:

* prototype drift = mean_k[1 - cosine(M_k_clean, M_k_noise)]
* accuracy drop = Acc(M_clean; held-out queries) - Acc(M_noise; held-out queries)

The held-out query accuracy is measured directly by nearest-prototype cosine
classification.  It is never derived from the drift value.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_POLLUTIONS = [0.0, 0.1, 0.2, 0.3, 0.4]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DRIFT_DEFINITION = "mean_prototype[1-cosine(M_clean,M_noise)]"
ACCURACY_DEFINITION = (
    "nearest-prototype cosine accuracy on a fixed held-out clean query bank"
)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def cosine_prototype_drift(clean_memory: np.ndarray, noisy_memory: np.ndarray) -> float:
    """Mean cosine distance over all class-phase prototypes."""

    if clean_memory.shape != noisy_memory.shape:
        raise ValueError("clean_memory and noisy_memory must have identical shapes")
    clean = _unit(clean_memory.reshape(-1, clean_memory.shape[-1]))
    noisy = _unit(noisy_memory.reshape(-1, noisy_memory.shape[-1]))
    cosine = np.clip(np.sum(clean * noisy, axis=-1), -1.0, 1.0)
    return float(np.mean(1.0 - cosine))


def make_base(args: argparse.Namespace, seed: int) -> np.ndarray:
    """Create a fixed, normalized class-phase prototype dictionary."""

    rng = np.random.default_rng(seed + 101)
    class_centres = _unit(rng.normal(size=(args.n_classes, args.embed_dim)))
    phase_offsets = _unit(rng.normal(size=(args.n_phases, args.embed_dim)))
    base = class_centres[:, None, :] + args.phase_separation * phase_offsets[None, :, :]
    return _unit(base)


def make_stream_plan(args: argparse.Namespace, base: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Pre-generate a nested pollution plan shared by all rho values."""

    rng = np.random.default_rng(seed + 10_003)
    y = rng.integers(0, args.n_classes, size=args.steps)
    k = rng.integers(0, args.n_phases, size=args.steps)
    y_offset = rng.integers(1, args.n_classes, size=args.steps)
    k_offset = rng.integers(1, args.n_phases, size=args.steps)
    source_y = (y + y_offset) % args.n_classes
    source_k = (k + k_offset) % args.n_phases
    corruption_uniform = rng.random(args.steps)
    clean_noise = rng.normal(0.0, args.noise, size=(args.steps, args.embed_dim))
    corrupt_noise = rng.normal(0.0, args.noise, size=(args.steps, args.embed_dim))

    clean_embedding = base[y, k] + clean_noise
    if args.pollution_type == "mislabeled_windows":
        corrupt_embedding = base[source_y, k] + corrupt_noise
    elif args.pollution_type == "wrong_phase_candidates":
        corrupt_embedding = base[y, source_k] + corrupt_noise
    elif args.pollution_type == "transition_inconsistent_replacement":
        corrupt_embedding = base[source_y, source_k] + corrupt_noise
    elif args.pollution_type == "burst_noise":
        corrupt_embedding = base[y, k] + args.burst_multiplier * corrupt_noise
    else:  # parser choices should make this unreachable
        raise ValueError(f"Unknown pollution type: {args.pollution_type}")
    return {
        "y": y,
        "k": k,
        "pollution_uniform": corruption_uniform,
        "clean_embedding": clean_embedding,
        "corrupt_embedding": corrupt_embedding,
    }


def make_query_bank(
    args: argparse.Namespace,
    base: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate fixed clean queries independent of every update stream."""

    rng = np.random.default_rng(seed + 20_003)
    flat = base.reshape(-1, args.embed_dim)
    queries = np.repeat(flat, args.n_queries_per_prototype, axis=0)
    queries = queries + rng.normal(0.0, args.query_noise, size=queries.shape)
    labels = np.repeat(np.arange(flat.shape[0]), args.n_queries_per_prototype)
    return _unit(queries), labels


def heldout_query_accuracy(
    memory: np.ndarray,
    queries: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Direct held-out nearest-prototype accuracy; independent of drift."""

    flat_memory = _unit(memory.reshape(-1, memory.shape[-1]))
    predictions = np.argmax(queries @ flat_memory.T, axis=1)
    return float(np.mean(predictions == labels))


class MemoryPolicy:
    """Direct EMA or reliability-gated confirmed memory."""

    def __init__(self, policy: str, initial: np.ndarray, args: argparse.Namespace):
        if policy not in {"direct_ema", "confirmed_memory"}:
            raise ValueError(f"Unknown policy: {policy}")
        self.policy = policy
        self.memory = initial.copy()
        self.args = args
        self.candidate = np.zeros_like(initial)
        self.counter = np.zeros(initial.shape[:2], dtype=np.int64)
        self.pending_polluted = np.zeros(initial.shape[:2], dtype=bool)
        self.total_commits = 0
        self.polluted_commits = 0

    def _commit(self, y: int, k: int, embedding: np.ndarray, polluted: bool) -> None:
        momentum = (
            self.args.ema_momentum if self.policy == "direct_ema" else self.args.momentum
        )
        self.memory[y, k] = momentum * self.memory[y, k] + (1.0 - momentum) * embedding
        self.total_commits += 1
        self.polluted_commits += int(polluted)

    def update(
        self,
        y: int,
        k: int,
        embedding: np.ndarray,
        reliability: float,
        polluted: bool,
    ) -> None:
        if self.policy == "direct_ema":
            self._commit(y, k, embedding, polluted)
            return

        if reliability >= self.args.reliability_threshold:
            if self.counter[y, k] == 0:
                self.candidate[y, k] = embedding
                self.pending_polluted[y, k] = polluted
            else:
                self.candidate[y, k] = 0.8 * self.candidate[y, k] + 0.2 * embedding
                self.pending_polluted[y, k] = self.pending_polluted[y, k] or polluted
            self.counter[y, k] += 1
            if self.counter[y, k] >= self.args.confirmation_threshold:
                self._commit(
                    y,
                    k,
                    self.candidate[y, k],
                    bool(self.pending_polluted[y, k]),
                )
                self.counter[y, k] = 0
                self.candidate[y, k] = 0.0
                self.pending_polluted[y, k] = False
        else:
            self.counter[y, k] = max(
                0, self.counter[y, k] - self.args.confirmation_decay
            )
            if self.counter[y, k] == 0:
                self.candidate[y, k] = 0.0
                self.pending_polluted[y, k] = False

    @property
    def false_commit_rate(self) -> float:
        return self.polluted_commits / self.total_commits if self.total_commits else 0.0


def run_policy_pair(
    args: argparse.Namespace,
    base: np.ndarray,
    initial: np.ndarray,
    plan: dict[str, np.ndarray],
    queries: np.ndarray,
    query_labels: np.ndarray,
    seed: int,
    pollution: float,
    policy: str,
) -> dict:
    clean_policy = MemoryPolicy(policy, initial, args)
    noisy_policy = MemoryPolicy(policy, initial, args)
    polluted_mask = plan["pollution_uniform"] < pollution
    for step in range(args.steps):
        y = int(plan["y"][step])
        k = int(plan["k"][step])
        clean_embedding = plan["clean_embedding"][step]
        clean_policy.update(
            y,
            k,
            clean_embedding,
            reliability=args.clean_reliability,
            polluted=False,
        )
        is_polluted = bool(polluted_mask[step])
        noisy_policy.update(
            y,
            k,
            plan["corrupt_embedding"][step] if is_polluted else clean_embedding,
            reliability=(
                args.polluted_reliability if is_polluted else args.clean_reliability
            ),
            polluted=is_polluted,
        )

    prototype_drift = cosine_prototype_drift(clean_policy.memory, noisy_policy.memory)
    clean_accuracy = heldout_query_accuracy(clean_policy.memory, queries, query_labels)
    corrupted_accuracy = heldout_query_accuracy(noisy_policy.memory, queries, query_labels)
    accuracy_drop = clean_accuracy - corrupted_accuracy
    method = "Direct EMA" if policy == "direct_ema" else "Confirmed memory"
    return {
        "policy": policy,
        "method": method,
        "pollution": float(pollution),
        "pollution_type": args.pollution_type,
        "seed": int(seed),
        "steps": int(args.steps),
        "n_prototypes": int(args.n_classes * args.n_phases),
        "n_queries": int(len(query_labels)),
        "realized_pollution": float(np.mean(polluted_mask)),
        "prototype_drift": float(prototype_drift),
        "clean_accuracy": float(clean_accuracy),
        "corrupted_accuracy": float(corrupted_accuracy),
        "accuracy_drop": float(accuracy_drop),
        "clean_commits": int(clean_policy.total_commits),
        "noisy_commits": int(noisy_policy.total_commits),
        "false_commit_rate": float(noisy_policy.false_commit_rate),
        "memory_momentum": float(
            args.ema_momentum if policy == "direct_ema" else args.momentum
        ),
        "reliability_threshold": float(args.reliability_threshold),
        "clean_reliability": float(args.clean_reliability),
        "polluted_reliability": float(args.polluted_reliability),
        "prototype_drift_definition": DRIFT_DEFINITION,
        "accuracy_definition": ACCURACY_DEFINITION,
    }


def run(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    for seed in args.seeds:
        base = make_base(args, seed)
        initial_rng = np.random.default_rng(seed + 30_013)
        initial = base + initial_rng.normal(0.0, args.initial_noise, size=base.shape)
        plan = make_stream_plan(args, base, seed)
        queries, query_labels = make_query_bank(args, base, seed)
        for pollution in args.pollutions:
            for policy in ["direct_ema", "confirmed_memory"]:
                rows.append(
                    run_policy_pair(
                        args,
                        base,
                        initial,
                        plan,
                        queries,
                        query_labels,
                        seed,
                        pollution,
                        policy,
                    )
                )
    return rows


def write_rows(
    rows: list[dict],
    output: str | Path,
    json_output: str | Path,
    args: argparse.Namespace,
) -> None:
    output = Path(output)
    json_output = Path(json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "prototype_drift_definition": DRIFT_DEFINITION,
        "accuracy_definition": ACCURACY_DEFINITION,
        "pollutions": list(args.pollutions),
        "seeds": list(args.seeds),
        "rows": rows,
    }
    json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pollution", type=float, default=None)
    parser.add_argument("--pollutions", nargs="+", type=float, default=DEFAULT_POLLUTIONS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--pollution_type",
        choices=[
            "mislabeled_windows",
            "burst_noise",
            "transition_inconsistent_replacement",
            "wrong_phase_candidates",
        ],
        default="mislabeled_windows",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--n_phases", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--phase_separation", type=float, default=0.65)
    parser.add_argument("--noise", type=float, default=0.10)
    parser.add_argument("--query_noise", type=float, default=0.35)
    parser.add_argument("--n_queries_per_prototype", type=int, default=100)
    parser.add_argument("--initial_noise", type=float, default=0.05)
    parser.add_argument("--burst_multiplier", type=float, default=5.0)
    parser.add_argument("--clean_reliability", type=float, default=0.95)
    parser.add_argument("--polluted_reliability", type=float, default=0.55)
    parser.add_argument("--reliability_threshold", type=float, default=0.7)
    parser.add_argument("--confirmation_threshold", type=int, default=5)
    parser.add_argument("--confirmation_decay", type=int, default=2)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--ema_momentum", type=float, default=0.95)
    parser.add_argument(
        "--output",
        default="diagnostics/paper_figures/source_data/fig6_memory_pollution.csv",
    )
    parser.add_argument(
        "--json_output",
        default="diagnostics/paper_figures/source_data/fig6_memory_pollution.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.pollution is not None:
        args.pollutions = [args.pollution]
    if args.seed is not None:
        args.seeds = [args.seed]
    if any(value < 0.0 or value > 1.0 for value in args.pollutions):
        raise ValueError("Pollution ratios must lie in [0,1].")
    if args.n_classes < 2 or args.n_phases < 2:
        raise ValueError("n_classes and n_phases must both be >= 2")

    rows = run(args)
    write_rows(rows, args.output, args.json_output, args)
    for row in rows:
        print(
            f"seed={row['seed']} rho={row['pollution']:.1f} {row['method']}: "
            f"drift={row['prototype_drift']:.5f}, "
            f"acc_clean={row['clean_accuracy']:.4f}, "
            f"acc_noise={row['corrupted_accuracy']:.4f}, "
            f"drop={row['accuracy_drop']:.4f}"
        )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
