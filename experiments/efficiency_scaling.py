"""Efficiency and scaling diagnostics for VETO models."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import PhasePathNet
from utils.common import count_parameters, set_seed


def estimate_veto_flops(
    n_classes: int,
    n_windows: int,
    n_phases: int,
    embed_dim: int,
) -> dict:
    prototype_flops = n_classes * n_windows * n_phases * embed_dim
    path_forward_flops = n_classes * max(n_windows - 1, 1) * n_phases * n_phases
    transition_generator_flops = n_classes * (16 * 64 + 64 * n_phases * n_phases)
    total = prototype_flops + path_forward_flops + transition_generator_flops
    return {
        "estimated_veto_head_flops": int(total),
        "estimated_prototype_flops": int(prototype_flops),
        "estimated_path_forward_flops": int(path_forward_flops),
        "estimated_transition_generator_flops": int(transition_generator_flops),
    }


def infer_n_windows(seq_length: int, window_size: int, stride: int) -> int:
    if seq_length <= window_size:
        return 1
    return 1 + int((seq_length - window_size + stride - 1) // stride)


def benchmark_one(args, n_classes: int, seq_length: int, n_phases: int) -> dict:
    window_size = args.window_size or max(10, seq_length // 10)
    stride = args.stride or max(1, window_size // 2)
    n_windows = infer_n_windows(seq_length, window_size, stride)
    model = PhasePathNet(
        n_classes=n_classes,
        n_channels=args.n_channels,
        seq_length=seq_length,
        n_phases=n_phases,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        stride=args.stride,
        backbone=args.backbone,
        transition_mode=args.transition_mode,
        prototype_mode=args.prototype_mode,
        head_mode=args.head_mode,
        path_score_mode=args.path_score_mode,
        path_weight_override=args.path_weight_override,
        use_memory=not args.no_memory,
    ).to(args.device)
    model.eval()

    x = torch.randn(args.batch_size, seq_length, args.n_channels, device=args.device)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x)["logits"]
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.repeats):
            _ = model(x)["logits"]
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    samples = args.batch_size * args.repeats
    latency_ms = elapsed / samples * 1000.0
    throughput = samples / elapsed if elapsed > 0 else 0.0
    peak_gpu = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else 0.0
    )
    flops = estimate_veto_flops(n_classes, n_windows, n_phases, args.embed_dim)
    return {
        "n_classes": n_classes,
        "seq_length": seq_length,
        "n_windows": n_windows,
        "n_phases": n_phases,
        "n_channels": args.n_channels,
        "embed_dim": args.embed_dim,
        "batch_size": args.batch_size,
        "backbone": args.backbone,
        "transition_mode": args.transition_mode,
        "prototype_mode": args.prototype_mode,
        "head_mode": args.head_mode,
        "path_score_mode": args.path_score_mode,
        "path_weight_override": args.path_weight_override if args.path_weight_override is not None else "",
        "parameters": count_parameters(model),
        "latency_ms_per_sample": latency_ms,
        "throughput_samples_per_sec": throughput,
        "peak_gpu_memory_mb": peak_gpu,
        **flops,
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
    parser = argparse.ArgumentParser(description="VETO efficiency/scaling diagnostics")
    parser.add_argument("--class_grid", nargs="*", type=int, default=[2, 5, 10, 26])
    parser.add_argument("--length_grid", nargs="*", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--phase_grid", nargs="*", type=int, default=[3, 5, 8, 12])
    parser.add_argument("--n_channels", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--backbone", choices=["inception", "resnet", "fcn"], default="fcn")
    parser.add_argument("--transition_mode", choices=["uniform", "free", "class_independent", "neural", "attention"], default="neural")
    parser.add_argument("--prototype_mode", choices=["class", "shared", "full"], default="class")
    parser.add_argument("--head_mode", choices=["backbone", "orderless", "prototype", "hmm", "veto"], default="veto")
    parser.add_argument("--path_score_mode", choices=["gain", "raw"], default="gain")
    parser.add_argument("--path_weight_override", type=float, default=None)
    parser.add_argument("--no_memory", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="diagnostics/efficiency_scaling.csv")
    parser.add_argument("--json_output", default="diagnostics/efficiency_scaling.json")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    set_seed(args.seed)
    rows = []
    baseline_classes = args.class_grid[-1]
    baseline_length = args.length_grid[1] if len(args.length_grid) > 1 else args.length_grid[0]
    baseline_phases = args.phase_grid[1] if len(args.phase_grid) > 1 else args.phase_grid[0]
    configs = []
    configs.extend((y, baseline_length, baseline_phases) for y in args.class_grid)
    configs.extend((baseline_classes, n, baseline_phases) for n in args.length_grid)
    configs.extend((baseline_classes, baseline_length, k) for k in args.phase_grid)
    seen = set()
    for n_classes, seq_length, n_phases in configs:
        key = (n_classes, seq_length, n_phases)
        if key in seen:
            continue
        seen.add(key)
        row = benchmark_one(args, n_classes, seq_length, n_phases)
        rows.append(row)
        print(
            f"Y={n_classes}, N={row['n_windows']}, K={n_phases}: "
            f"{row['latency_ms_per_sample']:.3f} ms/sample, "
            f"params={row['parameters']}"
        )
    write_rows(rows, args.output, args.json_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
