"""Summarize multi-dataset benchmark CSV files for paper tables."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


NEMENYI_Q_ALPHA_005 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.268,
    13: 3.313,
    14: 3.354,
    15: 3.391,
    16: 3.426,
    17: 3.458,
    18: 3.489,
    19: 3.517,
    20: 3.544,
}


def holm_correction(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    m = len(p_values)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p_values[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def rank_biserial_from_pairs(x, y):
    diff = np.asarray(x) - np.asarray(y)
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    pos = ranks[nonzero > 0].sum()
    neg = ranks[nonzero < 0].sum()
    denom = len(nonzero) * (len(nonzero) + 1) / 2
    return float((pos - neg) / denom)


def win_tie_loss(a, b, eps=1e-8):
    a = np.asarray(a)
    b = np.asarray(b)
    wins = int(np.sum(a > b + eps))
    ties = int(np.sum(np.abs(a - b) <= eps))
    losses = int(np.sum(a < b - eps))
    return wins, ties, losses


def load_and_aggregate(paths, metric):
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if "status" in df.columns:
            df = df[df["status"] == "ok"]
        if "experiment" not in df.columns:
            df["experiment"] = Path(path).stem
        if metric == "selected_test_acc" and "selected_test_acc" not in df.columns and "best_test_acc" in df.columns:
            df["selected_test_acc"] = df["best_test_acc"]
        if metric == "selected_test_macro_f1" and "selected_test_macro_f1" not in df.columns and "best_macro_f1" in df.columns:
            df["selected_test_macro_f1"] = df["best_macro_f1"]
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    if metric not in data.columns:
        if metric == "selected_test_acc" and "best_test_acc" in data.columns:
            data["selected_test_acc"] = data["best_test_acc"]
        elif metric == "selected_test_macro_f1" and "best_macro_f1" in data.columns:
            data["selected_test_macro_f1"] = data["best_macro_f1"]
        else:
            raise ValueError(f"Metric {metric!r} not found. Columns: {list(data.columns)}")

    grouped = (
        data.groupby(["experiment", "dataset"], as_index=False)
        .agg(
            mean=(metric, "mean"),
            std=(metric, "std"),
            n_seeds=(metric, "count"),
            params=("model_params", "mean") if "model_params" in data.columns else (metric, "count"),
            latency_ms=("single_sample_latency_ms", "mean")
            if "single_sample_latency_ms" in data.columns
            else (metric, "count"),
        )
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    return data, grouped


def make_metric_table(grouped):
    mean = grouped.pivot(index="dataset", columns="experiment", values="mean")
    std = grouped.pivot(index="dataset", columns="experiment", values="std")
    return mean, std


def compute_ranks(mean_table):
    ranks = mean_table.apply(lambda row: rankdata(-row.values, method="average"), axis=1, result_type="expand")
    ranks.columns = mean_table.columns
    ranks.index = mean_table.index
    return ranks


def nemenyi_cd(n_methods, n_datasets, alpha=0.05):
    if alpha != 0.05:
        raise ValueError("Only alpha=0.05 is tabulated for Nemenyi CD.")
    q_alpha = NEMENYI_Q_ALPHA_005.get(n_methods, 3.544 if n_methods > 20 else None)
    if q_alpha is None:
        raise ValueError(f"No Nemenyi q_alpha for {n_methods} methods")
    return float(q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_datasets)))


def write_cd_diagram(avg_rank, cd, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    methods = list(avg_rank.index)
    ranks = avg_rank.values.astype(float)
    y_positions = np.arange(len(methods))
    fig_width = max(7.0, 0.45 * len(methods) + 4.0)
    fig_height = max(3.0, 0.35 * len(methods) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.scatter(ranks, y_positions, s=36, color="#1f77b4", zorder=3)
    for y, method, rank in zip(y_positions, methods, ranks):
        ax.text(rank + 0.03, y, f"{method} ({rank:.2f})", va="center", fontsize=9)
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_yticks([])
    ax.set_ylim(-1, len(methods))
    ax.grid(axis="x", alpha=0.25)
    ax.invert_xaxis()

    best_rank = float(np.min(ranks))
    cd_y = -0.55
    ax.plot([best_rank, best_rank + cd], [cd_y, cd_y], color="black", lw=2)
    ax.plot([best_rank, best_rank], [cd_y - 0.08, cd_y + 0.08], color="black", lw=2)
    ax.plot([best_rank + cd, best_rank + cd], [cd_y - 0.08, cd_y + 0.08], color="black", lw=2)
    ax.text(best_rank + cd / 2, cd_y - 0.18, f"CD={cd:.2f}", ha="center", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def summarize(args):
    data, grouped = load_and_aggregate(args.inputs, args.metric)
    mean_table, std_table = make_metric_table(grouped)
    common = mean_table.dropna(axis=0, how="any")
    if common.empty:
        raise ValueError("No common datasets across experiments after aggregation.")

    ranks = compute_ranks(common)
    avg_rank = ranks.mean(axis=0).sort_values()
    methods = list(common.columns)
    cd = ""
    if len(methods) >= 2 and len(common) >= 2:
        cd = nemenyi_cd(len(methods), len(common))

    friedman = {"statistic": "", "p_value": ""}
    if len(methods) >= 3 and len(common) >= 2:
        stat, p = friedmanchisquare(*[common[m].values for m in methods])
        friedman = {"statistic": float(stat), "p_value": float(p)}

    reference = args.reference or methods[0]
    if reference not in methods:
        raise ValueError(f"Reference {reference!r} not present in experiments: {methods}")

    comparisons = []
    p_values = []
    for method in methods:
        if method == reference:
            continue
        ref_values = common[reference].values
        method_values = common[method].values
        try:
            _, p = wilcoxon(ref_values, method_values, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            p = 1.0
        p_values.append(p)
        wins, ties, losses = win_tie_loss(ref_values, method_values)
        comparisons.append(
            {
                "reference": reference,
                "method": method,
                "n_datasets": int(len(common)),
                "median_improvement": float(np.median(ref_values - method_values)),
                "win": wins,
                "tie": ties,
                "loss": losses,
                "wilcoxon_p": float(p),
                "rank_biserial": rank_biserial_from_pairs(ref_values, method_values),
            }
        )
    if comparisons:
        adjusted = holm_correction(np.asarray(p_values))
        for row, p_adj in zip(comparisons, adjusted):
            row["holm_p"] = float(p_adj)
            row["significant_0_05"] = bool(p_adj < 0.05)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_dir / "seed_aggregated_metrics.csv", index=False)
    common.to_csv(output_dir / f"{args.metric}_mean_table.csv")
    std_table.reindex(index=common.index, columns=common.columns).to_csv(
        output_dir / f"{args.metric}_std_table.csv"
    )
    ranks.to_csv(output_dir / "average_rank_table.csv")
    avg_rank.to_csv(output_dir / "average_rank_summary.csv", header=["average_rank"])
    if cd != "":
        with (output_dir / "nemenyi_cd.txt").open("w", encoding="utf-8") as file:
            file.write(f"alpha=0.05\nn_methods={len(methods)}\nn_datasets={len(common)}\ncd={cd:.6f}\n")
        write_cd_diagram(avg_rank, cd, output_dir / "cd_diagram.png")

    with (output_dir / "statistical_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "metric": args.metric,
                "reference": reference,
                "datasets": list(common.index),
                "average_rank": avg_rank.to_dict(),
                "nemenyi_cd_0_05": cd,
                "friedman": friedman,
                "comparisons": comparisons,
            },
            file,
            indent=2,
        )

    with (output_dir / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as file:
        fields = list(comparisons[0].keys()) if comparisons else ["reference", "method"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)

    print(f"Wrote summaries to {output_dir}")
    print("Average rank:")
    for method, rank in avg_rank.items():
        print(f"  {method}: {rank:.3f}")
    if friedman["p_value"] != "":
        print(f"Friedman p={friedman['p_value']:.6g}")


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark CSV files")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--metric", default="selected_test_acc")
    parser.add_argument("--reference", default="")
    parser.add_argument("--output_dir", default="diagnostics/summary")
    args = parser.parse_args()
    args.reference = args.reference or None
    summarize(args)


if __name__ == "__main__":
    main()
