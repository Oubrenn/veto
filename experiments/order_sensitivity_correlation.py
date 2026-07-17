"""Compute OSI versus VETO gain correlation from benchmark CSV files."""
import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


def aggregate(path, metric):
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    if metric == "selected_test_acc" and "selected_test_acc" not in df.columns and "best_test_acc" in df.columns:
        df = df.copy()
        df["selected_test_acc"] = df["best_test_acc"]
    return df.groupby("dataset", as_index=True)[metric].mean()


def main():
    parser = argparse.ArgumentParser(description="Order sensitivity correlation")
    parser.add_argument("--ordered_baseline", required=True, help="CSV for order-aware baseline/backbone")
    parser.add_argument("--orderless_baseline", required=True, help="CSV for orderless pooling model")
    parser.add_argument("--veto", required=True, help="CSV for full VETO")
    parser.add_argument("--metric", default="selected_test_acc")
    parser.add_argument("--output", default="diagnostics/order_sensitivity_correlation.json")
    args = parser.parse_args()

    ordered = aggregate(args.ordered_baseline, args.metric)
    orderless = aggregate(args.orderless_baseline, args.metric)
    veto = aggregate(args.veto, args.metric)

    common = sorted(set(ordered.index) & set(orderless.index) & set(veto.index))
    if len(common) < 2:
        raise ValueError("Need at least two common datasets for Spearman correlation.")

    ordered = ordered.loc[common]
    orderless = orderless.loc[common]
    veto = veto.loc[common]

    osi = ordered - orderless
    veto_gain = veto - ordered
    rho, p_value = spearmanr(osi.values, veto_gain.values)

    rows = [
        {
            "dataset": dataset,
            "ordered": float(ordered.loc[dataset]),
            "orderless": float(orderless.loc[dataset]),
            "veto": float(veto.loc[dataset]),
            "osi": float(osi.loc[dataset]),
            "veto_gain": float(veto_gain.loc[dataset]),
        }
        for dataset in common
    ]
    result = {
        "metric": args.metric,
        "n_datasets": len(common),
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "datasets": rows,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(
        f"Spearman rho={result['spearman_rho']:.4f}, "
        f"p={result['p_value']:.6g}, n={result['n_datasets']}"
    )


if __name__ == "__main__":
    main()
