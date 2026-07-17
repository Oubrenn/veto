"""Build manuscript sensitivity tables from saved diagnostic CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


DATASETS = [
    "Handwriting",
    "Heartbeat",
    "JapaneseVowels",
    "UWaveGestureLibrary",
]

K_ROWS = [
    ("5", "sensitivity_k_5.csv"),
    ("8", "sensitivity_k_8.csv"),
    ("16", "sensitivity_k_16.csv"),
    ("32", "sensitivity_k_32.csv"),
    ("64", "sensitivity_k_64.csv"),
]

LAMBDA_ROWS = [
    ("Learned (default)", "sensitivity_lg_learned.csv"),
    ("0", "sensitivity_lg_0.csv"),
    ("0.1", "sensitivity_lg_0p1.csv"),
    ("0.5", "sensitivity_lg_0p5.csv"),
    ("1.0", "sensitivity_lg_1.csv"),
    ("2.0", "sensitivity_lg_2.csv"),
]


def metric_column(df: pd.DataFrame) -> str:
    if "selected_test_acc" in df.columns:
        return "selected_test_acc"
    if "best_test_acc" in df.columns:
        return "best_test_acc"
    raise ValueError("No supported accuracy metric column found")


def load_row(input_dir: Path, filename: str) -> dict[str, float]:
    path = input_dir / filename
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"].eq("ok")].copy()
    df = df[df["dataset"].isin(DATASETS)].copy()
    if df.empty:
        return {}
    acc_col = metric_column(df)
    df[acc_col] = pd.to_numeric(df[acc_col], errors="coerce")
    df["diagnostic_delta_g"] = pd.to_numeric(df["diagnostic_delta_g"], errors="coerce")
    by_dataset = (
        df.groupby("dataset", as_index=True)
        .agg(acc=(acc_col, "mean"), delta_raw=("diagnostic_delta_g", "mean"))
    )
    return {
        "acc_by_dataset": by_dataset["acc"].to_dict(),
        "mean_acc": float(by_dataset["acc"].mean()),
        # Preserve the diagnostic definition used by the benchmark:
        # Delta_ord = G_real - mean(G_shuffle).  Negative values are failures,
        # not magnitudes to flip into positive evidence.
        "delta_ord": float(by_dataset["delta_raw"].mean()),
    }


def compute_rows(input_dir: Path, specs: list[tuple[str, str]]) -> list[dict]:
    loaded = {label: load_row(input_dir, filename) for label, filename in specs}
    ranks = {label: [] for label, _ in specs}
    for dataset in DATASETS:
        present = []
        for label, _ in specs:
            acc = loaded[label].get("acc_by_dataset", {}).get(dataset)
            if acc is not None and not np.isnan(acc):
                present.append((label, acc))
        if not present:
            continue
        rank_values = rankdata([-acc for _, acc in present], method="average")
        for rank, (label, _) in zip(rank_values, present):
            ranks[label].append(rank)

    rows = []
    for label, _ in specs:
        payload = loaded[label]
        rows.append(
            {
                "setting": label,
                "mean_acc": payload.get("mean_acc", np.nan),
                "mean_rank": float(np.mean(ranks[label])) if ranks[label] else np.nan,
                "delta_ord": payload.get("delta_ord", np.nan),
            }
        )
    return rows


def fmt(value: float, ndigits: int = 3) -> str:
    if value is None or np.isnan(value):
        return "--"
    if abs(value) < 0.5 * 10 ** (-ndigits):
        value = 0.0
    return f"{value:.{ndigits}f}"


def write_csv(path: Path, rows: list[dict], setting_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[setting_name, "Mean Acc.", "Mean Rank", "Delta_ord"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    setting_name: row["setting"],
                    "Mean Acc.": row["mean_acc"],
                    "Mean Rank": row["mean_rank"],
                    "Delta_ord": row["delta_ord"],
                }
            )


def write_tex(path: Path, rows: list[dict], setting_header: str) -> None:
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        f"{setting_header} & Mean Acc. $\\uparrow$ & Mean Rank $\\downarrow$ & $\\Delta_{{\\mathrm{{ord}}}}\\uparrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [
            row["setting"],
            fmt(row["mean_acc"]),
            fmt(row["mean_rank"], ndigits=2),
            fmt(row["delta_ord"]),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build VETO sensitivity tables")
    parser.add_argument("--input_dir", default="diagnostics/sensitivity_smoke")
    parser.add_argument("--output_dir", default="diagnostics/paper_artifacts/tables")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    k_rows = compute_rows(input_dir, K_ROWS)
    lambda_rows = compute_rows(input_dir, LAMBDA_ROWS)

    write_csv(output_dir / "table_sensitivity_k.csv", k_rows, "K")
    write_tex(output_dir / "table_sensitivity_k.tex", k_rows, "$K$")
    write_csv(output_dir / "table_sensitivity_lg.csv", lambda_rows, "lambda_g")
    write_tex(output_dir / "table_sensitivity_lg.tex", lambda_rows, "$\\lambda_g$")

    print(f"Wrote sensitivity tables to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
