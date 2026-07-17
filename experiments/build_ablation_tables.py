"""Aggregate VETO ablation CSVs into manuscript-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = [
    ("DuckDuckGeese", "DDG"),
    ("Handwriting", "Handwriting"),
    ("LSST", "LSST"),
    ("MotorImagery", "MotorImagery"),
    ("SelfRegulationSCP1", "SCP1"),
    ("SelfRegulationSCP2", "SCP2"),
]


COMPONENT_ROWS = [
    ("local_only", False, False, False, False),
    ("raw_nomem_nocf", True, False, False, False),
    ("gain_nomem_nocf", True, True, False, False),
    ("raw_mem_nocf", True, False, True, False),
    ("raw_nomem_cf", True, False, False, True),
    ("gain_mem_nocf", True, True, True, False),
    ("gain_nomem_cf", True, True, False, True),
    ("full_veto", True, True, True, True),
]


DESIGN_ROWS = [
    ("class_independent_transition", "Class-independent transition"),
    ("free_transition_matrix", "Free class-specific matrices"),
    ("raw_transition_score", "Raw transition score"),
    ("direct_ema_memory", "Direct EMA memory"),
    ("gain_mem_nocf", "No counterfactual objective"),
    ("full_veto", "VETO full model"),
]


def load_results(input_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(input_dir.glob("*.csv")):
        df = pd.read_csv(path)
        if not df.empty:
            df["source_file"] = path.name
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all[df_all["status"].eq("ok")].copy()
    if "selected_test_acc" not in df_all.columns and "best_test_acc" in df_all.columns:
        df_all["selected_test_acc"] = df_all["best_test_acc"]
    if "selected_test_acc" not in df_all.columns:
        df_all["selected_test_acc"] = np.nan
    df_all["selected_test_acc"] = pd.to_numeric(df_all["selected_test_acc"], errors="coerce")
    if "best_test_acc" in df_all.columns:
        df_all["best_test_acc"] = pd.to_numeric(df_all["best_test_acc"], errors="coerce")
    else:
        df_all["best_test_acc"] = np.nan
    df_all["diagnostic_delta_g"] = pd.to_numeric(df_all["diagnostic_delta_g"], errors="coerce")
    return df_all


def method_dataset_mean(df: pd.DataFrame, experiment: str, dataset: str, metric: str) -> float:
    values = df.loc[
        df["experiment"].eq(experiment) & df["dataset"].eq(dataset),
        metric,
    ].dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def fmt(value: float, bold: bool = False) -> str:
    if np.isnan(value):
        return "--"
    if abs(value) < 5e-4:
        value = 0.0
    text = f"{value:.3f}"
    return f"\\textbf{{{text}}}" if bold else text


def fmt_signed_delta(value: float, bold: bool = False) -> str:
    """Format the signed, length-normalized order gap without hiding its scale."""
    if np.isnan(value):
        return "--"
    text = f"{value:+.6f}"
    return f"\\textbf{{{text}}}" if bold else text


def marks(value: bool) -> str:
    return "\\cmark" if value else "\\xmark"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def component_table(df: pd.DataFrame) -> list[dict]:
    rows = []
    for experiment, pcc, iid, mem, cf in COMPONENT_ROWS:
        row = {
            "experiment": experiment,
            "P_cc": pcc,
            "R_iid": iid,
            "M_conf": mem,
            "L_cf": cf,
        }
        values = []
        for dataset, short in DATASETS:
            value = method_dataset_mean(df, experiment, dataset, "selected_test_acc")
            row[short] = value
            values.append(value)
        row["Avg."] = float(np.nanmean(values))
        rows.append(row)
    return rows


def design_table(df: pd.DataFrame) -> list[dict]:
    rows = []
    dataset_method_acc = {}
    for experiment, _ in DESIGN_ROWS:
        dataset_method_acc[experiment] = [
            method_dataset_mean(df, experiment, dataset, "selected_test_acc")
            for dataset, _ in DATASETS
        ]
    acc_matrix = np.asarray([dataset_method_acc[experiment] for experiment, _ in DESIGN_ROWS])
    ranks = np.full_like(acc_matrix, np.nan, dtype=np.float64)
    for col in range(acc_matrix.shape[1]):
        values = acc_matrix[:, col]
        valid = ~np.isnan(values)
        order = np.argsort(-values[valid])
        valid_indices = np.flatnonzero(valid)
        for rank, local_idx in enumerate(order, start=1):
            ranks[valid_indices[local_idx], col] = rank

    for idx, (experiment, label) in enumerate(DESIGN_ROWS):
        acc_values = acc_matrix[idx]
        delta_values = [
            method_dataset_mean(df, experiment, dataset, "diagnostic_delta_g")
            for dataset, _ in DATASETS
        ]
        rows.append(
            {
                "experiment": experiment,
                "Variant": label,
                "Mean Acc.": float(np.nanmean(acc_values)),
                "Mean Rank": float(np.nanmean(ranks[idx])),
                # diagnostic_delta_g is already
                # mean[(G_real - G_shuffle) / n_valid_transitions].  Preserve
                # its sign when aggregating seeds within a dataset and then
                # datasets within a variant.
                "Delta_ord": float(np.nanmean(delta_values)),
            }
        )
    return rows


def write_component_tex(path: Path, rows: list[dict]) -> None:
    best = {short: max(row[short] for row in rows if not np.isnan(row[short])) for _, short in DATASETS}
    best["Avg."] = max(row["Avg."] for row in rows if not np.isnan(row["Avg."]))
    lines = [
        "\\begin{tabular}{cccc|cccccc|c}",
        "\\toprule",
        "$\\mathcal{P}_{\\mathrm{cc}}$ & $\\mathcal{R}_{\\mathrm{iid}}$ & "
        "$\\mathcal{M}_{\\mathrm{conf}}$ & $\\mathcal{L}_{\\mathrm{cf}}$ & "
        "DDG & Handwriting & LSST & MotorImagery & SCP1 & SCP2 & Avg. \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [
            marks(row["P_cc"]),
            marks(row["R_iid"]),
            marks(row["M_conf"]),
            marks(row["L_cf"]),
        ]
        for _, short in DATASETS:
            cells.append(fmt(row[short], abs(row[short] - best[short]) < 5e-7))
        cells.append(fmt(row["Avg."], abs(row["Avg."] - best["Avg."]) < 5e-7))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_design_tex(path: Path, rows: list[dict]) -> None:
    best_acc = max(row["Mean Acc."] for row in rows)
    best_rank = min(row["Mean Rank"] for row in rows)
    best_delta = max(row["Delta_ord"] for row in rows)
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Variant & Mean Acc. $\\uparrow$ & Mean Rank $\\downarrow$ & $\\Delta_{\\mathrm{ord}}\\uparrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [
            row["Variant"],
            fmt(row["Mean Acc."], abs(row["Mean Acc."] - best_acc) < 5e-7),
            fmt(row["Mean Rank"], abs(row["Mean Rank"] - best_rank) < 5e-7),
            fmt_signed_delta(
                row["Delta_ord"],
                abs(row["Delta_ord"] - best_delta) < 5e-7,
            ),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build VETO ablation manuscript tables")
    parser.add_argument("--input_dir", default="diagnostics/ablation_tables/run_fast")
    parser.add_argument("--output_dir", default="diagnostics/ablation_tables")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    df = load_results(input_dir)
    comp_rows = component_table(df)
    design_rows = design_table(df)

    write_csv(output_dir / "component_ablation_summary.csv", comp_rows)
    write_csv(output_dir / "design_ablation_summary.csv", design_rows)
    write_component_tex(output_dir / "table_component_ablation.tex", comp_rows)
    write_design_tex(output_dir / "table_design_ablation.tex", design_rows)
    (output_dir / "ablation_table_payload.json").write_text(
        json.dumps({"component": comp_rows, "design": design_rows}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote ablation tables to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
