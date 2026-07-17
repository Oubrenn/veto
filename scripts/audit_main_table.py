"""Audit provenance and statistics for the 26-dataset main comparison table."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from merge_main_table_results import (
    DATASETS,
    EXTERNAL_COLUMNS,
    METHODS,
    MODEL_TO_METHOD,
    parse_float,
    selected_accuracy,
)


VETO_FILES = [
    "diagnostics/official_cuda_10ep.csv",
    "diagnostics/official_cuda_10ep_part2.csv",
    "diagnostics/official_cuda_10ep_uwave.csv",
    "diagnostics/main_table/veto_missingA_bs64_10ep.csv",
    "diagnostics/main_table/veto_missingB_bs64_10ep.csv",
    "diagnostics/main_table/veto_racketsports_bs64_10ep.csv",
    "diagnostics/main_table/veto_eigen_bs8_10ep.csv",
]


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def result_field(row: dict) -> str:
    return "selected_test_acc" if parse_float(row.get("selected_test_acc")) is not None else "best_test_acc"


def row_metadata(row: dict, source: Path, field: str) -> dict:
    train_file = row.get("train_file", "")
    test_file = row.get("test_file", "")
    official = bool(train_file and test_file and "_TRAIN" in train_file.upper() and "_TEST" in test_file.upper())
    preprocessing = (
        f"normalize={row.get('normalize', 'NOT RECORDED')}; "
        f"train_fraction={row.get('train_fraction', 'NOT RECORDED')}; "
        f"train_file={train_file or 'NOT RECORDED'}; test_file={test_file or 'NOT RECORDED'}"
    )
    return {
        "Source_Type": "internal_reproduction",
        "Source_Paper_or_File": source.as_posix(),
        "Official_Default_Split": str(official),
        "Published_or_Reproduced": "reproduced",
        "Seed_Count": "1",
        "Seeds": row.get("seed", "NOT RECORDED") or "NOT RECORDED",
        "Code_Version": "NOT FOUND (no commit SHA recorded)",
        "Preprocessing": preprocessing,
        "Notes": (
            f"field={field}; epochs={row.get('epochs_completed', 'NOT RECORDED')}; "
            f"batch_size={row.get('batch_size', 'NOT RECORDED')}"
        ),
    }


def build_provenance(root: Path) -> dict[tuple[str, str], dict]:
    provenance: dict[tuple[str, str], dict] = {}
    external = root / "diagnostics/external_baselines/multiverse_accuracy_mean.csv"
    with external.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        dataset_field = reader.fieldnames[0]
        for row in reader:
            dataset = row.get(dataset_field, "").strip()
            if dataset not in DATASETS:
                continue
            for method, column in EXTERNAL_COLUMNS.items():
                value = parse_float(row.get(column))
                if value is None:
                    continue
                provenance[(dataset, method)] = {
                    "Accuracy": value,
                    "Source_Type": "external_published_aggregate",
                    "Source_Paper_or_File": external.as_posix(),
                    "Official_Default_Split": "NOT FOUND in source file",
                    "Published_or_Reproduced": "published/external",
                    "Seed_Count": "NOT FOUND",
                    "Seeds": "NOT FOUND",
                    "Code_Version": "NOT FOUND",
                    "Preprocessing": "NOT FOUND",
                    "Notes": f"column={column}; external aggregate metadata absent",
                }

    baseline_dir = root / "diagnostics/main_table_baselines_10ep"
    for path in sorted(baseline_dir.glob("*.csv")):
        model = next((name for name in MODEL_TO_METHOD if path.stem.startswith(f"{name}_")), None)
        if model is None:
            continue
        method = MODEL_TO_METHOD[model]
        for row in read_rows(path):
            dataset = row.get("dataset", "").strip()
            if dataset not in DATASETS or (dataset, method) in provenance:
                continue
            value = selected_accuracy(row)
            if value is None or row.get("status", "ok").strip() not in {"", "ok"}:
                continue
            field = result_field(row)
            provenance[(dataset, method)] = {
                "Accuracy": value,
                **row_metadata(row, path, field),
            }

    candidates: dict[tuple[str, str], list[tuple[int, float, Path, dict, str]]] = defaultdict(list)
    for relative in VETO_FILES:
        path = root / relative
        if not path.exists():
            continue
        for row in read_rows(path):
            dataset = row.get("dataset", "").strip()
            value = selected_accuracy(row)
            if dataset not in DATASETS or value is None:
                continue
            if row.get("status", "ok").strip() not in {"", "ok"}:
                continue
            seed = row.get("seed", "0").strip() or "0"
            epochs = int(float(row.get("epochs_completed", "0") or 0))
            candidates[(dataset, seed)].append((epochs, value, path, row, result_field(row)))

    selected_by_dataset: dict[str, list[tuple[str, tuple[int, float, Path, dict, str], int]]] = defaultdict(list)
    for (dataset, seed), rows in candidates.items():
        chosen = max(rows, key=lambda item: (item[0], item[1]))
        if chosen[0] > 0:
            selected_by_dataset[dataset].append((seed, chosen, len(rows)))
    for dataset, selections in selected_by_dataset.items():
        scores = [selection[1][1] for selection in selections]
        seed_text = ";".join(selection[0] for selection in selections)
        files = ";".join(sorted({selection[1][2].as_posix() for selection in selections}))
        fields = ";".join(sorted({selection[1][4] for selection in selections}))
        first_row = selections[0][1][3]
        metadata = row_metadata(first_row, selections[0][1][2], fields)
        metadata.update(
            {
                "Source_Paper_or_File": files,
                "Seed_Count": str(len(selections)),
                "Seeds": seed_text,
                "Notes": (
                    f"mean across selected seeds; fields={fields}; "
                    "within each dataset/seed choose max by (epochs_completed, accuracy); "
                    f"candidate_counts={';'.join(str(item[2]) for item in selections)}"
                ),
            }
        )
        provenance[(dataset, "VETO")] = {"Accuracy": float(np.mean(scores)), **metadata}
    return provenance


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.zeros(len(p_values), dtype=float)
    running = 0.0
    for position, index in enumerate(order):
        value = min(1.0, (len(p_values) - position) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def compute_statistics(matrix: pd.DataFrame) -> dict:
    methods = list(matrix.columns)
    values = matrix.to_numpy(dtype=float)
    ranks = np.vstack([rankdata(-row, method="average") for row in values])
    row_best = values.max(axis=1)
    aggregates = {}
    for index, method in enumerate(methods):
        aggregates[method] = {
            "mean_accuracy": float(values[:, index].mean()),
            "mean_rank": float(ranks[:, index].mean()),
            "wins_including_ties": int(np.isclose(values[:, index], row_best, atol=5e-7).sum()),
        }
    statistic, p_value = friedmanchisquare(*[values[:, index] for index in range(values.shape[1])])
    ref_index = methods.index("VETO")
    ref = values[:, ref_index]
    comparisons = []
    raw_p = []
    for index, method in enumerate(methods):
        if method == "VETO":
            continue
        other = values[:, index]
        stat, p = wilcoxon(ref, other, zero_method="wilcox", alternative="two-sided")
        comparisons.append(
            {
                "comparison": f"VETO vs {method}",
                "wins": int(np.sum(ref > other + 5e-7)),
                "ties": int(np.sum(np.isclose(ref, other, atol=5e-7))),
                "losses": int(np.sum(ref < other - 5e-7)),
                "wilcoxon_statistic": float(stat),
                "raw_p": float(p),
            }
        )
        raw_p.append(float(p))
    for row, adjusted in zip(comparisons, holm_adjust(raw_p)):
        row["holm_p"] = adjusted
        row["reject_at_0.05"] = adjusted < 0.05
    return {
        "n_datasets": int(values.shape[0]),
        "n_methods": int(values.shape[1]),
        "aggregates": aggregates,
        "friedman": {"statistic": float(statistic), "p_value": float(p_value)},
        "wilcoxon_holm_reference": "VETO",
        "wilcoxon_holm": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output_dir", type=Path, default=Path("diagnostics/reproducibility_audit"))
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = pd.read_csv(root / "diagnostics/main_table/table_main_comparison_26uea_10ep_values.csv")
    matrix = merged[merged["Dataset"].isin(DATASETS)].set_index("Dataset").loc[DATASETS, METHODS].astype(float)
    matrix.to_csv(output_dir / "main_table_26x11_clean.csv", float_format="%.12g")

    provenance = build_provenance(root)
    provenance_rows = []
    mismatches = []
    for dataset in DATASETS:
        for method in METHODS:
            record = provenance.get((dataset, method))
            if record is None:
                record = {
                    "Accuracy": float(matrix.loc[dataset, method]),
                    "Source_Type": "NOT FOUND",
                    "Source_Paper_or_File": "NOT FOUND",
                    "Official_Default_Split": "NOT FOUND",
                    "Published_or_Reproduced": "NOT FOUND",
                    "Seed_Count": "NOT FOUND",
                    "Seeds": "NOT FOUND",
                    "Code_Version": "NOT FOUND",
                    "Preprocessing": "NOT FOUND",
                    "Notes": "No matching source under current merge rules",
                }
            expected = float(matrix.loc[dataset, method])
            if not np.isclose(float(record["Accuracy"]), expected, atol=5e-7):
                mismatches.append(
                    {"dataset": dataset, "method": method, "matrix": expected, "source": record["Accuracy"]}
                )
            provenance_rows.append({"Dataset": dataset, "Method": method, **record})
    pd.DataFrame(provenance_rows).to_csv(output_dir / "main_table_26x11_provenance.csv", index=False)

    statistics = compute_statistics(matrix)
    statistics["source_mismatches"] = mismatches
    statistics["full_hc2_datasets"] = [
        dataset
        for dataset in DATASETS
        if provenance[(dataset, "HC2")]["Source_Type"] == "external_published_aggregate"
    ]
    statistics["hc2_lite_datasets"] = [
        dataset
        for dataset in DATASETS
        if provenance[(dataset, "HC2")]["Source_Type"] == "internal_reproduction"
    ]
    statistics["published_entry_count"] = sum(
        row["Published_or_Reproduced"] == "published/external" for row in provenance_rows
    )
    statistics["reproduced_entry_count"] = sum(
        row["Published_or_Reproduced"] == "reproduced" for row in provenance_rows
    )
    statistics["published_entries"] = [
        f"{row['Dataset']}::{row['Method']}"
        for row in provenance_rows
        if row["Published_or_Reproduced"] == "published/external"
    ]
    statistics["reproduced_entries"] = [
        f"{row['Dataset']}::{row['Method']}"
        for row in provenance_rows
        if row["Published_or_Reproduced"] == "reproduced"
    ]
    (output_dir / "main_table_26x11_statistics.json").write_text(
        json.dumps(statistics, indent=2), encoding="utf-8"
    )
    print(f"Wrote audit artifacts to {output_dir}")
    print(f"Source mismatches: {len(mismatches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
