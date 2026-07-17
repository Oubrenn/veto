"""Build the UEA main comparison table from external baselines and VETO CSVs."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


TABLE_DATASETS = [
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "CharacterTrajectories",
    "Cricket",
    "DuckDuckGeese",
    "EigenWorms",
    "Epilepsy",
    "ERing",
    "EthanolConcentration",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "JapaneseVowels",
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PenDigits",
    "PEMS-SF",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
]

METHODS = [
    "HC2",
    "ROCKET",
    "MultiRocket",
    "InceptionTime",
    "Transformer",
    "MTS2Graph",
    "SimTSC",
    "TMA-GAT",
    "TapNet",
    "PDFTime",
    "VETO",
]

EXTERNAL_COLUMNS = {
    "HC2": "HC2",
    "ROCKET": "ROCKET",
    "InceptionTime": "h-inceptiontime",
}

TIE_TOLERANCE = 5e-7


def parse_float(value: str):
    if value is None:
        return None
    value = str(value).strip()
    if not value or value == "--":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def selected_accuracy(row: dict) -> float | None:
    acc = parse_float(row.get("selected_test_acc", ""))
    if acc is None:
        acc = parse_float(row.get("best_test_acc", ""))
    return acc


def read_external_baselines(path: Path) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {method: {} for method in METHODS}
    if not path.exists():
        return values

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        dataset_key = reader.fieldnames[0] if reader.fieldnames else None
        if dataset_key is None:
            return values
        for row in reader:
            dataset = row.get(dataset_key, "").strip()
            if dataset not in TABLE_DATASETS:
                continue
            for method, column in EXTERNAL_COLUMNS.items():
                score = parse_float(row.get(column, ""))
                if score is not None:
                    values[method][dataset] = score
    return values


def read_veto_results(paths: list[Path]) -> dict[str, float]:
    by_dataset_seed: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                dataset = row.get("dataset", "").strip()
                if dataset not in TABLE_DATASETS:
                    continue
                status = row.get("status", "ok").strip()
                if status and status != "ok":
                    continue
                acc = selected_accuracy(row)
                if acc is None:
                    continue
                seed = row.get("seed", "").strip() or "0"
                epochs = int(float(row.get("epochs_completed", "0") or 0))
                by_dataset_seed[(dataset, seed)].append((epochs, acc))

    by_dataset: dict[str, list[float]] = defaultdict(list)
    for (dataset, _seed), rows in by_dataset_seed.items():
        epochs, acc = max(rows, key=lambda item: (item[0], item[1]))
        if epochs > 0:
            by_dataset[dataset].append(acc)

    return {
        dataset: sum(scores) / len(scores)
        for dataset, scores in by_dataset.items()
        if scores
    }


def ranks_for_row(row: dict[str, float]) -> dict[str, float]:
    numeric = [(method, value) for method, value in row.items() if value is not None]
    numeric.sort(key=lambda item: item[1], reverse=True)
    ranks: dict[str, float] = {}
    idx = 0
    while idx < len(numeric):
        end = idx + 1
        while end < len(numeric) and abs(numeric[end][1] - numeric[idx][1]) <= TIE_TOLERANCE:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for method, _ in numeric[idx:end]:
            ranks[method] = avg_rank
        idx = end
    return ranks


def format_number(value: float) -> str:
    return f"{value:.3f}"


def latex_cell(value: float | None, row_values: dict[str, float | None]) -> str:
    if value is None:
        return r"\NA"
    numeric = sorted(
        {score for score in row_values.values() if score is not None},
        reverse=True,
    )
    text = format_number(value)
    if len(numeric) < 2:
        return text
    if numeric and abs(value - numeric[0]) <= TIE_TOLERANCE:
        return rf"\best{{{text}}}"
    if len(numeric) > 1 and abs(value - numeric[1]) <= TIE_TOLERANCE:
        return rf"\second{{{text}}}"
    return text


def build_values(external_path: Path, veto_paths: list[Path]):
    values = read_external_baselines(external_path)
    values["VETO"] = read_veto_results(veto_paths)
    return values


def write_numeric_csv(values, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", *METHODS])
        for dataset in TABLE_DATASETS:
            writer.writerow(
                [
                    dataset,
                    *[
                        "" if values.get(method, {}).get(dataset) is None else f"{values[method][dataset]:.6f}"
                        for method in METHODS
                    ],
                ]
            )


def build_latex(values) -> str:
    rows = []
    rank_sums = defaultdict(float)
    rank_counts = defaultdict(int)
    acc_sums = defaultdict(float)
    acc_counts = defaultdict(int)
    wins = defaultdict(int)

    for dataset in TABLE_DATASETS:
        row_values = {method: values.get(method, {}).get(dataset) for method in METHODS}
        comparable_values = [value for value in row_values.values() if value is not None]
        ranks = ranks_for_row(row_values) if len(comparable_values) >= 2 else {}
        for method, rank in ranks.items():
            rank_sums[method] += rank
            rank_counts[method] += 1
        best = max(comparable_values, default=None) if len(comparable_values) >= 2 else None
        for method, value in row_values.items():
            if value is None:
                continue
            acc_sums[method] += value
            acc_counts[method] += 1
            if best is not None and abs(value - best) <= TIE_TOLERANCE:
                wins[method] += 1
        rows.append(
            f"{dataset:<27} & "
            + " & ".join(latex_cell(row_values[method], row_values) for method in METHODS)
            + r" \\"
        )

    mean_acc = []
    mean_rank = []
    wins_row = []
    for method in METHODS:
        mean_acc.append(r"\NA" if acc_counts[method] == 0 else format_number(acc_sums[method] / acc_counts[method]))
        mean_rank.append(r"\NA" if rank_counts[method] == 0 else format_number(rank_sums[method] / rank_counts[method]))
        wins_row.append(r"\NA" if acc_counts[method] == 0 else str(wins[method]))

    lines = [
        r"\begin{table*}[t]",
        r"\caption{Per-dataset classification accuracy on the UEA multivariate time-series classification benchmark.",
        r"All methods are evaluated on the official train/test splits whenever reproducible results are available under the same protocol.",
        r"The best result on each dataset is shown in bold, and the second-best result is underlined.",
        r"Methods with unavailable or non-comparable results are marked as ``--'' and are excluded from mean-rank and win calculations.}",
        r"\label{tab:main}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccccccccccc}",
        r"\toprule",
        r"Dataset ",
        r"& HC2 ",
        r"& ROCKET ",
        r"& MultiRocket ",
        r"& InceptionTime ",
        r"& Transformer ",
        r"& MTS2Graph ",
        r"& SimTSC ",
        r"& TMA-GAT ",
        r"& TapNet ",
        r"& PDFTime ",
        r"& VETO \\",
        r"\midrule",
        *rows,
        r"\midrule",
        "Mean Acc. $\\uparrow$      & " + " & ".join(mean_acc) + r" \\",
        "Mean Rank $\\downarrow$    & " + " & ".join(mean_rank) + r" \\",
        "Wins $\\uparrow$           & " + " & ".join(wins_row) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\vspace{0.5em}",
        "",
        r"\footnotesize",
        r"HC2 denotes HIVE-COTE 2.0. All aggregate statistics are computed only over datasets where comparable results are available for the included methods.",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build main UEA comparison table")
    parser.add_argument(
        "--external",
        default="diagnostics/external_baselines/multiverse_accuracy_mean.csv",
    )
    parser.add_argument("--veto_csv", nargs="*", default=[])
    parser.add_argument(
        "--output_tex",
        default="diagnostics/main_table/table_main_comparison.tex",
    )
    parser.add_argument(
        "--output_csv",
        default="diagnostics/main_table/table_main_comparison_values.csv",
    )
    args = parser.parse_args()

    values = build_values(Path(args.external), [Path(path) for path in args.veto_csv])
    output_tex = Path(args.output_tex)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(build_latex(values), encoding="utf-8")
    write_numeric_csv(values, Path(args.output_csv))
    print(f"Wrote {output_tex}")
    print(f"Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
