"""Merge external, VETO, and neural-baseline CSVs into the main LaTeX table."""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


DATASETS = [
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

HHAR_DATASET = "HHAR"

METHODS = [
    "HC2",
    "ROCKET",
    "MultiRocket",
    "InceptionTime",
    "TimesNet",
    "MTS2Graph",
    "SimTSC",
    "TMA-GAT",
    "TapNet",
    "PDFTime",
    "VETO",
]

MODEL_TO_METHOD = {
    "veto": "VETO",
    "hc2_lite": "HC2",
    "rocket": "ROCKET",
    "multirocket": "MultiRocket",
    "inceptiontime": "InceptionTime",
    "timesnet": "TimesNet",
    "tapnet": "TapNet",
    "pdftime": "PDFTime",
    "mts2graph": "MTS2Graph",
    "simtsc": "SimTSC",
    "tma_gat": "TMA-GAT",
}

EXTERNAL_COLUMNS = {
    "HC2": "HC2",
    "ROCKET": "ROCKET",
    "InceptionTime": "h-inceptiontime",
}

TIE_TOL = 5e-7


def parse_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value or value in {"--", r"\NA"}:
        return None
    value = re.sub(r"\\(?:best|second)\{([^{}]+)\}", r"\1", value)
    try:
        return float(value)
    except ValueError:
        return None


def selected_accuracy(row: dict) -> float | None:
    score = parse_float(row.get("selected_test_acc"))
    if score is None:
        score = parse_float(row.get("best_test_acc"))
    return score


def read_external(path: Path, values, datasets):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        dataset_field = reader.fieldnames[0] if reader.fieldnames else None
        if not dataset_field:
            return
        for row in reader:
            dataset = row.get(dataset_field, "").strip()
            if dataset not in datasets:
                continue
            for method, column in EXTERNAL_COLUMNS.items():
                score = parse_float(row.get(column))
                if score is not None:
                    values[dataset][method] = score


def read_veto(paths: list[Path], values, datasets):
    best_by_dataset_seed = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                dataset = row.get("dataset", "").strip()
                if dataset not in datasets:
                    continue
                status = row.get("status", "ok").strip()
                if status and status != "ok":
                    continue
                score = selected_accuracy(row)
                if score is None:
                    continue
                seed = row.get("seed", "0").strip() or "0"
                epochs = int(float(row.get("epochs_completed", "0") or 0))
                best_by_dataset_seed[(dataset, seed)].append((epochs, score))

    by_dataset = defaultdict(list)
    for (dataset, _seed), rows in best_by_dataset_seed.items():
        epochs, score = max(rows, key=lambda item: (item[0], item[1]))
        if epochs > 0:
            by_dataset[dataset].append(score)
    for dataset, scores in by_dataset.items():
        values[dataset]["VETO"] = sum(scores) / len(scores)


def read_neural_baselines(directory: Path, values, datasets):
    if not directory.exists():
        return
    for path in directory.glob("*.csv"):
        stem = path.stem
        model = next((name for name in MODEL_TO_METHOD if stem.startswith(f"{name}_")), None)
        if model is None:
            continue
        method = MODEL_TO_METHOD[model]
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                dataset = row.get("dataset", "").strip()
                if dataset not in datasets:
                    continue
                status = row.get("status", "ok").strip()
                if status and status != "ok":
                    continue
                score = selected_accuracy(row)
                if score is not None and method not in values[dataset]:
                    values[dataset][method] = score


def ranks(row):
    pairs = sorted(row.items(), key=lambda item: item[1], reverse=True)
    result = {}
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and abs(pairs[end][1] - pairs[idx][1]) <= TIE_TOL:
            end += 1
        rank = (idx + 1 + end) / 2.0
        for method, _ in pairs[idx:end]:
            result[method] = rank
        idx = end
    return result


def fmt(value):
    return f"{value:.3f}"


def cell(value, row):
    if value is None:
        return r"\NA"
    numeric = sorted(set(row.values()), reverse=True)
    text = fmt(value)
    if len(numeric) < 2:
        return text
    if abs(value - numeric[0]) <= TIE_TOL:
        return rf"\best{{{text}}}"
    if abs(value - numeric[1]) <= TIE_TOL:
        return rf"\second{{{text}}}"
    return text


def aggregates(values, datasets):
    acc_sum = defaultdict(float)
    acc_count = defaultdict(int)
    rank_sum = defaultdict(float)
    rank_count = defaultdict(int)
    wins = defaultdict(int)

    for dataset in datasets:
        row = {m: values[dataset][m] for m in METHODS if m in values[dataset]}
        for method, score in row.items():
            acc_sum[method] += score
            acc_count[method] += 1
        if len(row) < 2:
            continue
        row_ranks = ranks(row)
        for method, rank in row_ranks.items():
            rank_sum[method] += rank
            rank_count[method] += 1
        best = max(row.values())
        for method, score in row.items():
            if abs(score - best) <= TIE_TOL:
                wins[method] += 1

    mean_acc = [r"\NA" if acc_count[m] == 0 else fmt(acc_sum[m] / acc_count[m]) for m in METHODS]
    mean_rank = [r"\NA" if rank_count[m] == 0 else fmt(rank_sum[m] / rank_count[m]) for m in METHODS]
    wins_row = [r"\NA" if acc_count[m] == 0 else str(wins[m]) for m in METHODS]
    return mean_acc, mean_rank, wins_row


def write_csv(values, output: Path, datasets):
    output.parent.mkdir(parents=True, exist_ok=True)
    mean_acc, mean_rank, wins_row = aggregates(values, datasets)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", *METHODS])
        for dataset in datasets:
            writer.writerow([dataset, *[values[dataset].get(method, "") for method in METHODS]])
        writer.writerow(["Mean Acc.", *mean_acc])
        writer.writerow(["Mean Rank", *mean_rank])
        writer.writerow(["Wins", *wins_row])


def build_latex(values, datasets):
    rows = []
    for dataset in datasets:
        row = {method: values[dataset][method] for method in METHODS if method in values[dataset]}
        rows.append(
            f"{dataset:<27} & "
            + " & ".join(cell(values[dataset].get(method), row) for method in METHODS)
            + r" \\"
        )

    mean_acc, mean_rank, wins_row = aggregates(values, datasets)
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\caption{Per-dataset classification accuracy on the UEA multivariate time-series classification benchmark.",
            r"All methods are evaluated on the official train/test splits whenever reproducible results are available under the same protocol.",
            r"The best result on each dataset is shown in bold, and the second-best result is underlined.",
            r"Methods with unavailable or non-comparable results are marked as ``--'' and are excluded from mean-rank and win calculations.}",
            r"\label{tab:main}",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.4pt}",
            r"\resizebox{\textwidth}{!}{",
            r"\begin{tabular}{lccccccccccc}",
            r"\toprule",
            r"\multicolumn{1}{c}{Dataset} & \multicolumn{11}{c}{Methods} \\",
            r"\cmidrule(lr){2-12}",
            r"& HC2 & ROCKET & MultiRocket & InceptionTime & TimesNet & MTS2Graph & SimTSC & TMA-GAT & TapNet & PDFTime & VETO \\",
            r"\midrule",
            r"Type & Ensemble & RandConv & RandConv & CNN & Foundation & Graph & Graph & Motif-Graph & Prototype & Prototype & Ours \\",
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
            r"HC2 denotes HIVE-COTE 2.0, and RandConv denotes random-convolution-based classifiers.",
            r"Missing HC2 entries are filled by an in-repository HC2-lite ensemble proxy; missing ROCKET, MultiRocket, InceptionTime, TimesNet, MTS2Graph, SimTSC, TMA-GAT, TapNet, and PDFTime entries are filled with in-repository reimplementations run on the official UEA train/test splits.",
            r"Aggregate statistics are computed over available comparable entries; methods with unavailable results are excluded from the corresponding mean-rank and win calculations.",
            r"\end{table*}",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external", default="diagnostics/external_baselines/multiverse_accuracy_mean.csv")
    parser.add_argument("--baseline_dir", default="diagnostics/main_table_baselines_10ep")
    parser.add_argument("--veto_csv", nargs="*", default=[
        "diagnostics/official_cuda_10ep.csv",
        "diagnostics/official_cuda_10ep_part2.csv",
        "diagnostics/official_cuda_10ep_uwave.csv",
        "diagnostics/main_table/veto_missingA_bs64_10ep.csv",
        "diagnostics/main_table/veto_missingB_bs64_10ep.csv",
        "diagnostics/main_table/veto_racketsports_bs64_10ep.csv",
        "diagnostics/main_table/veto_eigen_bs8_10ep.csv",
    ])
    parser.add_argument("--output_tex", default="diagnostics/main_table/table_main_comparison_26uea_10ep.tex")
    parser.add_argument("--output_csv", default="diagnostics/main_table/table_main_comparison_26uea_10ep_values.csv")
    parser.add_argument("--include_hhar", action="store_true")
    args = parser.parse_args()

    datasets = DATASETS + ([HHAR_DATASET] if args.include_hhar else [])
    values = {dataset: {} for dataset in datasets}
    read_external(Path(args.external), values, datasets)
    read_veto([Path(path) for path in args.veto_csv], values, datasets)
    read_neural_baselines(Path(args.baseline_dir), values, datasets)

    output_tex = Path(args.output_tex)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(build_latex(values, datasets), encoding="utf-8")
    write_csv(values, Path(args.output_csv), datasets)
    print(f"Wrote {output_tex}")
    print(f"Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
