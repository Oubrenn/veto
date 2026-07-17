"""Build paper-ready tables and plots from experiment CSV outputs."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def dataframe_to_markdown(df: pd.DataFrame, float_format: str) -> str:
    rendered = df.copy()
    for col in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[col]):
            rendered[col] = rendered[col].map(lambda value: float_format % value)
    rendered = rendered.astype(str)
    headers = list(rendered.columns)
    rows = rendered.values.tolist()
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in rows)) if rows else len(str(header))
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *row_lines])


def write_table(df, path_base: Path, float_format="%.4f"):
    path_base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_base.with_suffix(".csv"), index=False)
    with path_base.with_suffix(".md").open("w", encoding="utf-8") as file:
        file.write(dataframe_to_markdown(df, float_format))
        file.write("\n")
    with path_base.with_suffix(".tex").open("w", encoding="utf-8") as file:
        file.write(df.to_latex(index=False, float_format=lambda x: float_format % x))


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    return df


def _load_metric_frame(path: Path, metric: str) -> pd.DataFrame:
    df = _load_csv(path)
    if metric not in df.columns:
        if metric == "selected_test_acc" and "best_test_acc" in df.columns:
            df = df.copy()
            df["selected_test_acc"] = df["best_test_acc"]
        else:
            raise ValueError(f"Metric {metric!r} not found in {path}")
    return df


def synthetic_artifacts(input_dir: Path, output_dir: Path):
    path = input_dir / "synthetic_path_control.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    table = df[
        [
            "task",
            "mode",
            "accuracy",
            "macro_f1",
            "g_true_minus_g_wrong",
            "delta_g",
            "valid_path_vs_shuffled_auroc",
        ]
    ].copy()
    write_table(table, output_dir / "tables" / "table_synthetic_path_control")

    pivot = df.pivot(index="task", columns="mode", values="accuracy")
    ax = pivot.plot(kind="bar", figsize=(8.5, 4.5), rot=0)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.legend(title="", ncols=2, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig_path = output_dir / "figures" / "fig_synthetic_accuracy.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=220)
    plt.close()

    pivot = df.pivot(index="task", columns="mode", values="delta_g")
    ax = pivot.plot(kind="bar", figsize=(8.5, 4.5), rot=0)
    ax.set_ylabel("Delta G: valid - shuffled")
    ax.set_xlabel("")
    ax.legend(title="", ncols=2, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig_path2 = output_dir / "figures" / "fig_synthetic_delta_g.png"
    plt.savefig(fig_path2, dpi=220)
    plt.close()
    return [path, fig_path, fig_path2]


def memory_artifacts(input_dir: Path, output_dir: Path):
    paths = [
        ("label_noise", input_dir / "memory_pollution_label_noise.csv"),
        ("window_noise", input_dir / "memory_pollution_window_noise.csv"),
    ]
    written = []
    frames = []
    for noise_name, path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.insert(0, "stress", noise_name)
        frames.append(df)
    if not frames:
        return written
    df = pd.concat(frames, ignore_index=True)
    cols = [
        "stress",
        "pollution",
        "policy",
        "false_commit_rate",
        "accepted_sample_purity",
        "prototype_drift",
        "transition_consistency",
        "wrong_class_write_ratio",
        "commit_recall_proxy",
        "confirmation_delay",
    ]
    table = df[cols].copy()
    write_table(table, output_dir / "tables" / "table_memory_pollution")
    written.append(output_dir / "tables" / "table_memory_pollution.csv")

    for stress, stress_df in df.groupby("stress"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for policy, policy_df in stress_df.groupby("policy"):
            policy_df = policy_df.sort_values("pollution")
            axes[0].plot(policy_df["pollution"], policy_df["false_commit_rate"], marker="o", label=policy)
            axes[1].plot(policy_df["pollution"], policy_df["prototype_drift"], marker="o", label=policy)
        axes[0].set_ylabel("False commit rate")
        axes[1].set_ylabel("Prototype drift")
        for ax in axes:
            ax.set_xlabel("Pollution")
            ax.grid(alpha=0.25)
        axes[1].legend(fontsize=7, loc="best")
        fig.suptitle(stress.replace("_", " "))
        plt.tight_layout()
        fig_path = output_dir / "figures" / f"fig_memory_{stress}.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=220)
        plt.close()
        written.append(fig_path)
    return written


def efficiency_artifacts(input_dir: Path, output_dir: Path):
    path = input_dir / "efficiency_scaling.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    table = df[
        [
            "n_classes",
            "seq_length",
            "n_windows",
            "n_phases",
            "parameters",
            "estimated_veto_head_flops",
            "latency_ms_per_sample",
            "throughput_samples_per_sec",
            "peak_gpu_memory_mb",
        ]
    ].copy()
    write_table(table, output_dir / "tables" / "table_efficiency_scaling")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    baseline_k = df["n_phases"].mode().iloc[0]
    baseline_len = df["seq_length"].mode().iloc[0]
    by_y = df[(df["n_phases"] == baseline_k) & (df["seq_length"] == baseline_len)].sort_values("n_classes")
    axes[0].plot(by_y["n_classes"], by_y["latency_ms_per_sample"], marker="o")
    axes[0].set_xlabel("Classes Y")
    axes[0].set_ylabel("ms/sample")

    baseline_y = df["n_classes"].max()
    by_n = df[(df["n_classes"] == baseline_y) & (df["n_phases"] == baseline_k)].sort_values("n_windows")
    axes[1].plot(by_n["n_windows"], by_n["latency_ms_per_sample"], marker="o")
    axes[1].set_xlabel("Windows N")

    by_k = df[(df["n_classes"] == baseline_y) & (df["seq_length"] == baseline_len)].sort_values("n_phases")
    axes[2].plot(by_k["n_phases"], by_k["latency_ms_per_sample"], marker="o")
    axes[2].set_xlabel("Phases K")
    for ax in axes:
        ax.grid(alpha=0.25)
    plt.tight_layout()
    fig_path = output_dir / "figures" / "fig_efficiency_scaling.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=220)
    plt.close()
    return [path, fig_path]


def efficiency_comparison_artifacts(output_dir: Path):
    datasets = [
        "DuckDuckGeese",
        "MotorImagery",
        "SelfRegulationSCP1",
        "SelfRegulationSCP2",
    ]
    method_sources = [
        (
            "Backbone only",
            [
                Path("diagnostics/main_table_baselines_10ep/inceptiontime_DuckDuckGeese.csv"),
                Path("diagnostics/main_table_baselines_10ep/inceptiontime_MotorImagery.csv"),
                Path("diagnostics/main_table_baselines_10ep/inceptiontime_SelfRegulationSCP1.csv"),
                Path("diagnostics/main_table_baselines_10ep/inceptiontime_SelfRegulationSCP2.csv"),
            ],
            10,
        ),
        (
            "VETO w/o transition gain",
            [Path("diagnostics/ablation_tables/run_fast/raw_nomem_nocf.csv")],
            1,
        ),
        (
            "VETO w/o confirmed memory",
            [Path("diagnostics/ablation_tables/run_fast/direct_ema_memory.csv")],
            1,
        ),
        (
            "VETO full model",
            [Path("diagnostics/ablation_tables/run_fast/full_veto.csv")],
            1,
        ),
    ]

    rows = []
    for method, sources, epoch_divisor in method_sources:
        frames = []
        for source in sources:
            df = _load_csv(source)
            if "dataset" in df.columns:
                df = df[df["dataset"].isin(datasets)]
            if "seed" in df.columns:
                df = df[df["seed"].eq(42)]
            if df.empty:
                continue
            frames.append(df)
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        if "epochs_completed" not in df.columns:
            df["epochs_completed"] = epoch_divisor
        df["train_time_per_epoch"] = pd.to_numeric(df["train_seconds"], errors="coerce") / pd.to_numeric(
            df["epochs_completed"].replace(0, np.nan), errors="coerce"
        )
        df["inference_ms"] = pd.to_numeric(df["single_sample_latency_ms"], errors="coerce")
        df["params"] = pd.to_numeric(df["model_params"], errors="coerce")
        df["memory_mb"] = pd.to_numeric(df["peak_gpu_memory_mb"], errors="coerce")
        rows.append(
            {
                "Method": method,
                "Params": f"{int(round(df['params'].mean())):,}",
                "Train time (s/epoch) $\\downarrow$": float(df["train_time_per_epoch"].mean()),
                "Inference time (ms/sample) $\\downarrow$": float(df["inference_ms"].mean()),
                "GPU memory (MB) $\\downarrow$": float(df["memory_mb"].mean()),
            }
        )

    table = pd.DataFrame(rows)
    write_table(table, output_dir / "tables" / "table_efficiency_comparison", float_format="%.3f")
    return [output_dir / "tables" / "table_efficiency_comparison.csv"]


def order_artifacts(input_dir: Path, output_dir: Path):
    path = input_dir / "order_corruption.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    cols = [
        "dataset",
        "level",
        "strategy",
        "severity",
        "occurrence_invariance_error",
        "iid_reference_invariance_error",
        "transition_gain_drop",
        "accuracy_drop",
        "prediction_flip_rate",
        "valid_corrupt_auroc",
    ]
    table = df[cols].copy()
    write_table(table, output_dir / "tables" / "table_order_corruption")

    plot_df = df[df["severity"].astype(str) != "severity_auc"].copy()
    if not plot_df.empty:
        ax = plot_df.plot(
            x="severity",
            y=["transition_gain_drop", "accuracy_drop", "prediction_flip_rate"],
            marker="o",
            figsize=(7.5, 4),
        )
        ax.set_xlabel("Corruption severity")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        fig_path = output_dir / "figures" / "fig_order_corruption.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=220)
        plt.close()
        return [path, fig_path]
    return [path]


def main_comparison_artifacts(input_dir: Path, output_dir: Path):
    csv_paths = sorted(input_dir.glob("*.csv"))
    frames = []
    for path in csv_paths:
        if path.name.endswith("_commands.csv") or path.name.startswith("summary"):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        required = {"experiment", "dataset"}
        if not required.issubset(df.columns):
            continue
        if "status" in df.columns:
            df = df[df["status"] == "ok"]
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return []

    df = pd.concat(frames, ignore_index=True)
    if "selected_test_acc" not in df.columns and "best_test_acc" in df.columns:
        df["selected_test_acc"] = df["best_test_acc"]
    if "selected_test_acc" not in df.columns:
        df["selected_test_acc"] = pd.NA
    if "selected_test_macro_f1" not in df.columns and "best_macro_f1" in df.columns:
        df["selected_test_macro_f1"] = df["best_macro_f1"]
    grouped = (
        df.groupby(["experiment", "dataset"], as_index=False)
        .agg(
            accuracy=("selected_test_acc", "mean"),
            macro_f1=("selected_test_macro_f1", "mean"),
            n_seeds=("seed", "nunique") if "seed" in df.columns else ("selected_test_acc", "count"),
            params=("model_params", "mean") if "model_params" in df.columns else ("selected_test_acc", "count"),
            latency_ms=("single_sample_latency_ms", "mean")
            if "single_sample_latency_ms" in df.columns
            else ("selected_test_acc", "count"),
        )
    )
    write_table(grouped, output_dir / "tables" / f"table_main_{input_dir.name}")

    mean_table = grouped.pivot(index="dataset", columns="experiment", values="accuracy")
    mean_table.to_csv(output_dir / "tables" / f"matrix_main_acc_{input_dir.name}.csv")
    ax = mean_table.plot(kind="bar", figsize=(10, 4.8), rot=35)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.legend(title="", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig_path = output_dir / "figures" / f"fig_main_acc_{input_dir.name}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=220)
    plt.close()
    return [output_dir / "tables" / f"table_main_{input_dir.name}.csv", fig_path]


def main():
    parser = argparse.ArgumentParser(description="Build paper tables/figures from diagnostics CSVs")
    parser.add_argument("--input_dir", default="diagnostics/paper_run_fast")
    parser.add_argument("--output_dir", default="diagnostics/paper_artifacts")
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    written = []
    written.extend(synthetic_artifacts(input_dir, output_dir))
    written.extend(memory_artifacts(input_dir, output_dir))
    written.extend(efficiency_artifacts(input_dir, output_dir))
    written.extend(efficiency_comparison_artifacts(output_dir))
    written.extend(order_artifacts(input_dir, output_dir))
    written.extend(main_comparison_artifacts(input_dir, output_dir))
    print(f"Wrote artifacts under {output_dir}")
    for item in written:
        print(item)


if __name__ == "__main__":
    main()
