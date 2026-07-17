"""Create a clean table for one benchmark CSV."""
import argparse
from pathlib import Path

import pandas as pd

from make_paper_artifacts import write_table


def main():
    parser = argparse.ArgumentParser(description="Create single-method benchmark table")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_base", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    if "selected_test_acc" not in df.columns and "best_test_acc" in df.columns:
        df["selected_test_acc"] = df["best_test_acc"]
    if "selected_test_macro_f1" not in df.columns and "best_macro_f1" in df.columns:
        df["selected_test_macro_f1"] = df["best_macro_f1"]
    cols = [
        "dataset",
        "epochs_completed",
        "batch_size",
        "selected_test_acc",
        "selected_test_macro_f1",
        "diagnostic_delta_g",
        "diagnostic_auroc",
        "single_sample_latency_ms",
        "peak_gpu_memory_mb",
    ]
    cols = [col for col in cols if col in df.columns]
    write_table(df[cols], Path(args.output_base))
    print(f"Wrote {args.output_base}.csv/.md/.tex")


if __name__ == "__main__":
    main()
