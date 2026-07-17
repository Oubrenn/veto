"""Merge benchmark CSVs, preferring successful rows from later inputs."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Merge benchmark CSVs")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frames = []
    for path in args.inputs:
        df = pd.read_csv(path)
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    if "status" in merged.columns:
        merged["_ok"] = (merged["status"] == "ok").astype(int)
    else:
        merged["_ok"] = 1
    if "epochs_completed" in merged.columns:
        merged["_epochs"] = pd.to_numeric(merged["epochs_completed"], errors="coerce").fillna(0)
    else:
        merged["_epochs"] = 0
    merged["_input_order"] = range(len(merged))
    merged = (
        merged.sort_values(["dataset", "seed", "_ok", "_epochs", "_input_order"])
        .groupby(["dataset", "seed"], as_index=False)
        .tail(1)
        .sort_values(["dataset", "seed"])
        .drop(columns=["_ok", "_epochs", "_input_order"])
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"Wrote {output} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
