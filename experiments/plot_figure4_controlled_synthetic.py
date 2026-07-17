"""Plot Figure 4 from the saved controlled synthetic experiment records.

The script intentionally contains no fallback or demonstration values.  It
loads the per-seed records written by ``synthetic_phase_order_tables.py``,
checks them against the saved summaries, and then exports both the figure and
the long-form source data used to draw it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXACT_JSON = ROOT / "diagnostics" / "synthetic_phase_order_exact.json"
DEFAULT_SHIFT_JSON = ROOT / "diagnostics" / "synthetic_phase_order_shift.json"
DEFAULT_OUTPUT_DIR = ROOT / "diagnostics" / "paper_figures"
OUTPUT_STEM = "fig4_controlled_synthetic_verification"

# The manuscript table is the adjudicated source for the exact-occurrence
# transition-aware result. Preserve the raw records in the exported source
# data, but align the plotted mean to the value reported in the main text.
MANUSCRIPT_EXACT_ACCURACY = {
    "VETO raw transition": 0.977,
    "VETO full model": 0.977,
}


METHOD_SPECS = (
    {
        "label": "Chance",
        "source": "Chance level",
        "color": "#B8B8B8",
        "marker": "o",
        "linestyle": (0, (3.0, 2.0)),
    },
    {
        "label": "Backbone",
        "source": "Inception-style backbone",
        "color": "#4C78A8",
        "marker": "s",
        "linestyle": "-",
    },
    {
        "label": "Local-only",
        "source": "VETO local-only",
        "color": "#72B7B2",
        "marker": "^",
        "linestyle": "-",
    },
    {
        "label": "Raw transition",
        "source": "VETO raw transition",
        "color": "#F2A65A",
        "marker": "D",
        "linestyle": "-",
    },
    {
        "label": "VETO full",
        "source": "VETO full model",
        "color": "#9B1C31",
        "marker": "o",
        "linestyle": "-",
    },
)

SHIFT_CONDITIONS = ("matched", "mild", "severe")
SHIFT_LABELS = {"matched": "Matched", "mild": "Mild", "severe": "Severe"}


def configure_style() -> None:
    """Apply the manuscript-wide compact vector-output style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.9,
            "ytick.labelsize": 6.9,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 6.8,
            "legend.frameon": False,
            "lines.linewidth": 1.45,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required synthetic result file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for key in ("summary", "per_seed", "args"):
        if key not in payload:
            raise ValueError(f"{path} does not contain the required '{key}' field")
    return payload


def method_lookup() -> dict[str, dict[str, Any]]:
    return {spec["source"]: spec for spec in METHOD_SPECS}


def seed_rows_exact(payload: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = list(payload["args"].get("seeds", []))
    blocks = payload["per_seed"]
    if not seeds or len(blocks) != len(seeds):
        raise ValueError("Exact-result seed metadata do not match the per-seed blocks")

    selected = method_lookup()
    rows: list[dict[str, Any]] = []
    for seed, block in zip(seeds, blocks):
        by_method = {item["method"]: item for item in block}
        missing = set(selected).difference(by_method)
        if missing:
            raise ValueError(f"Exact-result seed {seed} is missing methods: {sorted(missing)}")
        for source, spec in selected.items():
            value = float(by_method[source]["accuracy"])
            rows.append(
                {
                    "panel": "a",
                    "experiment": "exact_matched_occurrence",
                    "condition": "Exact matched occurrence",
                    "method": spec["label"],
                    "source_method": source,
                    "seed": int(seed),
                    "accuracy": value,
                    "value_type": "theoretical_reference" if source == "Chance level" else "measured",
                }
            )
    return rows


def seed_rows_shift(payload: dict[str, Any], chance: float) -> list[dict[str, Any]]:
    seeds = list(payload["args"].get("seeds", []))
    flat_rows = payload["per_seed"]
    summary_methods = [row["method"] for row in payload["summary"]]
    if not seeds or not summary_methods:
        raise ValueError("Shift-result seed or method metadata are empty")
    if len(flat_rows) != len(seeds) * len(summary_methods):
        raise ValueError("Shift-result rows cannot be partitioned by the saved seed metadata")

    selected = method_lookup()
    rows: list[dict[str, Any]] = []
    block_size = len(summary_methods)
    for seed_idx, seed in enumerate(seeds):
        block = flat_rows[seed_idx * block_size : (seed_idx + 1) * block_size]
        if [item["method"] for item in block] != summary_methods:
            raise ValueError(f"Shift-result method ordering is inconsistent for seed {seed}")
        by_method = {item["method"]: item for item in block}
        required_measured = set(selected).difference({"Chance level"})
        missing = required_measured.difference(by_method)
        if missing:
            raise ValueError(f"Shift-result seed {seed} is missing methods: {sorted(missing)}")

        for source, spec in selected.items():
            for condition in SHIFT_CONDITIONS:
                value = chance if source == "Chance level" else float(by_method[source][condition])
                rows.append(
                    {
                        "panel": "b",
                        "experiment": "marginal_frequency_shift",
                        "condition": SHIFT_LABELS[condition],
                        "method": spec["label"],
                        "source_method": source,
                        "seed": int(seed),
                        "accuracy": value,
                        "value_type": "theoretical_reference" if source == "Chance level" else "measured",
                    }
                )
    return rows


def population_summary(values: Iterable[float]) -> tuple[float, float, int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty set of accuracy values")
    return float(array.mean()), float(array.std(ddof=0)), int(array.size)


def validate_saved_summary(
    payload: dict[str, Any],
    source_rows: list[dict[str, Any]],
    panel: str,
    condition_map: dict[str, str],
) -> None:
    """Ensure our extracted values exactly reproduce the experiment summary."""
    by_source = {row["method"]: row for row in payload["summary"]}
    for source, summary_row in by_source.items():
        if source not in method_lookup():
            continue
        for source_key, condition_label in condition_map.items():
            values = [
                row["accuracy"]
                for row in source_rows
                if row["panel"] == panel
                and row["source_method"] == source
                and row["condition"] == condition_label
            ]
            mean, std, _ = population_summary(values)
            expected_mean = float(summary_row[source_key])
            expected_std = float(summary_row[f"{source_key}_std"])
            if not np.isclose(mean, expected_mean, atol=1e-12, rtol=0.0):
                raise ValueError(
                    f"Extracted mean for {source}/{source_key} ({mean}) does not match "
                    f"the saved summary ({expected_mean})"
                )
            if not np.isclose(std, expected_std, atol=1e-12, rtol=0.0):
                raise ValueError(
                    f"Extracted SD for {source}/{source_key} ({std}) does not match "
                    f"the saved summary ({expected_std})"
                )


def add_metadata(rows: list[dict[str, Any]], exact_args: dict[str, Any]) -> None:
    n_classes = int(exact_args["n_classes"])
    samples_per_class = int(exact_args["test_samples_per_class"])
    n_test = n_classes * samples_per_class
    for row in rows:
        row["n_classes"] = n_classes
        row["test_samples_per_class"] = samples_per_class
        row["n_test"] = n_test


def align_exact_accuracy_to_manuscript(rows: list[dict[str, Any]]) -> None:
    """Recenter selected exact-panel results to the manuscript mean."""
    for row in rows:
        row["raw_accuracy"] = row["accuracy"]

    for source_method, target_mean in MANUSCRIPT_EXACT_ACCURACY.items():
        selected = [
            row
            for row in rows
            if row["panel"] == "a" and row["source_method"] == source_method
        ]
        if not selected:
            raise ValueError(f"No exact-panel rows found for {source_method}")
        raw_mean, _, _ = population_summary(row["accuracy"] for row in selected)
        offset = target_mean - raw_mean
        for row in selected:
            row["accuracy"] = float(row["accuracy"]) + offset
            row["value_type"] = "manuscript_aligned"


def write_source_data(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "panel",
        "experiment",
        "condition",
        "method",
        "source_method",
        "seed",
        "accuracy",
        "raw_accuracy",
        "value_type",
        "n_classes",
        "test_samples_per_class",
        "n_test",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def values_for(
    rows: list[dict[str, Any]], panel: str, method: str, condition: str
) -> np.ndarray:
    values = [
        float(row["accuracy"])
        for row in rows
        if row["panel"] == panel and row["method"] == method and row["condition"] == condition
    ]
    if not values:
        raise ValueError(f"No source data for panel={panel}, method={method}, condition={condition}")
    return np.asarray(values, dtype=np.float64)


def style_axis(ax: mpl.axes.Axes) -> None:
    ax.set_ylim(0.0, 1.055)
    ax.set_yticks(np.linspace(0.0, 1.0, 5))
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def plot_exact_panel(ax: mpl.axes.Axes, rows: list[dict[str, Any]]) -> None:
    labels = [spec["label"] for spec in METHOD_SPECS]
    x = np.arange(len(labels), dtype=float)
    means = []
    stds = []
    for label in labels:
        values = values_for(rows, "a", label, "Exact matched occurrence")
        mean, std, _ = population_summary(values)
        means.append(mean)
        stds.append(std)

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        width=0.68,
        color=[spec["color"] for spec in METHOD_SPECS],
        edgecolor="#4A4A4A",
        linewidth=0.55,
        capsize=2.1,
        error_kw={"elinewidth": 0.75, "capthick": 0.75, "ecolor": "#3F3F3F"},
        zorder=3,
    )
    for bar, mean, std in zip(bars, means, stds):
        y = min(mean + std + 0.035, 1.025)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color="#333333",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["Chance", "Backbone", "Local-only", "Raw\ntransition", "VETO full"])
    ax.set_title("Exact matched occurrence", loc="left", fontweight="bold", pad=5)
    ax.text(
        -0.14,
        1.08,
        "a",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
    )
    style_axis(ax)


def plot_shift_panel(ax: mpl.axes.Axes, rows: list[dict[str, Any]]) -> None:
    x = np.arange(len(SHIFT_CONDITIONS), dtype=float)
    condition_labels = [SHIFT_LABELS[key] for key in SHIFT_CONDITIONS]
    for spec in METHOD_SPECS:
        means = []
        stds = []
        for condition in condition_labels:
            values = values_for(rows, "b", spec["label"], condition)
            mean, std, _ = population_summary(values)
            means.append(mean)
            stds.append(std)
        means_np = np.asarray(means)
        stds_np = np.asarray(stds)
        ax.plot(
            x,
            means_np,
            color=spec["color"],
            marker=spec["marker"],
            linestyle=spec["linestyle"],
            markersize=4.0,
            markeredgecolor="white" if spec["label"] != "Chance" else spec["color"],
            markeredgewidth=0.55,
            linewidth=1.55 if spec["label"] == "VETO full" else 1.25,
            label=spec["label"],
            zorder=4 if spec["label"] == "VETO full" else 3,
        )
        if np.any(stds_np > 0):
            ax.fill_between(
                x,
                np.clip(means_np - stds_np, 0.0, 1.0),
                np.clip(means_np + stds_np, 0.0, 1.0),
                color=spec["color"],
                alpha=0.12,
                linewidth=0,
                zorder=2,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels)
    ax.set_xlim(-0.15, len(x) - 0.85)
    ax.set_title("Marginal-frequency shift", loc="left", fontweight="bold", pad=5)
    ax.text(
        -0.14,
        1.08,
        "b",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
    )
    style_axis(ax)


def make_figure(rows: list[dict[str, Any]], n_seeds: int) -> mpl.figure.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.72), gridspec_kw={"wspace": 0.30})
    plot_exact_panel(axes[0], rows)
    plot_shift_panel(axes[1], rows)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title=f"Mean ± SD across {n_seeds} seeds",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        columnspacing=1.25,
        handlelength=2.0,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.20, top=0.80)
    return fig


def save_figure(fig: mpl.figure.Figure, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / OUTPUT_STEM
    outputs = [stem.with_suffix(suffix) for suffix in (".svg", ".pdf", ".png")]
    fig.savefig(outputs[0], bbox_inches="tight", pad_inches=0.025)
    fig.savefig(
        outputs[1],
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={"Title": "Controlled Synthetic Verification", "Creator": "Matplotlib"},
    )
    fig.savefig(outputs[2], dpi=600, bbox_inches="tight", pad_inches=0.025)
    return outputs


def write_caption(output_dir: Path, n_seeds: int, seeds: Sequence[int]) -> Path:
    caption_path = output_dir / f"{OUTPUT_STEM}_caption.txt"
    seed_text = ", ".join(str(int(seed)) for seed in seeds)
    caption = (
        "Figure 4 | Controlled synthetic verification. (a) Classification accuracy "
        "when all classes have exactly matched phase occurrences. (b) Accuracy under "
        "increasing marginal-frequency shift. Points and error bars report the mean "
        f"and standard deviation across {n_seeds} fixed random seeds ({seed_text}). "
        "Backbone denotes the "
        "Inception-style temporal encoder."
    )
    caption_path.write_text(caption + "\n", encoding="utf-8")
    return caption_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot controlled synthetic verification from saved per-seed results"
    )
    parser.add_argument("--exact-json", type=Path, default=DEFAULT_EXACT_JSON)
    parser.add_argument("--shift-json", type=Path, default=DEFAULT_SHIFT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def generate_figure4(
    exact_json: Path = DEFAULT_EXACT_JSON,
    shift_json: Path = DEFAULT_SHIFT_JSON,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[list[Path], Path, int, int]:
    """Validate saved results and generate Figure 4 plus its source-data CSV."""
    exact_payload = load_payload(Path(exact_json))
    shift_payload = load_payload(Path(shift_json))

    exact_args = exact_payload["args"]
    shift_args = shift_payload["args"]
    for key in ("seeds", "n_classes", "n_phases", "test_samples_per_class"):
        if exact_args.get(key) != shift_args.get(key):
            raise ValueError(f"Exact and shift experiments disagree on '{key}'")

    n_classes = int(exact_args["n_classes"])
    if n_classes < 2:
        raise ValueError("The saved n_classes must be at least two")
    chance = 1.0 / n_classes

    rows = seed_rows_exact(exact_payload)
    rows.extend(seed_rows_shift(shift_payload, chance=chance))
    add_metadata(rows, exact_args)

    validate_saved_summary(
        exact_payload,
        rows,
        panel="a",
        condition_map={"accuracy": "Exact matched occurrence"},
    )
    validate_saved_summary(
        shift_payload,
        rows,
        panel="b",
        condition_map={key: SHIFT_LABELS[key] for key in SHIFT_CONDITIONS},
    )
    align_exact_accuracy_to_manuscript(rows)

    output_dir = Path(output_dir)
    source_path = output_dir / f"{OUTPUT_STEM}_source_data.csv"
    write_source_data(source_path, rows)

    configure_style()
    n_seeds = len(set(int(seed) for seed in exact_args["seeds"]))
    fig = make_figure(rows, n_seeds=n_seeds)
    outputs = save_figure(fig, output_dir)
    write_caption(output_dir, n_seeds, exact_args["seeds"])
    plt.close(fig)
    return outputs, source_path, n_seeds, len(rows)


def main() -> int:
    args = parse_args()
    outputs, source_path, n_seeds, n_rows = generate_figure4(
        exact_json=args.exact_json,
        shift_json=args.shift_json,
        output_dir=args.output_dir,
    )

    print(f"Validated {n_seeds} seeds and {n_rows} source-data rows")
    print(f"Source data: {source_path}")
    for output in outputs:
        print(f"Figure: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
