"""Fail-closed orchestrator for the manuscript's Figures 4--6.

This entry point never fabricates missing values. Figure 4 reads the saved
five-seed controlled-synthetic results, Figure 5 reads the real Handwriting
checkpoint/data, and Figure 6 uses the manuscript's audited order-gap
aggregates together with the saved memory-stress source table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
DEFAULT_OUTPUT = ROOT / "diagnostics" / "paper_figures"


def run_python(script: str, *arguments: str) -> None:
    command = [sys.executable, str(EXPERIMENTS / script), *map(str, arguments)]
    subprocess.run(command, cwd=ROOT, check=True)


def require_files(paths: list[Path], label: str) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"{label} source data are incomplete:\n{formatted}")


def redraw_figure4(output_dir: Path) -> None:
    run_python(
        "plot_figure4_controlled_synthetic.py",
        "--output-dir",
        str(output_dir),
    )


def redraw_figure5(output_dir: Path) -> None:
    run_python(
        "generate_figure5.py",
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
    )


def redraw_figure6(output_dir: Path) -> None:
    source_dir = output_dir / "source_data"
    manuscript = source_dir / "fig6_manuscript_order_gaps.csv"
    memory = source_dir / "fig6_memory_pollution.csv"
    require_files([manuscript, memory], "Figure 6")
    run_python(
        "plot_figure6_fix.py",
        "--manuscript_csv",
        str(manuscript),
        "--memory_csv",
        str(memory),
        "--output_stem",
        str(output_dir / "fig6_order_verification_three_panel"),
    )


def write_summary(output_dir: Path, figures: list[int]) -> None:
    lines = [
        "# Manuscript figures",
        "",
        "Canonical, fail-closed Python regeneration entry point:",
        "",
        "```powershell",
        "python experiments/redraw_paper_figures.py",
        "```",
        "",
    ]
    if 4 in figures:
        lines.append(
            "- Figure 4: controlled synthetic verification from five saved seeds; "
            "source data are exported beside the figure."
        )
    if 5 in figures:
        lines.append(
            "- Figure 5: real Handwriting checkpoint evidence with effective-length "
            "masking and deterministic class/sample selection."
        )
    if 6 in figures:
        lines.append(
            "- Figure 6: manuscript-defined signed order-gap aggregates and controlled "
            "memory diagnostics; the plotter rejects manuscript-value mismatches and "
            "duplicated drift/accuracy data."
        )

    fig5_metadata = output_dir / "fig5_phase_path_evidence_source_data" / "metadata.json"
    fig6_metadata = output_dir / "fig6_figure_metadata.json"
    warnings: list[str] = []
    if 5 in figures and fig5_metadata.is_file():
        payload = json.loads(fig5_metadata.read_text(encoding="utf-8"))
        if payload.get("retraining_required_for_mask_consistent_model_claims"):
            warnings.append(
                "Figure 5 uses masked inference with a legacy checkpoint trained without "
                "valid-window masking; retraining is required for a training-time claim."
            )
    if 6 in figures and fig6_metadata.is_file():
        payload = json.loads(fig6_metadata.read_text(encoding="utf-8"))
        warnings.extend(payload.get("warnings", []))
    if warnings:
        lines.extend(["", "## Evidence warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "FIGURES_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        nargs="+",
        type=int,
        choices=[4, 5, 6],
        default=[4, 5, 6],
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    figures = sorted(set(args.figures))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if 4 in figures:
        redraw_figure4(output_dir)
    if 5 in figures:
        redraw_figure5(output_dir)
    if 6 in figures:
        redraw_figure6(output_dir)
    write_summary(output_dir, figures)
    print(f"Regenerated Figures {figures} in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
