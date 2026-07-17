from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from experiments.build_ablation_tables import DATASETS, DESIGN_ROWS, design_table
from experiments.build_sensitivity_tables import load_row


def test_design_ablation_preserves_signed_order_gap():
    rows = []
    for variant_idx, (experiment, _) in enumerate(DESIGN_ROWS):
        for dataset, _ in DATASETS:
            rows.append(
                {
                    "experiment": experiment,
                    "dataset": dataset,
                    "selected_test_acc": 0.5 + 0.001 * variant_idx,
                    "diagnostic_delta_g": -0.125,
                }
            )

    result = design_table(pd.DataFrame(rows))
    assert result
    assert all(row["Delta_ord"] == -0.125 for row in result)


def test_sensitivity_table_does_not_flip_negative_gap():
    frame = pd.DataFrame(
        {
            "status": ["ok"] * 4,
            "dataset": [
                "Handwriting",
                "Heartbeat",
                "JapaneseVowels",
                "UWaveGestureLibrary",
            ],
            "selected_test_acc": [0.5, 0.6, 0.7, 0.8],
            "diagnostic_delta_g": [-0.1, -0.2, -0.3, -0.4],
        }
    )
    filename = "signed.csv"
    with TemporaryDirectory(dir=Path("tmp")) as directory:
        input_dir = Path(directory)
        frame.to_csv(input_dir / filename, index=False)
        result = load_row(input_dir, filename)
    assert result["delta_ord"] == -0.25
