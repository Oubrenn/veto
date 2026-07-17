import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "diagnostics" / "paper_figures"
SOURCE = OUTPUT / "fig5_phase_path_evidence_source_data"


def test_figure5_export_bundle_is_internally_consistent():
    metadata = json.loads((SOURCE / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataset"] == "Handwriting"
    assert metadata["data_are_placeholder"] is False
    assert metadata["selection_device"] == "cpu"
    assert metadata["deterministic_selection"] is True
    assert metadata["displayed_channels_one_based"] == [1, 3, 2]
    assert metadata["figure_inference_padding_masked"] is True
    assert metadata["checkpoint_training_padding_masked"] is False
    assert metadata["length_audit"]["test"]["min"] == 34
    assert metadata["length_audit"]["test"]["median"] == 90.0
    assert metadata["length_audit"]["test"]["max"] == 149

    marginals = pd.read_csv(SOURCE / "class_phase_marginals.csv")
    pi_columns = [column for column in marginals if column.startswith("pi_P")]
    np.testing.assert_allclose(marginals[pi_columns].sum(axis=1), 1.0, atol=1e-6)

    pairs = pd.read_csv(SOURCE / "class_pair_selection.csv")
    selected_pair = pairs.loc[pairs["selected"]]
    candidates = pairs.loc[pairs["high_similarity_candidate"]]
    assert len(selected_pair) == 1
    np.testing.assert_allclose(
        selected_pair["transition_divergence"].iloc[0],
        candidates["transition_divergence"].max(),
        atol=1e-12,
    )

    catalog = pd.read_csv(SOURCE / "correct_sample_catalog.csv")
    assert "transition_gain_per_valid_transition" in catalog
    selected_catalog = catalog.loc[catalog["selected"]]
    assert len(selected_catalog) == 2
    for row in selected_catalog.itertuples(index=False):
        class_rows = catalog.loc[catalog["class_idx"] == row.class_idx]
        np.testing.assert_allclose(
            row.absolute_distance_to_class_median,
            class_rows["absolute_distance_to_class_median"].min(),
            atol=1e-12,
        )

    for sample in metadata["selected_samples"]:
        prefix = f"class_{sample['class_label']}_sample_{sample['sample_idx']}"
        responsibility = pd.read_csv(SOURCE / f"{prefix}_responsibilities.csv")
        q_columns = [column for column in responsibility if column.startswith("q_P")]
        assert len(responsibility) == sample["valid_windows"]
        assert int(responsibility["window_end_exclusive"].max()) <= sample[
            "effective_length"
        ]
        np.testing.assert_allclose(
            responsibility[q_columns].sum(axis=1),
            1.0,
            atol=1e-6,
        )

        transition = pd.read_csv(
            SOURCE / f"class_{sample['class_label']}_transition_matrix.csv"
        )
        row_sums = transition.groupby("from_phase")["probability"].sum()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    svg = (OUTPUT / "fig5_phase_path_evidence.svg").read_text(encoding="utf-8")
    assert "<text" in svg
    assert "<image" not in svg

    pdf = (OUTPUT / "fig5_phase_path_evidence.pdf").read_bytes()
    assert pdf.startswith(b"%PDF")
    assert b"/Subtype /Image" not in pdf

    with Image.open(OUTPUT / "fig5_phase_path_evidence.png") as image:
        assert image.width >= 4000
        assert image.height >= 2000
        dpi = image.info.get("dpi", (0.0, 0.0))
        assert min(dpi) >= 590
