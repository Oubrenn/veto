from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def gaussian_pulse(x, center, width, amplitude):
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def sigmoid_step(x, center, width, amplitude):
    return amplitude / (1.0 + np.exp(-(x - center) / width))


def normalize_series(raw_series):
    series = []
    for y in raw_series:
        y = y - np.median(y)
        scale = np.quantile(np.abs(y), 0.985)
        series.append(y / scale)
    return series


def build_series(n=260, seed=7):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)

    source_a = 0.11 * rng.normal(size=n)
    for center, width, amplitude in [
        (0.09, 0.006, 0.95),
        (0.18, 0.004, 1.75),
        (0.205, 0.003, 2.55),
        (0.49, 0.005, 0.55),
        (0.76, 0.005, 1.55),
        (0.815, 0.004, 1.25),
    ]:
        source_a += gaussian_pulse(x, center, width, amplitude)

    source_b = (
        0.55 * np.sin(2 * np.pi * 5.2 * x + 0.4)
        + 0.22 * np.sin(2 * np.pi * 13.5 * x)
        + 0.09 * rng.normal(size=n)
    )
    source_b += gaussian_pulse(x, 0.35, 0.035, 0.45)
    source_b -= gaussian_pulse(x, 0.68, 0.04, 0.35)

    impulse_centers = np.array([0.12, 0.28, 0.57, 0.72, 0.88])
    source_c = 0.06 * rng.normal(size=n)
    for center in impulse_centers:
        tail = np.maximum(x - center, 0)
        source_c += (tail > 0) * np.exp(-tail / 0.035) * rng.uniform(0.55, 1.0)
        source_c -= gaussian_pulse(x, center - 0.012, 0.006, 0.25)

    return x, normalize_series([source_a, source_b, source_c])


def build_variant_one(n=280, seed=19):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)

    series_a = 0.08 * rng.normal(size=n)
    for center, width, amplitude in [
        (0.08, 0.012, -0.75),
        (0.16, 0.007, 1.45),
        (0.31, 0.010, 1.05),
        (0.52, 0.006, -1.20),
        (0.64, 0.012, 1.60),
        (0.84, 0.008, 1.05),
    ]:
        series_a += gaussian_pulse(x, center, width, amplitude)
    series_a += 0.14 * np.sin(2 * np.pi * 2.0 * x)

    chirp_phase = 2 * np.pi * (2.2 * x + 5.8 * x**2)
    series_b = (0.18 + 0.42 * x) * np.sin(chirp_phase)
    series_b += 0.22 * np.sin(2 * np.pi * 17.5 * x + 0.8)
    series_b += 0.08 * rng.normal(size=n)

    series_c = 0.05 * rng.normal(size=n)
    for start, end, height in [
        (0.10, 0.18, 0.80),
        (0.36, 0.44, -0.55),
        (0.58, 0.67, 1.05),
        (0.76, 0.90, 0.65),
    ]:
        series_c += sigmoid_step(x, start, 0.004, height)
        series_c -= sigmoid_step(x, end, 0.004, height)
    series_c += 0.12 * np.sin(2 * np.pi * 9.0 * x)

    return x, normalize_series([series_a, series_b, series_c])


def build_variant_two(n=300, seed=31):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)

    series_a = 0.08 * rng.normal(size=n)
    for center, width, amplitude in [
        (0.12, 0.035, 0.90),
        (0.30, 0.018, -0.65),
        (0.48, 0.028, 1.10),
        (0.70, 0.020, -0.85),
        (0.89, 0.030, 0.70),
    ]:
        series_a += gaussian_pulse(x, center, width, amplitude)
    series_a += 0.17 * np.sin(2 * np.pi * 11.0 * x + 0.5)

    step_levels = np.piecewise(
        x,
        [
            x < 0.18,
            (x >= 0.18) & (x < 0.34),
            (x >= 0.34) & (x < 0.52),
            (x >= 0.52) & (x < 0.75),
            x >= 0.75,
        ],
        [0.0, 0.65, -0.30, 0.95, 0.15],
    )
    series_b = step_levels + 0.09 * rng.normal(size=n)
    for center in [0.24, 0.53, 0.78]:
        series_b -= gaussian_pulse(x, center, 0.010, 0.65)

    series_c = 0.04 * rng.normal(size=n)
    for center, amplitude in [(0.09, 1.0), (0.39, 0.85), (0.62, 1.10), (0.86, 0.75)]:
        tail = np.maximum(x - center, 0)
        series_c += (tail > 0) * amplitude * np.exp(-tail / 0.055) * np.sin(
            2 * np.pi * 32.0 * tail
        )
    series_c += 0.16 * np.sin(2 * np.pi * 3.5 * x)

    return x, normalize_series([series_a, series_b, series_c])


def save_figure(fig, stem, dpi=600):
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.tiff", dpi=dpi, bbox_inches="tight", facecolor="white")


def plot_stacked_series(x, series, colors, stem):
    offsets = [4.8, 2.4, 0.0]
    fig, ax = plt.subplots(figsize=(6.4, 2.25), constrained_layout=True)

    for y, offset, color in zip(series, offsets, colors):
        y_plot = 0.55 * y + offset
        ax.plot(x, y_plot, color=color, lw=1.15, solid_capstyle="round")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.85, 5.95)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.margins(x=0)

    save_figure(fig, stem)
    plt.close(fig)


def main():
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    figure_specs = [
        (
            build_series,
            ["#1f5a96", "#d77a2f", "#2f8f5b"],
            "multisource_time_series",
        ),
        (
            build_variant_one,
            ["#6b4c9a", "#c58f2a", "#0b8f9c"],
            "multisource_time_series_variant_01",
        ),
        (
            build_variant_two,
            ["#b23a48", "#3d5a80", "#5a8f29"],
            "multisource_time_series_variant_02",
        ),
    ]

    for builder, colors, filename in figure_specs:
        x, series = builder()
        plot_stacked_series(x, series, colors, out_dir / filename)


if __name__ == "__main__":
    main()
