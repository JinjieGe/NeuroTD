"""Render revised figures from manuscript artwork, benchmark results and plot summaries."""

import argparse
import csv
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import Bbox
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d


ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "examples/revision/inputs"


def read_table(path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def schematic():
    image = Image.open(ROOT / "docs/assets/fig1.png")
    height = 1100 * image.height / image.width
    figure = plt.figure(figsize=(7.4, 7.4 * height / 1100))
    ax = figure.add_axes([0, 0, 1, 1])
    ax.imshow(image, extent=(0, 1100, height, 0), aspect="auto")
    ax.set(xlim=(0, 1100), ylim=(height, 0))
    ax.axis("off")
    ax.add_patch(Rectangle((639, 151), 193, 36, facecolor="white", edgecolor="none"))
    ax.text(734, 155, r"$\widehat{d}_k \in \mathrm{arg\,min}_{\tau}\, C_k(\tau)$",
            ha="center", va="center", fontsize=6.7)
    ax.text(734, 178, r"$\widehat{\Delta}(t_k)=\widehat{d}_k/f_s$", ha="center", va="center", fontsize=6.7)
    return figure


def simulation(table_dir):
    grid = read_table(INPUTS / "baseline_tables/simulation_results_discrete.csv")
    table = read_table(table_dir / "Table_S1.csv")
    methods = ["NeuroTD", "DTW", "CTW", "FFT"]
    fields = [("rmse_" + name.lower(), name) for name in methods] + [("quality_rmse_neurotd", "E")]
    if len(grid) != 66 or len(table) != 66:
        raise ValueError("Expected 66 noise-by-window combinations")
    for row, printed in zip(grid, table):
        if int(row["window_size"]) != int(printed["Window (samples, ms)"].split()[0]):
            raise ValueError("Different window sizes in benchmark grid and Table S1")
        np.testing.assert_allclose(float(row["noise_std"]), float(printed["Noise SD"]), rtol=0, atol=1e-12)
        for field, label in fields:
            value = float(row[field]) * (1 if label == "E" else 1000 / 5120)
            if f"{value:.3f}" != printed[label].rstrip("†"):
                raise ValueError(f"Mismatch in Table S1 for {label}")
    image = Image.open(ROOT / "docs/assets/fig2.png").convert("RGB")
    figure = plt.figure(figsize=(10.5, 5.885))
    bg = figure.add_axes([0, 0, 1, 1])
    bg.imshow(image, aspect="auto", extent=[0, 1050, 588.5, 0])
    bg.set(xlim=(0, 1050), ylim=(588.5, 0))
    bg.axis("off")
    for x, y, width, height in [(610, 318, 440, 272), (610, 302, 35, 25)]:
        bg.add_patch(Rectangle((x, y), width, height, facecolor="white", edgecolor="none"))
    bg.text(621, 324, "c", fontsize=22, weight="bold", va="baseline")
    ax = figure.add_axes([667/1050, (588.5-541)/588.5, (1036-667)/1050, (541-329)/588.5])
    levels = sorted({float(row["noise_std"]) for row in grid})
    for name, color, marker in zip(methods, ["#e66101", "#db68a7", "#009e73", "#087fb7"], ["s", "D", "^", "v"]):
        field = "rmse_" + name.lower()
        criterion = "quality_rmse_neurotd" if name == "NeuroTD" else field
        errors = []
        for level in levels:
            candidates = (row for row in grid if float(row["noise_std"]) == level)
            selected = min(candidates, key=lambda row: float(row[criterion]))
            errors.append(float(selected[field]) * 1000 / 5120)
        ax.plot(levels, errors, label=name, color=color, marker=marker, ms=4.7, lw=1.1,
                ls="--" if name == "NeuroTD" else ":", zorder=4 if name == "NeuroTD" else 3)
    ax.set(xlim=(-.007, .207), ylim=(-.16, 3.9), xlabel="Noise SD", ylabel="Delay RMSE (ms)")
    ax.xaxis.set_major_locator(MultipleLocator(.05))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(color="#cecece", lw=.65)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, width=.6, length=3)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    for spine in ax.spines.values():
        spine.set(color="#666666", linewidth=.65)
    ax.legend(loc="upper center", ncol=4, fontsize=8, framealpha=.94, handlelength=1.35, borderpad=.35, columnspacing=.9)
    return figure


def regional(table_dir, original):
    rows = read_table(table_dir / "Table_S6.csv")
    bins = read_table(INPUTS / "audit_v1/tables/fig4_violin_bins.csv")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.3, "axes.titlesize": 8.4,
        "axes.labelsize": 7.4, "axes.edgecolor": "#444444", "axes.linewidth": .75,
        "xtick.labelsize": 6.3, "ytick.labelsize": 6.4, "xtick.major.size": 2,
        "ytick.major.size": 2, "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(8.6, 2.04))
    if original:
        ax = fig.add_axes([.026, .115, .32, .815])
        with Image.open(original) as image:
            if image.size != (3129, 937):
                raise ValueError("Expected the original 3129 x 937 pixel fig4.png manuscript artwork")
            ax.imshow(image.convert("RGB").crop((0, 57, 1190, 880)), aspect="equal")
        ax.axis("off")
        fig.text(.005, .985, "a", va="top", weight="bold", fontsize=11.5)
    ax = fig.add_axes([.421, .22, .215, .64])
    labels = ["AMY", "HIP", "MTG", "STG"]
    for pos, (label, color) in enumerate(zip(labels, ["#d4e1ee", "#9dc2d5", "#5a94b9", "#296399"]), 1):
        subset = [row for row in bins if row["region"] == label]
        counts = np.array([int(row["count"]) for row in subset])
        if len(subset) != 100 or counts.sum() != 200000:
            raise ValueError(f"Incomplete display histogram for {label}")
        centers = [(float(row["bin_left_ms"]) + float(row["bin_right_ms"])) / 2 for row in subset]
        heights = gaussian_filter1d(counts.astype(float), 2.0, mode="nearest")
        spread = .40 * heights / heights.max()
        ax.fill_betweenx(centers, pos-spread, pos+spread, facecolor=color, edgecolor="#3a3a3a", linewidth=.62)
        median = float(subset[0]["display_sample_median_ms"])
        ax.plot([pos-.12, pos+.12], [median, median], color="black", lw=1.4)
    ax.set(ylim=(0, 16.2), xlim=(.4, 4.6), xticks=range(1, 5), xticklabels=labels,
           ylabel="Absolute delay (ms)", xlabel="Brain region", title="Within-region delays")
    ax.yaxis.set_major_locator(MultipleLocator(4))
    ax.set_axisbelow(True)
    ax.grid(color="#d0d0d0", lw=.55)
    ax = fig.add_axes([.743, .22, .247, .64])
    for pos, row in enumerate(rows):
        value = float(row["Mean difference\n(ms)"])
        lower, upper = map(float, re.findall(r"[-+]?[0-9]*\.?[0-9]+", row["Simultaneous\n95% interval (ms)"]))
        ax.errorbar(value, len(rows)-1-pos, xerr=[[value-lower], [upper-value]], fmt="o", color="#296399",
                    elinewidth=1.15, markersize=3.7, capsize=2.2)
    ax.axvline(0, color=".35", ls="--", lw=.85)
    ax.grid(color="#d0d0d0", lw=.55)
    ax.set_axisbelow(True)
    ax.set(yticks=range(6), yticklabels=[row["Comparison"] for row in rows][::-1], ylim=(-.6, 5.6),
           xlim=(-.22, 1.30), xlabel="Mean difference (ms)", xticks=[0, .4, .8, 1.2], title="Regional comparisons")
    for letter, left in [("b", .359), ("c", .653)]:
        fig.text(left, .985, letter, va="top", weight="bold", fontsize=11.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=("fig1", "fig2", "fig4"))
    parser.add_argument("--table-dir", type=Path, default=Path("revision_runs/current/supplement_tables"))
    parser.add_argument("--original-fig4", type=Path, help="Original composite fig4.png from the manuscript source for panel a.")
    parser.add_argument("--output", required=True, type=Path, help="Output PDF or PNG.")
    args = parser.parse_args()
    if args.figure == "fig1":
        figure = schematic()
    elif args.figure == "fig2":
        figure = simulation(args.table_dir)
    else:
        figure = regional(args.table_dir, args.original_fig4)
    options = {}
    if args.figure == "fig4" and args.original_fig4 is None:
        options["bbox_inches"] = Bbox.from_extents(2.99, 0, 8.6, 2.04)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=320 if args.figure == "fig2" else 450, **options)
    plt.close(figure)


if __name__ == "__main__":
    main()
