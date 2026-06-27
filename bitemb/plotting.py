"""Publication-quality figures for experiment phases.

Produces LaTeX-compatible PDF figures using matplotlib with a consistent
scientific style (serif fonts, appropriate sizing for two-column layouts).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Scientific plot style ----------

_STYLE = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
}

# Single-column width for academic papers (~3.5 inches)
_FIG_WIDTH = 5.5
_FIG_HEIGHT = 3.5


def _apply_style() -> None:
    plt.rcParams.update(_STYLE)


# ---------- Figure 1: Cumulative PCA Variance ----------


def plot_cumulative_variance(
    variances: dict[str, NDArray[np.float64]],
    output_path: Path,
    pca_dims: tuple[int, ...] = (64, 128, 256, 384),
) -> Path:
    """Plot cumulative explained variance for multiple datasets.

    Args:
        variances: {dataset_name: cumulative_variance_array (768,)}.
        output_path: Path for the output PDF.
        pca_dims: Vertical reference lines for PCA reduction targets.

    Returns:
        Path to saved figure.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    for name, cumvar in variances.items():
        ax.plot(range(1, len(cumvar) + 1), cumvar, label=name)

    # Reference lines at PCA reduction targets
    for d in pca_dims:
        ax.axvline(d, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.text(d + 3, 0.55, str(d), fontsize=7, color="gray")

    # 95% threshold
    ax.axhline(0.95, color="black", linestyle=":", linewidth=0.8)
    ax.text(10, 0.955, "95%", fontsize=7)

    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_xlim(1, 768)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------- Figure 2: Per-component Variance (sorted) ----------


def plot_variance_spectrum(
    variances: dict[str, NDArray[np.float64]],
    output_path: Path,
    n_components: int = 100,
) -> Path:
    """Plot per-component explained variance (first n components).

    Shows the asymmetry of variance distribution across dimensions,
    motivating TurboQuant's rotation step.

    Args:
        variances: {dataset_name: explained_variance_ratio array (768,)}.
        output_path: Path for the output PDF.
        n_components: Number of components to display.

    Returns:
        Path to saved figure.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    for name, var_ratio in variances.items():
        ax.plot(range(1, n_components + 1), var_ratio[:n_components], label=name)

    # Uniform baseline (what rotation achieves)
    uniform = 1.0 / 768
    ax.axhline(uniform, color="black", linestyle=":", linewidth=0.8)
    ax.text(n_components - 15, uniform * 1.3, "uniform (1/768)", fontsize=7)

    ax.set_xlabel("Principal component rank")
    ax.set_ylabel("Explained variance ratio")
    ax.set_xlim(1, n_components)
    ax.set_yscale("log")
    ax.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------- Phase 2: Distance Distortion ----------

# Colors per bit depth (consistent across plots)
_BIT_COLORS = {1: "#d62728", 2: "#ff7f0e", 4: "#2ca02c"}
_BIT_LABELS = {1: "Binary (1-bit)", 2: "TurboQuant 2-bit", 4: "TurboQuant 4-bit"}


def plot_distortion_pareto(
    results: list[dict],
    output_path: Path,
    metric: str = "spearman_rho",
    ylabel: str = "Spearman ρ",
) -> Path:
    """Pareto plot: distortion metric vs. total bits per vector.

    Shows the compression–quality trade-off across all (bit_depth, dim)
    combinations. Each dataset gets its own line style.

    Args:
        results: List of dataset result dicts from phase2 JSON.
        output_path: Path for the output PDF.
        metric: Key in result dict to plot on y-axis.
        ylabel: Label for y-axis.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    markers = ["o", "s", "^"]
    linestyles = ["-", "--", ":"]

    for ds_idx, ds in enumerate(results):
        name = ds["dataset"]
        for bit_depth in [4, 2, 1]:
            entries = [r for r in ds["results"] if r["bit_depth"] == bit_depth]
            entries.sort(key=lambda r: r["dim"])
            bits_per_vec = [r["dim"] * bit_depth for r in entries]
            values = [r[metric] for r in entries]
            ax.plot(
                bits_per_vec,
                values,
                color=_BIT_COLORS[bit_depth],
                linestyle=linestyles[ds_idx],
                marker=markers[ds_idx],
                markersize=4,
                label=f"{_BIT_LABELS[bit_depth]} ({name})" if ds_idx == 0 else None,
            )

    # Dataset legend (linestyle only)
    for ds_idx, ds in enumerate(results):
        ax.plot([], [], color="gray", linestyle=linestyles[ds_idx], label=ds["dataset"])

    ax.set_xlabel("Bits per vector")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ax.set_xlim(32, 4096)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", ncol=2, fontsize=7)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_distortion_heatmap(
    results: dict,
    output_path: Path,
    metric: str = "spearman_rho",
    title: str | None = None,
) -> Path:
    """Heatmap of distortion metric across (bit_depth × PCA_dim).

    Args:
        results: Single dataset result dict from phase2 JSON.
        output_path: Path for the output PDF.
        metric: Key to visualize.
        title: Optional figure title (defaults to dataset name).
    """
    _apply_style()

    dims = sorted({r["dim"] for r in results["results"]})
    bits = [4, 2, 1]

    matrix = np.zeros((len(bits), len(dims)))
    for r in results["results"]:
        row = bits.index(r["bit_depth"])
        col = dims.index(r["dim"])
        matrix[row, col] = r[metric]

    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, 2.2))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    # Labels
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_yticks(range(len(bits)))
    ax.set_yticklabels([_BIT_LABELS[b] for b in bits])
    ax.set_xlabel("PCA dimensions")

    # Annotate cells
    for i in range(len(bits)):
        for j in range(len(dims)):
            val = matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label=metric.replace("_", " ").title())
    ax.set_title(title or results["dataset"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
