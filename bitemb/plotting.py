"""Publication-quality figures for Phase 1 characterization.

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
