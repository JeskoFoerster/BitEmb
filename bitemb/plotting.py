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
    plt.rcParams.update(_STYLE)  # type: ignore[arg-type]


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


# ---------- Phase 1: Dimension Distribution (Skewness/Kurtosis) ----------


def plot_dimension_distribution(
    stats: dict[str, dict[str, NDArray[np.float64]]],
    output_path: Path,
) -> Path:
    """Two-panel plot: |skewness| and kurtosis distributions across 768 dims.

    Args:
        stats: {dataset_name: {"skewness": array(768,), "kurtosis": array(768,)}}.
        output_path: Path for the output PDF.
    """
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    for name, s in stats.items():
        ax1.hist(np.abs(s["skewness"]), bins=40, alpha=0.5, label=name, density=True)
        ax2.hist(s["kurtosis"], bins=40, alpha=0.5, label=name, density=True)

    ax1.set_xlabel("|Skewness|")
    ax1.set_ylabel("Density")
    ax1.axvline(1.0, color="black", linestyle=":", linewidth=0.8)
    ax1.legend(fontsize=7)

    ax2.set_xlabel("Excess kurtosis")
    ax2.set_ylabel("Density")
    ax2.axvline(3.0, color="black", linestyle=":", linewidth=0.8)
    ax2.legend(fontsize=7)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------- Phase 1: Summary Table (LaTeX) ----------


def generate_phase1_table(results: list[dict], output_path: Path) -> Path:
    """Generate a LaTeX table summarizing Phase 1 characterization.

    Args:
        results: List of dataset result dicts (from characterization.json).
        output_path: Path for the output .tex file.
    """
    header = (
        r"\begin{table}[ht]" "\n"
        r"\centering" "\n"
        r"\caption{Phase~1: Float-space characterization across datasets.}" "\n"
        r"\label{tab:phase1}" "\n"
        r"\begin{tabular}{l" + "r" * len(results) + "}\n"
        r"\toprule" "\n"
        r"Metric & " + " & ".join(r["dataset"] for r in results) + r" \\" "\n"
        r"\midrule" "\n"
    )

    rows = []
    rows.append("$N$ (corpus) & " + " & ".join(
        f"{r['n_corpus']:,}" for r in results) + r" \\")
    rows.append(r"Norm CV & " + " & ".join(
        f"{r['norm_distribution']['cv']:.2e}" for r in results) + r" \\")
    rows.append(r"Mean $|\text{skew}|$ & " + " & ".join(
        f"{r['dimension_stats']['mean_abs_skewness']:.4f}" for r in results) + r" \\")
    rows.append(r"Max $|\text{skew}|$ & " + " & ".join(
        f"{r['dimension_stats']['max_abs_skewness']:.4f}" for r in results) + r" \\")
    rows.append(r"Mean kurtosis & " + " & ".join(
        f"{r['dimension_stats']['mean_kurtosis']:.4f}" for r in results) + r" \\")
    rows.append(r"Max kurtosis & " + " & ".join(
        f"{r['dimension_stats']['max_kurtosis']:.4f}" for r in results) + r" \\")
    rows.append(r"TwoNN $\hat{d}$ & " + " & ".join(
        f"{r['intrinsic_dimensionality']['twonn']:.1f}" for r in results) + r" \\")
    rows.append(r"PCA (95\%) & " + " & ".join(
        f"{r['intrinsic_dimensionality']['pca_95']}" for r in results) + r" \\")

    footer = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )

    content = header + "\n".join(rows) + "\n" + footer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


# ---------- Phase 2: Distance Distortion ----------

# Colors per bit depth (consistent across plots)
_BIT_COLORS = {1: "#d62728", 2: "#ff7f0e", 4: "#2ca02c"}
_BIT_LABELS = {1: "Binary (1-bit)", 2: "TurboQuant 2-bit", 4: "TurboQuant 4-bit"}


# ---------- Phase 2: Scatter Plot (Float vs. Quantized Distance) ----------


def plot_distance_scatter(
    d_float: NDArray[np.float64],
    d_quant: dict[int, NDArray[np.float64]],
    output_path: Path,
    dataset_name: str = "",
    dim: int = 768,
) -> Path:
    """Scatter plot: float distance vs. quantized distance for each bit depth.

    Args:
        d_float: Normalized float distances (n_pairs,).
        d_quant: {bit_depth: normalized quantized distances}.
        output_path: Path for the output PDF.
        dataset_name: Dataset label for title.
        dim: PCA dimension used.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(_FIG_WIDTH * 1.3, _FIG_HEIGHT), sharey=True)

    for ax, bit_depth in zip(axes, [4, 2, 1]):
        ax.scatter(d_float, d_quant[bit_depth], s=1, alpha=0.15, color=_BIT_COLORS[bit_depth])
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Float distance (norm.)")
        ax.set_title(_BIT_LABELS[bit_depth])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

    axes[0].set_ylabel("Quantized distance (norm.)")
    title = f"{dataset_name} (d={dim})" if dataset_name else f"d={dim}"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------- Phase 2: Error Distribution Histogram ----------


def plot_error_histogram(
    d_float: NDArray[np.float64],
    d_quant: dict[int, NDArray[np.float64]],
    output_path: Path,
    dataset_name: str = "",
    dim: int = 768,
) -> Path:
    """Histogram of absolute errors |d_float - d_quant| per bit depth.

    Args:
        d_float: Normalized float distances (n_pairs,).
        d_quant: {bit_depth: normalized quantized distances}.
        output_path: Path for the output PDF.
        dataset_name: Dataset label for title.
        dim: PCA dimension used.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    for bit_depth in [4, 2, 1]:
        errors = np.abs(d_float - d_quant[bit_depth])
        ax.hist(errors, bins=60, alpha=0.5, color=_BIT_COLORS[bit_depth],
                label=_BIT_LABELS[bit_depth], density=True)

    ax.set_xlabel("Absolute error |d_float − d_quant|")
    ax.set_ylabel("Density")
    title = f"{dataset_name} (d={dim})" if dataset_name else f"d={dim}"
    ax.set_title(title)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


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

    # For error metrics (lower is better), invert colormap and auto-scale
    is_error_metric = metric in ("mae", "rmse")
    cmap = "RdYlGn" if not is_error_metric else "RdYlGn_r"
    vmin = matrix.min() * 0.9 if is_error_metric else 0.0
    vmax = matrix.max() * 1.1 if is_error_metric else 1.0
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

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


# ---------- Phase 3: Neighborhood Preservation ----------


def plot_neighborhood_heatmap(
    results: dict,
    output_path: Path,
    metric: str = "overlap",
    k: int = 10,
    title: str | None = None,
) -> Path:
    """Heatmap of neighborhood metric across (bit_depth × PCA_dim).

    Args:
        results: Single dataset result dict from phase3 JSON.
        output_path: Path for the output PDF.
        metric: "overlap" or "trustworthiness".
        k: Which k value to display.
        title: Optional figure title.
    """
    _apply_style()

    entries = [r for r in results["results"] if r["k"] == k]
    dims = sorted({r["dim"] for r in entries})
    bits = [4, 2, 1]

    matrix = np.zeros((len(bits), len(dims)))
    for r in entries:
        row = bits.index(r["bit_depth"])
        col = dims.index(r["dim"])
        matrix[row, col] = r[metric]

    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, 2.2))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_yticks(range(len(bits)))
    ax.set_yticklabels([_BIT_LABELS[b] for b in bits])
    ax.set_xlabel("PCA dimensions")

    for i in range(len(bits)):
        for j in range(len(dims)):
            val = matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=color)

    label = metric.replace("_", " ").title()
    fig.colorbar(im, ax=ax, shrink=0.8, label=label)
    ax.set_title(title or f"{results['dataset']} — {label} (k={k})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_neighborhood_overlap_by_k(
    results: list[dict],
    output_path: Path,
    dim: int = 768,
    metric: str = "overlap",
    ylabel: str | None = None,
) -> Path:
    """Line plot: neighborhood metric as function of k for each bit depth.

    One line per (dataset, bit_depth) combination at fixed PCA dim.

    Args:
        results: List of dataset result dicts from phase3 JSON.
        output_path: Path for the output PDF.
        dim: PCA dimension to display.
        metric: Key in result dict ("overlap" or "trustworthiness").
        ylabel: Y-axis label (defaults based on metric).
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    linestyles = ["-", "--", ":"]

    for ds_idx, ds in enumerate(results):
        entries = [r for r in ds["results"] if r["dim"] == dim]
        for bit_depth in [4, 2, 1]:
            bit_entries = sorted(
                [r for r in entries if r["bit_depth"] == bit_depth],
                key=lambda r: r["k"],
            )
            ks = [r["k"] for r in bit_entries]
            values = [r[metric] for r in bit_entries]
            ax.plot(
                ks, values,
                color=_BIT_COLORS[bit_depth],
                linestyle=linestyles[ds_idx],
                marker="o", markersize=4,
                label=f"{_BIT_LABELS[bit_depth]}" if ds_idx == 0 else None,
            )

    # Dataset legend
    for ds_idx, ds in enumerate(results):
        ax.plot([], [], color="gray", linestyle=linestyles[ds_idx],
                label=ds["dataset"])

    label = ylabel or metric.replace("_", " ").title()
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel(label)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([10, 50, 100])
    ax.legend(loc="lower right", ncol=2, fontsize=7)
    ax.set_title(f"{label} (d={dim})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
