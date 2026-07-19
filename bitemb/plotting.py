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

_BIT_COLORS = {
    16: "#aec7e8",
    8: "#9467bd",
    4: "#2ca02c",
    2: "#ff7f0e",
    1: "#d62728",
}
_BIT_LABELS = {
    16: "Float16 (16-bit)",
    8: "TurboQuant 8-bit",
    4: "TurboQuant 4-bit",
    2: "TurboQuant 2-bit",
    1: "Binary (1-bit)",
}


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
        for bit_depth in [16, 8, 4, 2, 1]:
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
    bits = [16, 8, 4, 2, 1]

    matrix = np.zeros((len(bits), len(dims)))
    for r in results["results"]:
        row = bits.index(r["bit_depth"])
        col = dims.index(r["dim"])
        matrix[row, col] = r[metric]

    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, 3.2))

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
    bits = [16, 8, 4, 2, 1]

    matrix = np.zeros((len(bits), len(dims)))
    for r in entries:
        row = bits.index(r["bit_depth"])
        col = dims.index(r["dim"])
        matrix[row, col] = r[metric]

    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, 3.2))
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
        for bit_depth in [16, 8, 4, 2, 1]:
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
    ax.set_xticks([5, 10, 20])
    ax.legend(loc="lower right", ncol=2, fontsize=7)
    ax.set_title(f"{label} (d={dim})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------- Figure: Neighborhood Pareto (bits vs. quality) ----------


def plot_neighborhood_pareto(
    results: list[dict],
    output_path: Path,
    metric: str = "overlap",
    k: int = 10,
    ylabel: str | None = None,
) -> Path:
    """Pareto plot: neighborhood metric vs. total bits per vector.

    Args:
        results: List of dataset result dicts from phase3 JSON.
        output_path: Path for the output PDF.
        metric: "overlap" or "trustworthiness".
        k: Which k value to display.
        ylabel: Y-axis label.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    markers = ["o", "s", "^"]
    linestyles = ["-", "--", ":"]

    for ds_idx, ds in enumerate(results):
        for bit_depth in [16, 8, 4, 2, 1]:
            entries = sorted(
                [r for r in ds["results"] if r["bit_depth"] == bit_depth and r["k"] == k],
                key=lambda r: r["dim"],
            )
            bits_per_vec = [r["dim"] * bit_depth for r in entries]
            values = [r[metric] for r in entries]
            ax.plot(
                bits_per_vec, values,
                color=_BIT_COLORS[bit_depth],
                linestyle=linestyles[ds_idx],
                marker=markers[ds_idx],
                markersize=4,
                label=f"{_BIT_LABELS[bit_depth]}" if ds_idx == 0 else None,
            )

    for ds_idx, ds in enumerate(results):
        ax.plot([], [], color="gray", linestyle=linestyles[ds_idx], label=ds["dataset"])

    label = ylabel or metric.replace("_", " ").title()
    ax.set_xlabel("Bits per vector")
    ax.set_ylabel(label)
    ax.set_xscale("log", base=2)
    ax.set_xlim(32, 4096)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", ncol=2, fontsize=7)
    ax.set_title(f"{label} vs. Compression (k={k})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------- Figure: Neighborhood by PCA dimension ----------


def plot_neighborhood_by_dim(
    results: list[dict],
    output_path: Path,
    metric: str = "overlap",
    k: int = 10,
    ylabel: str | None = None,
) -> Path:
    """Line plot: neighborhood metric as function of PCA dimension.

    Args:
        results: List of dataset result dicts from phase3 JSON.
        output_path: Path for the output PDF.
        metric: "overlap" or "trustworthiness".
        k: Which k value to display.
        ylabel: Y-axis label.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    linestyles = ["-", "--", ":"]

    for ds_idx, ds in enumerate(results):
        for bit_depth in [16, 8, 4, 2, 1]:
            entries = sorted(
                [r for r in ds["results"] if r["bit_depth"] == bit_depth and r["k"] == k],
                key=lambda r: r["dim"],
            )
            dims = [r["dim"] for r in entries]
            values = [r[metric] for r in entries]
            ax.plot(
                dims, values,
                color=_BIT_COLORS[bit_depth],
                linestyle=linestyles[ds_idx],
                marker="o", markersize=4,
                label=f"{_BIT_LABELS[bit_depth]}" if ds_idx == 0 else None,
            )

    for ds_idx, ds in enumerate(results):
        ax.plot([], [], color="gray", linestyle=linestyles[ds_idx], label=ds["dataset"])

    label = ylabel or metric.replace("_", " ").title()
    ax.set_xlabel("PCA dimensions")
    ax.set_ylabel(label)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([64, 128, 256, 384, 768])
    ax.legend(loc="lower right", ncol=2, fontsize=7)
    ax.set_title(f"{label} vs. Dimension (k={k})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------- Phase 5: Runtime and Memory Efficiency ----------

_REP_ORDER = [
    "float32",
    "16bit",
    "naive_8bit",
    "tq_8bit",
    "naive_4bit",
    "tq_4bit",
    "naive_2bit",
    "tq_2bit",
    "naive_1bit",
    "tq_1bit",
]
_REP_COLORS = {
    "float32": "#1f77b4",
    "16bit": "#aec7e8",
    "naive_8bit": "#c5b0d5",
    "tq_8bit": "#9467bd",
    "naive_4bit": "#98df8a",
    "tq_4bit": "#2ca02c",
    "naive_2bit": "#ffbb78",
    "tq_2bit": "#ff7f0e",
    "naive_1bit": "#ff9896",
    "tq_1bit": "#d62728",
}
_REP_LABELS = {
    "float32": "Float32",
    "16bit": "Naive 16-bit",
    "naive_8bit": "Naive 8-bit",
    "tq_8bit": "TurboQuant 8-bit",
    "naive_4bit": "Naive 4-bit",
    "tq_4bit": "TurboQuant 4-bit",
    "naive_2bit": "Naive 2-bit",
    "tq_2bit": "TurboQuant 2-bit",
    "naive_1bit": "Naive 1-bit",
    "tq_1bit": "TurboQuant 1-bit",
}


def _phase5_entry_label(entry: dict) -> str:
    return f"{entry['dataset']} (n={entry['n_vectors']}, d={entry['dim']})"


def _phase5_series_label(entry: dict) -> str:
    return f"{entry['dataset']} (n={entry['n_vectors']})"


def _phase5_legend_below(ax: plt.Axes, ncol: int = 4) -> float:
    """Place a compact legend below a Phase 5 plot and return the needed bottom margin."""
    _, labels = ax.get_legend_handles_labels()
    rows = max(1, (len(labels) + ncol - 1) // ncol)
    bottom_margin = min(0.46, 0.18 + 0.06 * (rows - 1))
    anchor_y = -0.24 - 0.08 * (rows - 1)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, anchor_y),
        ncol=ncol,
        fontsize=7,
        frameon=False,
        borderaxespad=0.0,
    )
    return bottom_margin


def _format_factor(value: float, _pos: float) -> str:
    """Format compression ratios as multiplicative factors."""
    if value >= 10 or float(value).is_integer():
        return f"{value:.0f}x"
    return f"{value:.1f}x"


_SHORT_REP_LABELS = {
    "float32": "Float32",
    "16bit": "Naive 16b",
    "naive_8bit": "Naive 8b",
    "tq_8bit": "TQ 8b",
    "naive_4bit": "Naive 4b",
    "tq_4bit": "TQ 4b",
    "naive_2bit": "Naive 2b",
    "tq_2bit": "TQ 2b",
    "naive_1bit": "Naive 1b",
    "tq_1bit": "TQ 1b",
}


def plot_phase5_memory_theoretical_vs_numpy(
    results: list[dict],
    output_path: Path,
) -> Path:
    """Bar plot comparing theoretical and NumPy bytes per vector.

    Args:
        results: List of records from phase5 memory_metrics.json.
        output_path: Path for the output PDF.
    """
    _apply_style()
    entries = sorted(results, key=lambda r: (r["dataset"], r["dim"], r["n_vectors"]))
    fig, axes = plt.subplots(
        len(entries), 1,
        figsize=(_FIG_WIDTH, max(2.4, 2.0 * len(entries))),
        squeeze=False,
    )

    for ax, entry in zip(axes[:, 0], entries):
        theoretical = {r["representation"]: r["bytes_per_vector"] for r in entry["theoretical"]}
        numpy_mem = {r["representation"]: r["bytes_per_vector"] for r in entry["numpy_vectorized"]}
        x = np.arange(len(_REP_ORDER))
        width = 0.36
        ax.bar(
            x - width / 2,
            [theoretical[r] for r in _REP_ORDER],
            width,
            label="theoretical",
            color="#9ecae1",
        )
        ax.bar(
            x + width / 2,
            [numpy_mem[r] for r in _REP_ORDER],
            width,
            label="NumPy layout",
            color="#3182bd",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([
            ("\n" + _SHORT_REP_LABELS[r] if i % 2 == 1 else _SHORT_REP_LABELS[r])
            for i, r in enumerate(_REP_ORDER)
        ], fontsize=7)
        ax.set_ylabel("Bytes / vector")
        ax.set_title(_phase5_entry_label(entry))
        ax.set_yscale("log", base=2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.045),
    )
    # Single note at the very bottom center
    fig.text(0.5, 0.01, "TQ: TurboQuant", ha="center", fontsize=6.5, color="dimgray", alpha=0.8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_phase5_memory_compression_by_dim(
    results: list[dict],
    output_path: Path,
) -> Path:
    """Line plot of NumPy layout compression ratio by PCA dimension."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    series_labels = sorted({_phase5_series_label(r) for r in results})
    linestyles = ["-", "--", ":", "-."]
    for ds_idx, series_label in enumerate(series_labels):
        ds_entries = sorted(
            [r for r in results if _phase5_series_label(r) == series_label],
            key=lambda r: r["dim"],
        )
        for rep in _REP_ORDER:
            dims = [r["dim"] for r in ds_entries]
            ratios = []
            for entry in ds_entries:
                numpy_mem = {r["representation"]: r for r in entry["numpy_vectorized"]}
                ratios.append(numpy_mem[rep]["compression_ratio_vs_float768"])
            ax.plot(
                dims,
                ratios,
                color=_REP_COLORS[rep],
                linestyle=linestyles[ds_idx % len(linestyles)],
                marker="o",
                markersize=4,
                label=_REP_LABELS[rep] if ds_idx == 0 else None,
            )
        ax.plot(
            [], [],
            color="gray",
            linestyle=linestyles[ds_idx % len(linestyles)],
            label=series_label,
        )

    ax.set_xlabel("PCA dimensions")
    ax.set_ylabel("Compression factor", labelpad=8)
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(_format_factor)
    ax.set_xticks(sorted({r["dim"] for r in results}))
    bottom_margin = _phase5_legend_below(ax, ncol=4)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=bottom_margin, top=0.95)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_phase5_runtime_by_dim(
    results: list[dict],
    output_path: Path,
    operation: str = "pairwise_distance",
    metric: str = "median_ms",
) -> Path:
    """Line plot of NumPy runtime metric by dimension.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    series_labels = sorted({_phase5_series_label(r) for r in results})
    linestyles = ["-", "--", ":", "-."]
    plotted = False
    for ds_idx, series_label in enumerate(series_labels):
        ds_entries = sorted(
            [r for r in results if _phase5_series_label(r) == series_label],
            key=lambda r: r["dim"],
        )
        for rep in _REP_ORDER:
            dims = []
            values = []
            for entry in ds_entries:
                matches = [
                    r for r in entry["results"]
                    if r.get("representation") == rep
                    and r.get("operation") == operation
                    and r.get("implementation") == "numpy_vectorized"
                    and r.get("status") == "ok"
                    and r.get(metric) is not None
                ]
                if matches:
                    dims.append(entry["dim"])
                    values.append(matches[0][metric])
            if values:
                plotted = True
                ax.plot(
                    dims,
                    values,
                    color=_REP_COLORS[rep],
                    linestyle=linestyles[ds_idx % len(linestyles)],
                    marker="o",
                    markersize=4,
                    label=_REP_LABELS[rep] if ds_idx == 0 else None,
                )
        if ds_entries:
            ax.plot(
                [], [],
                color="gray",
                linestyle=linestyles[ds_idx % len(linestyles)],
                label=series_label,
            )

    ax.set_xlabel("PCA dimensions")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"NumPy {operation.replace('_', ' ')}")
    if plotted:
        bottom_margin = _phase5_legend_below(ax, ncol=4)
        fig.subplots_adjust(left=0.15, right=0.98, bottom=bottom_margin, top=0.88)
    else:
        ax.text(
            0.5,
            0.5,
            "No NumPy runtime measurements available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path

# ---------- Phase 4: Retrieval Evaluation ----------

_RETRIEVAL_REP_ORDER = [
    "float32",
    "16bit",
    "naive_8bit",
    "tq_8bit",
    "naive_4bit",
    "tq_4bit",
    "naive_2bit",
    "tq_2bit",
    "naive_1bit",
    "tq_1bit",
]
_RETRIEVAL_REP_LABELS = {
    "float32": "Float32",
    "16bit": "Naive 16-bit",
    "naive_8bit": "Naive 8-bit",
    "tq_8bit": "TurboQuant 8-bit",
    "naive_4bit": "Naive 4-bit",
    "tq_4bit": "TurboQuant 4-bit",
    "naive_2bit": "Naive 2-bit",
    "tq_2bit": "TurboQuant 2-bit",
    "naive_1bit": "Naive 1-bit",
    "tq_1bit": "TurboQuant 1-bit",
}
_RETRIEVAL_REP_COLORS = {
    "float32": "#1f77b4",
    "16bit": "#aec7e8",
    "naive_8bit": "#c5b0d5",
    "tq_8bit": "#9467bd",
    "naive_4bit": "#98df8a",
    "tq_4bit": "#2ca02c",
    "naive_2bit": "#ffbb78",
    "tq_2bit": "#ff7f0e",
    "naive_1bit": "#ff9896",
    "tq_1bit": "#d62728",
}


def _retrieval_legend_below(ax: plt.Axes, ncol: int = 4) -> None:
    """Place a compact legend below a Phase 4 retrieval plot."""
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=ncol,
        fontsize=7,
        frameon=False,
        borderaxespad=0.0,
    )


def _retrieval_bits_per_vector(representation: str, dim: int) -> int:
    if representation == "float32":
        return dim * 32
    if representation == "16bit":
        return dim * 16
    if representation in ("naive_8bit", "tq_8bit"):
        return dim * 8
    if representation in ("naive_4bit", "tq_4bit"):
        return dim * 4
    if representation in ("naive_2bit", "tq_2bit"):
        return dim * 2
    if representation in ("naive_1bit", "tq_1bit"):
        return dim
    raise ValueError(f"Unknown representation: {representation}")


def plot_retrieval_heatmap(
    results: dict,
    output_path: Path,
    metric: str = "ndcg_at_10",
    title: str | None = None,
) -> Path:
    """Heatmap of a retrieval metric across representation and PCA dimension."""
    _apply_style()

    dims = sorted({r["dim"] for r in results["results"]})
    matrix = np.zeros((len(_RETRIEVAL_REP_ORDER), len(dims)))
    for r in results["results"]:
        row = _RETRIEVAL_REP_ORDER.index(r["representation"])
        col = dims.index(r["dim"])
        matrix[row, col] = r[metric]

    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, 2.8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_yticks(range(len(_RETRIEVAL_REP_ORDER)))
    ax.set_yticklabels([_RETRIEVAL_REP_LABELS[r] for r in _RETRIEVAL_REP_ORDER])
    ax.set_xlabel("PCA dimensions")

    for i in range(len(_RETRIEVAL_REP_ORDER)):
        for j in range(len(dims)):
            val = matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color)

    label = metric.replace("_", " ").upper()
    cax = fig.add_axes((0.22, 0.12, 0.62, 0.045))
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(label, labelpad=4)
    ax.set_title(title or f"{results['dataset']} - {label}")
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.36, top=0.86)


    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_retrieval_by_dim(
    results: list[dict],
    output_path: Path,
    metric: str = "ndcg_at_10",
    ylabel: str | None = None,
) -> Path:
    """Line plot of a retrieval metric as function of PCA dimension."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    linestyles = ["-", "--", ":"]
    for ds_idx, ds in enumerate(results):
        for representation in _RETRIEVAL_REP_ORDER:
            entries = sorted(
                [r for r in ds["results"] if r["representation"] == representation],
                key=lambda r: r["dim"],
            )
            dims = [r["dim"] for r in entries]
            values = [r[metric] for r in entries]
            ax.plot(
                dims,
                values,
                color=_RETRIEVAL_REP_COLORS[representation],
                linestyle=linestyles[ds_idx % len(linestyles)],
                marker="o",
                markersize=4,
                label=_RETRIEVAL_REP_LABELS[representation] if ds_idx == 0 else None,
            )
        ax.plot(
            [], [],
            color="gray",
            linestyle=linestyles[ds_idx % len(linestyles)],
            label=ds["dataset"],
        )

    label = ylabel or metric.replace("_", " ").upper()
    ax.set_xlabel("PCA dimensions")
    ax.set_ylabel(label)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(sorted({r["dim"] for ds in results for r in ds["results"]}))
    _retrieval_legend_below(ax, ncol=4)
    ax.set_title(f"Retrieval {label} by Dimension")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.24, top=0.88)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_retrieval_pareto(
    results: list[dict],
    output_path: Path,
    metric: str = "ndcg_at_10",
    ylabel: str | None = None,
) -> Path:
    """Pareto plot of retrieval quality against bits per vector."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    linestyles = ["-", "--", ":"]
    markers = ["o", "s", "^"]
    for ds_idx, ds in enumerate(results):
        for representation in _RETRIEVAL_REP_ORDER:
            entries = sorted(
                [r for r in ds["results"] if r["representation"] == representation],
                key=lambda r: r["dim"],
            )
            bits = [_retrieval_bits_per_vector(representation, r["dim"]) for r in entries]
            values = [r[metric] for r in entries]
            ax.plot(
                bits,
                values,
                color=_RETRIEVAL_REP_COLORS[representation],
                linestyle=linestyles[ds_idx % len(linestyles)],
                marker=markers[ds_idx % len(markers)],
                markersize=4,
                label=_RETRIEVAL_REP_LABELS[representation] if ds_idx == 0 else None,
            )
        ax.plot(
            [], [],
            color="gray",
            linestyle=linestyles[ds_idx % len(linestyles)],
            label=ds["dataset"],
        )

    label = ylabel or metric.replace("_", " ").upper()
    ax.set_xlabel("Bits per vector")
    ax.set_ylabel(label)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.0, 1.05)
    _retrieval_legend_below(ax, ncol=4)
    ax.set_title(f"Retrieval {label} vs. Compression")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.24, top=0.88)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
