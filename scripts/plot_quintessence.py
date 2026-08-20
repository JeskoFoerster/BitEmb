#!/usr/bin/env python3
"""Generate the central Quintessence / Key Takeaway figure for Phase 5.

Combines:
1. Quality vs. Compression Pareto Curve with shaded 'Sweet Spot Zone' (>= 95% Quality).
2. Heatmap Matrix of Relative Quality across PCA Dimensions and Bit Depths.
3. Callout note in Subplot B on dimension compensation.

Saves PDF and high-res PNG into results/phase5/figures/ and copies PNG to artifact directory.
"""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


def main():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    metrics_path = base_dir / "results/phase4/retrieval_metrics.json"
    memory_path = base_dir / "results/phase5/memory_metrics.json"
    output_dir = base_dir / "results/phase5/figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    with open(memory_path, "r", encoding="utf-8") as f:
        memory_data = json.load(f)

    scifact_results = next(e["results"] for e in metrics_data if e["dataset"] == "scifact")
    quality_map = {(r["representation"], r["dim"]): r["ndcg_at_10"] for r in scifact_results}

    memory_map = {}
    for entry in memory_data:
        if entry["dataset"] == "scifact":
            dim = entry["dim"]
            for nv in entry["numpy_vectorized"]:
                rep = nv["representation"]
                memory_map[(rep, dim)] = nv["total_bytes"] / entry["n_vectors"]

    q_base = quality_map[("float32", 1024)]
    c_base = memory_map[("float32", 1024)]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.8, 5.4), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    # ---------------- Subplot 1: Quality vs Compression Pareto ----------------
    tq_reps = [
        ("tq_8bit", "8-Bit (TurboQuant)", "#9467bd", "P", "--"),
        ("tq_4bit", "4-Bit (TurboQuant)", "#2ca02c", "s", "-"),
        ("tq_2bit", "2-Bit (TurboQuant)", "#ff7f0e", "^", "-"),
        ("tq_1bit", "1-Bit (TurboQuant)", "#d62728", "D", ":"),
    ]

    # Plot Float32 baseline reference line
    ax1.axhline(
        100.0,
        color="#1f77b4",
        linestyle="-.",
        linewidth=1.2,
        label="Float32 1024d Baseline (100%)",
        alpha=0.8,
    )

    # Shaded Sweet Spot Region (strictly >= 95.0% quality, 8x to 31x compression)
    sweet_spot_box = mpatches.FancyBboxPatch(
        (7.0, 95.0),
        25.0,
        5.0,
        boxstyle="round,pad=0.3",
        facecolor="#e6f542",
        alpha=0.3,
        edgecolor="#b5c400",
        linewidth=1.5,
        zorder=1,
    )
    ax1.add_patch(sweet_spot_box)

    # Callout 1: Sweet Spot Zone (>= 95% Quality)
    ax1.annotate(
        "SWEET SPOT ZONE (2-bit bis 8-bit)\n≥ 95% Qualität | 8x bis 31x Kompression",
        xy=(15.0, 95.0),
        xytext=(15.0, 76.0),
        ha="center",
        va="top",
        fontsize=7.8,
        fontweight="bold",
        color="#5c6300",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.95, edgecolor="#b5c400"),
        arrowprops=dict(
            arrowstyle="->",
            connectionstyle="arc3,rad=-0.15",
            color="#5c6300",
            lw=1.2,
        ),
        zorder=6,
    )

    # Plot curves
    for rep, label, color, marker, ls in tq_reps:
        dims = [1024, 768, 512, 384, 256, 128, 64]
        sub_dims = [d for d in dims if (rep, d) in quality_map]
        comps = [c_base / memory_map[(rep, d)] for d in sub_dims]
        q_rels = [(quality_map[(rep, d)] / q_base) * 100.0 for d in sub_dims]

        ax1.plot(
            comps,
            q_rels,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=1.8,
            markersize=6,
            label=label,
            zorder=4,
        )

        # Annotate specific key points (256d & 1024d)
        for d, comp, q_rel in zip(sub_dims, comps, q_rels):
            if rep in ["tq_4bit", "tq_2bit"] and d in [1024, 256]:
                ax1.annotate(
                    f"{d}d ({q_rel:.1f}%)",
                    (comp, q_rel),
                    textcoords="offset points",
                    xytext=(0, 7 if rep == "tq_4bit" else -12),
                    ha="center",
                    fontsize=7.5,
                    fontweight="bold",
                    color=color,
                    zorder=5,
                )

    ax1.set_xscale("log")
    ax1.set_xlim(3.5, 140)
    ax1.set_ylim(60, 102)
    ax1.set_xlabel("Kompressionsfaktor (vs. Float32 1024d, Log-Skala)", fontweight="bold")
    ax1.set_ylabel("Relative Retrieval-Qualität (% NDCG@10)", fontweight="bold")
    ax1.set_title("A) Qualität vs. Kompression & Dimension", fontweight="bold", fontsize=11, pad=10)
    ax1.set_xticks([4, 8, 15, 31, 60, 120])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.legend(loc="lower left", framealpha=0.9, fontsize=8)

    # ---------------- Subplot 2: Heatmap Matrix ----------------
    dims_matrix = [1024, 768, 512, 384, 256, 128]
    reps_matrix = ["float32", "tq_8bit", "tq_4bit", "tq_2bit", "tq_1bit"]
    rep_labels_matrix = [
        "Float32 (32-bit)",
        "TurboQuant 8-bit",
        "TurboQuant 4-bit",
        "TurboQuant 2-bit",
        "TurboQuant 1-bit",
    ]

    matrix_data = np.zeros((len(reps_matrix), len(dims_matrix)))
    for i, r in enumerate(reps_matrix):
        for j, d in enumerate(dims_matrix):
            if (r, d) in quality_map:
                matrix_data[i, j] = (quality_map[(r, d)] / q_base) * 100.0

    im = ax2.imshow(matrix_data, cmap="YlGnBu", vmin=70, vmax=100, aspect="auto")

    ax2.set_xticks(np.arange(len(dims_matrix)))
    ax2.set_yticks(np.arange(len(reps_matrix)))
    ax2.set_xticklabels([f"{d}d" for d in dims_matrix], fontweight="bold")
    ax2.set_yticklabels(rep_labels_matrix, fontweight="bold")
    ax2.set_xlabel("PCA Dimensionen", fontweight="bold")
    ax2.set_title("B) Qualitäts-Matrix (% Baseline)", fontweight="bold", fontsize=11, pad=10)

    # Text overlay in cells
    for i in range(len(reps_matrix)):
        for j in range(len(dims_matrix)):
            val = matrix_data[i, j]
            color = "white" if val > 95.0 or val < 75.0 else "black"
            fontw = "bold" if val >= 95.0 else "normal"
            ax2.text(
                j,
                i,
                f"{val:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontweight=fontw,
                fontsize=8.5,
            )

    # Polygon strictly surrounding cells >= 95.0% quality:
    # 8-bit and 4-bit (rows 1 & 2): cols 0 to 4 (1024d to 256d)
    # 2-bit (row 3): cols 0 and 1 (1024d & 768d)
    poly_verts = [
        (-0.45, 0.55),
        (4.45, 0.55),
        (4.45, 2.45),
        (1.45, 2.45),
        (1.45, 3.45),
        (-0.45, 3.45),
    ]
    poly = mpatches.Polygon(
        poly_verts,
        closed=True,
        fill=False,
        edgecolor="#d95f02",
        linewidth=2.5,
        linestyle="--",
    )
    ax2.add_patch(poly)

    # Label box for Subplot B
    ax2.text(
        2.0,
        3.65,
        " OPTIMALER BEREICH (≥ 95% Qualität)\n"
        " 8-bit & 4-bit (256d–1024d) sowie 2-bit (768d–1024d)",
        ha="center",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color="#d95f02",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.95, edgecolor="#d95f02"),
    )

    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Relative Qualität (%)", fontweight="bold", fontsize=8.5)

    # Main Title over entire figure
    fig.suptitle(
        "Phase 5 Quintessenz: Optimaler Kompromiss aus Bit-Tiefe (2-bit bis 8-bit) & "
        "Dimensionen (256d–1024d)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    pdf_path = output_dir / "bitemb_quintessence_overview.pdf"
    png_path = output_dir / "bitemb_quintessence_overview.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")

    # Copy PNG to artifact directory for markdown display
    artifact_dir = Path(
        "/home/jesko/.gemini/antigravity-cli/brain/5343d8d5-7a49-4f93-9b63-9187d8753a34"
    )
    if artifact_dir.exists():
        dst_png = artifact_dir / "bitemb_quintessence_overview.png"
        shutil.copy(png_path, dst_png)
        print(f"Copied PNG to artifact dir: {dst_png}")


if __name__ == "__main__":
    main()
