#!/usr/bin/env python3
"""Calculate trade-off metrics and generate figures for the BitEmb evaluation.

Calculates Relative Quality-to-Space Ratio (RQSR), Compression-Quality Score (CQ_beta),
and Quality Elasticity of Storage (QES) for the SciFact dataset and generates
publication-quality figures (PNG and PDF). All paths are resolved dynamically relative
to this script, making it fully portable and reproducible.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Consistent styling matching bitemb.plotting
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

_REP_MARKERS = {
    "float32": "o",
    "16bit": "v",
    "naive_8bit": "x",
    "tq_8bit": "P",
    "naive_4bit": "x",
    "tq_4bit": "s",
    "naive_2bit": "x",
    "tq_2bit": "^",
    "naive_1bit": "x",
    "tq_1bit": "D",
}

def apply_style():
    plt.rcParams.update(_STYLE)

def main():
    apply_style()
    
    # Paths resolved relative to this script's location
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    
    metrics_path = base_dir / "results/phase4/retrieval_metrics.json"
    memory_path = base_dir / "results/phase5/memory_metrics.json"
    output_dir = base_dir / "results/phase5/figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not metrics_path.exists() or not memory_path.exists():
        print(f"Error: Data files not found.")
        print(f"Expected retrieval metrics at: {metrics_path}")
        print(f"Expected memory metrics at: {memory_path}")
        print(f"Please ensure Phase 4 and Phase 5 experiments have been run.")
        sys.exit(1)
        
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)
        
    with open(memory_path, "r") as f:
        memory_data = json.load(f)
        
    # Get scifact retrieval results
    scifact_results = None
    for entry in metrics_data:
        if entry["dataset"] == "scifact":
            scifact_results = entry["results"]
            break
            
    if not scifact_results:
        print("Error: SciFact retrieval results not found in json.")
        sys.exit(1)
        
    quality_map = {}
    for r in scifact_results:
        rep = r["representation"]
        dim = r["dim"]
        quality_map[(rep, dim)] = r["ndcg_at_10"]
        
    # Get memory usage
    memory_map = {}
    for entry in memory_data:
        if entry["dataset"] == "scifact":
            dim = entry["dim"]
            for nv in entry["numpy_vectorized"]:
                rep = nv["representation"]
                memory_map[(rep, dim)] = nv["total_bytes"] / entry["n_vectors"]
                
    # Baseline (float32, 768d)
    q_baseline = quality_map.get(("float32", 768))
    c_baseline = memory_map.get(("float32", 768))
    
    if not q_baseline or not c_baseline:
        print("Error: Baseline (float32, 768d) not found.")
        sys.exit(1)
        
    # Calculate metrics
    results = []
    for (rep, dim), q_val in sorted(quality_map.items()):
        c_val = memory_map.get((rep, dim))
        if c_val is None:
            continue
            
        q_rel = q_val / q_baseline
        c_rel = c_val / c_baseline
        s_rel = 1.0 - c_rel
        
        # RQSR
        rqsr = q_rel / c_rel if c_rel > 0 else float('inf')
        
        # CQ_1
        cq1 = 2.0 * (q_rel * s_rel) / (q_rel + s_rel) if (q_rel + s_rel) > 0 else 0.0
        
        # QES (Elasticity)
        q_loss = 1.0 - q_rel
        qes = q_loss / s_rel if s_rel > 0 else 0.0
        
        results.append({
            "rep": rep,
            "dim": dim,
            "ndcg": q_val,
            "q_rel": q_rel,
            "bytes_per_vec": c_val,
            "c_rel": c_rel,
            "s_rel": s_rel,
            "rqsr": rqsr,
            "cq1": cq1,
            "qes": qes
        })

    # Output text results
    print(f"Calculated trade-off metrics for SciFact dataset against baseline (float32, 768d, NDCG@10={q_baseline:.4f}).")
    
    # ------------------ Plot 1: CQ1-Score by PCA Dimension ------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for rep in _REP_ORDER:
        rep_results = sorted([r for r in results if r["rep"] == rep], key=lambda x: x["dim"])
        if not rep_results:
            continue
        dims = [r["dim"] for r in rep_results]
        cq1_vals = [r["cq1"] for r in rep_results]
        ax.plot(
            dims, cq1_vals, 
            color=_REP_COLORS[rep], 
            marker=_REP_MARKERS[rep], 
            linestyle="-" if "naive" not in rep else ":",
            label=_REP_LABELS[rep]
        )
    ax.set_xlabel("PCA Dimension")
    ax.set_ylabel("$CQ_1$-Score (Balanced Trade-off)")
    ax.set_title("Compression-Quality Score ($CQ_1$) by Dimension")
    ax.set_xticks([64, 128, 256, 384, 768])
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    
    # Save Plot 1
    p1_png = output_dir / "tradeoff_cq1_by_dim.png"
    p1_pdf = output_dir / "tradeoff_cq1_by_dim.pdf"
    fig.savefig(p1_png, bbox_inches="tight")
    fig.savefig(p1_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p1_png}")
    
    # ------------------ Plot 2: Elasticity (QES) by PCA Dimension ------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for rep in _REP_ORDER:
        rep_results = sorted([r for r in results if r["rep"] == rep], key=lambda x: x["dim"])
        if not rep_results:
            continue
        dims = [r["dim"] for r in rep_results]
        qes_vals = [r["qes"] for r in rep_results]
        ax.plot(
            dims, qes_vals, 
            color=_REP_COLORS[rep], 
            marker=_REP_MARKERS[rep], 
            linestyle="-" if "naive" not in rep else ":",
            label=_REP_LABELS[rep]
        )
    
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("PCA Dimension")
    ax.set_ylabel(r"Quality Elasticity of Storage ($\epsilon_{Q,C}$)")
    ax.set_title("Storage Quality Elasticity (QES) by Dimension")
    ax.set_xticks([64, 128, 256, 384, 768])
    ax.set_ylim(-0.05, 0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    
    # Save Plot 2
    p2_png = output_dir / "tradeoff_elasticity_by_dim.png"
    p2_pdf = output_dir / "tradeoff_elasticity_by_dim.pdf"
    fig.savefig(p2_png, bbox_inches="tight")
    fig.savefig(p2_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p2_png}")
    
    # ------------------ Plot 3: Quality vs. Savings (Pareto Space) ------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for rep in _REP_ORDER:
        rep_results = sorted([r for r in results if r["rep"] == rep], key=lambda x: x["dim"])
        if not rep_results:
            continue
        # Y: Relative Quality, X: Relative Savings
        savings = [r["s_rel"] for r in rep_results]
        quality = [r["q_rel"] for r in rep_results]
        ax.plot(
            savings, quality, 
            color=_REP_COLORS[rep], 
            marker=_REP_MARKERS[rep], 
            linestyle="-" if "naive" not in rep else ":",
            label=_REP_LABELS[rep]
        )
        
        # Annotate selected points with dimension
        for r in rep_results:
            if r["dim"] in [64, 256, 768]:
                ax.annotate(
                    f"{r['dim']}d",
                    (r["s_rel"], r["q_rel"]),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=7,
                    alpha=0.8
                )
                
    ax.text(0.02, 0.02, "Points on curve (L to R): 768d -> 384d -> 256d -> 128d -> 64d", transform=ax.transAxes, fontsize=6.5, color="dimgray", alpha=0.8)
    ax.set_xlabel("Relative Storage Savings ($S_{rel}$)")
    ax.set_ylabel("Relative Quality ($Q_{rel}$)")
    ax.set_title("Normalized Trade-off Space (Quality vs. Savings)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    
    # Save Plot 3
    p3_png = output_dir / "tradeoff_quality_vs_savings.png"
    p3_pdf = output_dir / "tradeoff_quality_vs_savings.pdf"
    fig.savefig(p3_png, bbox_inches="tight")
    fig.savefig(p3_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p3_png}")
    
    # ------------------ Plot 4a: Zoomed Quality vs. Savings (TurboQuant Only) ------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for rep in _REP_ORDER:
        if not rep.startswith("tq_"):
            continue
        rep_results = sorted([r for r in results if r["rep"] == rep], key=lambda x: x["dim"])
        if not rep_results:
            continue
        savings = [r["s_rel"] for r in rep_results]
        quality = [r["q_rel"] for r in rep_results]
        ax.plot(
            savings, quality, 
            color=_REP_COLORS[rep], 
            marker=_REP_MARKERS[rep], 
            linestyle="-",
            label=_REP_LABELS[rep]
        )
        
        for r in rep_results:
            if r["dim"] in [64, 256, 768]:
                ax.annotate(
                    f"{r['dim']}d",
                    (r["s_rel"], r["q_rel"]),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=7,
                    alpha=0.8
                )
                
    ax.text(0.02, 0.02, "Points on curve (L to R): 768d -> 384d -> 256d -> 128d -> 64d", transform=ax.transAxes, fontsize=6.5, color="dimgray", alpha=0.8)
    ax.set_xlabel("Relative Storage Savings ($S_{rel}$)")
    ax.set_ylabel("Relative Quality ($Q_{rel}$)")
    ax.set_title("Zoomed Trade-off Space (TurboQuant Only)")
    ax.set_xlim(0.70, 1.01)
    ax.set_ylim(0.43, 1.03)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    
    # Save Plot 4a
    p4a_png = output_dir / "tradeoff_quality_vs_savings_zoomed_tq.png"
    p4a_pdf = output_dir / "tradeoff_quality_vs_savings_zoomed_tq.pdf"
    fig.savefig(p4a_png, bbox_inches="tight")
    fig.savefig(p4a_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p4a_png}")

    # ------------------ Plot 4b: Zoomed Quality vs. Savings (Naive Only) ------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for rep in _REP_ORDER:
        if not rep.startswith("naive_"):
            continue
        rep_results = sorted([r for r in results if r["rep"] == rep], key=lambda x: x["dim"])
        if not rep_results:
            continue
        savings = [r["s_rel"] for r in rep_results]
        quality = [r["q_rel"] for r in rep_results]
        ax.plot(
            savings, quality, 
            color=_REP_COLORS[rep], 
            marker=_REP_MARKERS[rep], 
            linestyle=":",
            label=_REP_LABELS[rep]
        )
        
        for r in rep_results:
            if r["dim"] in [64, 256, 768]:
                ax.annotate(
                    f"{r['dim']}d",
                    (r["s_rel"], r["q_rel"]),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=7,
                    alpha=0.8
                )
                
    ax.text(0.02, 0.02, "Points on curve (L to R): 768d -> 384d -> 256d -> 128d -> 64d", transform=ax.transAxes, fontsize=6.5, color="dimgray", alpha=0.8)
    ax.set_xlabel("Relative Storage Savings ($S_{rel}$)")
    ax.set_ylabel("Relative Quality ($Q_{rel}$)")
    ax.set_title("Zoomed Trade-off Space (Naive Only)")
    ax.set_xlim(0.70, 1.01)
    ax.set_ylim(0.43, 1.03)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    
    # Save Plot 4b
    p4b_png = output_dir / "tradeoff_quality_vs_savings_zoomed_naive.png"
    p4b_pdf = output_dir / "tradeoff_quality_vs_savings_zoomed_naive.pdf"
    fig.savefig(p4b_png, bbox_inches="tight")
    fig.savefig(p4b_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p4b_png}")

    # ------------------ Plot 5a: Quality vs. Compression Ratio (TurboQuant Only) ------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for rep in _REP_ORDER:
        if not rep.startswith("tq_"):
            continue
        rep_results = sorted([r for r in results if r["rep"] == rep], key=lambda x: x["dim"])
        if not rep_results:
            continue
        comp_ratios = [1.0 / r["c_rel"] for r in rep_results]
        quality = [r["q_rel"] for r in rep_results]
        ax.plot(
            comp_ratios, quality, 
            color=_REP_COLORS[rep], 
            marker=_REP_MARKERS[rep], 
            linestyle="-",
            label=_REP_LABELS[rep]
        )
        
        for r in rep_results:
            if r["dim"] in [64, 256, 768]:
                ax.annotate(
                    f"{r['dim']}d",
                    (1.0 / r["c_rel"], r["q_rel"]),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=7,
                    alpha=0.8
                )
                
    ax.text(0.02, 0.02, "Points on curve (L to R): 768d -> 384d -> 256d -> 128d -> 64d", transform=ax.transAxes, fontsize=6.5, color="dimgray", alpha=0.8)
    ax.set_xlabel("Compression Factor (Multiplier vs. Float32 768d, Log-Scale)")
    ax.set_ylabel("Relative Quality ($Q_{rel}$)")
    ax.set_title("Quality vs. Compression Factor (TurboQuant Only)")
    ax.set_xscale("log")
    ax.set_xlim(3, 450)
    ax.set_ylim(0.43, 1.03)
    ax.set_xticks([4, 8, 12, 15, 23, 30, 45, 64, 96, 180, 384])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xticks(rotation=45)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    
    # Save Plot 5a
    p5a_png = output_dir / "tradeoff_quality_vs_compression_ratio_tq.png"
    p5a_pdf = output_dir / "tradeoff_quality_vs_compression_ratio_tq.pdf"
    fig.savefig(p5a_png, bbox_inches="tight")
    fig.savefig(p5a_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p5a_png}")

    # ------------------ Plot 5b: Quality vs. Compression Ratio (Naive Only) ------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for rep in _REP_ORDER:
        if not rep.startswith("naive_"):
            continue
        rep_results = sorted([r for r in results if r["rep"] == rep], key=lambda x: x["dim"])
        if not rep_results:
            continue
        comp_ratios = [1.0 / r["c_rel"] for r in rep_results]
        quality = [r["q_rel"] for r in rep_results]
        ax.plot(
            comp_ratios, quality, 
            color=_REP_COLORS[rep], 
            marker=_REP_MARKERS[rep], 
            linestyle=":",
            label=_REP_LABELS[rep]
        )
        
        for r in rep_results:
            if r["dim"] in [64, 256, 768]:
                ax.annotate(
                    f"{r['dim']}d",
                    (1.0 / r["c_rel"], r["q_rel"]),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=7,
                    alpha=0.8
                )
                
    ax.text(0.02, 0.02, "Points on curve (L to R): 768d -> 384d -> 256d -> 128d -> 64d", transform=ax.transAxes, fontsize=6.5, color="dimgray", alpha=0.8)
    ax.set_xlabel("Compression Factor (Multiplier vs. Float32 768d, Log-Scale)")
    ax.set_ylabel("Relative Quality ($Q_{rel}$)")
    ax.set_title("Quality vs. Compression Factor (Naive Only)")
    ax.set_xscale("log")
    ax.set_xlim(3, 450)
    ax.set_ylim(0.43, 1.03)
    ax.set_xticks([4, 8, 12, 15, 23, 30, 45, 64, 96, 180, 384])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xticks(rotation=45)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    
    # Save Plot 5b
    p5b_png = output_dir / "tradeoff_quality_vs_compression_ratio_naive.png"
    p5b_pdf = output_dir / "tradeoff_quality_vs_compression_ratio_naive.pdf"
    fig.savefig(p5b_png, bbox_inches="tight")
    fig.savefig(p5b_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p5b_png}")
    
    # Remove old combined files
    for filename in ["tradeoff_quality_vs_savings_zoomed.png", "tradeoff_quality_vs_savings_zoomed.pdf", "tradeoff_quality_vs_compression_ratio.png", "tradeoff_quality_vs_compression_ratio.pdf"]:
        filepath = output_dir / filename
        if filepath.exists():
            os.remove(filepath)
            print(f"Deleted old combined file: {filename}")
            
    print("All figures successfully saved in the repository under results/phase5/figures/")

if __name__ == "__main__":
    main()
