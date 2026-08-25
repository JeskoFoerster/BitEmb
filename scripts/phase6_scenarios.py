#!/usr/bin/env python3
"""Phase 6: Practical Application Scenarios & AWS Hosting Cost Analysis.

Evaluates three real-world deployment scenarios using empirical BitEmb results:
1. Edge / Mobile (Minimal Memory Footprint)
2. Business Sweet Spot (Balanced Cost-Performance for SaaS)
3. Enterprise Precision (High Precision / Low Loss)

Generates publication-quality figures (PDF) and JSON summary with AWS cost calculations.
"""

import json
import sys
from pathlib import Path

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
    "float32": "Float32 Baseline",
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

# AWS EC2 On-Demand Instance Pricing (Region: eu-central-1, Frankfurt).
# Prices are Linux On-Demand list prices from the AWS Price List Bulk API
# (AmazonEC2/current/eu-central-1/index.csv), converted with 730 hours/month.
# Price list publication date: 2026-08-24T19:31:47Z; effective date: 2026-08-01.
# These are rounded order-of-magnitude estimates, not a procurement quote; list
# prices change over time and exclude storage, network and reserved/spot discounts.
# Footprints are reported as binary units (MiB/GiB), matching the calculations below.
# Baseline (38.15 GiB RAM for 10M vecs): r6i.2xlarge (64 GiB RAM) -> $0.6080/hr
# Enterprise (4.92 GiB RAM): m6g.large (8 GiB RAM) -> $0.0920/hr
# Business Sweet Spot (2.54 GiB RAM): t4g.medium (4 GiB RAM) -> $0.0384/hr
# Edge / Mobile (0.89 GiB RAM): t4g.small (2 GiB RAM) -> $0.0192/hr
AWS_PRICING_REGION = "eu-central-1"
AWS_PRICING_RETRIEVED = "2026-08-25"
AWS_PRICING_PUBLICATION_DATE = "2026-08-24T19:31:47Z"
AWS_PRICING_EFFECTIVE_DATE = "2026-08-01"
AWS_PRICING_HOURS_PER_MONTH = 730
AWS_PRICING = {
    "baseline": {"instance": "r6i.2xlarge (64 GiB RAM)", "hourly_usd": 0.6080},
    "enterprise": {"instance": "m6g.large (8 GiB RAM)", "hourly_usd": 0.0920},
    "business": {"instance": "t4g.medium (4 GiB RAM)", "hourly_usd": 0.0384},
    "edge": {"instance": "t4g.small (2 GiB RAM)", "hourly_usd": 0.0192},
}

for _pricing in AWS_PRICING.values():
    _pricing["monthly_usd"] = _pricing["hourly_usd"] * AWS_PRICING_HOURS_PER_MONTH

MOBILE_VECTOR_COUNTS = [
    10_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    5_000_000,
]
MOBILE_DIMS = [64, 128, 256, 384, 512, 768, 1024]
MOBILE_TQ_REPS = ["tq_1bit", "tq_2bit", "tq_4bit", "tq_8bit"]


def apply_style() -> None:
    plt.rcParams.update(_STYLE)


def main() -> None:
    apply_style()

    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent

    metrics_path = base_dir / "results/phase4/retrieval_metrics.json"
    memory_path = base_dir / "results/phase5/memory_metrics.json"
    output_dir = base_dir / "results/phase6/figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists() or not memory_path.exists():
        print("Error: Data files not found.")
        sys.exit(1)

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

    # Defined Scenarios
    scenarios = {
        "Baseline": {
            "rep": "float32",
            "dim": 1024,
            "title": "Float32 Uncompressed Baseline",
            "aws": AWS_PRICING["baseline"],
        },
        "Enterprise Precision": {
            "rep": "tq_4bit",
            "dim": 1024,
            "title": "Enterprise Precision (High Accuracy)",
            "aws": AWS_PRICING["enterprise"],
        },
        "Business Sweet Spot": {
            "rep": "tq_2bit",
            "dim": 1024,
            "title": "Business Sweet Spot (Optimal ROI)",
            "aws": AWS_PRICING["business"],
        },
        "Edge / Mobile": {
            "rep": "tq_1bit",
            "dim": 768,
            "title": "Edge / Mobile (Ultra Low Memory)",
            "aws": AWS_PRICING["edge"],
        },
    }

    results = {}
    for s_name, s_info in scenarios.items():
        rep = s_info["rep"]
        dim = s_info["dim"]
        q_val = quality_map[(rep, dim)]
        b_val = memory_map[(rep, dim)]
        q_rel = (q_val / q_base) * 100.0
        comp_ratio = c_base / b_val

        # Scaling RAM for 100k, 1M, 10M in binary units.
        ram_100k_mib = (b_val * 100_000) / (1024 * 1024)
        ram_1m_gib = (b_val * 1_000_000) / (1024 * 1024 * 1024)
        ram_10m_gib = (b_val * 10_000_000) / (1024 * 1024 * 1024)

        aws_cost = s_info["aws"]["monthly_usd"]
        base_aws_cost = AWS_PRICING["baseline"]["monthly_usd"]
        aws_savings_pct = (
            (1.0 - (aws_cost / base_aws_cost)) * 100.0 if s_name != "Baseline" else 0.0
        )
        aws_savings_mo = base_aws_cost - aws_cost if s_name != "Baseline" else 0.0

        results[s_name] = {
            "representation": rep,
            "dim": dim,
            "ndcg_at_10": q_val,
            "relative_quality_pct": q_rel,
            "quality_loss_pct": 100.0 - q_rel,
            "bytes_per_vector": b_val,
            "compression_ratio": comp_ratio,
            "ram_100k_mib": ram_100k_mib,
            "ram_1m_gib": ram_1m_gib,
            "ram_10m_gib": ram_10m_gib,
            "aws_instance": s_info["aws"]["instance"],
            "aws_hourly_usd": s_info["aws"]["hourly_usd"],
            "aws_monthly_usd": aws_cost,
            "aws_annual_usd": aws_cost * 12,
            "aws_savings_monthly_usd": aws_savings_mo,
            "aws_savings_pct": aws_savings_pct,
            "aws_pricing_region": AWS_PRICING_REGION,
            "aws_pricing_retrieved": AWS_PRICING_RETRIEVED,
            "aws_pricing_publication_date": AWS_PRICING_PUBLICATION_DATE,
            "aws_pricing_effective_date": AWS_PRICING_EFFECTIVE_DATE,
        }

    edge_bytes = results["Edge / Mobile"]["bytes_per_vector"]
    baseline_bytes = results["Baseline"]["bytes_per_vector"]
    results["Edge / Mobile"]["sub_10m_scaling"] = [
        {
            "n_vectors": n_vectors,
            "mobile_ram_mib": (edge_bytes * n_vectors) / (1024 * 1024),
            "mobile_ram_gib": (edge_bytes * n_vectors) / (1024 * 1024 * 1024),
            "float32_ram_mib": (baseline_bytes * n_vectors) / (1024 * 1024),
            "float32_ram_gib": (baseline_bytes * n_vectors) / (1024 * 1024 * 1024),
            "compression_ratio": baseline_bytes / edge_bytes,
        }
        for n_vectors in MOBILE_VECTOR_COUNTS
    ]

    results["Edge / Mobile"]["sub_10m_by_dim"] = []
    for dim in MOBILE_DIMS:
        rep = "tq_1bit"
        q_val = quality_map[(rep, dim)]
        b_val = memory_map[(rep, dim)]
        dim_entry = {
            "dim": dim,
            "representation": rep,
            "ndcg_at_10": q_val,
            "relative_quality_pct": (q_val / q_base) * 100.0,
            "quality_loss_pct": 100.0 - ((q_val / q_base) * 100.0),
            "bytes_per_vector": b_val,
            "compression_ratio": c_base / b_val,
            "scaling": [
                {
                    "n_vectors": n_vectors,
                    "mobile_ram_mib": (b_val * n_vectors) / (1024 * 1024),
                    "mobile_ram_gib": (b_val * n_vectors) / (1024 * 1024 * 1024),
                }
                for n_vectors in MOBILE_VECTOR_COUNTS
            ],
        }
        results["Edge / Mobile"]["sub_10m_by_dim"].append(dim_entry)

    results["Edge / Mobile"]["sub_10m_by_turboquant"] = []
    for rep in MOBILE_TQ_REPS:
        rep_entry = {
            "representation": rep,
            "dimensions": [],
        }
        for dim in MOBILE_DIMS:
            q_val = quality_map[(rep, dim)]
            b_val = memory_map[(rep, dim)]
            rep_entry["dimensions"].append(
                {
                    "dim": dim,
                    "ndcg_at_10": q_val,
                    "relative_quality_pct": (q_val / q_base) * 100.0,
                    "quality_loss_pct": 100.0 - ((q_val / q_base) * 100.0),
                    "bytes_per_vector": b_val,
                    "compression_ratio": c_base / b_val,
                    "scaling": [
                        {
                            "n_vectors": n_vectors,
                            "ram_mib": (b_val * n_vectors) / (1024 * 1024),
                            "ram_gib": (b_val * n_vectors) / (1024 * 1024 * 1024),
                        }
                        for n_vectors in MOBILE_VECTOR_COUNTS
                    ],
                }
            )
        results["Edge / Mobile"]["sub_10m_by_turboquant"].append(rep_entry)

    # Save JSON summary
    json_path = base_dir / "results/phase6/scenarios_summary.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved scenario analysis JSON to {json_path}")

    # ------------------ Plot 1: RAM Footprint Comparison (10M Vectors) ------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    names = list(scenarios.keys())
    ram_vals = [results[n]["ram_10m_gib"] for n in names]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    bars = ax.bar(names, ram_vals, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("RAM Footprint for 10M Vectors (GiB)")
    ax.set_title("Memory Footprint Comparison across Application Scenarios (10M Vectors)")
    ax.set_ylim(0, 48)

    for bar, val in zip(bars, ram_vals):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.2,
            f"{val:.2f} GiB",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.text(
        0.02,
        0.92,
        "* Enterprise: 7.8x compression (99.5% quality)\n"
        "* Business: 15.0x compression (98.1% quality)\n"
        "* Mobile: 42.7x compression (93.9% quality)",
        transform=ax.transAxes,
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"),
    )
    fig.tight_layout()
    p1_pdf = output_dir / "phase6_scenario_memory_comparison.pdf"
    fig.savefig(p1_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p1_pdf}")

    # ------------------ Plot 2: Highlighted Trade-off Space (3 Zones) ------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    rep_results = sorted(
        [
            r
            for r in scifact_results
            if r["representation"].startswith("tq_") or r["representation"] == "float32"
        ],
        key=lambda x: x["dim"],
    )

    tq_reps = ["tq_8bit", "tq_4bit", "tq_2bit", "tq_1bit"]
    for rep in tq_reps:
        sub = sorted([r for r in rep_results if r["representation"] == rep], key=lambda x: x["dim"])
        if not sub:
            continue
        c_ratios = [c_base / memory_map[(rep, r["dim"])] for r in sub]
        q_rels = [(r["ndcg_at_10"] / q_base) * 100.0 for r in sub]
        ax.plot(
            c_ratios,
            q_rels,
            color=_REP_COLORS[rep],
            marker="o",
            linestyle="-",
            label=_REP_LABELS[rep],
        )

    # Highlight Scenarios
    callouts = [
        ("Enterprise Precision\n(tq_4bit 1024d, 99.5% Q)", 7.8, 99.5, (-30, 15)),
        ("Business Sweet Spot\n(tq_2bit 1024d, 98.1% Q)", 15.0, 98.1, (20, -25)),
        ("Edge / Mobile\n(tq_1bit 768d, 93.9% Q)", 42.7, 93.9, (15, 15)),
    ]

    for label, comp, q_val, offset in callouts:
        ax.scatter([comp], [q_val], color="black", s=50, zorder=5)
        ax.annotate(
            label,
            (comp, q_val),
            textcoords="offset points",
            xytext=offset,
            ha="center",
            fontsize=7.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3, edgecolor="orange"),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="black"),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Compression Factor (Multiplier vs. Float32 1024d, Log-Scale)")
    ax.set_ylabel("Relative Retrieval Quality (% of Float32)")
    ax.set_title("TurboQuant Trade-off Space with Highlighted Application Scenarios")
    ax.set_xlim(3, 100)
    ax.set_ylim(80, 102)
    ax.set_xticks([4, 8, 15, 30, 43, 64, 85])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()

    p2_pdf = output_dir / "phase6_tradeoff_scenarios_highlighted.pdf"
    fig.savefig(p2_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p2_pdf}")

    # ------------------ Plot 3: AWS Monthly Cost Savings (Cloud Scenarios Only) ------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    cloud_names = ["Baseline", "Enterprise Precision", "Business Sweet Spot"]
    cloud_costs = [results[n]["aws_monthly_usd"] for n in cloud_names]
    cloud_colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    bars = ax.bar(
        cloud_names, cloud_costs, color=cloud_colors, width=0.45, edgecolor="black", linewidth=0.8
    )
    ax.set_ylabel("Estimated Monthly AWS EC2 RAM Cost (USD)")
    ax.set_title("Monthly AWS Hosting Cost Comparison (10M Vector Index)")
    ax.set_ylim(0, max(cloud_costs) * 1.24)

    for bar, val, n in zip(bars, cloud_costs, cloud_names):
        height = bar.get_height()
        savings_text = f"{val:.2f} USD/mo"
        if n != "Baseline":
            savings_pct = results[n]["aws_savings_pct"]
            savings_text += f"\n(-{savings_pct:.1f}%)"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 8.0,
            savings_text,
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )

    fig.text(
        0.5,
        0.01,
        "* AWS EC2 Estimates for Cloud Hosting (10M Vector Index): "
        f"Baseline (r6i.2xlarge, {results['Baseline']['aws_monthly_usd']:.2f} USD/mo), "
        f"Enterprise (m6g.large, {results['Enterprise Precision']['aws_monthly_usd']:.2f} USD/mo), "
        f"Business (t4g.medium, {results['Business Sweet Spot']['aws_monthly_usd']:.2f} USD/mo). "
        f"Linux On-Demand list prices, {AWS_PRICING_REGION}, retrieved {AWS_PRICING_RETRIEVED} "
        f"({AWS_PRICING_HOURS_PER_MONTH} h/month).",
        ha="center",
        fontsize=6.0,
        color="dimgray",
        alpha=0.9,
    )
    fig.subplots_adjust(bottom=0.18)

    p3_pdf = output_dir / "phase6_aws_cost_savings.pdf"
    fig.savefig(p3_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p3_pdf}")

    # ------------------ Plot 4: Edge / Mobile Scaling by TurboQuant Variant ------------------
    tq_scaling = results["Edge / Mobile"]["sub_10m_by_turboquant"]
    x_labels = ["10k", "50k", "100k", "250k", "500k", "1M", "5M"]
    x_pos = list(range(len(x_labels)))
    bar_width = 0.11
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(
        len(tq_scaling),
        1,
        figsize=(8.4, 14.0),
        sharex=False,
    )

    for ax, rep_entry in zip(axes, tq_scaling):
        rep = rep_entry["representation"]
        dim_entries = rep_entry["dimensions"]

        for idx, dim_entry in enumerate(dim_entries):
            dim = dim_entry["dim"]
            rel_q = dim_entry["relative_quality_pct"]
            ram_vals = [row["ram_mib"] for row in dim_entry["scaling"]]
            offset = (idx - (len(dim_entries) - 1) / 2.0) * bar_width
            bar_x = [x_val + offset for x_val in x_pos]
            ax.bar(
                bar_x,
                ram_vals,
                width=bar_width,
                color=cmap(idx / max(1, len(dim_entries) - 1)),
                edgecolor="black",
                linewidth=0.35,
                label=f"{dim}d ({rel_q:.1f}% Q)",
            )

        y_max = max(
            row["ram_mib"]
            for dim_entry in dim_entries
            for row in dim_entry["scaling"]
        )
        ax.set_ylim(0, y_max * 1.18)
        if 128 < y_max * 1.18:
            ax.axhline(128, color="dimgray", linestyle="--", linewidth=0.8, alpha=0.75)
        if 512 < y_max * 1.18:
            ax.axhline(512, color="dimgray", linestyle=":", linewidth=0.8, alpha=0.75)
        ax.set_ylabel("RAM Footprint (MiB)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis="x", labelbottom=True, pad=3)
        ax.set_title(
            f"{_REP_LABELS[rep]} - grouped by dimension",
            loc="left",
            fontsize=9.0,
            fontweight="bold",
        )

    fig.supxlabel("Number of Embedding Vectors", y=0.035, fontsize=9.5)
    axes[0].legend(loc="upper left", ncol=2, fontsize=6.7)


    fig.suptitle(
        "Edge / Mobile Memory Scaling below 10M Vectors by TurboQuant Variant",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "* Values include only the raw vector representation, excluding metadata, text chunks, "
        "ANN structures, and runtime overhead.",
        ha="center",
        fontsize=6.2,
        color="dimgray",
        alpha=0.9,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.965), h_pad=2.4)
    fig.subplots_adjust(hspace=0.55)

    p4_pdf = output_dir / "phase6_mobile_sub10m_scaling.pdf"
    fig.savefig(p4_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p4_pdf}")



if __name__ == "__main__":
    main()
