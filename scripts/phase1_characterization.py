"""Phase 1: Characterization of the float embedding space.

Computes and reports:
  1. Norm distribution (is the space a hypersphere?)
  2. Per-dimension statistics (skewness, kurtosis → quantization predictions)
  3. Intrinsic dimensionality (TwoNN + PCA 95% variance)

Usage:
    python scripts/phase1_characterization.py --dataset scifact
    python scripts/phase1_characterization.py --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bitemb.analysis import (
    compute_dimension_stats,
    compute_intrinsic_dimensionality,
    compute_norm_distribution,
)
from bitemb.cache import load_or_encode
from bitemb.config import DATASETS
from bitemb.dataset import load_beir
from bitemb.engine import EmbeddingEngine
from bitemb.plotting import (
    generate_phase1_table,
    plot_cumulative_variance,
    plot_dimension_distribution,
    plot_variance_spectrum,
)

OUTPUT_DIR = Path("results/phase1")


def run(dataset_name: str, engine: EmbeddingEngine, max_docs: int | None = None) -> dict:
    """Run Phase 1 characterization for one dataset."""
    print(f"\n{'='*60}")
    print(f"  Phase 1: {dataset_name}")
    print(f"{'='*60}")

    # Load & encode (cached)
    print("\n  Loading dataset...")
    ds = load_beir(dataset_name)
    texts = ds.corpus_texts
    print(f"  Encoding {len(texts)} documents...")
    corpus_embs = load_or_encode(dataset_name, texts, engine, show_progress=True)

    if max_docs and max_docs < len(texts):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(texts), size=max_docs, replace=False)
        corpus_embs = corpus_embs[idx]
        print(f"  Sampled {max_docs}/{ds.n_corpus} documents for characterization")

    # 1. Norm distribution
    print("\n  [1/3] Norm distribution")
    norms = compute_norm_distribution(corpus_embs)
    print(f"    Mean: {norms.mean:.6f}")
    print(f"    Std:  {norms.std:.6f}")
    print(f"    CV:   {norms.cv:.6f}")
    print(f"    Range: [{norms.min:.6f}, {norms.max:.6f}]")
    print(f"    Near unit sphere: {norms.is_near_unit_sphere()}")

    # 2. Dimension statistics
    print("\n  [2/3] Dimension statistics (768 dims)")
    dimstats = compute_dimension_stats(corpus_embs)
    print(f"    Mean of means:     {dimstats.mean.mean():.6f}")
    print(f"    Mean of stds:      {dimstats.std.mean():.6f}")
    print(f"    Mean |skewness|:   {np.abs(dimstats.skewness).mean():.4f}")
    print(f"    Max |skewness|:    {np.abs(dimstats.skewness).max():.4f}")
    print(f"    Mean kurtosis:     {dimstats.kurtosis.mean():.4f}")
    print(f"    Max kurtosis:      {dimstats.kurtosis.max():.4f}")
    n_skewed = int((np.abs(dimstats.skewness) > 1.0).sum())
    n_heavy_tail = int((dimstats.kurtosis > 3.0).sum())
    print(f"    Dims with |skew| > 1:    {n_skewed}/768")
    print(f"    Dims with kurtosis > 3:  {n_heavy_tail}/768")

    # 3. Intrinsic dimensionality
    print("\n  [3/3] Intrinsic dimensionality")
    intdim = compute_intrinsic_dimensionality(corpus_embs)
    print(f"    TwoNN estimate:   {intdim.twonn:.1f}")
    print(f"    PCA (95% var):    {intdim.pca_95} components")
    print("    Nominal dim:      768")
    print(f"    Redundancy ratio: {1 - intdim.pca_95/768:.1%}")

    # Assemble results
    return {
        "dataset": dataset_name,
        "n_corpus": ds.n_corpus,
        "norm_distribution": {
            "mean": norms.mean,
            "std": norms.std,
            "min": norms.min,
            "max": norms.max,
            "cv": norms.cv,
        },
        "dimension_stats": {
            "mean_of_means": float(dimstats.mean.mean()),
            "mean_of_stds": float(dimstats.std.mean()),
            "mean_abs_skewness": float(np.abs(dimstats.skewness).mean()),
            "max_abs_skewness": float(np.abs(dimstats.skewness).max()),
            "mean_kurtosis": float(dimstats.kurtosis.mean()),
            "max_kurtosis": float(dimstats.kurtosis.max()),
            "n_skewed_dims": n_skewed,
            "n_heavy_tail_dims": n_heavy_tail,
        },
        "intrinsic_dimensionality": {
            "twonn": intdim.twonn,
            "pca_95": intdim.pca_95,
            "pca_cumulative_variance_50": float(intdim.pca_cumulative_variance[49]),
            "pca_cumulative_variance_100": float(intdim.pca_cumulative_variance[99]),
            "pca_cumulative_variance_256": float(intdim.pca_cumulative_variance[255]),
        },
        "_pca_cumulative_variance": intdim.pca_cumulative_variance,
        "_dimstats": {"skewness": dimstats.skewness, "kurtosis": dimstats.kurtosis},
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Float space characterization")
    parser.add_argument("--dataset", choices=list(DATASETS), default="scifact")
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Max corpus docs to encode (subsample for speed)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    engine = EmbeddingEngine()
    datasets = list(DATASETS) if args.all else [args.dataset]
    results = []

    for name in datasets:
        result = run(name, engine, max_docs=args.max_docs)
        results.append(result)

    # Generate figures
    cumulative = {r["dataset"]: r["_pca_cumulative_variance"] for r in results}
    per_component = {
        r["dataset"]: np.diff(r["_pca_cumulative_variance"], prepend=0) for r in results
    }

    fig_dir = args.output / "figures"
    p1 = plot_cumulative_variance(cumulative, fig_dir / "pca_cumulative_variance.pdf")
    p2 = plot_variance_spectrum(per_component, fig_dir / "pca_variance_spectrum.pdf")
    print(f"\n  Figures saved to {p1} and {p2}")

    # Dimension distribution (skewness/kurtosis)
    dim_data = {r["dataset"]: r["_dimstats"] for r in results}
    p4 = plot_dimension_distribution(dim_data, fig_dir / "dimension_distribution.pdf")
    print(f"  Dimension distribution saved to {p4}")

    # Summary table (LaTeX)
    json_results = []
    for r in results:
        out = {k: v for k, v in r.items() if not k.startswith("_")}
        json_results.append(out)
    table_path = generate_phase1_table(json_results, args.output / "phase1_summary.tex")
    print(f"  Summary table saved to {table_path}")

    # Save JSON (without numpy arrays)
    out_path = args.output / "characterization.json"
    out_path.write_text(json.dumps(json_results, indent=2), encoding="utf-8")
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
