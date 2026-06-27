"""Phase 2: Pairwise distance correlation and distortion analysis.

Computes the full 2D experimental matrix (bit_depth × PCA_dim) for each dataset.
For each combination, 10,000 random pairs are drawn and distance preservation
is measured via Pearson r, Spearman ρ, MAE, and RMSE.

Usage:
    python scripts/phase2_distance_analysis.py --dataset scifact
    python scripts/phase2_distance_analysis.py --all
    python scripts/phase2_distance_analysis.py --all --max-docs 10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bitemb.config import DATASETS, PCA_DIMS, SEED
from bitemb.dataset import load_beir
from bitemb.distance import compute_distance_distortion
from bitemb.engine import EmbeddingEngine
from bitemb.plotting import plot_distortion_heatmap, plot_distortion_pareto
from bitemb.quantization import PCAReducer

OUTPUT_DIR = Path("results/phase2")


def run(dataset_name: str, engine: EmbeddingEngine, max_docs: int | None = None) -> dict:
    """Run Phase 2 distance analysis for one dataset across all PCA dims."""
    print(f"\n{'='*60}")
    print(f"  Phase 2: {dataset_name}")
    print(f"{'='*60}")

    # Load & encode
    print("\n  Loading dataset...")
    ds = load_beir(dataset_name)
    texts = ds.corpus_texts
    if max_docs and max_docs < len(texts):
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(texts), size=max_docs, replace=False)
        texts = [texts[i] for i in idx]
        print(f"  Subsampled {max_docs}/{ds.n_corpus} documents")

    print(f"  Encoding {len(texts)} documents...")
    corpus_embs = engine.encode_passages(texts, show_progress=True)

    all_results = []

    for dim in PCA_DIMS:
        print(f"\n  --- PCA dim = {dim} ---")

        # Fit PCA if dim < 768
        if dim < 768:
            pca = PCAReducer(n_components=dim).fit(corpus_embs)
        else:
            pca = None

        results = compute_distance_distortion(
            corpus_embs, dim=dim, pca_reducer=pca, seed=SEED,
        )

        for r in results:
            print(f"    {r.bit_depth}-bit: Pearson={r.pearson_r:.4f}  "
                  f"Spearman={r.spearman_rho:.4f}  MAE={r.mae:.4f}  RMSE={r.rmse:.4f}")
            all_results.append({
                "bit_depth": r.bit_depth,
                "dim": r.dim,
                "pearson_r": r.pearson_r,
                "pearson_p": r.pearson_p,
                "spearman_rho": r.spearman_rho,
                "spearman_p": r.spearman_p,
                "mae": r.mae,
                "rmse": r.rmse,
                "n_pairs": r.n_pairs,
            })

    return {
        "dataset": dataset_name,
        "n_corpus": len(texts),
        "n_pairs": 10_000,
        "seed": SEED,
        "results": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Distance distortion analysis")
    parser.add_argument("--dataset", choices=list(DATASETS), default="scifact")
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Max corpus docs (subsample for speed)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    engine = EmbeddingEngine()
    datasets = list(DATASETS) if args.all else [args.dataset]
    all_outputs = []

    for name in datasets:
        result = run(name, engine, max_docs=args.max_docs)
        all_outputs.append(result)

    # Save JSON
    out_path = args.output / "distance_distortion.json"
    out_path.write_text(json.dumps(all_outputs, indent=2), encoding="utf-8")
    print(f"\n  Results saved to {out_path}")

    # Generate figures
    fig_dir = args.output / "figures"
    p1 = plot_distortion_pareto(all_outputs, fig_dir / "distortion_pareto.pdf")
    print(f"  Pareto plot saved to {p1}")
    for ds_result in all_outputs:
        name = ds_result["dataset"]
        p = plot_distortion_heatmap(ds_result, fig_dir / f"distortion_heatmap_{name}.pdf")
        print(f"  Heatmap saved to {p}")


if __name__ == "__main__":
    main()
