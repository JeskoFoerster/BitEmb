"""Phase 3: Neighborhood preservation analysis.

Computes neighborhood overlap and trustworthiness across the full 2D
experimental matrix (bit_depth × PCA_dim) for k ∈ {10, 50, 100}.

Usage:
    python scripts/phase3_neighborhood.py --dataset scifact
    python scripts/phase3_neighborhood.py --all
    python scripts/phase3_neighborhood.py --all --max-docs 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bitemb.cache import load_or_encode
from bitemb.config import DATASETS, PCA_DIMS, SEED
from bitemb.dataset import load_beir
from bitemb.engine import EmbeddingEngine
from bitemb.neighborhood import compute_neighborhood_preservation
from bitemb.plotting import plot_neighborhood_heatmap, plot_neighborhood_overlap_by_k
from bitemb.quantization import PCAReducer

OUTPUT_DIR = Path("results/phase3")


def run(
    dataset_name: str, engine: EmbeddingEngine, max_docs: int | None = None
) -> dict:
    """Run Phase 3 neighborhood analysis for one dataset."""
    print(f"\n{'='*60}")
    print(f"  Phase 3: {dataset_name}")
    print(f"{'='*60}")

    print("\n  Loading dataset...")
    ds = load_beir(dataset_name)
    texts = ds.corpus_texts
    print(f"  Encoding {len(texts)} documents...")
    corpus_embs = load_or_encode(dataset_name, texts, engine, show_progress=True)

    if max_docs and max_docs < len(texts):
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(texts), size=max_docs, replace=False)
        corpus_embs = corpus_embs[idx]
        print(f"  Subsampled {max_docs}/{ds.n_corpus} documents")

    all_results = []

    for dim in PCA_DIMS:
        print(f"\n  --- PCA dim = {dim} ---")

        if dim < 768:
            pca = PCAReducer(n_components=dim).fit(corpus_embs)
        else:
            pca = None

        results = compute_neighborhood_preservation(
            corpus_embs, dim=dim, pca_reducer=pca, seed=SEED,
        )

        for r in results:
            print(
                f"    {r.bit_depth}-bit k={r.k:3d}: "
                f"overlap={r.overlap:.4f}  T={r.trustworthiness:.4f}"
            )
            all_results.append({
                "bit_depth": r.bit_depth,
                "dim": r.dim,
                "k": r.k,
                "overlap": r.overlap,
                "trustworthiness": r.trustworthiness,
                "n_docs": r.n_docs,
                "random_baseline": r.random_baseline,
            })

    return {
        "dataset": dataset_name,
        "n_corpus": len(texts),
        "seed": SEED,
        "results": all_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Neighborhood preservation analysis"
    )
    parser.add_argument("--dataset", choices=list(DATASETS), default="scifact")
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument(
        "--max-docs", type=int, default=None,
        help="Max corpus docs (subsample for speed)",
    )
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
    out_path = args.output / "neighborhood_preservation.json"
    out_path.write_text(json.dumps(all_outputs, indent=2), encoding="utf-8")
    print(f"\n  Results saved to {out_path}")

    # Generate figures
    fig_dir = args.output / "figures"

    # Metric-by-k line plots (full dim)
    p = plot_neighborhood_overlap_by_k(
        all_outputs, fig_dir / "overlap_by_k.pdf", metric="overlap",
    )
    print(f"  Overlap-by-k plot saved to {p}")

    p = plot_neighborhood_overlap_by_k(
        all_outputs, fig_dir / "trustworthiness_by_k.pdf",
        metric="trustworthiness",
    )
    print(f"  Trustworthiness-by-k plot saved to {p}")

    # Heatmaps per dataset, per metric, per k
    for ds_result in all_outputs:
        name = ds_result["dataset"]
        for metric in ("overlap", "trustworthiness"):
            for k in (10, 50, 100):
                fname = f"neighborhood_{metric}_k{k}_{name}.pdf"
                p = plot_neighborhood_heatmap(
                    ds_result, fig_dir / fname, metric=metric, k=k,
                )
                print(f"  Heatmap ({metric}, k={k}) saved to {p}")


if __name__ == "__main__":
    main()
