"""Phase 4: Exact BEIR retrieval evaluation.

Computes search quality across the full representation matrix:
float32, TurboQuant 4-bit, TurboQuant 2-bit, and binary 1-bit, each evaluated
for the configured PCA dimensions.

Usage:
    python scripts/phase4_retrieval.py --dataset scifact
    python scripts/phase4_retrieval.py --all
    python scripts/phase4_retrieval.py --all --max-docs 10000
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from bitemb.cache import load_or_encode, load_or_encode_queries  # noqa: E402
from bitemb.config import DATASETS, PCA_DIMS, SEED  # noqa: E402
from bitemb.dataset import BeirDataset, load_beir  # noqa: E402
from bitemb.engine import EmbeddingEngine  # noqa: E402
from bitemb.plotting import (  # noqa: E402
    plot_retrieval_by_dim,
    plot_retrieval_heatmap,
    plot_retrieval_pareto,
)
from bitemb.quantization import PCAReducer  # noqa: E402
from bitemb.retrieval import compute_retrieval_evaluation, compute_significance_tests  # noqa: E402

OUTPUT_DIR = Path("results/phase4")


def _subsample_corpus(
    ds: BeirDataset,
    corpus_embs: np.ndarray,
    max_docs: int | None,
) -> tuple[np.ndarray, dict[int, dict[int, int]], int]:
    """Apply deterministic corpus subsampling and remap qrels accordingly."""
    if max_docs is None or max_docs >= ds.n_corpus:
        return corpus_embs, ds.qrels, ds.n_corpus

    rng = np.random.default_rng(SEED)
    selected = np.sort(rng.choice(ds.n_corpus, size=max_docs, replace=False))
    old_to_new = {int(old): new for new, old in enumerate(selected)}

    remapped_qrels: dict[int, dict[int, int]] = {}
    for q_idx, rels in ds.qrels.items():
        kept = {
            old_to_new[doc_idx]: rel
            for doc_idx, rel in rels.items()
            if doc_idx in old_to_new
        }
        if kept:
            remapped_qrels[q_idx] = kept

    return corpus_embs[selected], remapped_qrels, int(selected.shape[0])


def _filter_queries_after_subsample(
    query_embs: np.ndarray,
    qrels: dict[int, dict[int, int]],
) -> tuple[np.ndarray, dict[int, dict[int, int]], list[int]]:
    """Drop queries that lost all relevant documents after corpus subsampling."""
    active = sorted(qrels)
    old_to_new = {old: new for new, old in enumerate(active)}
    filtered_qrels = {old_to_new[old]: rels for old, rels in qrels.items() if old in old_to_new}
    return query_embs[active], filtered_qrels, active


def _per_query_to_json(
    per_query_by_dim: dict[int, dict[str, dict[str, np.ndarray]]],
) -> list[dict]:
    records = []
    for dim, by_rep in per_query_by_dim.items():
        for representation, metrics in by_rep.items():
            record = {
                "dim": dim,
                "representation": representation,
            }
            record.update({metric: values.tolist() for metric, values in metrics.items()})
            records.append(record)
    return records


def run(
    dataset_name: str,
    engine: EmbeddingEngine,
    max_docs: int | None = None,
    batch_size: int = 32,
) -> dict:
    """Run Phase 4 retrieval evaluation for one dataset."""
    print(f"\n{'='*60}")
    print(f"  Phase 4: {dataset_name}")
    print(f"{'='*60}")

    print("\n  Loading dataset...")
    ds = load_beir(dataset_name)

    print(f"  Encoding/loading {ds.n_corpus} corpus documents...")
    corpus_embs = load_or_encode(dataset_name, ds.corpus_texts, engine, show_progress=True)

    print(f"  Encoding/loading {ds.n_queries} queries...")
    query_embs = load_or_encode_queries(dataset_name, ds.queries, engine, show_progress=True)

    corpus_embs, qrels, n_corpus_eval = _subsample_corpus(ds, corpus_embs, max_docs)
    if max_docs and max_docs < ds.n_corpus:
        query_embs, qrels, active_queries = _filter_queries_after_subsample(query_embs, qrels)
        print(
            f"  Subsampled {n_corpus_eval}/{ds.n_corpus} documents; "
            f"kept {len(active_queries)}/{ds.n_queries} queries with qrels"
        )
    else:
        active_queries = list(range(ds.n_queries))

    if not qrels:
        raise ValueError("No relevance judgments remain after corpus subsampling")

    all_results = []
    all_significance = []
    per_query_by_dim = {}

    for dim in PCA_DIMS:
        print(f"\n  --- PCA dim = {dim} ---")
        if dim < corpus_embs.shape[1]:
            pca = PCAReducer(n_components=dim).fit(corpus_embs)
        else:
            pca = None

        results, per_query = compute_retrieval_evaluation(
            corpus_embs,
            query_embs,
            qrels,
            dim=dim,
            pca_reducer=pca,
            batch_size=batch_size,
        )
        significance = compute_significance_tests(per_query, dim=dim)

        for r in results:
            print(
                f"    {r.representation:7s}: "
                f"NDCG@10={r.ndcg_at_10:.4f}  "
                f"R@10={r.recall_at_10:.4f}  "
                f"R@100={r.recall_at_100:.4f}  "
                f"MRR={r.mrr:.4f}"
            )
            all_results.append(asdict(r))

        all_significance.extend(asdict(r) for r in significance)
        per_query_by_dim[dim] = per_query

    return {
        "dataset": dataset_name,
        "n_corpus": n_corpus_eval,
        "n_queries": query_embs.shape[0],
        "original_n_corpus": ds.n_corpus,
        "original_n_queries": ds.n_queries,
        "active_query_indices": active_queries,
        "seed": SEED,
        "results": all_results,
        "significance": all_significance,
        "_per_query": per_query_by_dim,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Exact retrieval evaluation")
    parser.add_argument("--dataset", choices=list(DATASETS), default="scifact")
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Max corpus docs (deterministic subsample for speed)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    engine = EmbeddingEngine()
    datasets = list(DATASETS) if args.all else [args.dataset]

    outputs = []
    per_query_outputs = []
    for name in datasets:
        result = run(name, engine, max_docs=args.max_docs, batch_size=args.batch_size)
        per_query_outputs.append({
            "dataset": name,
            "per_query": _per_query_to_json(result["_per_query"]),
        })
        outputs.append({k: v for k, v in result.items() if not k.startswith("_")})

    metrics_path = args.output / "retrieval_metrics.json"
    per_query_path = args.output / "retrieval_per_query.json"
    metrics_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    per_query_path.write_text(json.dumps(per_query_outputs, indent=2), encoding="utf-8")
    print(f"\n  Results saved to {metrics_path}")
    print(f"  Per-query metrics saved to {per_query_path}")

    fig_dir = args.output / "figures"
    for metric, label in [
        ("ndcg_at_10", "NDCG@10"),
        ("recall_at_10", "Recall@10"),
        ("recall_at_100", "Recall@100"),
        ("mrr", "MRR"),
    ]:
        p = plot_retrieval_by_dim(outputs, fig_dir / f"retrieval_{metric}_by_dim.pdf", metric)
        print(f"  By-dim ({label}) saved to {p}")
        p = plot_retrieval_pareto(outputs, fig_dir / f"retrieval_pareto_{metric}.pdf", metric)
        print(f"  Pareto ({label}) saved to {p}")

    for ds_result in outputs:
        for metric, label in [
            ("ndcg_at_10", "NDCG@10"),
            ("recall_at_100", "Recall@100"),
        ]:
            p = plot_retrieval_heatmap(
                ds_result,
                fig_dir / f"retrieval_heatmap_{metric}_{ds_result['dataset']}.pdf",
                metric=metric,
                title=f"{ds_result['dataset']} - {label}",
            )
            print(f"  Heatmap ({label}) saved to {p}")


if __name__ == "__main__":
    main()




