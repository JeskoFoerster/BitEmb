"""BEIR Natural Questions evaluation for float and binary embeddings."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

DEFAULT_DIMS = [768, 256, 64]
DEFAULT_KS = [1, 5, 10]
OUTPUT_PATH = Path("beir_eval_results.json")
_POPCOUNT_TABLE = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("_id", row.get("id", "")))


def _query_id(row: dict[str, Any]) -> str:
    return str(row.get("query-id", row.get("query_id", row.get("qid", ""))))


def _corpus_id(row: dict[str, Any]) -> str:
    return str(row.get("corpus-id", row.get("corpus_id", row.get("doc_id", ""))))


def _score(row: dict[str, Any]) -> float:
    return float(row.get("score", 1.0))


def _text(row: dict[str, Any]) -> str:
    title = str(row.get("title", "")).strip()
    text = str(row.get("text", "")).strip()
    if title and text:
        return f"{title}\n{text}"
    return text or title


def load_nq_subset(
    n_docs: int, n_queries: int
) -> tuple[list[str], list[str], dict[int, set[int]]]:
    """Load a filtered BEIR/NQ subset with qrels mapped to local integer ids."""
    if n_docs <= 0:
        raise ValueError("n_docs must be greater than 0")
    if n_queries <= 0:
        raise ValueError("n_queries must be greater than 0")

    datasets = importlib.import_module("datasets")
    load_dataset = datasets.load_dataset

    corpus_ds = load_dataset("BeIR/nq", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/nq", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/nq-qrels", split="test")

    corpus_rows = [cast(dict[str, Any], corpus_ds[i]) for i in range(min(n_docs, len(corpus_ds)))]
    corpus_texts = [_text(row) for row in corpus_rows]
    corpus_id_to_idx = {_row_id(row): idx for idx, row in enumerate(corpus_rows)}

    qrels_by_query_id: dict[str, set[int]] = {}
    for row_obj in qrels_ds:
        row = cast(dict[str, Any], row_obj)
        if _score(row) <= 0:
            continue
        corpus_idx = corpus_id_to_idx.get(_corpus_id(row))
        if corpus_idx is None:
            continue
        qrels_by_query_id.setdefault(_query_id(row), set()).add(corpus_idx)

    queries: list[str] = []
    qrels: dict[int, set[int]] = {}

    for row_obj in queries_ds:
        row = cast(dict[str, Any], row_obj)
        relevant = qrels_by_query_id.get(_row_id(row))
        if not relevant:
            continue
        query_idx = len(queries)
        queries.append(_text(row))
        qrels[query_idx] = relevant
        if len(queries) >= n_queries:
            break

    return corpus_texts, queries, qrels


def recall_at_k(
    retrieved_ndarray: NDArray[np.integer[Any]], relevant_set: set[int], k: int
) -> float:
    """Compute recall@k for one ranked result list."""
    if k <= 0 or not relevant_set:
        return 0.0
    top_k = {int(idx) for idx in retrieved_ndarray[:k]}
    return len(top_k & relevant_set) / len(relevant_set)


def ndcg_at_k(retrieved: NDArray[np.integer[Any]], relevant: set[int], k: int) -> float:
    """Compute binary-relevance nDCG@k for one ranked result list."""
    if k <= 0 or not relevant:
        return 0.0

    dcg = 0.0
    for rank, idx in enumerate(retrieved[:k]):
        if int(idx) in relevant:
            dcg += 1.0 / math.log2(rank + 2)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def _mean_metrics(
    rankings: NDArray[np.integer[Any]], qrels: dict[int, set[int]], ks: list[int]
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not qrels:
        for k in ks:
            metrics[f"recall@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0
        return metrics

    for k in ks:
        recalls = [
            recall_at_k(rankings[query_idx], relevant, k) for query_idx, relevant in qrels.items()
        ]
        ndcgs = [
            ndcg_at_k(rankings[query_idx], relevant, k) for query_idx, relevant in qrels.items()
        ]
        metrics[f"recall@{k}"] = float(np.mean(recalls))
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs))

    return metrics


def evaluate_float(
    corpus_embs: NDArray[np.float32],
    query_embs: NDArray[np.float32],
    qrels: dict[int, set[int]],
    ks: list[int],
) -> dict[str, float]:
    """Evaluate normalized float embeddings with cosine similarity."""
    scores = query_embs @ corpus_embs.T
    rankings = np.argsort(scores, axis=1)[:, ::-1]
    return _mean_metrics(rankings, qrels, ks)


def evaluate_bit_hamming(
    corpus_bits: NDArray[np.uint8],
    query_bits: NDArray[np.uint8],
    qrels: dict[int, set[int]],
    ks: list[int],
    dim: int,
) -> dict[str, float]:
    """Evaluate packed binary embeddings with Hamming distance."""
    del dim

    rankings: list[NDArray[np.integer[Any]]] = []
    for query in query_bits:
        xor = np.bitwise_xor(corpus_bits, query)
        distances = _POPCOUNT_TABLE[xor].sum(axis=1)
        rankings.append(np.argsort(distances))

    return _mean_metrics(np.vstack(rankings), qrels, ks)


def evaluate_asymmetric(
    corpus_bits: NDArray[np.uint8],
    query_embs: NDArray[np.float32],
    qrels: dict[int, set[int]],
    ks: list[int],
    dim: int,
) -> dict[str, float]:
    """Evaluate float queries against unpacked normalized binary corpus vectors."""
    unpacked = np.unpackbits(corpus_bits, axis=1)[:, :dim].astype(np.float32)
    signed = unpacked * 2 - 1
    signed = signed / np.linalg.norm(signed, axis=1, keepdims=True)
    scores = query_embs @ signed.T
    rankings = np.argsort(scores, axis=1)[:, ::-1]
    return _mean_metrics(rankings, qrels, ks)


def _format_table(results: dict[str, dict[str, dict[str, float]]], ks: list[int]) -> str:
    metric_names = [metric for k in ks for metric in (f"recall@{k}", f"ndcg@{k}")]
    header = f"{'Dim':>5}  {'Mode':<10}  " + "  ".join(f"{name:>10}" for name in metric_names)
    lines = [header, "-" * len(header)]

    for dim_key, mode_results in results.items():
        for mode, metrics in mode_results.items():
            values = "  ".join(f"{metrics[name]:>10.4f}" for name in metric_names)
            lines.append(f"{dim_key:>5}  {mode:<10}  {values}")

    return "\n".join(lines)


def _json_ready(results: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    return {
        dim: {
            mode: {metric: value for metric, value in metrics.items()}
            for mode, metrics in modes.items()
        }
        for dim, modes in results.items()
    }


def main() -> None:
    """Run BEIR/NQ evaluation and save metrics as JSON."""
    parser = argparse.ArgumentParser(description="Evaluate BitEmb on a BEIR/NQ subset.")
    parser.add_argument("--n-docs", type=int, default=5_000)
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--dims", type=int, nargs="+", default=DEFAULT_DIMS)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    from engine import EmbeddingEngine

    corpus_texts, queries, qrels = load_nq_subset(args.n_docs, args.n_queries)
    engine = EmbeddingEngine()
    results: dict[str, dict[str, dict[str, float]]] = {}

    for dim in args.dims:
        corpus_embs = engine.embed_float(corpus_texts, dim)
        query_embs = engine.embed_float(queries, dim)
        corpus_bits = np.packbits(corpus_embs > 0, axis=1)
        query_bits = np.packbits(query_embs > 0, axis=1)

        results[str(dim)] = {
            "float": evaluate_float(corpus_embs, query_embs, qrels, DEFAULT_KS),
            "bit": evaluate_bit_hamming(corpus_bits, query_bits, qrels, DEFAULT_KS, dim),
            "asymmetric": evaluate_asymmetric(corpus_bits, query_embs, qrels, DEFAULT_KS, dim),
        }

    print(_format_table(results, DEFAULT_KS))
    args.output.write_text(json.dumps(_json_ready(results), indent=2), encoding="utf-8")
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
