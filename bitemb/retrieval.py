"""Phase 4: Exact retrieval evaluation on BEIR qrels.

Evaluates whether geometric preservation from Phases 2 and 3 translates into
actual search quality. Rankings are computed exactly by exhaustive comparison
against the full corpus for each representation:

  - float32: cosine similarity
  - TurboQuant 4-bit / 2-bit: asymmetric L2^2 distance
  - binary 1-bit: Hamming distance
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import wilcoxon

from bitemb.quantization import (
    PCAReducer,
    binarize,
    binarize_rotated,
    hamming_distance,
    quantize_distance,
    quantize_encode,
)

RETRIEVAL_METRICS = ("ndcg_at_10", "recall_at_10", "recall_at_100", "mrr")
REPRESENTATIONS = (
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
)
ALPHA = 0.05
N_PAIRWISE_COMPARISONS = 45
BONFERRONI_ALPHA = ALPHA / N_PAIRWISE_COMPARISONS


@dataclass
class RetrievalResult:
    """Mean retrieval metrics for one representation and PCA dimension."""

    representation: str
    dim: int
    ndcg_at_10: float
    recall_at_10: float
    recall_at_100: float
    mrr: float
    n_queries: int


@dataclass
class SignificanceResult:
    """Wilcoxon signed-rank result for one pairwise metric comparison."""

    dim: int
    metric: str
    representation_a: str
    representation_b: str
    statistic: float
    p_value: float
    alpha: float
    corrected_alpha: float
    significant: bool


def _dcg(relevances: NDArray[np.float64]) -> float:
    """Discounted cumulative gain with graded relevance."""
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2, dtype=np.float64))
    gains = np.power(2.0, relevances) - 1.0
    return float((gains / discounts).sum())


def _ndcg_at_k(ranked_doc_ids: NDArray[np.int64], rels: dict[int, int], k: int) -> float:
    """NDCG@k for a single query."""
    if not rels:
        return 0.0
    retrieved_rels = np.array([rels.get(int(doc_id), 0) for doc_id in ranked_doc_ids[:k]])
    dcg = _dcg(retrieved_rels.astype(np.float64))
    ideal_rels = np.array(sorted(rels.values(), reverse=True)[:k], dtype=np.float64)
    idcg = _dcg(ideal_rels)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def _recall_at_k(ranked_doc_ids: NDArray[np.int64], rels: dict[int, int], k: int) -> float:
    """Recall@k for a single query, treating all positive qrels as relevant."""
    if not rels:
        return 0.0
    relevant = set(rels)
    retrieved = set(int(doc_id) for doc_id in ranked_doc_ids[:k])
    return len(relevant & retrieved) / len(relevant)


def _metrics_for_query(
    ranked_doc_ids: NDArray[np.int64],
    reciprocal_rank: float,
    rels: dict[int, int],
) -> dict[str, float]:
    """Compute all Phase 4 metrics for one query ranking."""
    return {
        "ndcg_at_10": _ndcg_at_k(ranked_doc_ids, rels, 10),
        "recall_at_10": _recall_at_k(ranked_doc_ids, rels, 10),
        "recall_at_100": _recall_at_k(ranked_doc_ids, rels, 100),
        "mrr": reciprocal_rank,
    }


def _topk_desc(scores: NDArray[np.float64], k: int) -> NDArray[np.int64]:
    """Top-k indices sorted by descending score and ascending doc id for ties."""
    k_eff = min(k, scores.shape[0])
    cutoff = np.partition(scores, -k_eff)[-k_eff]
    candidates = np.flatnonzero(scores >= cutoff)
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order[:k_eff]].astype(np.int64)


def _topk_asc(distances: NDArray[np.float64], k: int) -> NDArray[np.int64]:
    """Top-k indices sorted by ascending distance and ascending doc id for ties."""
    k_eff = min(k, distances.shape[0])
    cutoff = np.partition(distances, k_eff - 1)[k_eff - 1]
    candidates = np.flatnonzero(distances <= cutoff)
    order = np.lexsort((candidates, distances[candidates]))
    return candidates[order[:k_eff]].astype(np.int64)


def _reciprocal_rank_desc(scores: NDArray[np.float64], rels: dict[int, int]) -> float:
    """Reciprocal rank with descending-score, ascending-doc-id tie-breaking."""
    if not rels:
        return 0.0
    rel_idx = np.fromiter(rels.keys(), dtype=np.int64)
    doc_ids = np.arange(scores.shape[0], dtype=np.int64)
    ranks = []
    for doc_id in rel_idx:
        score = scores[doc_id]
        before = (scores > score) | ((scores == score) & (doc_ids < doc_id))
        ranks.append(int(np.count_nonzero(before)) + 1)
    return 1.0 / min(ranks)


def _reciprocal_rank_asc(distances: NDArray[np.float64], rels: dict[int, int]) -> float:
    """Reciprocal rank with ascending-distance, ascending-doc-id tie-breaking."""
    if not rels:
        return 0.0
    rel_idx = np.fromiter(rels.keys(), dtype=np.int64)
    doc_ids = np.arange(distances.shape[0], dtype=np.int64)
    ranks = []
    for doc_id in rel_idx:
        distance = distances[doc_id]
        before = (distances < distance) | ((distances == distance) & (doc_ids < doc_id))
        ranks.append(int(np.count_nonzero(before)) + 1)
    return 1.0 / min(ranks)


def _mean_result(
    representation: str,
    dim: int,
    per_query: dict[str, NDArray[np.float64]],
) -> RetrievalResult:
    """Aggregate per-query arrays into a RetrievalResult."""
    return RetrievalResult(
        representation=representation,
        dim=dim,
        ndcg_at_10=float(per_query["ndcg_at_10"].mean()),
        recall_at_10=float(per_query["recall_at_10"].mean()),
        recall_at_100=float(per_query["recall_at_100"].mean()),
        mrr=float(per_query["mrr"].mean()),
        n_queries=int(per_query["mrr"].shape[0]),
    )


def _empty_per_query(n_queries: int) -> dict[str, NDArray[np.float64]]:
    return {metric: np.zeros(n_queries, dtype=np.float64) for metric in RETRIEVAL_METRICS}


def _evaluate_float(
    corpus_embs: NDArray[np.float32],
    query_embs: NDArray[np.float32],
    qrels: dict[int, dict[int, int]],
    *,
    top_k: int,
    batch_size: int,
) -> dict[str, NDArray[np.float64]]:
    """Evaluate exact cosine-similarity retrieval."""
    per_query = _empty_per_query(query_embs.shape[0])

    for start in range(0, query_embs.shape[0], batch_size):
        end = min(start + batch_size, query_embs.shape[0])
        sims = (query_embs[start:end] @ corpus_embs.T).astype(np.float64)
        for local_idx, scores in enumerate(sims):
            q_idx = start + local_idx
            rels = qrels.get(q_idx, {})
            ranked = _topk_desc(scores, top_k)
            rr = _reciprocal_rank_desc(scores, rels)
            metrics = _metrics_for_query(ranked, rr, rels)
            for metric, value in metrics.items():
                per_query[metric][q_idx] = value

    return per_query


def _evaluate_float16(
    corpus_embs: NDArray[np.float32],
    query_embs: NDArray[np.float32],
    qrels: dict[int, dict[int, int]],
    *,
    top_k: int,
    batch_size: int,
) -> dict[str, NDArray[np.float64]]:
    """Evaluate exact float16 cosine-similarity retrieval."""
    per_query = _empty_per_query(query_embs.shape[0])
    c_f16 = corpus_embs.astype(np.float16)
    q_f16 = query_embs.astype(np.float16)

    for start in range(0, q_f16.shape[0], batch_size):
        end = min(start + batch_size, q_f16.shape[0])
        sims = (q_f16[start:end] @ c_f16.T).astype(np.float64)
        for local_idx, scores in enumerate(sims):
            q_idx = start + local_idx
            rels = qrels.get(q_idx, {})
            ranked = _topk_desc(scores, top_k)
            rr = _reciprocal_rank_desc(scores, rels)
            metrics = _metrics_for_query(ranked, rr, rels)
            for metric, value in metrics.items():
                per_query[metric][q_idx] = value

    return per_query


def _evaluate_quantized(
    corpus_embs: NDArray[np.float32],
    query_embs: NDArray[np.float32],
    qrels: dict[int, dict[int, int]],
    *,
    bits: int,
    is_rotated: bool,
    top_k: int,
    batch_size: int,
) -> dict[str, NDArray[np.float64]]:
    """Evaluate exact quantized retrieval (naive or rotated) with asymmetric query distances."""
    per_query = _empty_per_query(query_embs.shape[0])
    index = quantize_encode(corpus_embs, bits=bits, is_rotated=is_rotated)

    for start in range(0, query_embs.shape[0], batch_size):
        end = min(start + batch_size, query_embs.shape[0])
        dists = quantize_distance(index, query_embs[start:end])
        for local_idx, distances in enumerate(dists):
            q_idx = start + local_idx
            rels = qrels.get(q_idx, {})
            ranked = _topk_asc(distances, top_k)
            rr = _reciprocal_rank_asc(distances, rels)
            metrics = _metrics_for_query(ranked, rr, rels)
            for metric, value in metrics.items():
                per_query[metric][q_idx] = value

    return per_query


def _evaluate_binary(
    corpus_embs: NDArray[np.float32],
    query_embs: NDArray[np.float32],
    qrels: dict[int, dict[int, int]],
    *,
    is_rotated: bool,
    top_k: int,
) -> dict[str, NDArray[np.float64]]:
    """Evaluate exact binary retrieval with Hamming distance (naive or rotated)."""
    per_query = _empty_per_query(query_embs.shape[0])
    if is_rotated:
        corpus_packed = binarize_rotated(corpus_embs)
        query_packed = binarize_rotated(query_embs)
    else:
        corpus_packed = binarize(corpus_embs)
        query_packed = binarize(query_embs)

    for q_idx in range(query_packed.shape[0]):
        rels = qrels.get(q_idx, {})
        dists = hamming_distance(corpus_packed, query_packed[q_idx:q_idx + 1]).astype(np.float64)
        ranked = _topk_asc(dists, top_k)
        rr = _reciprocal_rank_asc(dists, rels)
        metrics = _metrics_for_query(ranked, rr, rels)
        for metric, value in metrics.items():
            per_query[metric][q_idx] = value

    return per_query


def compute_retrieval_evaluation(
    corpus_embeddings: NDArray[np.float32],
    query_embeddings: NDArray[np.float32],
    qrels: dict[int, dict[int, int]],
    *,
    dim: int = 1024,
    pca_reducer: PCAReducer | None = None,
    top_k: int = 100,
    batch_size: int = 32,
) -> tuple[list[RetrievalResult], dict[str, dict[str, NDArray[np.float64]]]]:
    """Run exact Phase 4 retrieval evaluation for one PCA dimension.

    Args:
        corpus_embeddings: L2-normalized float32 corpus embeddings.
        query_embeddings: L2-normalized float32 query embeddings.
        qrels: Mapping query index -> {corpus index: relevance_score}.
        dim: Target dimensionality after optional PCA.
        pca_reducer: Fitted reducer for dim < 1024.
        top_k: Largest cutoff needed for top-k metrics.
        batch_size: Query batch size for dense score/distance matrices.

    Returns:
        Mean results and per-query metric arrays keyed by representation.
    """
    if pca_reducer is not None and dim < corpus_embeddings.shape[1]:
        corpus = pca_reducer.transform(corpus_embeddings)
        queries = pca_reducer.transform(query_embeddings)
    else:
        corpus = corpus_embeddings
        queries = query_embeddings

    if top_k < 100:
        raise ValueError("top_k must be at least 100 for Recall@100")
    if queries.shape[0] != len(qrels):
        raise ValueError(
            f"Expected qrels for {queries.shape[0]} filtered queries, got {len(qrels)}"
        )

    per_query = {
        "float32": _evaluate_float(corpus, queries, qrels, top_k=top_k, batch_size=batch_size),
        "16bit": _evaluate_float16(corpus, queries, qrels, top_k=top_k, batch_size=batch_size),
        "naive_8bit": _evaluate_quantized(
            corpus, queries, qrels, bits=8, is_rotated=False, top_k=top_k, batch_size=batch_size,
        ),
        "tq_8bit": _evaluate_quantized(
            corpus, queries, qrels, bits=8, is_rotated=True, top_k=top_k, batch_size=batch_size,
        ),
        "naive_4bit": _evaluate_quantized(
            corpus, queries, qrels, bits=4, is_rotated=False, top_k=top_k, batch_size=batch_size,
        ),
        "tq_4bit": _evaluate_quantized(
            corpus, queries, qrels, bits=4, is_rotated=True, top_k=top_k, batch_size=batch_size,
        ),
        "naive_2bit": _evaluate_quantized(
            corpus, queries, qrels, bits=2, is_rotated=False, top_k=top_k, batch_size=batch_size,
        ),
        "tq_2bit": _evaluate_quantized(
            corpus, queries, qrels, bits=2, is_rotated=True, top_k=top_k, batch_size=batch_size,
        ),
        "naive_1bit": _evaluate_binary(corpus, queries, qrels, is_rotated=False, top_k=top_k),
        "tq_1bit": _evaluate_binary(corpus, queries, qrels, is_rotated=True, top_k=top_k),
    }

    results = [
        _mean_result(representation, dim, per_query[representation])
        for representation in REPRESENTATIONS
    ]
    return results, per_query


def compute_significance_tests(
    per_query: dict[str, dict[str, NDArray[np.float64]]],
    *,
    dim: int,
    alpha: float = ALPHA,
) -> list[SignificanceResult]:
    """Run pairwise Wilcoxon signed-rank tests on per-query metric arrays."""
    corrected_alpha = alpha / N_PAIRWISE_COMPARISONS
    results: list[SignificanceResult] = []

    for metric in RETRIEVAL_METRICS:
        for rep_a, rep_b in combinations(REPRESENTATIONS, 2):
            values_a = per_query[rep_a][metric]
            values_b = per_query[rep_b][metric]
            if np.allclose(values_a, values_b):
                statistic = 0.0
                p_value = 1.0
            else:
                test = wilcoxon(values_a, values_b, zero_method="wilcox", alternative="two-sided")
                statistic = float(test.statistic)
                p_value = float(test.pvalue)

            results.append(SignificanceResult(
                dim=dim,
                metric=metric,
                representation_a=rep_a,
                representation_b=rep_b,
                statistic=statistic,
                p_value=p_value,
                alpha=alpha,
                corrected_alpha=corrected_alpha,
                significant=p_value < corrected_alpha,
            ))

    return results
