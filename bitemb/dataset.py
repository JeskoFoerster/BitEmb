"""BEIR dataset loading for SciFact, FiQA, and TREC-COVID.

Uses the HuggingFace `datasets` library to load corpus, queries, and qrels
from the official BeIR collection. Returns clean Python structures with
integer-indexed mappings suitable for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import datasets as hf_datasets

from bitemb.config import DATASETS

# Mapping from short name → HuggingFace dataset identifiers
_CORPUS_DATASETS = {
    "scifact": ("BeIR/scifact", "corpus", "corpus"),
    "fiqa": ("BeIR/fiqa", "corpus", "corpus"),
    "trec-covid": ("BeIR/trec-covid", "corpus", "corpus"),
}

_QUERY_DATASETS = {
    "scifact": ("BeIR/scifact", "queries", "queries"),
    "fiqa": ("BeIR/fiqa", "queries", "queries"),
    "trec-covid": ("BeIR/trec-covid", "queries", "queries"),
}

_QREL_DATASETS = {
    "scifact": ("BeIR/scifact-qrels", None, "test"),
    "fiqa": ("BeIR/fiqa-qrels", None, "test"),
    "trec-covid": ("BeIR/trec-covid-qrels", None, "test"),
}


@dataclass
class BeirDataset:
    """A loaded BEIR subset ready for evaluation.

    Attributes:
        name: Dataset identifier (e.g. "scifact").
        corpus_texts: List of document texts, indexed by corpus position.
        corpus_ids: Original string IDs from BEIR, parallel to corpus_texts.
        queries: List of query texts, indexed by query position.
        query_ids: Original string IDs from BEIR, parallel to queries.
        qrels: Mapping from query index → dict of {corpus_index: relevance_score}.
    """

    name: str
    corpus_texts: list[str] = field(default_factory=list)
    corpus_ids: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    query_ids: list[str] = field(default_factory=list)
    qrels: dict[int, dict[int, int]] = field(default_factory=dict)

    @property
    def n_corpus(self) -> int:
        return len(self.corpus_texts)

    @property
    def n_queries(self) -> int:
        return len(self.queries)


def _extract_text(row: dict[str, Any]) -> str:
    """Combine title and text fields from a BEIR corpus row."""
    title = str(row.get("title", "")).strip()
    text = str(row.get("text", "")).strip()
    if title and text:
        return f"{title} {text}"
    return text or title


def _get_id(row: dict[str, Any]) -> str:
    """Extract document/query ID from a row."""
    return str(row.get("_id", row.get("id", "")))


def load_beir(name: str) -> BeirDataset:
    """Load a BEIR dataset by name.

    Args:
        name: One of "scifact", "fiqa", "trec-covid".

    Returns:
        BeirDataset with corpus, queries, and relevance judgments.

    Raises:
        ValueError: If name is not a supported dataset.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Supported: {DATASETS}")

    # Load corpus
    repo, config, split = _CORPUS_DATASETS[name]
    corpus_ds = hf_datasets.load_dataset(repo, config, split=split)
    corpus_texts: list[str] = []
    corpus_ids: list[str] = []
    corpus_id_to_idx: dict[str, int] = {}

    for row in corpus_ds:
        doc_id = _get_id(row)
        corpus_id_to_idx[doc_id] = len(corpus_texts)
        corpus_ids.append(doc_id)
        corpus_texts.append(_extract_text(row))

    # Load queries
    repo, config, split = _QUERY_DATASETS[name]
    queries_ds = hf_datasets.load_dataset(repo, config, split=split)
    queries: list[str] = []
    query_ids: list[str] = []
    query_id_to_idx: dict[str, int] = {}

    for row in queries_ds:
        qid = _get_id(row)
        query_id_to_idx[qid] = len(queries)
        query_ids.append(qid)
        queries.append(str(row.get("text", "")).strip())

    # Load qrels
    repo, config, split = _QREL_DATASETS[name]
    if config:
        qrels_ds = hf_datasets.load_dataset(repo, config, split=split)
    else:
        qrels_ds = hf_datasets.load_dataset(repo, split=split)

    qrels: dict[int, dict[int, int]] = {}
    for row in qrels_ds:
        qid = str(row.get("query-id", row.get("query_id", "")))
        cid = str(row.get("corpus-id", row.get("corpus_id", "")))
        score = int(row.get("score", 0))
        if score <= 0:
            continue
        q_idx = query_id_to_idx.get(qid)
        c_idx = corpus_id_to_idx.get(cid)
        if q_idx is None or c_idx is None:
            continue
        qrels.setdefault(q_idx, {})[c_idx] = score

    # Filter queries to only those with relevance judgments
    active_q_indices = sorted(qrels.keys())
    filtered_queries = [queries[i] for i in active_q_indices]
    filtered_query_ids = [query_ids[i] for i in active_q_indices]

    # Remap qrels to new contiguous indices
    old_to_new = {old: new for new, old in enumerate(active_q_indices)}
    remapped_qrels = {old_to_new[old]: rels for old, rels in qrels.items()}

    return BeirDataset(
        name=name,
        corpus_texts=corpus_texts,
        corpus_ids=corpus_ids,
        queries=filtered_queries,
        query_ids=filtered_query_ids,
        qrels=remapped_qrels,
    )
