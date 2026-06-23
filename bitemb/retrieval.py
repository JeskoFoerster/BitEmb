"""Retriever – Float and binary embedding search."""

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class SearchResult:
    """Single search result."""

    idx: int
    text: str
    score: float
    latency_ms: float = 0.0


@dataclass
class Index:
    """Stored embedding index."""

    texts: list[str]
    float_embs: NDArray[np.float32]
    bit_embs: NDArray[np.uint8]
    dim: int = 768
    metadata: dict = field(default_factory=dict)


class Retriever:
    """Search engine supporting float, binary, and asymmetric retrieval."""

    def __init__(self) -> None:
        """Initialize with empty index store."""
        self.indices: dict[str, Index] = {}

    def create_index(
        self,
        name: str,
        texts: list[str],
        float_embs: NDArray[np.float32],
        bit_embs: NDArray[np.uint8],
        dim: int = 768,
    ) -> None:
        """Create and store a named index.

        Args:
            name: Index identifier.
            texts: Original texts.
            float_embs: Normalized float embeddings.
            bit_embs: Packed binary embeddings.
            dim: Embedding dimensionality.
        """
        self.indices[name] = Index(texts=texts, float_embs=float_embs, bit_embs=bit_embs, dim=dim)

    def search_float(
        self, index_name: str, query_emb: NDArray[np.float32], k: int = 10
    ) -> list[SearchResult]:
        """Cosine similarity search using float embeddings.

        Args:
            index_name: Target index name.
            query_emb: Query embedding (1, dim).
            k: Number of results.

        Returns:
            Top-k search results sorted by score descending.
        """
        idx = self.indices[index_name]
        t0 = time.perf_counter()
        scores = (idx.float_embs @ query_emb.T).ravel()
        top_k = np.argsort(scores)[::-1][:k]
        latency = (time.perf_counter() - t0) * 1000
        return [SearchResult(int(i), idx.texts[i], float(scores[i]), latency) for i in top_k]

    def search_bit(
        self, index_name: str, query_bits: NDArray[np.uint8], k: int = 10
    ) -> list[SearchResult]:
        """Hamming similarity search using binary embeddings.

        Args:
            index_name: Target index name.
            query_bits: Packed query bits (1, ceil(dim/8)).
            k: Number of results.

        Returns:
            Top-k search results sorted by similarity descending.
        """
        idx = self.indices[index_name]
        t0 = time.perf_counter()
        xor = np.bitwise_xor(idx.bit_embs, query_bits)
        hamming_dist = np.unpackbits(xor, axis=1)[:, : idx.dim].sum(axis=1)
        similarity = 1.0 - hamming_dist / idx.dim
        top_k = np.argsort(similarity)[::-1][:k]
        latency = (time.perf_counter() - t0) * 1000
        return [SearchResult(int(i), idx.texts[i], float(similarity[i]), latency) for i in top_k]

    def search_asymmetric(
        self, index_name: str, query_float: NDArray[np.float32], k: int = 10
    ) -> list[SearchResult]:
        """Asymmetric search: float query against binary index.

        Unpacks binary embeddings to {-1, +1}, normalizes, then computes
        dot product with the float query.

        Args:
            index_name: Target index name.
            query_float: Float query embedding (1, dim).
            k: Number of results.

        Returns:
            Top-k search results sorted by score descending.
        """
        idx = self.indices[index_name]
        t0 = time.perf_counter()
        unpacked = np.unpackbits(idx.bit_embs, axis=1)[:, : idx.dim].astype(np.float32)
        signed = unpacked * 2 - 1
        norms = np.linalg.norm(signed, axis=1, keepdims=True)
        signed = signed / norms
        scores = (signed @ query_float.T).ravel()
        top_k = np.argsort(scores)[::-1][:k]
        latency = (time.perf_counter() - t0) * 1000
        return [SearchResult(int(i), idx.texts[i], float(scores[i]), latency) for i in top_k]

    def compare_search(
        self,
        index_name: str,
        query_float: NDArray[np.float32],
        query_bits: NDArray[np.uint8],
        k: int = 10,
    ) -> dict:
        """Compare float and bit search results.

        Args:
            index_name: Target index name.
            query_float: Float query embedding (1, dim).
            query_bits: Packed query bits (1, ceil(dim/8)).
            k: Number of results.

        Returns:
            Dict with float_results, bit_results, and overlap ratio.
        """
        float_results = self.search_float(index_name, query_float, k)
        bit_results = self.search_bit(index_name, query_bits, k)
        float_ids = {r.idx for r in float_results}
        bit_ids = {r.idx for r in bit_results}
        overlap = len(float_ids & bit_ids) / k if k > 0 else 0.0
        return {"float_results": float_results, "bit_results": bit_results, "overlap": overlap}
