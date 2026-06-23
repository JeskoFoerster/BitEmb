"""FastMCP server exposing BitEmb embedding, retrieval, and analysis tools."""

from __future__ import annotations

import importlib
from dataclasses import asdict
from typing import Any, cast

import numpy as np
from fastmcp import FastMCP

from bitemb.benchmark import BenchmarkResult, benchmark_scaling, format_benchmark_results
from bitemb.chunking import STRATEGIES, Chunk, load_file
from bitemb.engine import EmbeddingEngine
from bitemb.retrieval import Retriever, SearchResult

mcp = FastMCP("BitEmb")
_engine: EmbeddingEngine | None = None
_retriever = Retriever()


def _get_engine() -> EmbeddingEngine:
    """Create the embedding engine on first use."""
    global _engine
    if _engine is None:
        _engine = EmbeddingEngine()
    return _engine


def _result_to_dict(result: SearchResult) -> dict[str, Any]:
    return {
        "idx": result.idx,
        "text": result.text,
        "score": result.score,
        "latency_ms": result.latency_ms,
    }


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "text": chunk.text,
        "source": chunk.source,
        "strategy": chunk.strategy,
        "index": chunk.index,
    }


def _benchmark_result_to_dict(value: Any) -> Any:
    if isinstance(value, BenchmarkResult):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _benchmark_result_to_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_benchmark_result_to_dict(item) for item in value]
    return value


def _embed_texts(texts: list[str], dim: int) -> tuple[np.ndarray, np.ndarray]:
    engine = _get_engine()
    float_embs = engine.embed_float(texts, dim=dim)
    bit_embs = engine.embed_bit_sign(texts, dim=dim)
    return float_embs, bit_embs


@mcp.tool()
def embed(texts: list[str], dim: int = 768, mode: str = "both") -> dict[str, Any]:
    """Embed texts as float vectors, packed bit vectors, or both."""
    engine = _get_engine()
    response: dict[str, Any] = {"dim": dim, "count": len(texts), "mode": mode}

    if mode in {"float", "both"}:
        response["float_embeddings"] = engine.embed_float(texts, dim=dim).tolist()
    if mode in {"bit", "both"}:
        response["bit_embeddings"] = engine.embed_bit_sign(texts, dim=dim).tolist()
    if mode not in {"float", "bit", "both"}:
        raise ValueError("mode must be one of: float, bit, both")

    return response


@mcp.tool()
def index_documents(
    index_name: str,
    paths: list[str],
    strategy: str = "semantic",
    dim: int = 768,
) -> dict[str, Any]:
    """Load files, chunk them, embed chunks, and store an in-memory index."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown chunking strategy: {strategy}")

    chunks: list[Chunk] = []
    for path in paths:
        text = load_file(path)
        chunks.extend(STRATEGIES[strategy](text, source=path))

    texts = [chunk.text for chunk in chunks]
    float_embs, bit_embs = _embed_texts(texts, dim)
    _retriever.create_index(index_name, texts, float_embs, bit_embs, dim)
    _retriever.indices[index_name].metadata = {
        "sources": paths,
        "strategy": strategy,
        "chunks": [_chunk_to_dict(chunk) for chunk in chunks],
    }

    return {"index": index_name, "documents": len(paths), "chunks": len(chunks), "dim": dim}


@mcp.tool()
def index_texts(index_name: str, texts: list[str], dim: int = 768) -> dict[str, Any]:
    """Embed raw texts and store an in-memory index."""
    float_embs, bit_embs = _embed_texts(texts, dim)
    _retriever.create_index(index_name, texts, float_embs, bit_embs, dim)
    _retriever.indices[index_name].metadata = {"source": "texts"}
    return {"index": index_name, "texts": len(texts), "dim": dim}


@mcp.tool()
def search(index_name: str, query: str, k: int = 10, mode: str = "float") -> dict[str, Any]:
    """Search an existing index using float, bit, or asymmetric retrieval."""
    idx = _retriever.indices[index_name]
    engine = _get_engine()

    if mode == "float":
        query_emb = engine.embed_float([query], dim=idx.dim)
        results = _retriever.search_float(index_name, query_emb, k)
    elif mode == "bit":
        query_bits = engine.embed_bit_sign([query], dim=idx.dim)
        results = _retriever.search_bit(index_name, query_bits, k)
    elif mode == "asymmetric":
        query_emb = engine.embed_float([query], dim=idx.dim)
        results = _retriever.search_asymmetric(index_name, query_emb, k)
    else:
        raise ValueError("mode must be one of: float, bit, asymmetric")

    return {"index": index_name, "mode": mode, "results": [_result_to_dict(r) for r in results]}


@mcp.tool()
def compare_embeddings(text_a: str, text_b: str, dim: int = 768) -> dict[str, Any]:
    """Compare two texts with float cosine and binary Hamming similarity."""
    engine = _get_engine()
    float_embs = engine.embed_float([text_a, text_b], dim=dim)
    bit_embs = engine.embed_bit_sign([text_a, text_b], dim=dim)
    float_similarity = float(float_embs[0] @ float_embs[1])
    hamming_distance = int(np.unpackbits(np.bitwise_xor(bit_embs[0], bit_embs[1]))[:dim].sum())
    bit_similarity = 1.0 - hamming_distance / dim

    return {
        "dim": dim,
        "float_cosine": float_similarity,
        "hamming_distance": hamming_distance,
        "bit_similarity": bit_similarity,
    }


@mcp.tool()
def visualize(index_name: str, method: str = "pca", title: str | None = None) -> dict[str, Any]:
    """Create a PCA or t-SNE visualization for an existing index."""
    analysis = importlib.import_module("bitemb.analysis")
    idx = _retriever.indices[index_name]
    plot_title = title or f"{index_name} {method}"

    if method == "pca":
        path = analysis.plot_pca_2d(idx.float_embs, plot_title, labels=idx.texts)
    elif method == "tsne":
        unpacked = np.unpackbits(idx.bit_embs, axis=1)[:, : idx.dim]
        path = analysis.plot_tsne_2d(idx.float_embs, unpacked, plot_title)
    else:
        raise ValueError("method must be one of: pca, tsne")

    return {"index": index_name, "method": method, "path": str(path)}


@mcp.tool()
def analyze_space(index_name: str) -> dict[str, Any]:
    """Analyze the float vector space for an existing index."""
    analysis = importlib.import_module("bitemb.analysis")
    idx = _retriever.indices[index_name]
    return cast(dict[str, Any], analysis.analyze_vector_space(idx.float_embs))


@mcp.tool()
def analyze_loss(index_name: str) -> dict[str, Any]:
    """Analyze information loss from float to packed binary embeddings."""
    analysis = importlib.import_module("bitemb.analysis")
    idx = _retriever.indices[index_name]
    return cast(
        dict[str, Any],
        analysis.analyze_information_loss(idx.float_embs, idx.bit_embs, idx.dim),
    )


@mcp.tool()
def compare_chunking(text_or_path: str, is_path: bool = False) -> dict[str, Any]:
    """Compare all configured chunking strategies for a text or file."""
    text = load_file(text_or_path) if is_path else text_or_path
    comparisons: dict[str, Any] = {}

    for name, strategy in STRATEGIES.items():
        chunks = strategy(text, source=text_or_path if is_path else "")
        lengths = [len(chunk.text) for chunk in chunks]
        comparisons[name] = {
            "n_chunks": len(chunks),
            "avg_chars": float(np.mean(lengths)) if lengths else 0.0,
            "min_chars": min(lengths) if lengths else 0,
            "max_chars": max(lengths) if lengths else 0,
            "preview": [_chunk_to_dict(chunk) for chunk in chunks[:3]],
        }

    return comparisons


@mcp.tool()
def run_benchmark(sizes: list[int] | None = None, dim: int = 768) -> dict[str, Any]:
    """Run the synthetic float-vs-bit benchmark suite."""
    results = benchmark_scaling(engine=None, sizes=sizes, dim=dim)
    return {
        "results": _benchmark_result_to_dict(results),
        "table": format_benchmark_results(results),
    }


@mcp.tool()
def list_indices() -> list[dict[str, Any]]:
    """List in-memory retrieval indices."""
    return [
        {
            "name": name,
            "texts": len(index.texts),
            "dim": index.dim,
            "float_memory_bytes": int(index.float_embs.nbytes),
            "bit_memory_bytes": int(index.bit_embs.nbytes),
            "metadata": index.metadata,
        }
        for name, index in _retriever.indices.items()
    ]


def main() -> None:
    """Run the BitEmb MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
