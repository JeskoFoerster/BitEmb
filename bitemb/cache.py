"""Deterministic embedding cache for avoiding redundant model inference.

Caches the full corpus embeddings per (model, dataset) pair as .npy files.
Since model inference is deterministic (no dropout at eval time), cached
embeddings are bit-for-bit identical to freshly computed ones.

Cache key: {model_name_slug}_{dataset_name}.npy
The cache stores the FULL corpus embedding matrix. Subsampling (--max-docs)
is applied AFTER loading from cache, preserving methodological equivalence.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bitemb.config import MODEL_NAME

if TYPE_CHECKING:
    from bitemb.engine import EmbeddingEngine

CACHE_DIR = Path("cache/embeddings")


def _model_slug(model_name: str) -> str:
    """Convert model name to a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def _cache_path(dataset_name: str, model_name: str = MODEL_NAME) -> Path:
    """Compute the cache file path for a (model, dataset) pair."""
    return CACHE_DIR / f"{_model_slug(model_name)}_{dataset_name}.npy"


def load_or_encode(
    dataset_name: str,
    texts: list[str],
    engine: EmbeddingEngine,
    *,
    model_name: str = MODEL_NAME,
    show_progress: bool = False,
) -> NDArray[np.float32]:
    """Load cached embeddings or compute and cache them.

    Args:
        dataset_name: BEIR dataset identifier (e.g. "scifact").
        texts: Full corpus texts (used for encoding if cache miss).
        engine: EmbeddingEngine instance with encode_passages method.
        model_name: Model identifier for cache key.
        show_progress: Show encoding progress bar.

    Returns:
        Float32 embedding matrix of shape (len(texts), dim).

    Raises:
        ValueError: If cached embeddings have mismatched row count,
                    indicating the corpus has changed.
    """
    path = _cache_path(dataset_name, model_name)

    if path.exists():
        embs = np.load(path)
        if embs.shape[0] != len(texts):
            raise ValueError(
                f"Cache mismatch for '{dataset_name}': cached {embs.shape[0]} rows, "
                f"but corpus has {len(texts)} texts. Delete {path} to re-encode."
            )
        print(f"  Loaded cached embeddings from {path}")
        return embs

    # Cache miss — encode and save
    print(f"  No cache found, encoding {len(texts)} documents...")
    embs = engine.encode_passages(texts, show_progress=show_progress)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embs)
    print(f"  Cached embeddings to {path}")
    return embs


def clear_cache(dataset_name: str | None = None, model_name: str = MODEL_NAME) -> None:
    """Remove cached embeddings.

    Args:
        dataset_name: If given, remove only that dataset's cache.
                      If None, remove all cached embeddings.
    """
    if dataset_name:
        path = _cache_path(dataset_name, model_name)
        if path.exists():
            path.unlink()
    else:
        if CACHE_DIR.exists():
            for f in CACHE_DIR.glob("*.npy"):
                f.unlink()
