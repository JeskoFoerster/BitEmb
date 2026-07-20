"""EmbeddingEngine – generates float embeddings with BAAI/bge-large-en-v1.5.

BGE models use instruction prefixes to distinguish queries from passages.
For bge-large-en-v1.5:
  - Queries: "Represent this sentence for searching relevant passages: {text}"
  - Passages: no prefix (raw text)

Reference: https://huggingface.co/BAAI/bge-large-en-v1.5
"""

import os as _os

# Disable hf_transfer if the package is missing – must patch both the env var
# AND the cached constant in huggingface_hub.constants (read at import time).
try:
    import hf_transfer  # noqa: F401
except ModuleNotFoundError:
    _os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    try:
        import huggingface_hub.constants
        huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = False
    except ImportError:
        pass

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from bitemb.config import MODEL_DIM, MODEL_NAME

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingEngine:
    """Generates normalized float embeddings using bge-large-en-v1.5 (1024d).

    This model was chosen because it is *not* trained with Matryoshka or binary
    objectives, making it suitable for studying post-hoc quantization effects
    on an unoptimized float space.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.dim = MODEL_DIM

    def encode_passages(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> NDArray[np.float32]:
        """Encode corpus passages to L2-normalized float32 embeddings (n, 1024).

        No prefix is applied — BGE encodes passages as-is.
        """
        emb = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_queries(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> NDArray[np.float32]:
        """Encode queries to L2-normalized float32 embeddings (n, 1024).

        Prepends the BGE query instruction prefix automatically.
        """
        prefixed = [f"{_QUERY_PREFIX}{t}" for t in texts]
        emb = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )
        return np.asarray(emb, dtype=np.float32)
