"""EmbeddingEngine – Float and binary embedding generation."""

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
MATRYOSHKA_DIMS = [768, 512, 256, 128, 64]


class EmbeddingEngine:
    """Generates float and binary embeddings using SentenceTransformer models."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """Initialize the embedding engine.

        Args:
            model_name: HuggingFace model identifier.
        """
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed_float(self, texts: list[str], dim: int | None = None) -> NDArray[np.float32]:
        """Encode texts to normalized float embeddings.

        Args:
            texts: Input texts to embed.
            dim: Optional dimensionality truncation.

        Returns:
            Normalized float32 embeddings of shape (n, dim).
        """
        emb = self.model.encode(texts, normalize_embeddings=True)
        if dim and dim < emb.shape[1]:
            emb = emb[:, :dim]
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb.astype(np.float32)

    def embed_bit_sign(self, texts: list[str], dim: int | None = None) -> NDArray[np.uint8]:
        """Binary embeddings using sign threshold at 0.

        Args:
            texts: Input texts to embed.
            dim: Optional dimensionality truncation.

        Returns:
            Packed uint8 binary embeddings.
        """
        emb = self.embed_float(texts, dim)
        return self._pack_bits(emb > 0)

    def embed_bit_matryoshka(self, texts: list[str], dim: int = 768) -> NDArray[np.uint8]:
        """Binary embeddings with Matryoshka truncation then sign.

        Args:
            texts: Input texts to embed.
            dim: Truncation dimension (should be in MATRYOSHKA_DIMS).

        Returns:
            Packed uint8 binary embeddings.
        """
        emb = self.embed_float(texts, dim)
        return self._pack_bits(emb > 0)

    def embed_bit_median(self, texts: list[str], dim: int | None = None) -> NDArray[np.uint8]:
        """Binary embeddings using per-dimension median threshold.

        Args:
            texts: Input texts to embed.
            dim: Optional dimensionality truncation.

        Returns:
            Packed uint8 binary embeddings.
        """
        emb = self.embed_float(texts, dim)
        thresholds = np.median(emb, axis=0)
        return self._pack_bits(emb > thresholds)

    def embed_bit_mean(self, texts: list[str], dim: int | None = None) -> NDArray[np.uint8]:
        """Binary embeddings using per-dimension mean threshold.

        Args:
            texts: Input texts to embed.
            dim: Optional dimensionality truncation.

        Returns:
            Packed uint8 binary embeddings.
        """
        emb = self.embed_float(texts, dim)
        thresholds = np.mean(emb, axis=0)
        return self._pack_bits(emb > thresholds)

    def embed_bit_optimal(
        self, texts: list[str], dim: int | None = None
    ) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
        """Binary embeddings with per-dimension optimal percentile threshold.

        Searches percentiles [30, 40, 50, 60, 70] to maximize correlation
        between float cosine similarity and Hamming similarity.

        Args:
            texts: Input texts to embed.
            dim: Optional dimensionality truncation.

        Returns:
            Tuple of (packed binary embeddings, optimal thresholds per dim).
        """
        emb = self.embed_float(texts, dim)
        n, d = emb.shape
        percentiles = [30, 40, 50, 60, 70]
        # Float cosine sim (already normalized)
        float_sim = emb @ emb.T
        best_thresholds = np.zeros(d)
        best_bits = np.zeros((n, d), dtype=bool)

        for i in range(d):
            best_corr = -1.0
            for p in percentiles:
                t = np.percentile(emb[:, i], p)
                bits_col = emb[:, i] > t
                # Hamming similarity for this dimension
                agree = bits_col[:, None] == bits_col[None, :]
                corr = np.corrcoef(float_sim.ravel(), agree.astype(float).ravel())[0, 1]
                if corr > best_corr:
                    best_corr = corr
                    best_thresholds[i] = t
                    best_bits[:, i] = bits_col

        return self._pack_bits(best_bits), best_thresholds

    def embed_bit_weighted(self, texts: list[str], dim: int | None = None) -> NDArray[np.uint8]:
        """Binary embeddings with low-confidence bits set to median.

        Bits where |x| < 25th percentile of absolute values are considered
        low-confidence and set to the median threshold instead.

        Args:
            texts: Input texts to embed.
            dim: Optional dimensionality truncation.

        Returns:
            Packed uint8 binary embeddings.
        """
        emb = self.embed_float(texts, dim)
        abs_emb = np.abs(emb)
        threshold_25 = np.percentile(abs_emb, 25, axis=0)
        low_confidence = abs_emb < threshold_25
        bits = emb > 0
        median_bits = emb > np.median(emb, axis=0)
        bits[low_confidence] = median_bits[low_confidence]
        return self._pack_bits(bits)

    def get_raw_float(self, texts: list[str], dim: int | None = None) -> NDArray[np.float32]:
        """Alias for embed_float.

        Args:
            texts: Input texts to embed.
            dim: Optional dimensionality truncation.

        Returns:
            Normalized float32 embeddings.
        """
        return self.embed_float(texts, dim)

    @staticmethod
    def _pack_bits(bool_array: NDArray) -> NDArray[np.uint8]:
        """Pack boolean array to uint8 bitwise.

        Args:
            bool_array: Boolean array of shape (n, d).

        Returns:
            Packed uint8 array of shape (n, ceil(d/8)).
        """
        return np.packbits(bool_array.astype(np.uint8), axis=1)

    @staticmethod
    def unpack_bits(packed: NDArray[np.uint8], dim: int) -> NDArray[np.uint8]:
        """Unpack uint8 bit array back to binary values.

        Args:
            packed: Packed uint8 array.
            dim: Original dimensionality.

        Returns:
            Unpacked binary array of shape (n, dim).
        """
        return np.unpackbits(packed, axis=1)[:, :dim]
