"""Shared test fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_texts():
    return [
        "Machine learning models learn patterns from data",
        "Neural networks use layers of interconnected nodes",
        "Binary quantization reduces embeddings to single bits",
        "The cat sat on the mat",
        "Quantum computing uses qubits for computation",
    ]


@pytest.fixture
def random_float_embs():
    """10 normalized random 1024-d vectors."""
    rng = np.random.default_rng(42)
    embs = rng.normal(size=(10, 1024)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return embs
