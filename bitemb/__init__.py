"""BitEmb – Quantization effects on embedding retrieval quality."""

from bitemb.config import DATASETS, MODEL_DIM, MODEL_NAME
from bitemb.dataset import BeirDataset, load_beir
from bitemb.engine import EmbeddingEngine
from bitemb.quantization import PCAReducer, TurboQuantIndex, binarize, turboquant_encode

__all__ = [
    "MODEL_NAME",
    "MODEL_DIM",
    "DATASETS",
    "EmbeddingEngine",
    "binarize",
    "turboquant_encode",
    "TurboQuantIndex",
    "PCAReducer",
    "load_beir",
    "BeirDataset",
]
