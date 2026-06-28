"""BitEmb – Quantization effects on embedding retrieval quality."""

from bitemb.analysis import (
    compute_dimension_stats,
    compute_intrinsic_dimensionality,
    compute_norm_distribution,
)
from bitemb.config import DATASETS, MODEL_DIM, MODEL_NAME
from bitemb.dataset import BeirDataset, load_beir
from bitemb.distance import DistortionResult, RawDistances, compute_distance_distortion, compute_raw_distances
from bitemb.engine import EmbeddingEngine
from bitemb.plotting import (
    plot_cumulative_variance,
    plot_distance_scatter,
    plot_distortion_heatmap,
    plot_distortion_pareto,
    plot_error_histogram,
    plot_variance_spectrum,
)
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
    "compute_norm_distribution",
    "compute_dimension_stats",
    "compute_intrinsic_dimensionality",
    "compute_distance_distortion",
    "compute_raw_distances",
    "DistortionResult",
    "RawDistances",
    "plot_cumulative_variance",
    "plot_variance_spectrum",
    "plot_distance_scatter",
    "plot_error_histogram",
    "plot_distortion_heatmap",
    "plot_distortion_pareto",
]
