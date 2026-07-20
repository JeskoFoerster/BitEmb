"""Central configuration for BitEmb experiments."""

# Embedding model (BGE-large, 1024 dimensions, no Matryoshka training)
MODEL_NAME = "BAAI/bge-large-en-v1.5"
MODEL_DIM = 1024

# BEIR datasets used for evaluation
DATASETS = ("scifact", "fiqa", "trec-covid")

# PCA dimension reduction targets (Section 3.3.3 of Methodik)
PCA_DIMS = (64, 128, 256, 384, 512, 768, 1024)

# Quantization bit depths
BIT_DEPTHS = (1, 2, 4)

# Random seed for reproducibility (rotation matrix, PCA)
SEED = 42
