# BitEmb

Empirical analysis of quantization effects on embedding retrieval quality.

## Research Question

Which semantic properties of a float embedding space are preserved under quantization, which are lost, and how does this loss affect retrieval quality?

## Structure

```
bitemb/              Core library
├── config.py        Experiment parameters
├── engine.py        Embedding generation (BAAI/bge-large-en-v1.5)
├── quantization.py  Binary + TurboQuant (2-bit, 4-bit) + PCA reduction
├── dataset.py       BEIR data loading (SciFact, FiQA, TREC-COVID)
├── analysis.py      Phase 1: float space characterization
├── distance.py      Phase 2: pairwise distance & distortion analysis
└── plotting.py      Publication-quality figures (matplotlib)

scripts/             Experiment runners
├── phase1_characterization.py
└── phase2_distance_analysis.py

tests/               Unit tests
```

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

```bash
make test-fast   # unit tests (no model download)
make test        # all tests including integration
make lint        # ruff + mypy

# Run Phase 1 (requires model download + dataset)
python scripts/phase1_characterization.py --dataset scifact
python scripts/phase1_characterization.py --all
python scripts/phase1_characterization.py --all --max-docs 10000  # subsample large corpora

# Run Phase 2 (requires model download + dataset)
python scripts/phase2_distance_analysis.py --dataset scifact
python scripts/phase2_distance_analysis.py --all
python scripts/phase2_distance_analysis.py --all --max-docs 10000
```

## Methods

- **Model**: `BAAI/bge-large-en-v1.5` (768d, not Matryoshka-trained)
- **Datasets**: BEIR SciFact, FiQA, TREC-COVID
- **Quantization**: Naive binary (1-bit), TurboQuant 2-bit, TurboQuant 4-bit
- **Dimension Reduction**: PCA to d ∈ {64, 128, 256, 384, 768}
