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
└── dataset.py       BEIR data loading (SciFact, FiQA, TREC-COVID)

scripts/             Experiment runners (to be implemented)
tests/               Unit tests
docs/                LaTeX source of the thesis
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
```

## Methods

- **Model**: `BAAI/bge-large-en-v1.5` (768d, not Matryoshka-trained)
- **Datasets**: BEIR SciFact, FiQA, TREC-COVID
- **Quantization**: Naive binary (1-bit), TurboQuant 2-bit, TurboQuant 4-bit
- **Dimension Reduction**: PCA to d ∈ {64, 128, 256, 384, 768}
