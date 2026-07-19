# BitEmb

Empirical analysis of quantization effects on embedding retrieval quality.

## Research Question

Which semantic properties of a float embedding space are preserved under quantization, which are lost, and how does this loss affect retrieval quality?

## Structure

```text
bitemb/              Core library
├── config.py        Experiment parameters
├── engine.py        Embedding generation (BAAI/bge-large-en-v1.5)
├── cache.py         Deterministic embedding cache (corpus + queries)
├── quantization.py  Binary + TurboQuant (2-bit, 4-bit) + PCA reduction
├── dataset.py       BEIR data loading (SciFact, FiQA, TREC-COVID)
├── analysis.py      Phase 1: float space characterization
├── distance.py      Phase 2: pairwise distance & distortion analysis
├── neighborhood.py  Phase 3: neighborhood overlap & trustworthiness
├── retrieval.py     Phase 4: exact BEIR retrieval evaluation
├── efficiency.py    Phase 5: runtime & memory efficiency analysis
└── plotting.py      Publication-quality figures (matplotlib)

scripts/             Experiment runners
├── phase1_characterization.py
├── phase2_distance_analysis.py
├── phase3_neighborhood.py
├── phase4_retrieval.py
└── phase5_efficiency.py

tests/               Unit tests
```

## Setup

```bash
pip install -e ".[dev]"
```

Phase 5 uses vectorized NumPy runtime measurements and does not require a C compiler, CFFI build step, or Microsoft C++ Build Tools. Generated `results/phase5/` outputs are ignored by Git; comparison runs are stored below that directory, e.g. `results/phase5/scaling_n/` and `results/phase5/scaling_dim/`.

## Usage

```bash
make test-fast   # unit tests (no model download)
make test        # all tests including integration
make lint        # ruff + mypy

# Run Phase 1 (requires model download + dataset)
python scripts/phase1_characterization.py --dataset scifact
python scripts/phase1_characterization.py --all
python scripts/phase1_characterization.py --all --max-docs 10000

# Run Phase 2 (requires model download + dataset)
python scripts/phase2_distance_analysis.py --dataset scifact
python scripts/phase2_distance_analysis.py --all
python scripts/phase2_distance_analysis.py --all --max-docs 10000

# Run Phase 3 (requires model download + dataset)
python scripts/phase3_neighborhood.py --dataset scifact
python scripts/phase3_neighborhood.py --all
python scripts/phase3_neighborhood.py --all --max-docs 5000 # recommended

# Run Phase 4 (requires model download + dataset)
python scripts/phase4_retrieval.py --dataset scifact
python scripts/phase4_retrieval.py --all
python scripts/phase4_retrieval.py --all --max-docs 10000

# Run Phase 5 smoke test (no dataset download)
python scripts/phase5_efficiency.py --synthetic --max-docs 1000 --dims 64

# Run Phase 5 on a real dataset
python scripts/phase5_efficiency.py --dataset scifact --max-docs 5000
```

## Embedding Cache

Corpus embeddings are cached to `cache/embeddings/` on first run. Subsequent runs (including across phases) load from cache instead of re-encoding.
The cache always stores the full corpus. `--max-docs` subsamples from the cached matrix after loading.
This holds regardless of run order: even if the first run uses `--max-docs`, the full corpus is still encoded and cached.

Query embeddings for Phase 4 are cached separately in `cache/query_embeddings/`, because BGE uses a query instruction prefix. Delete `cache/` to force re-encoding.

## Documentation

Detailed documentation for each phase is in `docs/`:

| File | Description |
|------|-------------|
| [`docs/overview.md`](docs/overview.md) | Project overview, experiment structure, datasets |
| [`docs/phase1.md`](docs/phase1.md) | Float space characterization: norms, skewness, kurtosis, intrinsic dimensionality |
| [`docs/phase2.md`](docs/phase2.md) | Distance analysis: Pearson r, Spearman ρ, MAE, RMSE |
| [`docs/phase3.md`](docs/phase3.md) | Neighborhood preservation: overlap, trustworthiness |
| [`docs/phase4.md`](docs/phase4.md) | Retrieval evaluation: NDCG@10, Recall@10/100, MRR, Wilcoxon tests |
| [`docs/phase5.md`](docs/phase5.md) | Runtime and memory efficiency: theoretical estimates and NumPy-vectorized measurements |

## Methods

- **Model**: `BAAI/bge-large-en-v1.5` (1024d, not Matryoshka-trained)
- **Datasets**: BEIR SciFact, FiQA, TREC-COVID
- **Quantization**: Naive binary (1-bit), TurboQuant 2-bit, TurboQuant 4-bit
- **Dimension Reduction**: PCA to d ∈ {64, 128, 256, 384, 512, 768, 1024}
