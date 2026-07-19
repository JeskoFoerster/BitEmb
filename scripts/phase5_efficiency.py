"""Phase 5: NumPy runtime and memory efficiency analysis.

Examples:
    python scripts/phase5_efficiency.py --synthetic --max-docs 1000
    python scripts/phase5_efficiency.py --dataset scifact --max-docs 5000
    python scripts/phase5_efficiency.py --all --max-docs-list 250 500 1000 \
        --output results/phase5/scaling_n
    python scripts/phase5_efficiency.py --all --max-docs 1000 \
        --dims 64 128 256 384 1024 --output results/phase5/scaling_dim
    python scripts/phase5_efficiency.py --dataset scifact --full-corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Work around a multiprocess shutdown bug on Python 3.12 that can be triggered
# by transitive scientific/ML imports. Phase 5 does not use multiprocess.
try:
    import multiprocess.resource_tracker as _mp_resource_tracker

    _mp_resource_tracker.ResourceTracker.__del__ = lambda self: None
except Exception:
    pass

import numpy as np  # noqa: E402

from bitemb.config import DATASETS, MODEL_NAME, PCA_DIMS, SEED  # noqa: E402
from bitemb.distance import _sample_pairs  # noqa: E402
from bitemb.efficiency import (  # noqa: E402
    REPRESENTATIONS,
    build_numpy_layouts,
    cosine_distance_pairs_numpy,
    dataclasses_to_dicts,
    hamming_distance_pairs_numpy,
    knn_cosine_numpy,
    knn_hamming_numpy,
    knn_turboquant_numpy,
    measure_runtime,
    practical_memory_for_layouts,
    theoretical_memory,
    theoretical_work,
    turboquant_distance_pairs_numpy,
)
from bitemb.plotting import (  # noqa: E402
    plot_phase5_memory_compression_by_dim,
    plot_phase5_memory_theoretical_vs_numpy,
    plot_phase5_runtime_by_dim,
)
from bitemb.quantization import PCAReducer  # noqa: E402

OUTPUT_DIR = Path("results/phase5")
PHASE5_CACHE_DIR = Path("cache/phase5_embeddings")


def _model_slug(model_name: str) -> str:
    return "_".join(part for part in model_name.lower().replace("/", "_").split("_") if part)


def _phase5_cache_path(dataset_name: str, max_docs: int) -> Path:
    filename = f"{_model_slug(MODEL_NAME)}_{dataset_name}_max{max_docs}_seed{SEED}.npy"
    return PHASE5_CACHE_DIR / filename


def _synthetic_embeddings(n: int, dim: int = 1024, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    embs = rng.normal(size=(n, dim)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def _load_embeddings(dataset_name: str, max_docs: int | None, synthetic: bool) -> np.ndarray:
    if synthetic:
        n = max_docs or 1000
        return _synthetic_embeddings(n)

    from bitemb.cache import _cache_path, load_or_encode
    from bitemb.dataset import load_beir
    from bitemb.engine import EmbeddingEngine

    ds = load_beir(dataset_name)
    n_total = len(ds.corpus_texts)
    engine = EmbeddingEngine()

    if max_docs is None or max_docs >= n_total:
        embs = load_or_encode(dataset_name, ds.corpus_texts, engine, show_progress=True)
        return np.ascontiguousarray(embs, dtype=np.float32)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(n_total, size=max_docs, replace=False)

    full_cache = _cache_path(dataset_name)
    if full_cache.exists():
        embs = load_or_encode(dataset_name, ds.corpus_texts, engine, show_progress=True)
        return np.ascontiguousarray(embs[idx], dtype=np.float32)

    subset_cache = _phase5_cache_path(dataset_name, max_docs)
    if subset_cache.exists():
        embs = np.load(subset_cache)
        if embs.shape[0] != max_docs:
            raise ValueError(
                f"Phase 5 cache mismatch for '{dataset_name}': cached {embs.shape[0]} rows, "
                f"expected {max_docs}. Delete {subset_cache} to re-encode."
            )
        print(f"  Loaded Phase 5 subsample embeddings from {subset_cache}")
        return np.ascontiguousarray(embs, dtype=np.float32)

    texts = [ds.corpus_texts[i] for i in idx]
    print(f"  Encoding Phase 5 subsample: {max_docs} of {n_total} documents...")
    embs = engine.encode_passages(texts, show_progress=True)
    subset_cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(subset_cache, embs)
    print(f"  Cached Phase 5 subsample embeddings to {subset_cache}")
    return np.ascontiguousarray(embs, dtype=np.float32)


def _embeddings_for_dim(embeddings: np.ndarray, dim: int) -> np.ndarray:
    if dim == embeddings.shape[1]:
        return np.ascontiguousarray(embeddings, dtype=np.float32)
    pca = PCAReducer(n_components=dim).fit(embeddings)
    return np.ascontiguousarray(pca.transform(embeddings), dtype=np.float32)


def _theoretical_records(n_vectors: int, dim: int) -> tuple[list[dict], list[dict]]:
    memory = [theoretical_memory(n_vectors, dim, rep) for rep in REPRESENTATIONS]
    work = [theoretical_work(dim, rep) for rep in REPRESENTATIONS]
    return dataclasses_to_dicts(memory), dataclasses_to_dicts(work)



def _numpy_runtime_records(
    embeddings: np.ndarray,
    pairs: np.ndarray,
    k: int,
    warmup_runs: int,
    measurement_runs: int,
) -> list[dict]:
    n, dim = embeddings.shape
    layouts = build_numpy_layouts(embeddings)

    runtime = []

    # ------------------ PAIRWISE DISTANCE ------------------
    # float32
    runtime.append(measure_runtime(
        lambda: cosine_distance_pairs_numpy(layouts.float32, pairs),
        representation="float32",
        operation="pairwise_distance",
        implementation="numpy_vectorized",
        n_vectors=n,
        dim=dim,
        n_pairs=pairs.shape[0],
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        throughput_units=pairs.shape[0],
        notes="NumPy einsum over sampled pairs",
    ))
    # float16
    runtime.append(measure_runtime(
        lambda: cosine_distance_pairs_numpy(layouts.float16, pairs),
        representation="16bit",
        operation="pairwise_distance",
        implementation="numpy_vectorized",
        n_vectors=n,
        dim=dim,
        n_pairs=pairs.shape[0],
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        throughput_units=pairs.shape[0],
        notes="NumPy einsum over sampled pairs (float16)",
    ))

    # Multi-bit Naive
    for b in (8, 4, 2):
        layout_name = f"naive_{b}bit"
        layout_obj = getattr(layouts, f"naive_{b}bit")
        runtime.append(measure_runtime(
            lambda lo=layout_obj: turboquant_distance_pairs_numpy(lo, pairs),
            representation=layout_name,
            operation="pairwise_distance",
            implementation="numpy_vectorized",
            n_vectors=n,
            dim=dim,
            n_pairs=pairs.shape[0],
            warmup_runs=warmup_runs,
            measurement_runs=measurement_runs,
            throughput_units=pairs.shape[0],
            notes=f"NumPy vectorized distance over packed {b}-bit layout (naive)",
        ))

    # Multi-bit TurboQuant
    for b in (8, 4, 2):
        layout_name = f"tq_{b}bit"
        layout_obj = getattr(layouts, f"tq_{b}bit")
        runtime.append(measure_runtime(
            lambda lo=layout_obj: turboquant_distance_pairs_numpy(lo, pairs),
            representation=layout_name,
            operation="pairwise_distance",
            implementation="numpy_vectorized",
            n_vectors=n,
            dim=dim,
            n_pairs=pairs.shape[0],
            warmup_runs=warmup_runs,
            measurement_runs=measurement_runs,
            throughput_units=pairs.shape[0],
            notes=f"NumPy vectorized distance over packed {b}-bit layout (rotated)",
        ))

    # 1-bit Naive
    runtime.append(measure_runtime(
        lambda: hamming_distance_pairs_numpy(layouts.naive_1bit, pairs),
        representation="naive_1bit",
        operation="pairwise_distance",
        implementation="numpy_vectorized",
        n_vectors=n,
        dim=dim,
        n_pairs=pairs.shape[0],
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        throughput_units=pairs.shape[0],
        notes="NumPy vectorized XOR plus lookup-table popcount (naive)",
    ))
    # 1-bit TurboQuant
    runtime.append(measure_runtime(
        lambda: hamming_distance_pairs_numpy(layouts.tq_1bit, pairs),
        representation="tq_1bit",
        operation="pairwise_distance",
        implementation="numpy_vectorized",
        n_vectors=n,
        dim=dim,
        n_pairs=pairs.shape[0],
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        throughput_units=pairs.shape[0],
        notes="NumPy vectorized XOR plus lookup-table popcount (rotated)",
    ))

    # ------------------ TOP-K SEARCH ------------------
    # float32
    runtime.append(measure_runtime(
        lambda: knn_cosine_numpy(layouts.float32, k),
        representation="float32",
        operation="top_k",
        implementation="numpy_vectorized",
        n_vectors=n,
        dim=dim,
        k=k,
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        throughput_units=n,
        notes="NumPy matrix multiplication plus argpartition",
    ))
    # float16
    runtime.append(measure_runtime(
        lambda: knn_cosine_numpy(layouts.float16, k),
        representation="16bit",
        operation="top_k",
        implementation="numpy_vectorized",
        n_vectors=n,
        dim=dim,
        k=k,
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        throughput_units=n,
        notes="NumPy matrix multiplication plus argpartition (float16)",
    ))

    # Multi-bit Naive
    for b in (8, 4, 2):
        layout_name = f"naive_{b}bit"
        layout_obj = getattr(layouts, f"naive_{b}bit")
        runtime.append(measure_runtime(
            lambda lo=layout_obj: knn_turboquant_numpy(lo, k),
            representation=layout_name,
            operation="top_k",
            implementation="numpy_vectorized",
            n_vectors=n,
            dim=dim,
            k=k,
            warmup_runs=warmup_runs,
            measurement_runs=measurement_runs,
            throughput_units=n,
            notes=f"Batched NumPy top-k over dequantized {b}-bit values (naive)",
        ))

    # Multi-bit TurboQuant
    for b in (8, 4, 2):
        layout_name = f"tq_{b}bit"
        layout_obj = getattr(layouts, f"tq_{b}bit")
        runtime.append(measure_runtime(
            lambda lo=layout_obj: knn_turboquant_numpy(lo, k),
            representation=layout_name,
            operation="top_k",
            implementation="numpy_vectorized",
            n_vectors=n,
            dim=dim,
            k=k,
            warmup_runs=warmup_runs,
            measurement_runs=measurement_runs,
            throughput_units=n,
            notes=f"Batched NumPy top-k over dequantized {b}-bit values (rotated)",
        ))

    # 1-bit Naive
    runtime.append(measure_runtime(
        lambda: knn_hamming_numpy(layouts.naive_1bit, k),
        representation="naive_1bit",
        operation="top_k",
        implementation="numpy_vectorized",
        n_vectors=n,
        dim=dim,
        k=k,
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        throughput_units=n,
        notes="Batched NumPy XOR plus lookup-table popcount and argpartition (naive)",
    ))
    # 1-bit TurboQuant
    runtime.append(measure_runtime(
        lambda: knn_hamming_numpy(layouts.tq_1bit, k),
        representation="tq_1bit",
        operation="top_k",
        implementation="numpy_vectorized",
        n_vectors=n,
        dim=dim,
        k=k,
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        throughput_units=n,
        notes="Batched NumPy XOR plus lookup-table popcount and argpartition (rotated)",
    ))

    return dataclasses_to_dicts(runtime)


def _sample_sizes(args: argparse.Namespace) -> list[int | None]:
    sizes: list[int | None] = []
    if args.max_docs_list:
        sizes.extend(args.max_docs_list)
    else:
        sizes.append(args.max_docs)
    if args.full_corpus:
        sizes.append(None)
    # Preserve order while removing duplicates. None means full corpus.
    deduped: list[int | None] = []
    for size in sizes:
        if size not in deduped:
            deduped.append(size)
    return deduped


def _sample_label(max_docs: int | None, n_vectors: int) -> str:
    return "full" if max_docs is None else f"max_docs={max_docs}"


def run(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    if args.synthetic:
        datasets = ["synthetic"]
    elif args.all:
        datasets = list(DATASETS)
    else:
        datasets = [args.dataset]
    memory_outputs: list[dict] = []
    runtime_outputs: list[dict] = []

    for dataset_name in datasets:
        for max_docs in _sample_sizes(args):
            sample_hint = "full corpus" if max_docs is None else f"max_docs={max_docs}"
            print(f"\n=== Phase 5: {dataset_name} ({sample_hint}) ===")
            embs_full = _load_embeddings(dataset_name, max_docs, args.synthetic)
            dims = tuple(
                d for d in args.dims
                if d <= embs_full.shape[1] and (d == embs_full.shape[1] or d <= embs_full.shape[0])
            )
            if not dims:
                print("  No requested PCA dimensions are valid for this sample size; skipping.")
                continue

            for dim in dims:
                print(f"  dim={dim}")
                embs = _embeddings_for_dim(embs_full, dim)
                n = embs.shape[0]
                n_pairs = min(args.n_pairs, n * (n - 1) // 2)
                pairs = _sample_pairs(n, n_pairs=n_pairs, seed=SEED)
                layouts = build_numpy_layouts(embs)

                theoretical_mem, theoretical_ops = _theoretical_records(n, dim)
                numpy_mem = dataclasses_to_dicts(practical_memory_for_layouts(layouts))
                sample_label = _sample_label(max_docs, n)

                memory_outputs.append({
                    "dataset": dataset_name,
                    "sample": sample_label,
                    "max_docs": max_docs,
                    "n_vectors": n,
                    "dim": dim,
                    "theoretical": theoretical_mem,
                    "numpy_vectorized": numpy_mem,
                })

                runtime_records = []
                runtime_records.extend({
                    "representation": rec["representation"],
                    "operation": "pairwise_distance",
                    "implementation": "theoretical",
                    "n_vectors": n,
                    "dim": dim,
                    "n_pairs": n_pairs,
                    "k": None,
                    "work_model": rec,
                } for rec in theoretical_ops)
                runtime_records.extend(_numpy_runtime_records(
                    embs,
                    pairs,
                    args.k,
                    args.warmup_runs,
                    args.measurement_runs,
                ))
                runtime_outputs.append({
                    "dataset": dataset_name,
                    "sample": sample_label,
                    "max_docs": max_docs,
                    "n_vectors": n,
                    "dim": dim,
                    "n_pairs": n_pairs,
                    "k": args.k,
                    "results": runtime_records,
                })

    return memory_outputs, runtime_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: NumPy runtime and memory efficiency")
    parser.add_argument("--dataset", choices=list(DATASETS), default="scifact")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic " \
    "normalized embeddings")
    parser.add_argument("--max-docs", type=int, default=1000)
    parser.add_argument(
        "--max-docs-list",
        type=int,
        nargs="+",
        help="Run Phase 5 for several controlled corpus-size caps in one invocation.",
    )
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Additionally run each selected dataset without a max-docs cap.",
    )
    parser.add_argument("--dims", type=int, nargs="+", default=list(PCA_DIMS))
    parser.add_argument("--n-pairs", type=int, default=10_000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--measurement-runs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    memory_outputs, runtime_outputs = run(args)

    memory_path = args.output / "memory_metrics.json"
    runtime_path = args.output / "runtime_metrics.json"
    memory_path.write_text(json.dumps(memory_outputs, indent=2), encoding="utf-8")
    runtime_path.write_text(json.dumps(runtime_outputs, indent=2), encoding="utf-8")
    print(f"\nSaved memory results to {memory_path}")
    print(f"Saved runtime results to {runtime_path}")

    fig_dir = args.output / "figures"
    p = plot_phase5_memory_theoretical_vs_numpy(
        memory_outputs, fig_dir / "memory_theoretical_vs_numpy.pdf"
    )
    print(f"Saved memory comparison plot to {p}")
    p = plot_phase5_memory_compression_by_dim(
        memory_outputs, fig_dir / "memory_compression_by_dim.pdf"
    )
    print(f"Saved compression plot to {p}")
    p = plot_phase5_runtime_by_dim(
        runtime_outputs, fig_dir / "runtime_pairs_by_dim.pdf", operation="pairwise_distance"
    )
    print(f"Saved pairwise runtime plot to {p}")
    p = plot_phase5_runtime_by_dim(
        runtime_outputs, fig_dir / "runtime_knn_by_dim.pdf", operation="top_k"
    )
    print(f"Saved top-k runtime plot to {p}")



if __name__ == "__main__":
    main()


