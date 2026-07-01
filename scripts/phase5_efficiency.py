"""Phase 5: native runtime and memory efficiency analysis.

Examples:
    python scripts/phase5_efficiency.py --synthetic --max-docs 1000
    python scripts/phase5_efficiency.py --dataset scifact --max-docs 5000
    python -m bitemb.native.build_native
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

from bitemb.config import DATASETS, PCA_DIMS, SEED  # noqa: E402
from bitemb.distance import _sample_pairs  # noqa: E402
from bitemb.efficiency import (  # noqa: E402
    REPRESENTATIONS,
    build_native_layouts,
    dataclasses_to_dicts,
    measure_runtime,
    numpy_payload_memory,
    practical_memory_for_layouts,
    theoretical_memory,
    theoretical_work,
    unavailable_runtime,
)
from bitemb.native import (  # noqa: E402
    NativeBackendUnavailableError,
    NativePackedBackend,
    native_backend_available,
)
from bitemb.plotting import (  # noqa: E402
    plot_phase5_memory_compression_by_dim,
    plot_phase5_memory_theoretical_vs_native,
    plot_phase5_runtime_by_dim,
)
from bitemb.quantization import PCAReducer  # noqa: E402

OUTPUT_DIR = Path("results/phase5")


def _synthetic_embeddings(n: int, dim: int = 768, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    embs = rng.normal(size=(n, dim)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def _load_embeddings(dataset_name: str, max_docs: int | None, synthetic: bool) -> np.ndarray:
    if synthetic:
        n = max_docs or 1000
        return _synthetic_embeddings(n)

    from bitemb.cache import load_or_encode
    from bitemb.dataset import load_beir
    from bitemb.engine import EmbeddingEngine

    ds = load_beir(dataset_name)
    engine = EmbeddingEngine()
    embs = load_or_encode(dataset_name, ds.corpus_texts, engine, show_progress=True)
    if max_docs and max_docs < embs.shape[0]:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(embs.shape[0], size=max_docs, replace=False)
        embs = embs[idx]
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


def _native_runtime_records(
    embeddings: np.ndarray,
    pairs: np.ndarray,
    k: int,
    warmup_runs: int,
    measurement_runs: int,
) -> list[dict]:
    n, dim = embeddings.shape
    layouts = build_native_layouts(embeddings)

    if not native_backend_available():
        note = (
            "Native backend unavailable. Run `python -m bitemb.native.build_native` "
            "after installing a C compiler."
        )
        records = []
        for rep in REPRESENTATIONS:
            records.append(unavailable_runtime(
                representation=rep,
                operation="pairwise_distance",
                implementation="native_packed",
                n_vectors=n,
                dim=dim,
                n_pairs=pairs.shape[0],
                notes=note,
            ))
            records.append(unavailable_runtime(
                representation=rep,
                operation="top_k",
                implementation="native_packed",
                n_vectors=n,
                dim=dim,
                k=k,
                notes=note,
            ))
        return dataclasses_to_dicts(records)

    try:
        backend = NativePackedBackend()
    except NativeBackendUnavailableError as exc:
        note = str(exc)
        return dataclasses_to_dicts([
            unavailable_runtime(
                representation=rep,
                operation="pairwise_distance",
                implementation="native_packed",
                n_vectors=n,
                dim=dim,
                n_pairs=pairs.shape[0],
                notes=note,
            )
            for rep in REPRESENTATIONS
        ])

    runtime = []
    runtime.append(measure_runtime(
        lambda: backend.cosine_distance_pairs(layouts.float32, pairs),
        representation="float32",
        operation="pairwise_distance",
        implementation="native_packed",
        n_vectors=n,
        dim=dim,
        n_pairs=pairs.shape[0],
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        throughput_units=pairs.shape[0],
    ))
    runtime.append(measure_runtime(
        lambda: backend.turboquant_distance_pairs(layouts.tq4, pairs),
        representation="4bit",
        operation="pairwise_distance",
        implementation="native_packed",
        n_vectors=n,
        dim=dim,
        n_pairs=pairs.shape[0],
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        throughput_units=pairs.shape[0],
    ))
    runtime.append(measure_runtime(
        lambda: backend.turboquant_distance_pairs(layouts.tq2, pairs),
        representation="2bit",
        operation="pairwise_distance",
        implementation="native_packed",
        n_vectors=n,
        dim=dim,
        n_pairs=pairs.shape[0],
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        throughput_units=pairs.shape[0],
    ))
    runtime.append(measure_runtime(
        lambda: backend.hamming_distance_pairs(layouts.binary1, pairs),
        representation="1bit",
        operation="pairwise_distance",
        implementation="native_packed",
        n_vectors=n,
        dim=dim,
        n_pairs=pairs.shape[0],
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        throughput_units=pairs.shape[0],
    ))

    runtime.append(measure_runtime(
        lambda: backend.knn_cosine(layouts.float32, k),
        representation="float32",
        operation="top_k",
        implementation="native_packed",
        n_vectors=n,
        dim=dim,
        k=k,
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        throughput_units=n,
    ))
    runtime.append(measure_runtime(
        lambda: backend.knn_turboquant(layouts.tq4, k),
        representation="4bit",
        operation="top_k",
        implementation="native_packed",
        n_vectors=n,
        dim=dim,
        k=k,
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        throughput_units=n,
    ))
    runtime.append(measure_runtime(
        lambda: backend.knn_turboquant(layouts.tq2, k),
        representation="2bit",
        operation="top_k",
        implementation="native_packed",
        n_vectors=n,
        dim=dim,
        k=k,
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        throughput_units=n,
    ))
    runtime.append(measure_runtime(
        lambda: backend.knn_hamming(layouts.binary1, k),
        representation="1bit",
        operation="top_k",
        implementation="native_packed",
        n_vectors=n,
        dim=dim,
        k=k,
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        throughput_units=n,
    ))
    return dataclasses_to_dicts(runtime)


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
        print(f"\n=== Phase 5: {dataset_name} ===")
        embs_full = _load_embeddings(dataset_name, args.max_docs, args.synthetic)
        dims = tuple(d for d in args.dims if d <= embs_full.shape[1])

        for dim in dims:
            print(f"  dim={dim}")
            embs = _embeddings_for_dim(embs_full, dim)
            n = embs.shape[0]
            n_pairs = min(args.n_pairs, n * (n - 1) // 2)
            pairs = _sample_pairs(n, n_pairs=n_pairs, seed=SEED)
            layouts = build_native_layouts(embs)

            theoretical_mem, theoretical_ops = _theoretical_records(n, dim)
            native_mem = dataclasses_to_dicts(practical_memory_for_layouts(layouts))
            numpy_mem = dataclasses_to_dicts(numpy_payload_memory(embs))

            memory_outputs.append({
                "dataset": dataset_name,
                "n_vectors": n,
                "dim": dim,
                "theoretical": theoretical_mem,
                "native_packed": native_mem,
                "python_numpy_reference": numpy_mem,
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
            runtime_records.extend(_native_runtime_records(
                embs,
                pairs,
                args.k,
                args.warmup_runs,
                args.measurement_runs,
            ))
            runtime_outputs.append({
                "dataset": dataset_name,
                "n_vectors": n,
                "dim": dim,
                "n_pairs": n_pairs,
                "k": args.k,
                "results": runtime_records,
            })

    return memory_outputs, runtime_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: native runtime and memory efficiency")
    parser.add_argument("--dataset", choices=list(DATASETS), default="scifact")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic " \
    "normalized embeddings")
    parser.add_argument("--max-docs", type=int, default=1000)
    parser.add_argument("--dims", type=int, nargs="+", default=list(PCA_DIMS))
    parser.add_argument("--n-pairs", type=int, default=10_000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--measurement-runs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    memory_outputs, runtime_outputs = run(args)

    memory_path = args.output / "memory.json"
    runtime_path = args.output / "runtime.json"
    memory_path.write_text(json.dumps(memory_outputs, indent=2), encoding="utf-8")
    runtime_path.write_text(json.dumps(runtime_outputs, indent=2), encoding="utf-8")
    print(f"\nSaved memory results to {memory_path}")
    print(f"Saved runtime results to {runtime_path}")

    fig_dir = args.output / "figures"
    p = plot_phase5_memory_theoretical_vs_native(
        memory_outputs, fig_dir / "memory_theoretical_vs_native.pdf"
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

    if not native_backend_available():
        print("\nNative backend is not built; runtime.json contains unavailable " \
        "native runtime records.")
        print("Build it with: python -m bitemb.native.build_native")


if __name__ == "__main__":
    main()


