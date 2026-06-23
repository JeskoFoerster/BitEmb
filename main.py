"""Demo script showcasing BitEmb capabilities with rich output."""

from rich.console import Console
from rich.table import Table

from bitemb.engine import EmbeddingEngine, MATRYOSHKA_DIMS
from bitemb.analysis import analyze_information_loss, analyze_vector_space
from bitemb.benchmark import benchmark_scaling

SAMPLE_TEXTS = [
    "Machine learning models learn patterns from data",
    "Neural networks use layers of interconnected nodes",
    "Binary quantization reduces embeddings to single bits",
    "Hamming distance counts differing bit positions",
    "Cosine similarity measures angle between vectors",
    "Matryoshka embeddings support multiple dimensions",
    "Information retrieval finds relevant documents",
    "Vector databases enable semantic search at scale",
    "The cat sat on the mat",
    "Quantum computing uses qubits for computation",
]

console = Console()


def show_information_loss(engine: EmbeddingEngine) -> None:
    """Display information loss across Matryoshka dimensions."""
    table = Table(title="Information Loss by Dimension")
    table.add_column("Dim", justify="right")
    table.add_column("Cosine Preservation", justify="right")
    table.add_column("Rank Correlation", justify="right")
    table.add_column("Compression", justify="right")

    for dim in MATRYOSHKA_DIMS:
        float_embs = engine.embed_float(SAMPLE_TEXTS, dim=dim)
        bit_packed = engine.embed_bit_sign(SAMPLE_TEXTS, dim=dim)
        loss = analyze_information_loss(float_embs, bit_packed, dim)
        table.add_row(
            str(dim),
            f"{loss['cosine_preservation_mean']:.4f} ± {loss['cosine_preservation_std']:.4f}",
            f"{loss['rank_correlation']:.4f}",
            f"{loss['compression_ratio']:.1f}x",
        )

    console.print(table)


def show_vector_space(engine: EmbeddingEngine) -> None:
    """Display vector space utilization at 768d."""
    float_embs = engine.embed_float(SAMPLE_TEXTS, dim=768)
    stats = analyze_vector_space(float_embs)

    table = Table(title="Vector Space Utilization (768d)")
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")

    table.add_row("Num vectors", str(stats["num_vectors"]))
    table.add_row("Dimension", str(stats["dimension"]))
    table.add_row("Mean pairwise sim", f"{stats['mean_pairwise_sim']:.4f}")
    table.add_row("Std pairwise sim", f"{stats['std_pairwise_sim']:.4f}")
    table.add_row("Effective dim (95%)", str(stats["effective_dim_95pct"]))
    table.add_row("Isotropy score", f"{stats['isotropy_score']:.4f}")

    console.print(table)


def show_benchmark() -> None:
    """Display performance benchmark results."""
    results = benchmark_scaling(engine=None, sizes=[1000, 5000, 10000])

    table = Table(title="Performance Benchmark")
    table.add_column("Docs", justify="right")
    table.add_column("Mode", justify="left")
    table.add_column("Search ms", justify="right")
    table.add_column("Memory", justify="right")
    table.add_column("QPS", justify="right")
    table.add_column("Speedup", justify="right")

    for r in results:
        f, b = r["float"], r["bit"]
        table.add_row(
            f"{f.n_docs:,}", "float", f"{f.search_time_ms:.2f}",
            f"{f.memory_bytes:,}", f"{f.throughput_qps:.0f}", "",
        )
        table.add_row(
            "", "bit", f"{b.search_time_ms:.2f}",
            f"{b.memory_bytes:,}", f"{b.throughput_qps:.0f}", f"{r['speedup']:.1f}x",
        )

    console.print(table)


def main() -> None:
    """Run all demos."""
    console.rule("[bold]BitEmb Demo")
    engine = EmbeddingEngine()

    console.print("\n[bold]1. Information Loss by Dimension[/bold]")
    show_information_loss(engine)

    console.print("\n[bold]2. Vector Space Utilization[/bold]")
    show_vector_space(engine)

    console.print("\n[bold]3. Performance Benchmark[/bold]")
    show_benchmark()


if __name__ == "__main__":
    main()
