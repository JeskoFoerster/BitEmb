"""Reproduction tests for the Phase 6 scenario analysis.

Phase 6 does not run a new evaluation. It re-derives every published number from
the Phase 4 retrieval metrics and the Phase 5 memory metrics via simple,
deterministic arithmetic (relative quality, compression ratio, linear RAM
scaling). These tests recompute the core numbers independently from the raw JSON
inputs and assert that the committed ``results/phase6/scenarios_summary.json``
matches. If someone edits the summary by hand, changes the upstream metrics, or
breaks the scaling formula, these tests fail.

The tests intentionally avoid importing the plotting script so they do not depend
on matplotlib or write any files; they only read the committed JSON artifacts.
"""

import json
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = BASE_DIR / "results/phase4/retrieval_metrics.json"
MEMORY_PATH = BASE_DIR / "results/phase5/memory_metrics.json"
SUMMARY_PATH = BASE_DIR / "results/phase6/scenarios_summary.json"

# Scenario -> (representation, dim), mirroring scripts/phase6_scenarios.py.
SCENARIOS = {
    "Baseline": ("float32", 1024),
    "Enterprise Precision": ("tq_4bit", 1024),
    "Business Sweet Spot": ("tq_2bit", 1024),
    "Edge / Mobile": ("tq_1bit", 768),
}

MOBILE_VECTOR_COUNTS = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 5_000_000]
MOBILE_DIMS = [64, 128, 256, 384, 512, 768, 1024]

_MIB = 1024 * 1024
_GIB = 1024 * 1024 * 1024

pytestmark = pytest.mark.skipif(
    not (METRICS_PATH.exists() and MEMORY_PATH.exists() and SUMMARY_PATH.exists()),
    reason="Phase 4/5/6 result artifacts not available",
)


@pytest.fixture(scope="module")
def raw_maps():
    """Build (quality_map, memory_map) from the SciFact raw metrics."""
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics_data = json.load(f)
    with open(MEMORY_PATH, encoding="utf-8") as f:
        memory_data = json.load(f)

    scifact = next(e["results"] for e in metrics_data if e["dataset"] == "scifact")
    quality_map = {(r["representation"], r["dim"]): r["ndcg_at_10"] for r in scifact}

    memory_map = {}
    for entry in memory_data:
        if entry["dataset"] == "scifact":
            dim = entry["dim"]
            for nv in entry["numpy_vectorized"]:
                memory_map[(nv["representation"], dim)] = nv["total_bytes"] / entry["n_vectors"]

    return quality_map, memory_map


@pytest.fixture(scope="module")
def summary():
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_summary_has_all_scenarios(summary):
    assert set(summary) == set(SCENARIOS)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_scenario_core_numbers_match_raw_data(scenario, summary, raw_maps):
    quality_map, memory_map = raw_maps
    rep, dim = SCENARIOS[scenario]

    q_base = quality_map[("float32", 1024)]
    c_base = memory_map[("float32", 1024)]

    expected_q = quality_map[(rep, dim)]
    expected_bytes = memory_map[(rep, dim)]
    expected_rel_q = (expected_q / q_base) * 100.0
    expected_comp = c_base / expected_bytes

    entry = summary[scenario]
    assert entry["representation"] == rep
    assert entry["dim"] == dim
    assert entry["ndcg_at_10"] == pytest.approx(expected_q)
    assert entry["relative_quality_pct"] == pytest.approx(expected_rel_q)
    assert entry["quality_loss_pct"] == pytest.approx(100.0 - expected_rel_q)
    assert entry["bytes_per_vector"] == pytest.approx(expected_bytes)
    assert entry["compression_ratio"] == pytest.approx(expected_comp)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_scenario_linear_ram_scaling(scenario, summary):
    """RAM figures must be a linear function of bytes/vector and count."""
    entry = summary[scenario]
    b = entry["bytes_per_vector"]
    assert entry["ram_100k_mib"] == pytest.approx((b * 100_000) / _MIB)
    assert entry["ram_1m_gib"] == pytest.approx((b * 1_000_000) / _GIB)
    assert entry["ram_10m_gib"] == pytest.approx((b * 10_000_000) / _GIB)


def test_aws_savings_are_consistent(summary):
    base_cost = summary["Baseline"]["aws_monthly_usd"]
    assert summary["Baseline"]["aws_savings_pct"] == pytest.approx(0.0)

    for scenario in ("Enterprise Precision", "Business Sweet Spot", "Edge / Mobile"):
        entry = summary[scenario]
        cost = entry["aws_monthly_usd"]
        assert entry["aws_annual_usd"] == pytest.approx(cost * 12)
        assert entry["aws_savings_monthly_usd"] == pytest.approx(base_cost - cost)
        assert entry["aws_savings_pct"] == pytest.approx((1.0 - cost / base_cost) * 100.0)


def test_aws_pricing_provenance_present(summary):
    """Every scenario must document the pricing region and retrieval date."""
    for scenario, entry in summary.items():
        assert entry.get("aws_pricing_region"), scenario
        assert entry.get("aws_pricing_retrieved"), scenario


def test_published_headline_numbers(summary):
    """Guard the exact numbers cited in the report and docs."""
    assert summary["Enterprise Precision"]["relative_quality_pct"] == pytest.approx(99.5, abs=0.1)
    assert summary["Enterprise Precision"]["compression_ratio"] == pytest.approx(7.8, abs=0.05)
    assert summary["Business Sweet Spot"]["relative_quality_pct"] == pytest.approx(98.1, abs=0.1)
    assert summary["Business Sweet Spot"]["compression_ratio"] == pytest.approx(15.0, abs=0.05)
    assert summary["Edge / Mobile"]["relative_quality_pct"] == pytest.approx(93.9, abs=0.1)
    assert summary["Edge / Mobile"]["compression_ratio"] == pytest.approx(42.7, abs=0.05)


def test_mobile_by_dim_matches_raw_data(summary, raw_maps):
    quality_map, memory_map = raw_maps
    q_base = quality_map[("float32", 1024)]
    c_base = memory_map[("float32", 1024)]

    by_dim = {e["dim"]: e for e in summary["Edge / Mobile"]["sub_10m_by_dim"]}
    assert set(by_dim) == set(MOBILE_DIMS)

    for dim, entry in by_dim.items():
        expected_bytes = memory_map[("tq_1bit", dim)]
        expected_rel_q = (quality_map[("tq_1bit", dim)] / q_base) * 100.0
        assert entry["bytes_per_vector"] == pytest.approx(expected_bytes)
        assert entry["compression_ratio"] == pytest.approx(c_base / expected_bytes)
        assert entry["relative_quality_pct"] == pytest.approx(expected_rel_q)

        scaling = {row["n_vectors"]: row for row in entry["scaling"]}
        assert set(scaling) == set(MOBILE_VECTOR_COUNTS)
        for n, row in scaling.items():
            assert row["mobile_ram_mib"] == pytest.approx((expected_bytes * n) / _MIB)
