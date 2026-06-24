"""Tests for Retriever."""

import numpy as np
import pytest

from bitemb.retrieval import Retriever


@pytest.fixture
def retriever_with_index(random_float_embs, random_bit_embs):
    r = Retriever()
    texts = [f"doc_{i}" for i in range(10)]
    r.create_index("test", texts, random_float_embs, random_bit_embs, dim=768)
    return r, random_float_embs, random_bit_embs


def test_create_index(retriever_with_index):
    r, _, _ = retriever_with_index
    assert "test" in r.indices
    assert r.indices["test"].float_embs.shape == (10, 768)


def test_search_float_returns_k(retriever_with_index):
    r, embs, _ = retriever_with_index
    results = r.search_float("test", embs[0:1].T.T, k=5)
    assert len(results) == 5


def test_search_float_top1_is_self(retriever_with_index):
    r, embs, _ = retriever_with_index
    results = r.search_float("test", embs[0], k=5)
    assert results[0].idx == 0
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_search_bit_returns_k(retriever_with_index):
    r, _, bits = retriever_with_index
    results = r.search_bit("test", bits[0:1], k=5)
    assert len(results) == 5


def test_search_bit_top1_is_self(retriever_with_index):
    r, _, bits = retriever_with_index
    results = r.search_bit("test", bits[0:1], k=5)
    assert results[0].idx == 0
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_search_asymmetric_returns_k(retriever_with_index):
    r, embs, _ = retriever_with_index
    results = r.search_asymmetric("test", embs[0], k=5)
    assert len(results) == 5


def test_scores_sorted_descending(retriever_with_index):
    r, embs, bits = retriever_with_index
    for results in [
        r.search_float("test", embs[0], k=10),
        r.search_bit("test", bits[0:1], k=10),
        r.search_asymmetric("test", embs[0], k=10),
    ]:
        scores = [res.score for res in results]
        assert scores == sorted(scores, reverse=True)
