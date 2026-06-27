"""Tests for bitemb.dataset – BEIR loading logic."""

import pytest

from bitemb.config import DATASETS
from bitemb.dataset import BeirDataset, load_beir


class TestDatasetConfig:
    def test_supported_datasets(self):
        assert "scifact" in DATASETS
        assert "fiqa" in DATASETS
        assert "trec-covid" in DATASETS

    def test_invalid_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_beir("nonexistent")


@pytest.mark.slow
class TestLoadBeir:
    """Integration tests that download data from HuggingFace."""

    def test_scifact_structure(self):
        ds = load_beir("scifact")
        assert isinstance(ds, BeirDataset)
        assert ds.name == "scifact"
        assert ds.n_corpus > 0
        assert ds.n_queries > 0
        assert len(ds.qrels) > 0
        # All query indices should be valid
        for q_idx, rels in ds.qrels.items():
            assert 0 <= q_idx < ds.n_queries
            for c_idx, score in rels.items():
                assert 0 <= c_idx < ds.n_corpus
                assert score > 0
