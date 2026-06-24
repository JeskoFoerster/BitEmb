"""Tests for bitemb.chunking – chunking strategies."""

import pytest

from bitemb.chunking import chunk_fixed, chunk_semantic, chunk_sentence


class TestChunkFixed:
    def test_basic(self):
        text = "a" * 1024
        chunks = chunk_fixed(text, size=512, overlap=64)
        assert len(chunks) == 3

    def test_overlap_content(self):
        text = "a" * 1024
        chunks = chunk_fixed(text, size=512, overlap=64)
        assert chunks[1].text == text[448 : 448 + 512]

    def test_empty(self):
        assert chunk_fixed("", size=512, overlap=64) == []

    def test_strategy_label(self):
        chunks = chunk_fixed("hello world", size=512, overlap=0)
        assert chunks[0].strategy == "fixed"


class TestChunkSentence:
    def test_basic(self):
        text = "One. Two. Three. Four. Five. Six."
        chunks = chunk_sentence(text, max_sentences=3)
        assert len(chunks) == 2

    def test_single(self):
        chunks = chunk_sentence("Only one.", max_sentences=5)
        assert len(chunks) == 1

    def test_strategy_label(self):
        chunks = chunk_sentence("Hello. World.", max_sentences=5)
        assert chunks[0].strategy == "sentence"


class TestChunkSemantic:
    def test_paragraph_split(self):
        text = "Para one content.\n\nPara two content.\n\nPara three content."
        chunks = chunk_semantic(text, min_size=10)
        assert len(chunks) == 3

    def test_merges_short(self):
        text = "Short.\n\nAlso short.\n\nTiny."
        chunks = chunk_semantic(text, min_size=100)
        assert len(chunks) == 1

    def test_strategy_label(self):
        chunks = chunk_semantic("Some paragraph.", min_size=1)
        assert chunks[0].strategy == "semantic"
