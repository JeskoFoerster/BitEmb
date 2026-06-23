"""Document loading and chunking utilities."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Chunk:
    """A text chunk with provenance and strategy metadata."""

    text: str
    source: str
    strategy: str
    index: int


def load_text(path: str | Path) -> str:
    """Load a UTF-8 text file."""
    return Path(path).read_text(encoding="utf-8")


def load_pdf(path: str | Path) -> str:
    """Load text from a PDF file using pypdf."""
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_file(path: str | Path) -> str:
    """Load text from a file, detecting PDFs by extension."""
    file_path = Path(path)
    if file_path.suffix.lower() == ".pdf":
        return load_pdf(file_path)
    return load_text(file_path)


def chunk_fixed(text: str, size: int = 512, overlap: int = 64, source: str = "") -> list[Chunk]:
    """Chunk text by character count with overlap."""
    if size <= 0:
        raise ValueError("size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    chunks: list[Chunk] = []
    step = size - overlap

    for index, start in enumerate(range(0, len(text), step)):
        chunk_text = text[start : start + size]
        if not chunk_text:
            break
        chunks.append(Chunk(text=chunk_text, source=source, strategy="fixed", index=index))
        if start + size >= len(text):
            break

    return chunks


def chunk_sentence(text: str, max_sentences: int = 5, source: str = "") -> list[Chunk]:
    """Chunk text by groups of sentences split with a simple punctuation regex."""
    if max_sentences <= 0:
        raise ValueError("max_sentences must be greater than 0")

    sentences = [sentence for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence]
    chunks: list[Chunk] = []

    for index, start in enumerate(range(0, len(sentences), max_sentences)):
        chunk_text = " ".join(sentences[start : start + max_sentences])
        chunks.append(Chunk(text=chunk_text, source=source, strategy="sentence", index=index))

    return chunks


def chunk_semantic(text: str, min_size: int = 100, source: str = "") -> list[Chunk]:
    """Chunk text by paragraphs, merging adjacent blocks until min_size is reached."""
    if min_size <= 0:
        raise ValueError("min_size must be greater than 0")

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text.strip())
        if paragraph.strip()
    ]
    chunks: list[Chunk] = []
    current: list[str] = []
    current_size = 0

    for paragraph in paragraphs:
        current.append(paragraph)
        current_size += len(paragraph)

        if current_size >= min_size:
            chunk_text = "\n\n".join(current)
            chunks.append(
                Chunk(text=chunk_text, source=source, strategy="semantic", index=len(chunks))
            )
            current = []
            current_size = 0

    if current:
        chunk_text = "\n\n".join(current)
        chunks.append(Chunk(text=chunk_text, source=source, strategy="semantic", index=len(chunks)))

    return chunks


STRATEGIES: dict[str, Callable[..., list[Chunk]]] = {
    "fixed": chunk_fixed,
    "sentence": chunk_sentence,
    "semantic": chunk_semantic,
}
