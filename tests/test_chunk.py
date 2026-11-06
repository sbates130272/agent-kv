"""Tests for text chunking logic."""

import pytest

from agentkv.chunk import Tokenizer, chunk_text


class _FakeTokenizer:
    """Deterministic tokenizer: one token per word."""

    backend = "whitespace"

    def count(self, text: str) -> int:
        return len(text.split())


def test_chunk_basic():
    tok = _FakeTokenizer()
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, tok, chunk_tokens=100, token_align=10, max_chunks=5)
    assert 1 <= len(chunks) <= 5
    for c in chunks:
        assert tok.count(c) >= 100


def test_chunk_max_chunks():
    tok = _FakeTokenizer()
    text = " ".join(f"word{i}" for i in range(5000))
    chunks = chunk_text(text, tok, chunk_tokens=100, token_align=10, max_chunks=3)
    assert len(chunks) == 3


def test_chunk_empty():
    tok = _FakeTokenizer()
    assert chunk_text("", tok, 100, 10, 5) == []


def test_chunk_alignment():
    tok = _FakeTokenizer()
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_text(text, tok, chunk_tokens=100, token_align=50, max_chunks=10)
    for c in chunks:
        count = tok.count(c)
        assert count % 50 == 0 or count >= 100, f"misaligned chunk size {count}"


def test_tokenizer_whitespace_fallback(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "tiktoken", None)
    tok = Tokenizer("some-model")
    assert tok.count("hello world") == 2
