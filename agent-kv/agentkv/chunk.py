"""Text chunking with optional tiktoken-based token counting."""

from __future__ import annotations


def _whitespace_count(text: str) -> int:
    return len(text.split())


class Tokenizer:
    """Thin wrapper around tiktoken (or whitespace fallback)."""

    def __init__(self, model: str):
        self._enc = None
        try:
            import tiktoken  # type: ignore

            try:
                self._enc = tiktoken.encoding_for_model(model)
            except Exception:
                self._enc = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            pass

    @property
    def backend(self) -> str:
        return "tiktoken" if self._enc else "whitespace"

    def count(self, text: str) -> int:
        if self._enc:
            return len(self._enc.encode(text))
        return _whitespace_count(text)


def chunk_text(
    text: str,
    tokenizer: Tokenizer,
    chunk_tokens: int,
    token_align: int,
    max_chunks: int,
) -> list[str]:
    """Split text into at most max_chunks chunks of ~chunk_tokens tokens each.

    Each chunk boundary is extended to the next multiple of token_align so
    that chunk starts sit on predictable prefix offsets for the KV cache.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    pos = 0

    while pos < len(words) and len(chunks) < max_chunks:
        approx_end = min(pos + int(chunk_tokens * 1.2), len(words))

        lo, hi = pos, approx_end
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if tokenizer.count(" ".join(words[pos:mid])) < chunk_tokens:
                lo = mid
            else:
                hi = mid

        base_count = tokenizer.count(" ".join(words[pos:hi]))
        target = ((base_count + token_align - 1) // token_align) * token_align
        while hi < len(words):
            if tokenizer.count(" ".join(words[pos : hi + 1])) >= target:
                hi += 1
                break
            hi += 1

        chunk = " ".join(words[pos:hi])
        if chunk:
            chunks.append(chunk)
        pos = hi

    return chunks
