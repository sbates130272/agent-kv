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

    Chunk boundaries are aligned to the nearest token_align multiple at or
    below chunk_tokens, so KV cache prefixes land on predictable offsets.
    The target is derived from chunk_tokens (not from the actual chunk size)
    to prevent alignment rounding from pushing chunks past the context limit.
    """
    # Align DOWN so the target never exceeds chunk_tokens.
    target = (chunk_tokens // token_align) * token_align if token_align > 0 else chunk_tokens

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    pos = 0

    while pos < len(words) and len(chunks) < max_chunks:
        approx_end = min(pos + int(target * 1.2), len(words))

        lo, hi = pos, approx_end
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if tokenizer.count(" ".join(words[pos:mid])) < target:
                lo = mid
            else:
                hi = mid

        chunk = " ".join(words[pos:hi])
        if chunk:
            chunks.append(chunk)
        pos = hi

    return chunks
