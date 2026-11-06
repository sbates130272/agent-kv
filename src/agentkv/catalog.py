"""Book catalog loading and filtering."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

_BUILTIN_CATALOG = files("agentkv.data").joinpath("books.jsonl")


def load_catalog(path: Path | None = None) -> list[dict]:
    """Load the JSONL book catalog from path or the built-in one."""
    if path is not None:
        src = path.read_text()
    else:
        src = _BUILTIN_CATALOG.read_text(encoding="utf-8")  # type: ignore[attr-defined]
    return [json.loads(line) for line in src.splitlines() if line.strip()]


def filter_catalog(
    books: list[dict],
    book_ids: list[int] | None = None,
    num_books: int | None = None,
) -> list[dict]:
    if book_ids:
        wanted = set(book_ids)
        books = [b for b in books if b["id"] in wanted]
    if num_books:
        books = books[:num_books]
    return books
