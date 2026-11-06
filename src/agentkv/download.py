"""Project Gutenberg book downloader with local disk cache."""

import re
import urllib.request
from pathlib import Path

_MIRRORS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt",
]

_START_RE = re.compile(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG", re.IGNORECASE)
_END_RE = re.compile(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG", re.IGNORECASE)


def fetch_book(book_id: int, cache_dir: Path) -> str:
    """Return the body text of a Gutenberg book, downloading if needed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{book_id}.txt"
    if dest.exists():
        return dest.read_text(encoding="utf-8", errors="replace")
    for url_tmpl in _MIRRORS:
        url = url_tmpl.format(id=book_id)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            dest.write_text(text, encoding="utf-8")
            return _strip_gutenberg_boilerplate(text)
        except Exception:
            pass
    raise RuntimeError(f"Could not download Gutenberg book {book_id}")


def _strip_gutenberg_boilerplate(text: str) -> str:
    m = _START_RE.search(text)
    if m:
        text = text[m.end() :]
    m = _END_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


def get_cached_text(book_id: int, cache_dir: Path) -> str:
    """Return stripped body text; assumes the raw file is already cached."""
    dest = cache_dir / f"{book_id}.txt"
    if not dest.exists():
        raise FileNotFoundError(f"Book {book_id} not in cache: {dest}")
    raw = dest.read_text(encoding="utf-8", errors="replace")
    return _strip_gutenberg_boilerplate(raw)
