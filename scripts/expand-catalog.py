#!/usr/bin/env python3
"""Expand books.jsonl to N entries using the Gutenberg catalog CSV.

Usage:
    python scripts/expand-catalog.py [--count 5000] [--output src/agentkv/data/books.jsonl]

Steps:
  1. Download the Gutenberg catalog CSV (cached at /tmp/pg_catalog.csv).
  2. Filter for English prose texts, sort by download count (popularity proxy).
  3. For each new book not already in the catalog, call the Anthropic API to
     generate 5 comprehension questions.
  4. Download the book text to measure character length, then discard it
     (the benchmark downloads on demand; this just captures the metadata).
  5. Write the updated JSONL in-place (existing entries preserved as-is).

Requirements:
    pip install anthropic httpx
    export ANTHROPIC_API_KEY=...

The Gutenberg catalog CSV schema (pg_catalog.csv):
    Text#, Type, Issued, Title, Language, Authors, Subjects, LoCC, Bookshelves
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

CATALOG_CSV_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
CATALOG_CSV_CACHE = Path("/tmp/pg_catalog.csv")
BOOK_TEXT_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

_START_RE = re.compile(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG", re.IGNORECASE)
_END_RE = re.compile(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG", re.IGNORECASE)

_QUESTION_PROMPT = """\
You are creating comprehension questions for a machine-learning benchmark.
Given the title and author of a book, write exactly 5 short comprehension
questions whose answers require understanding the actual content of the book.

Rules:
- Each question must be answerable from the book text alone.
- Questions must be specific — no "What is the theme?" or "Who is the main character?"
- Output ONLY a JSON array of 5 strings, no other text.

Book title: {title}
Author: {author}

Output (JSON array of 5 strings):"""


def fetch_catalog_csv() -> list[dict]:
    if not CATALOG_CSV_CACHE.exists():
        print("Downloading Gutenberg catalog CSV ...", flush=True)
        with urllib.request.urlopen(CATALOG_CSV_URL, timeout=60) as resp:
            data = resp.read().decode("utf-8", errors="replace")
        CATALOG_CSV_CACHE.write_text(data)
    else:
        print(f"Using cached catalog at {CATALOG_CSV_CACHE}", flush=True)
        data = CATALOG_CSV_CACHE.read_text()

    rows = []
    reader = csv.DictReader(io.StringIO(data))
    for row in reader:
        rows.append(row)
    return rows


def filter_rows(rows: list[dict], existing_ids: set[int]) -> list[dict]:
    """Return English prose texts not already in the catalog, sorted by ID asc.

    Gutenberg doesn't publish download counts in the CSV; we use ID as a rough
    popularity proxy (lower IDs = older, more popular classics).  For a better
    ranking, fetch https://gutenberg.org/browse/scores/top and scrape IDs, or
    use the Gutenberg mirror that provides download stats.
    """
    good = []
    for row in rows:
        try:
            book_id = int(row.get("Text#", 0))
        except ValueError:
            continue
        if book_id in existing_ids:
            continue
        if row.get("Type", "").strip().lower() != "text":
            continue
        lang = row.get("Language", "")
        if "en" not in lang.lower():
            continue
        title = row.get("Title", "").strip()
        authors = row.get("Authors", "").strip()
        if not title or not authors:
            continue
        good.append({"id": book_id, "title": title, "author": authors})

    # Sort by ID ascending (lower = more canonical/popular classic)
    good.sort(key=lambda r: r["id"])
    return good


def fetch_book_charcount(book_id: int) -> int | None:
    """Download the book, measure stripped character length, return it."""
    for url_tmpl in [BOOK_TEXT_URL]:
        url = url_tmpl.format(id=book_id)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            m = _START_RE.search(text)
            if m:
                text = text[m.end():]
            m = _END_RE.search(text)
            if m:
                text = text[:m.start()]
            return len(text.strip())
        except Exception:
            pass
    return None


def generate_questions(client, title: str, author: str, retries: int = 3) -> list[str] | None:
    prompt = _QUESTION_PROMPT.format(title=title, author=author)
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model="claude-haiku-5-5",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            # Extract JSON array from response
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if not m:
                raise ValueError(f"No JSON array in response: {raw[:200]}")
            questions = json.loads(m.group(0))
            if isinstance(questions, list) and len(questions) == 5:
                return [str(q) for q in questions]
        except Exception as exc:
            print(f"    attempt {attempt+1} failed: {exc}", flush=True)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--count", type=int, default=5000, help="Target total catalog size (default: 5000)"
    )
    ap.add_argument(
        "--output",
        default="src/agentkv/data/books.jsonl",
        help="Path to books.jsonl (default: src/agentkv/data/books.jsonl)",
    )
    ap.add_argument(
        "--no-charcount",
        action="store_true",
        help="Skip downloading books to measure character length (faster, char_count=null).",
    )
    ap.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Seconds between Anthropic API calls (default: 0.5).",
    )
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set")

    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)

    out_path = Path(args.output)
    existing: list[dict] = []
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            line = line.strip()
            if line:
                existing.append(json.loads(line))
    existing_ids = {b["id"] for b in existing}
    print(f"Existing catalog: {len(existing)} books", flush=True)

    need = args.count - len(existing)
    if need <= 0:
        print(f"Already at {len(existing)} books — nothing to do.")
        return

    print(f"Need {need} more books to reach {args.count} total.", flush=True)

    rows = fetch_catalog_csv()
    candidates = filter_rows(rows, existing_ids)
    print(f"Candidates from Gutenberg CSV: {len(candidates)}", flush=True)

    new_entries: list[dict] = []
    for i, cand in enumerate(candidates):
        if len(new_entries) >= need:
            break

        book_id = cand["id"]
        title = cand["title"]
        author = cand["author"]
        print(f"[{i+1}/{need}] id={book_id} — {title[:60]}", flush=True)

        questions = generate_questions(client, title, author)
        if not questions:
            print("    skipping — could not generate questions", flush=True)
            continue

        char_count: int | None = None
        if not args.no_charcount:
            char_count = fetch_book_charcount(book_id)
            if char_count is None:
                print("    warning: could not fetch book text; char_count=null", flush=True)

        entry = {
            "id": book_id,
            "title": title,
            "author": author,
            "char_count": char_count,
            "questions": questions,
        }
        new_entries.append(entry)
        print(f"    ok  char_count={char_count}  q[0]={questions[0][:60]}", flush=True)

        time.sleep(args.rate_limit)

    # Write updated catalog
    all_entries = existing + new_entries
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(e) for e in all_entries) + "\n")
    print(f"\nWrote {len(all_entries)} books to {out_path}", flush=True)
    print(f"Added {len(new_entries)} new entries.", flush=True)


if __name__ == "__main__":
    main()
