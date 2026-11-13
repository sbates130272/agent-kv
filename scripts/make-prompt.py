#!/usr/bin/env python3
"""Build a Gutenberg-sourced prompt of a specified token length.

Outputs the prompt to stdout so hsa-snoop-capture.sh can consume it.
Writes a one-line diagnostic (actual token count, book id) to stderr.

Usage:
    python3 scripts/make-prompt.py [options]
    uv run python3 scripts/make-prompt.py --tokens 2048 --book-id 84 --salt "abc123"
"""

import argparse
import sys
from pathlib import Path

# agentkv is installed in the project venv; run via .venv/bin/python3 or uv run.
from agentkv.download import fetch_book, _strip_gutenberg_boilerplate
from agentkv.chunk import Tokenizer, chunk_text

_TEMPLATE = (
    "Read the following passage carefully, then provide a detailed summary "
    "of its key themes, arguments, and narrative elements.\n\n"
    "Passage:\n{chunk}\n\n"
    "Summary:"
)

_TEMPLATE_OVERHEAD = len(_TEMPLATE.format(chunk="").split())  # whitespace fallback guard


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--book-id", type=int, default=84,
                   help="Gutenberg book ID (default: 84 = Frankenstein)")
    p.add_argument("--tokens", type=int, default=512,
                   help="Target total prompt token count (default: 512)")
    p.add_argument("--token-align", type=int, default=32,
                   help="Align chunk to this token multiple (default: 32)")
    p.add_argument("--model", default="Qwen/Qwen3.5-2B",
                   help="Model name used to select tokenizer")
    p.add_argument("--cache-dir",
                   default=str(Path.home() / ".cache/agent-kv/gutenberg"),
                   help="Directory for cached Gutenberg .txt files")
    p.add_argument("--salt", default="",
                   help="Salt string prepended to the prompt; prevents cross-run cache hits")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    try:
        raw = _strip_gutenberg_boilerplate(fetch_book(args.book_id, cache_dir))
    except Exception as exc:
        print(f"ERROR: failed to fetch book {args.book_id}: {exc}", file=sys.stderr)
        sys.exit(1)

    tok = Tokenizer(args.model)

    # Reserve tokens for the template wrapper and optional salt prefix.
    salt_prefix = f"[salt:{args.salt}]\n\n" if args.salt else ""
    salt_tokens = tok.count(salt_prefix) if args.salt else 0
    template_tokens = tok.count(_TEMPLATE.format(chunk=""))
    overhead = salt_tokens + template_tokens
    target = max(args.token_align, args.tokens - overhead)

    chunks = chunk_text(raw, tok, target, token_align=args.token_align, max_chunks=1)
    if not chunks:
        print("ERROR: no text chunks produced — check book ID and cache", file=sys.stderr)
        sys.exit(1)

    prompt = salt_prefix + _TEMPLATE.format(chunk=chunks[0])
    actual = tok.count(prompt)
    print(
        f"make-prompt: book={args.book_id} target={args.tokens} "
        f"actual={actual} backend={tok.backend}",
        file=sys.stderr,
    )
    print(prompt, end="")


if __name__ == "__main__":
    main()
