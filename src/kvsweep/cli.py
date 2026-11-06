"""CLI entry point for kv-sweep."""

from __future__ import annotations

import asyncio
import random
import sys
import time
from pathlib import Path

import click
from openai import AsyncOpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

from kvsweep import __version__
from kvsweep.catalog import load_catalog, filter_catalog
from kvsweep.chunk import Tokenizer, chunk_text
from kvsweep.download import fetch_book
from kvsweep.metrics import RequestResult, RoundMetrics, aggregate_round
from kvsweep.report import (
    console,
    print_round_header,
    print_round_summary,
    print_sweep_summary,
    write_results,
)
from kvsweep.runner import RequestSpec, run_round

_PROMPT_TEMPLATE = (
    "Read the following passage carefully, then answer the question.\n\n"
    "Passage:\n{chunk}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

_DEFAULT_CACHE = Path.home() / ".cache" / "kv-sweep" / "gutenberg"
_DEFAULT_OUTPUT = Path("kvsweep-results.json")


@click.group()
@click.version_option(__version__, prog_name="kvsweep")
def main() -> None:
    """kv-sweep — KV-cache reuse ramp benchmark for LLM inference servers.

    Sends repeated rounds of identical prompts to a vLLM-compatible endpoint
    and tracks how TTFT, TPOT, and E2E latency change as the prefix KV cache
    fills from cold (0%) toward fully warm (100%).
    """


@main.command("run")
@click.option("--host", default="localhost", show_default=True, help="Inference server host.")
@click.option("--port", default=8000, show_default=True, type=int, help="Inference server port.")
@click.option("--base-url", default="", help="Full base URL — overrides --host/--port.")
@click.option("--model", default="Qwen/Qwen2.5-3B-Instruct", show_default=True)
@click.option("--catalog", "catalog_path", default=None, type=click.Path(path_type=Path),
              help="Path to a custom JSONL book catalog (default: built-in 100-book catalog).")
@click.option("--book-ids", default="", help="Comma-separated Gutenberg IDs to include.")
@click.option("--num-books", default=0, type=int, help="Max books (0 = all).")
@click.option("--chunk-tokens", default=4096, show_default=True, type=int,
              help="Target tokens per chunk.")
@click.option("--token-align", default=256, show_default=True, type=int,
              help="Round chunk size to this token multiple for cache alignment.")
@click.option("--max-chunks", default=4, show_default=True, type=int,
              help="Max chunks per book.")
@click.option("--questions-per-chunk", default=2, show_default=True, type=int,
              help="Questions asked per chunk per round.")
@click.option("--output-len", default=200, show_default=True, type=int,
              help="Max output tokens per answer.")
@click.option("--max-inflight", default=4, show_default=True, type=int,
              help="Max concurrent in-flight requests.")
@click.option("--rounds", "repeat_count", default=3, show_default=True, type=int,
              help="Number of query rounds after the warmup round.")
@click.option("--sleep-between", default=2.0, show_default=True, type=float,
              help="Seconds between rounds.")
@click.option("--shuffle/--no-shuffle", default=False, show_default=True,
              help="Shuffle request order before each round.")
@click.option("--shuffle-seed", default=42, show_default=True, type=int)
@click.option("--cache-dir", default=str(_DEFAULT_CACHE), show_default=True,
              help="Directory for cached Gutenberg text files.")
@click.option("--output", "-o", default=str(_DEFAULT_OUTPUT), show_default=True,
              type=click.Path(path_type=Path),
              help="JSON output file for results.")
@click.option("--expected-ttft-gain", default=None, type=float,
              help="Exit non-zero if TTFT speedup (warmup/last-round) is below this factor.")
def run_cmd(
    host: str,
    port: int,
    base_url: str,
    model: str,
    catalog_path: Path | None,
    book_ids: str,
    num_books: int,
    chunk_tokens: int,
    token_align: int,
    max_chunks: int,
    questions_per_chunk: int,
    output_len: int,
    max_inflight: int,
    repeat_count: int,
    sleep_between: float,
    shuffle: bool,
    shuffle_seed: int,
    cache_dir: str,
    output: Path,
    expected_ttft_gain: float | None,
) -> None:
    """Run the KV-cache reuse sweep benchmark."""
    asyncio.run(
        _run(
            host=host,
            port=port,
            base_url=base_url,
            model=model,
            catalog_path=catalog_path,
            book_ids=[int(x) for x in book_ids.split(",") if x.strip()],
            num_books=num_books or None,
            chunk_tokens=chunk_tokens,
            token_align=token_align,
            max_chunks=max_chunks,
            questions_per_chunk=questions_per_chunk,
            output_len=output_len,
            max_inflight=max_inflight,
            repeat_count=repeat_count,
            sleep_between=sleep_between,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            cache_dir=Path(cache_dir),
            output=output,
            expected_ttft_gain=expected_ttft_gain,
        )
    )


async def _run(
    host: str,
    port: int,
    base_url: str,
    model: str,
    catalog_path: Path | None,
    book_ids: list[int],
    num_books: int | None,
    chunk_tokens: int,
    token_align: int,
    max_chunks: int,
    questions_per_chunk: int,
    output_len: int,
    max_inflight: int,
    repeat_count: int,
    sleep_between: float,
    shuffle: bool,
    shuffle_seed: int,
    cache_dir: Path,
    output: Path,
    expected_ttft_gain: float | None,
) -> None:
    books = load_catalog(catalog_path)
    books = filter_catalog(books, book_ids=book_ids or None, num_books=num_books)
    if not books:
        console.print("[red]No books selected. Check --book-ids / --num-books.[/]")
        sys.exit(1)

    tok = Tokenizer(model)
    console.print(f"Token counting backend: [bold]{tok.backend}[/]")

    # Build chunks for each book
    console.print(f"\nPreparing {len(books)} book(s)...")
    chunks_by_book: dict[int, list[str]] = {}
    for book in books:
        bid = book["id"]
        console.print(f"  Book {bid}: {book.get('title', '?')}")
        raw = fetch_book(bid, cache_dir)
        chunks = chunk_text(raw, tok, chunk_tokens, token_align, max_chunks)
        console.print(f"    {len(chunks)} chunk(s), sizes: {[tok.count(c) for c in chunks]} tokens")
        chunks_by_book[bid] = chunks

    # Build request specs
    specs: list[RequestSpec] = []
    for book in books:
        bid = book["id"]
        questions: list[str] = book.get("questions", [])
        for ci, chunk in enumerate(chunks_by_book.get(bid, [])):
            for qi in range(min(questions_per_chunk, len(questions))):
                specs.append(
                    RequestSpec(
                        book_id=bid,
                        title=book.get("title", str(bid)),
                        chunk_idx=ci,
                        question_idx=qi,
                        prompt=_PROMPT_TEMPLATE.format(chunk=chunk, question=questions[qi]),
                    )
                )

    if not specs:
        console.print("[red]No requests generated — check your catalog has 'questions' fields.[/]")
        sys.exit(1)

    rng = random.Random(shuffle_seed)
    console.print(f"\nTotal prompts per round: {len(specs)}")

    endpoint = base_url or f"http://{host}:{port}/v1"
    client = AsyncOpenAI(base_url=endpoint, api_key="dummy")

    config = {
        "endpoint": endpoint,
        "model": model,
        "chunk_tokens": chunk_tokens,
        "token_align": token_align,
        "max_chunks": max_chunks,
        "questions_per_chunk": questions_per_chunk,
        "output_len": output_len,
        "max_inflight": max_inflight,
        "repeat_count": repeat_count,
        "sleep_between": sleep_between,
        "shuffle": shuffle,
        "shuffle_seed": shuffle_seed,
        "n_books": len(books),
        "n_specs": len(specs),
    }

    all_results: list[RequestResult] = []
    round_metrics: list[RoundMetrics] = []
    warmup_ttfts: dict[tuple, float] = {}

    # Round 0: warmup (cold cache)
    round_specs = list(specs)
    if shuffle:
        rng.shuffle(round_specs)

    print_round_header(0, "warmup (cold cache)", len(round_specs))

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as prog:
        task = prog.add_task("warmup", total=len(round_specs))

        def on_done(r: RequestResult) -> None:
            prog.advance(task)

        warmup_results = await run_round(
            client, model, round_specs, output_len, max_inflight,
            round_idx=0, round_label="warmup",
            progress_callback=on_done,
        )

    all_results.extend(warmup_results)
    for r in warmup_results:
        warmup_ttfts[(r.book_id, r.chunk_idx, r.question_idx)] = r.ttft_s

    m0 = aggregate_round(warmup_results)
    round_metrics.append(m0)
    print_round_summary(m0)

    # Query rounds — prefix cache progressively warms
    for qi in range(repeat_count):
        if sleep_between > 0:
            console.print(f"\nSleeping {sleep_between:.1f}s ...", style="dim")
            await asyncio.sleep(sleep_between)

        round_specs = list(specs)
        if shuffle:
            rng.shuffle(round_specs)

        label = f"query-{qi+1}"
        print_round_header(qi + 1, label, len(round_specs))

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as prog:
            task = prog.add_task(label, total=len(round_specs))

            def on_done(r: RequestResult, _t=task, _p=prog) -> None:
                _p.advance(_t)

            qr_results = await run_round(
                client, model, round_specs, output_len, max_inflight,
                round_idx=qi + 1, round_label=label,
                progress_callback=on_done,
            )

        all_results.extend(qr_results)
        m = aggregate_round(qr_results, warmup_ttfts=warmup_ttfts)
        round_metrics.append(m)
        print_round_summary(m)

    print_sweep_summary(round_metrics)
    write_results(output, config, round_metrics, all_results)

    # Optional pass/fail gate
    if expected_ttft_gain is not None:
        warmup_m = round_metrics[0]
        last_m = round_metrics[-1]
        if warmup_m.ttft and last_m.ttft:
            speedup = warmup_m.ttft["mean"] / last_m.ttft["mean"]
            if speedup < expected_ttft_gain:
                console.print(
                    f"[red]FAIL: TTFT speedup {speedup:.2f}× < required {expected_ttft_gain}×[/]"
                )
                sys.exit(1)
