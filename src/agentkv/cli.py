"""CLI entry point for agent-kv."""

from __future__ import annotations

import asyncio
import os
import random
import sys
import time
from pathlib import Path

import click
from openai import AsyncOpenAI
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TimeElapsedColumn

from agentkv import __version__
from agentkv.catalog import filter_catalog, load_catalog
from agentkv.chunk import Tokenizer, chunk_text
from agentkv.download import fetch_book
from agentkv.metrics import RequestResult, RoundMetrics, aggregate_round
from agentkv.report import (
    console,
    print_round_header,
    print_round_summary,
    print_sweep_summary,
    write_results,
)
from agentkv.runner import RequestSpec, run_round

_PROMPT_TEMPLATE = (
    "Read the following passage carefully, then answer the question.\n\n"
    "Passage:\n{chunk}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

_DEFAULT_CACHE = Path.home() / ".cache" / "agent-kv" / "gutenberg"
_DEFAULT_OUTPUT = Path("agent-kv-results.json")

_SHELLS = ("bash", "zsh", "fish")


@click.group()
@click.version_option(__version__, prog_name="agent-kv")
def main() -> None:
    """agent-kv — KV-cache reuse ramp benchmark for LLM inference servers.

    Sends repeated rounds of identical prompts to a vLLM-compatible endpoint
    and tracks how TTFT, TPOT, and E2E latency change as the prefix KV cache
    fills from cold (0%) toward fully warm (100%).
    """


@main.command("run")
@click.option("--host", default="localhost", show_default=True, help="Inference server host.")
@click.option("--port", default=8000, show_default=True, type=int, help="Inference server port.")
@click.option("--base-url", default="", help="Full base URL — overrides --host/--port.")
@click.option("--model", default="Qwen/Qwen3.5-2B", show_default=True)
@click.option(
    "--catalog",
    "catalog_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to a custom JSONL book catalog (default: built-in 100-book catalog).",
)
@click.option("--book-ids", default="", help="Comma-separated Gutenberg IDs to include.")
@click.option("--num-books", default=0, type=int, help="Max books (0 = all).")
@click.option(
    "--chunk-tokens", default=4096, show_default=True, type=int, help="Target tokens per chunk."
)
@click.option(
    "--token-align",
    default=256,
    show_default=True,
    type=int,
    help="Round chunk size to this token multiple for cache alignment.",
)
@click.option("--max-chunks", default=4, show_default=True, type=int, help="Max chunks per book.")
@click.option(
    "--questions-per-chunk",
    default=2,
    show_default=True,
    type=int,
    help="Questions asked per chunk per round.",
)
@click.option(
    "--output-len", default=200, show_default=True, type=int, help="Max output tokens per answer."
)
@click.option(
    "--max-inflight",
    default=4,
    show_default=True,
    type=int,
    help="Max concurrent in-flight requests.",
)
@click.option(
    "--rounds",
    "repeat_count",
    default=3,
    show_default=True,
    type=int,
    help="Number of query rounds after the warmup round.",
)
@click.option(
    "--sleep-between", default=2.0, show_default=True, type=float, help="Seconds between rounds."
)
@click.option(
    "--shuffle/--no-shuffle",
    default=False,
    show_default=True,
    help="Shuffle request order before each round.",
)
@click.option("--shuffle-seed", default=42, show_default=True, type=int)
@click.option(
    "--cache-dir",
    default=str(_DEFAULT_CACHE),
    show_default=True,
    help="Directory for cached Gutenberg text files.",
)
@click.option(
    "--output",
    "-o",
    default=str(_DEFAULT_OUTPUT),
    show_default=True,
    type=click.Path(path_type=Path),
    help="JSON output file for results.",
)
@click.option(
    "--expected-ttft-gain",
    default=None,
    type=float,
    help="Exit non-zero if TTFT speedup (warmup/last-round) is below this factor.",
)
@click.option(
    "--api-key",
    default=None,
    help="API key for the inference server. Falls back to $LEMONADE_API_KEY then $OPENAI_API_KEY.",
)
@click.option(
    "--wait-for-health/--no-wait-for-health",
    default=False,
    show_default=True,
    help="Poll GET /health until 200 before starting the benchmark.",
)
@click.option(
    "--health-timeout",
    default=300,
    show_default=True,
    type=int,
    help="Seconds to wait for /health when --wait-for-health is set.",
)
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
    api_key: str | None,
    wait_for_health: bool,
    health_timeout: int,
) -> None:
    """Run the KV-cache reuse sweep benchmark."""
    resolved_key = (
        api_key or os.environ.get("LEMONADE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "dummy"
    )
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
            api_key=resolved_key,
            wait_for_health=wait_for_health,
            health_timeout=health_timeout,
        )
    )


async def _wait_for_health(health_url: str, timeout: int) -> None:
    import httpx

    deadline = time.monotonic() + timeout
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task(f"Waiting for [bold]{health_url}[/] ...", total=None)
        async with httpx.AsyncClient() as hc:
            while True:
                try:
                    r = await hc.get(health_url, timeout=5.0)
                    if r.status_code == 200:
                        prog.update(task, description=f"[green]{health_url} is healthy[/]")
                        return
                except Exception:
                    pass
                if time.monotonic() >= deadline:
                    console.print(f"[red]Timed out after {timeout}s waiting for {health_url}[/]")
                    sys.exit(1)
                await asyncio.sleep(5)


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
    api_key: str = "dummy",
    wait_for_health: bool = False,
    health_timeout: int = 300,
) -> None:
    endpoint = base_url or f"http://{host}:{port}/v1"

    if wait_for_health:
        health_url = endpoint.removesuffix("/v1") + "/health"
        await _wait_for_health(health_url, health_timeout)

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

    client = AsyncOpenAI(base_url=endpoint, api_key=api_key)

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
            client,
            model,
            round_specs,
            output_len,
            max_inflight,
            round_idx=0,
            round_label="warmup",
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

        label = f"query-{qi + 1}"
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
                client,
                model,
                round_specs,
                output_len,
                max_inflight,
                round_idx=qi + 1,
                round_label=label,
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


# ---------------------------------------------------------------------------
# nvme-sweep subcommand
# ---------------------------------------------------------------------------


@main.command("nvme-sweep")
@click.option("--host", default="localhost", show_default=True, help="vLLM server host.")
@click.option("--port", default=8000, show_default=True, type=int, help="vLLM server port.")
@click.option("--base-url", default="", help="Full base URL — overrides --host/--port.")
@click.option("--model", default="Qwen/Qwen3.5-2B", show_default=True)
@click.option(
    "--lmcache-http-url",
    default="http://localhost:8080",
    show_default=True,
    help="LMCache HTTP management API base URL (--http-port).",
)
@click.option(
    "--dm-device",
    default=None,
    help="Name of the dm-delay device (e.g. kv-delay).  Omit to skip delay injection.",
)
@click.option(
    "--delays",
    default="0,10,50,100,200",
    show_default=True,
    help="Comma-separated read delay values in milliseconds.",
)
@click.option(
    "--fill-fractions",
    default="0.25,0.50,0.75,1.0",
    show_default=True,
    help="Comma-separated NVMe fill fractions (0.0–1.0).",
)
@click.option(
    "--catalog",
    "catalog_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to a custom JSONL book catalog.",
)
@click.option("--book-ids", default="", help="Comma-separated Gutenberg IDs to include.")
@click.option("--num-books", default=2, show_default=True, type=int, help="Max books (0 = all).")
@click.option("--chunk-tokens", default=4096, show_default=True, type=int)
@click.option("--token-align", default=256, show_default=True, type=int)
@click.option("--max-chunks", default=4, show_default=True, type=int)
@click.option("--questions-per-chunk", default=2, show_default=True, type=int)
@click.option("--output-len", default=200, show_default=True, type=int)
@click.option("--max-inflight", default=4, show_default=True, type=int)
@click.option(
    "--rounds",
    default=2,
    show_default=True,
    type=int,
    help="Measurement rounds per (delay, fill) point.",
)
@click.option(
    "--output",
    "-o",
    default="nvme-sweep-results.json",
    show_default=True,
    type=click.Path(path_type=Path),
    help="JSON output file.",
)
@click.option(
    "--api-key",
    default=None,
    help="API key for the inference server.",
)
@click.option(
    "--cache-dir",
    default=str(_DEFAULT_CACHE),
    show_default=True,
    help="Directory for cached Gutenberg text files.",
)
@click.option(
    "--wait-for-health/--no-wait-for-health",
    default=False,
    show_default=True,
    help="Poll GET /health until 200 before starting the sweep.",
)
@click.option(
    "--health-timeout",
    default=300,
    show_default=True,
    type=int,
    help="Seconds to wait for /health when --wait-for-health is set.",
)
def nvme_sweep_cmd(
    host: str,
    port: int,
    base_url: str,
    model: str,
    lmcache_http_url: str,
    dm_device: str | None,
    delays: str,
    fill_fractions: str,
    catalog_path: Path | None,
    book_ids: str,
    num_books: int,
    chunk_tokens: int,
    token_align: int,
    max_chunks: int,
    questions_per_chunk: int,
    output_len: int,
    max_inflight: int,
    rounds: int,
    output: Path,
    api_key: str | None,
    cache_dir: str,
    wait_for_health: bool,
    health_timeout: int,
) -> None:
    """Sweep NVMe read latency vs KV block fill fraction and measure TTFT/E2E.

    For each (delay_ms, fill_fraction) point the sweep:
      1. Clears VRAM and L1 caches.
      2. Warms NVMe L2 by running fill_fraction of prompts with delay=0.
      3. Clears VRAM+L1 again so only NVMe holds the blocks.
      4. Sets the dm-delay read latency to delay_ms.
      5. Runs `--rounds` measurement passes and records TTFT/E2E.

    Requires VLLM_SERVER_DEV_MODE=1 (POST /reset_prefix_cache) and
    LMCache --http-port (DELETE /cache/clear).  sudo dmsetup must work
    without a password if --dm-device is set.
    """
    from agentkv.nvme_sweep import run_nvme_sweep

    resolved_key = (
        api_key or os.environ.get("LEMONADE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "dummy"
    )
    vllm_base = base_url or f"http://{host}:{port}/v1"

    if wait_for_health:
        health_url = vllm_base.removesuffix("/v1") + "/health"
        asyncio.run(_wait_for_health(health_url, health_timeout))

    delays_ms = [int(x.strip()) for x in delays.split(",") if x.strip()]
    fractions = [float(x.strip()) for x in fill_fractions.split(",") if x.strip()]

    books = load_catalog(catalog_path)
    books = filter_catalog(
        books,
        book_ids=[int(x) for x in book_ids.split(",") if x.strip()] or None,
        num_books=num_books or None,
    )
    if not books:
        console.print("[red]No books selected.[/]")
        sys.exit(1)

    tok = Tokenizer(model)
    specs: list[RequestSpec] = []
    for book in books:
        bid = book["id"]
        raw = fetch_book(bid, Path(cache_dir))
        chunks = chunk_text(raw, tok, chunk_tokens, token_align, max_chunks)
        questions: list[str] = book.get("questions", [])
        for ci, chunk in enumerate(chunks):
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
        console.print("[red]No request specs generated.[/]")
        sys.exit(1)

    n_points = len(delays_ms) * len(fractions)
    console.print(
        f"nvme-sweep: {len(specs)} prompts, {len(delays_ms)} delays"
        f" × {len(fractions)} fills = {n_points} points"
    )

    results = asyncio.run(
        run_nvme_sweep(
            vllm_base=vllm_base,
            lmcache_http=lmcache_http_url,
            dm_device=dm_device,
            model=model,
            specs=specs,
            delays_ms=delays_ms,
            fill_fractions=fractions,
            output_len=output_len,
            max_inflight=max_inflight,
            rounds=rounds,
            api_key=resolved_key,
        )
    )

    # Serialize results
    out_data = {
        "schema_version": 1,
        "command": "nvme-sweep",
        "config": {
            "vllm_base": vllm_base,
            "lmcache_http": lmcache_http_url,
            "dm_device": dm_device,
            "delays_ms": delays_ms,
            "fill_fractions": fractions,
            "model": model,
            "n_specs": len(specs),
            "rounds": rounds,
        },
        "sweep": [
            {
                "delay_ms": p.delay_ms,
                "fill_fraction": p.fill_fraction,
                "rounds": p.round_metrics,
            }
            for p in results
        ],
    }
    import json as _json

    out_path = Path(output)
    out_path.write_text(_json.dumps(out_data, indent=2))
    console.print(f"\n[green]Results written to {out_path}[/]")


# ---------------------------------------------------------------------------
# man subcommand
# ---------------------------------------------------------------------------


@main.command("man")
@click.pass_context
def man_cmd(ctx: click.Context) -> None:
    """Show the full reference manual in the pager."""
    import textwrap

    def _fmt_cmd(cmd: click.Command, name: str, indent: int = 0) -> str:
        pad = " " * indent
        lines = [f"{pad}COMMAND: agent-kv {name}"]
        if cmd.help:
            for line in textwrap.wrap(cmd.help, width=72):
                lines.append(f"{pad}  {line}")
        lines.append("")
        if hasattr(cmd, "params"):
            for p in cmd.params:
                if isinstance(p, click.Option):
                    opt_str = ", ".join(p.opts)
                    default = f"  [default: {p.default}]" if p.default is not None else ""
                    help_str = p.help or ""
                    lines.append(f"{pad}  {opt_str}{default}")
                    if help_str:
                        for hl in textwrap.wrap(help_str, width=68):
                            lines.append(f"{pad}      {hl}")
        return "\n".join(lines)

    page = [
        "NAME",
        "    agent-kv — KV-cache reuse ramp benchmark for LLM inference servers",
        "",
        "SYNOPSIS",
        "    agent-kv [--version] [--help] COMMAND [OPTIONS]",
        "",
        "DESCRIPTION",
        "    agent-kv sends repeated rounds of identical long-document prompts to a",
        "    vLLM-compatible endpoint and measures TTFT, TPOT, and E2E latency as the",
        "    prefix KV cache fills from cold (0%) toward fully warm (~100%).",
        "    Results are written as structured JSON for database ingestion or plotting.",
        "",
        "COMMANDS",
        "",
    ]

    group = ctx.find_root().command
    for name, cmd in sorted(group.commands.items()):  # type: ignore[attr-defined]
        page.append(_fmt_cmd(cmd, name, indent=4))

    page += [
        "SHELL COMPLETION",
        "    bash:",
        '        eval "$(_AGENT_KV_COMPLETE=bash_source agent-kv)"',
        "    zsh:",
        '        eval "$(_AGENT_KV_COMPLETE=zsh_source agent-kv)"',
        "    fish:",
        "        eval (env _AGENT_KV_COMPLETE=fish_source agent-kv)",
        "",
        "OUTPUT FORMAT",
        "    Each run writes a JSON file (schema_version=1) with per-round stats",
        "    (mean/p50/p90/p99 for TTFT, TPOT, E2E) and a per_request array",
        "    suitable for DuckDB or pandas ingestion.",
        "",
        "SEE ALSO",
        "    https://github.com/sbates130272/agent-kv",
        "    https://inferencex.semianalysis.com/agentx",
    ]

    click.echo_via_pager("\n".join(page))


# ---------------------------------------------------------------------------
# completion subcommand
# ---------------------------------------------------------------------------


@main.command("completion")
@click.argument("shell", type=click.Choice(list(_SHELLS)), default="bash", required=False)
def completion_cmd(shell: str) -> None:
    """Print shell completion setup instructions and the activation snippet.

    SHELL must be one of: bash, zsh, fish  [default: bash]

    Add the printed line to your shell startup file (~/.bashrc, ~/.zshrc, etc.)
    to enable tab-completion for all agent-kv commands and options.
    """
    var = "_AGENT_KV_COMPLETE"
    snippets = {
        "bash": f'eval "$({var}=bash_source agent-kv)"',
        "zsh": f'eval "$({var}=zsh_source agent-kv)"',
        "fish": f"eval (env {var}=fish_source agent-kv)",
    }
    rc_files = {"bash": "~/.bashrc", "zsh": "~/.zshrc", "fish": "~/.config/fish/config.fish"}
    snippet = snippets[shell]
    rc = rc_files[shell]
    click.echo(f"\n# Add to {rc}:")
    click.echo(snippet)
    click.echo("\n# Or activate for this session only:")
    click.echo(snippet)
