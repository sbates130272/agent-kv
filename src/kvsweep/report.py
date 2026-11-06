"""Rich terminal output and structured JSON result writer."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich import box

from kvsweep.metrics import RoundMetrics, RequestResult

console = Console()


def print_round_header(round_idx: int, label: str, n: int) -> None:
    style = "bold cyan" if round_idx == 0 else "bold green"
    console.rule(f"[{style}]Round {round_idx}: {label} ({n} requests)[/{style}]")


def print_round_summary(m: RoundMetrics) -> None:
    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    t.add_column("Metric", style="dim")
    t.add_column("mean", justify="right")
    t.add_column("p50", justify="right")
    t.add_column("p90", justify="right")
    t.add_column("p99", justify="right")

    def row(name: str, d: dict) -> None:
        if not d:
            return
        t.add_row(
            name,
            f"{d['mean']:.3f}s",
            f"{d['p50']:.3f}s",
            f"{d['p90']:.3f}s",
            f"{d['p99']:.3f}s",
        )

    row("TTFT", m.ttft)
    row("TPOT", m.tpot)
    row("E2E", m.e2e)
    console.print(t)

    hit_str = (
        f"  estimated cache hit rate: [bold yellow]{m.estimated_cache_hit_rate*100:.1f}%[/]"
        if m.estimated_cache_hit_rate is not None
        else ""
    )
    errors = f"  errors: [red]{m.n_errors}[/]" if m.n_errors else ""
    console.print(f"  requests: {m.n_total}{errors}{hit_str}")


def print_sweep_summary(rounds: list[RoundMetrics]) -> None:
    console.rule("[bold white]Sweep Summary[/]")
    t = Table(box=box.MARKDOWN, show_header=True, header_style="bold magenta")
    t.add_column("Round")
    t.add_column("Label")
    t.add_column("Cache hit%", justify="right")
    t.add_column("TTFT mean", justify="right")
    t.add_column("TPOT mean", justify="right")
    t.add_column("E2E mean", justify="right")
    t.add_column("Errors", justify="right")

    for m in rounds:
        hit = f"{m.estimated_cache_hit_rate*100:.1f}%" if m.estimated_cache_hit_rate is not None else "—"
        ttft = f"{m.ttft['mean']:.3f}s" if m.ttft else "—"
        tpot = f"{m.tpot['mean']:.3f}s" if m.tpot else "—"
        e2e = f"{m.e2e['mean']:.3f}s" if m.e2e else "—"
        t.add_row(str(m.round_idx), m.label, hit, ttft, tpot, e2e, str(m.n_errors))

    console.print(t)

    warmup = rounds[0] if rounds else None
    query_rounds = [r for r in rounds if r.round_idx > 0]
    if warmup and query_rounds and warmup.ttft and query_rounds[-1].ttft:
        speedup = warmup.ttft["mean"] / query_rounds[-1].ttft["mean"]
        console.print(
            f"\n  TTFT speedup (warmup → last query round): "
            f"[bold green]{speedup:.2f}×[/]  "
            f"({warmup.ttft['mean']:.3f}s → {query_rounds[-1].ttft['mean']:.3f}s)"
        )


def write_results(
    output_path: Path,
    config: dict[str, Any],
    rounds: list[RoundMetrics],
    all_results: list[RequestResult],
) -> None:
    """Write a single JSON result file."""
    warmup = next((r for r in rounds if r.round_idx == 0), None)
    query = [r for r in rounds if r.round_idx > 0]

    ttft_speedup: float | None = None
    if warmup and query and warmup.ttft and query[-1].ttft:
        ttft_speedup = warmup.ttft["mean"] / query[-1].ttft["mean"]

    doc = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "schema_version": 1,
        "config": config,
        "rounds": [r.to_dict() for r in rounds],
        "summary": {
            "ttft_speedup_warmup_to_last_round": ttft_speedup,
            "final_cache_hit_rate": query[-1].estimated_cache_hit_rate if query else None,
            "n_rounds_total": len(rounds),
        },
        "per_request": [
            {
                "round": r.round_idx,
                "label": r.round_label,
                "book_id": r.book_id,
                "title": r.title,
                "chunk_idx": r.chunk_idx,
                "question_idx": r.question_idx,
                "prompt_tokens": r.prompt_tokens,
                "output_tokens": r.output_tokens,
                "ttft_s": r.ttft_s,
                "tpot_s": r.tpot_s,
                "e2e_s": r.e2e_s,
                "error": r.error,
            }
            for r in all_results
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(doc, indent=2))
    console.print(f"\n[dim]Results written to[/] [bold]{output_path}[/]")
