"""Metric aggregation: TTFT, TPOT, E2E latency statistics."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass
class RequestResult:
    round_idx: int
    round_label: str
    book_id: int
    title: str
    chunk_idx: int
    question_idx: int
    prompt_tokens: int
    output_tokens: int
    ttft_s: float
    e2e_s: float
    tpot_s: float | None  # None when output_tokens < 2
    error: str | None = None


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, int(len(sorted_vals) * p / 100) - 1)
    return sorted_vals[idx]


@dataclass
class RoundMetrics:
    round_idx: int
    label: str
    n_total: int
    n_errors: int
    estimated_cache_hit_rate: float | None  # None for warmup (round 0)
    ttft: dict = field(default_factory=dict)   # mean, p50, p90, p99
    tpot: dict = field(default_factory=dict)
    e2e: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "round": self.round_idx,
            "label": self.label,
            "n_total": self.n_total,
            "n_errors": self.n_errors,
            "estimated_cache_hit_rate": self.estimated_cache_hit_rate,
            "ttft": self.ttft,
            "tpot": self.tpot,
            "e2e": self.e2e,
        }


def _stat_block(vals: list[float]) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    return {
        "mean": statistics.mean(s),
        "p50": _percentile(s, 50),
        "p90": _percentile(s, 90),
        "p99": _percentile(s, 99),
        "min": s[0],
        "max": s[-1],
    }


def aggregate_round(
    results: list[RequestResult],
    warmup_ttfts: dict[tuple, float] | None = None,
) -> RoundMetrics:
    """Compute per-round metric stats.

    warmup_ttfts maps (book_id, chunk_idx, question_idx) → warmup TTFT;
    when provided, cache hit rate is estimated as fraction of requests
    whose TTFT < 50% of the warmup TTFT for that prompt.
    """
    good = [r for r in results if not r.error]
    n_errors = len(results) - len(good)

    cache_hit_rate: float | None = None
    if warmup_ttfts and good:
        hits = sum(
            1
            for r in good
            if (key := (r.book_id, r.chunk_idx, r.question_idx)) in warmup_ttfts
            and r.ttft_s < warmup_ttfts[key] * 0.5
        )
        cache_hit_rate = hits / len(good)

    tpot_vals = [r.tpot_s for r in good if r.tpot_s is not None]

    round_idx = results[0].round_idx if results else 0
    label = results[0].round_label if results else ""

    return RoundMetrics(
        round_idx=round_idx,
        label=label,
        n_total=len(results),
        n_errors=n_errors,
        estimated_cache_hit_rate=cache_hit_rate,
        ttft=_stat_block([r.ttft_s for r in good]),
        tpot=_stat_block(tpot_vals),
        e2e=_stat_block([r.e2e_s for r in good]),
    )
