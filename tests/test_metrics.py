"""Tests for metric aggregation."""

import pytest

from agentkv.metrics import RequestResult, aggregate_round, _stat_block


def _make_result(ttft: float, e2e: float, tpot: float | None = None, error: str | None = None) -> RequestResult:
    return RequestResult(
        round_idx=1,
        round_label="query-1",
        book_id=1,
        title="Test",
        chunk_idx=0,
        question_idx=0,
        prompt_tokens=100,
        output_tokens=50,
        ttft_s=ttft,
        e2e_s=e2e,
        tpot_s=tpot,
        error=error,
    )


def test_stat_block_basic():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = _stat_block(vals)
    assert b["min"] == 1.0
    assert b["max"] == 5.0
    assert abs(b["mean"] - 3.0) < 1e-9


def test_stat_block_empty():
    assert _stat_block([]) == {}


def test_aggregate_round_no_errors():
    results = [_make_result(ttft=1.0, e2e=2.0), _make_result(ttft=0.5, e2e=1.5)]
    m = aggregate_round(results)
    assert m.n_errors == 0
    assert m.n_total == 2
    assert abs(m.ttft["mean"] - 0.75) < 1e-9
    assert m.estimated_cache_hit_rate is None  # no warmup_ttfts provided


def test_aggregate_round_errors_excluded():
    results = [
        _make_result(ttft=1.0, e2e=2.0),
        _make_result(ttft=0.5, e2e=1.5, error="timeout"),
    ]
    m = aggregate_round(results)
    assert m.n_errors == 1
    assert m.n_total == 2
    assert m.ttft["mean"] == 1.0  # only the successful one


def test_cache_hit_rate_estimate():
    warmup = {(1, 0, 0): 2.0, (1, 0, 1): 2.0}
    # first result has TTFT < 50% of warmup → hit; second > 50% → miss
    results = [
        RequestResult(1, "query-1", 1, "T", 0, 0, 100, 50, 0.9, 1.5, None, None),
        RequestResult(1, "query-1", 1, "T", 0, 1, 100, 50, 1.5, 2.0, None, None),
    ]
    m = aggregate_round(results, warmup_ttfts=warmup)
    assert m.estimated_cache_hit_rate == 0.5


def test_aggregate_all_errors():
    results = [_make_result(ttft=1.0, e2e=2.0, error="err")]
    m = aggregate_round(results)
    assert m.n_errors == 1
    assert m.ttft == {}
