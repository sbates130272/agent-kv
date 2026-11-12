"""Probabilistic KV-cache ramp benchmark.

Fires requests drawn uniformly at random from a finite corpus so the
server-side cache fills gradually.  With corpus size N and request rate R:

    P(hit at time t) ≈ 1 - exp(-R·t / N)
    90% coverage reached after ≈ N · ln(10) / R seconds

Smaller corpus or higher rate = faster ramp; larger corpus or lower rate =
slower ramp.  Set --corpus-size and --request-rate to target a ramp duration.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass

from openai import AsyncOpenAI

from agentkv.runner import RequestSpec


@dataclass
class RampSample:
    t: float            # wall-clock seconds since ramp start
    spec_idx: int       # index into corpus list
    ttft_s: float
    e2e_s: float
    corpus_coverage: float  # fraction of unique corpus entries seen so far
    n_total: int            # cumulative requests fired (including this one)
    error: str | None

    def to_dict(self) -> dict:
        return {
            "t": self.t,
            "spec_idx": self.spec_idx,
            "ttft_s": self.ttft_s,
            "e2e_s": self.e2e_s,
            "corpus_coverage": self.corpus_coverage,
            "n_total": self.n_total,
            "error": self.error,
        }


async def run_ramp(
    client: AsyncOpenAI,
    model: str,
    corpus: list[RequestSpec],
    duration_s: float,
    max_inflight: int,
    output_len: int,
    rng_state: int,
    request_rate: float | None = None,
    progress_callback=None,
) -> list[RampSample]:
    """
    Fire corpus entries randomly until *duration_s* seconds elapse.

    *request_rate* caps throughput to that many requests per second across
    all workers.  None means unlimited (bounded only by *max_inflight*).
    """
    import random

    rng = random.Random(rng_state)
    start = time.monotonic()
    deadline = start + duration_s

    samples: list[RampSample] = []
    seen: set[int] = set()
    n_lock = asyncio.Lock()
    shared_n = [0]

    # Rate limiting: shared token bucket state
    fire_interval = (1.0 / request_rate) if request_rate else 0.0
    rate_lock = asyncio.Lock()
    next_fire = [start]

    async def worker(worker_rng: random.Random) -> None:
        while time.monotonic() < deadline:
            if fire_interval > 0:
                async with rate_lock:
                    now = time.monotonic()
                    wait = next_fire[0] - now
                    next_fire[0] = max(now, next_fire[0]) + fire_interval
                if wait > 0:
                    await asyncio.sleep(wait)
                if time.monotonic() >= deadline:
                    break

            spec_idx = worker_rng.randrange(len(corpus))
            spec = corpus[spec_idx]

            t_req_start = time.monotonic()
            first_token_t: float | None = None
            error: str | None = None
            try:
                stream = await client.completions.create(
                    model=model,
                    prompt=spec.prompt,
                    max_tokens=output_len,
                    temperature=0.0,
                    stream=True,
                )
                async for chunk in stream:
                    if first_token_t is None and chunk.choices:
                        first_token_t = time.monotonic()
            except Exception as exc:
                error = str(exc)
                if first_token_t is None:
                    first_token_t = time.monotonic()
            t_req_end = time.monotonic()

            ttft = (first_token_t - t_req_start) if first_token_t else (t_req_end - t_req_start)
            e2e = t_req_end - t_req_start

            async with n_lock:
                shared_n[0] += 1
                seen.add(spec_idx)
                sample = RampSample(
                    t=t_req_end - start,
                    spec_idx=spec_idx,
                    ttft_s=ttft,
                    e2e_s=e2e,
                    corpus_coverage=len(seen) / len(corpus),
                    n_total=shared_n[0],
                    error=error,
                )
                samples.append(sample)

            if progress_callback:
                progress_callback(sample)

    # Each worker gets its own RNG so they don't block each other.
    worker_tasks = [
        asyncio.create_task(worker(random.Random(rng.randint(0, 2**31))))
        for _ in range(max_inflight)
    ]
    await asyncio.gather(*worker_tasks)

    return sorted(samples, key=lambda s: s.t)


def ramp_window_stats(
    samples: list[RampSample],
    window_start: float,
    window_end: float,
) -> dict:
    """Compute TTFT stats over a time window [window_start, window_end)."""
    window = [s for s in samples if window_start <= s.t < window_end and not s.error]
    if not window:
        return {}
    ttfts = sorted(s.ttft_s for s in window)
    n = len(ttfts)
    return {
        "n": n,
        "mean": statistics.mean(ttfts),
        "p50": ttfts[int(n * 0.50)],
        "p90": ttfts[int(n * 0.90)],
        "p99": ttfts[min(n - 1, int(n * 0.99))],
        "corpus_coverage": window[-1].corpus_coverage,
        "n_total": window[-1].n_total,
    }
