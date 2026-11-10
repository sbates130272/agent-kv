"""NVMe latency sweep: measure TTFT/E2E as a function of dm-delay read latency
and the fraction of KV blocks resident only on NVMe (not in VRAM or L1 CPU RAM).

Requires:
  - vLLM running with VLLM_SERVER_DEV_MODE=1  (enables /reset_prefix_cache)
  - LMCache running with --http-port 8080      (enables DELETE /cache/clear)
  - KV_TIER=file, KV_STORE on a dm-delay device
  - `sudo dmsetup` available without a password prompt (NOPASSWD in sudoers)
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import AsyncOpenAI

from agentkv.metrics import RequestResult, aggregate_round
from agentkv.report import console
from agentkv.runner import RequestSpec, run_round


@dataclass
class SweepPoint:
    delay_ms: int
    fill_fraction: float
    round_metrics: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


async def reset_vllm_prefix_cache(vllm_base: str) -> None:
    """Clear vLLM GPU KV cache only (no reset_external — keeps NVMe L2 intact)."""
    url = vllm_base.rstrip("/").removesuffix("/v1") + "/reset_prefix_cache"
    async with httpx.AsyncClient() as hc:
        r = await hc.post(url, timeout=30)
        if r.status_code not in (200, 204):
            raise RuntimeError(f"reset_prefix_cache failed {r.status_code}: {r.text[:200]}")
    console.print("  [dim]vLLM VRAM KV cache cleared[/]")


async def clear_lmcache_l1(lmcache_http: str) -> None:
    """Clear LMCache L1 (CPU RAM) only, leaving L2 (NVMe) intact.

    Tries DELETE /cache/clear?locations=LocalCPUBackend first; falls back to
    POST /clear-cache which wipes all L1 on the standalone MP server.
    """
    base = lmcache_http.rstrip("/")
    async with httpx.AsyncClient() as hc:
        try:
            r = await hc.delete(
                f"{base}/cache/clear",
                params={"locations": "LocalCPUBackend"},
                timeout=30,
            )
            if r.status_code in (200, 204):
                console.print("  [dim]LMCache L1 cleared (internal API)[/]")
                return
        except Exception:
            pass
        # Fallback
        r = await hc.post(f"{base}/clear-cache", timeout=30)
        if r.status_code not in (200, 204):
            raise RuntimeError(f"LMCache L1 clear failed {r.status_code}: {r.text[:200]}")
    console.print("  [dim]LMCache L1 cleared (clear-cache endpoint)[/]")


async def clear_all_caches(vllm_base: str, lmcache_http: str) -> None:
    await reset_vllm_prefix_cache(vllm_base)
    await clear_lmcache_l1(lmcache_http)


# ---------------------------------------------------------------------------
# dm-delay control
# ---------------------------------------------------------------------------


def set_dm_delay(device: str, read_delay_ms: int) -> None:
    """Update the read delay on a dm-delay device (requires sudo dmsetup)."""
    # Get current table: "0 <sectors> delay <dev> 0 <r_ms> <dev> 0 <w_ms>"
    result = subprocess.run(
        ["sudo", "dmsetup", "table", device],
        capture_output=True, text=True, check=True,
    )
    fields = result.stdout.strip().split()
    # Field 5 is the read delay; replace it
    fields[5] = str(read_delay_ms)
    new_table = " ".join(fields)

    subprocess.run(
        ["sudo", "dmsetup", "load", device],
        input=new_table, text=True, check=True,
    )
    subprocess.run(["sudo", "dmsetup", "resume", device], check=True)
    console.print(f"  [dim]dm-delay {device}: read latency → {read_delay_ms} ms[/]")


# ---------------------------------------------------------------------------
# Warm and measure helpers
# ---------------------------------------------------------------------------


async def warm_nvme(
    client: AsyncOpenAI,
    model: str,
    specs: list[RequestSpec],
    fill_fraction: float,
    output_len: int,
    max_inflight: int,
) -> None:
    """Run fill_fraction of specs so their KV blocks land on NVMe L2."""
    n = max(1, round(len(specs) * fill_fraction))
    warm_specs = specs[:n]
    console.print(f"  Warming NVMe: {n}/{len(specs)} prompts (fill={fill_fraction:.0%})")
    await run_round(
        client, model, warm_specs, output_len, max_inflight,
        round_idx=0, round_label="nvme-warm",
    )


async def measure_point(
    client: AsyncOpenAI,
    model: str,
    specs: list[RequestSpec],
    output_len: int,
    max_inflight: int,
    rounds: int,
) -> list[dict[str, Any]]:
    """Run `rounds` passes over all specs; return per-round aggregated metrics."""
    results = []
    for i in range(rounds):
        label = f"measure-{i + 1}"
        raw = await run_round(
            client, model, specs, output_len, max_inflight,
            round_idx=i + 1, round_label=label,
        )
        m = aggregate_round(raw)
        results.append({
            "round": i + 1,
            "ttft": m.ttft,
            "tpot": m.tpot,
            "e2e": m.e2e,
            "n_requests": len(raw),
        })
    return results


# ---------------------------------------------------------------------------
# Top-level sweep
# ---------------------------------------------------------------------------


async def run_nvme_sweep(
    *,
    vllm_base: str,
    lmcache_http: str,
    dm_device: str | None,
    model: str,
    specs: list[RequestSpec],
    delays_ms: list[int],
    fill_fractions: list[float],
    output_len: int,
    max_inflight: int,
    rounds: int,
    api_key: str,
) -> list[SweepPoint]:
    client = AsyncOpenAI(base_url=vllm_base, api_key=api_key)
    sweep_results: list[SweepPoint] = []

    total = len(delays_ms) * len(fill_fractions)
    done = 0

    for fill in fill_fractions:
        for delay in delays_ms:
            done += 1
            console.rule(
                f"[bold]Point {done}/{total}: fill={fill:.0%}, delay={delay} ms[/]"
            )

            # 1. Reset dm-delay to 0 for the write (warm) phase
            if dm_device:
                set_dm_delay(dm_device, 0)

            # 2. Clear both caches before warming
            console.print("  Clearing caches...")
            await clear_all_caches(vllm_base, lmcache_http)
            await asyncio.sleep(2)

            # 3. Warm NVMe — issue fill_fraction of prompts; KV blocks go to L2
            await warm_nvme(client, model, specs, fill, output_len, max_inflight)

            # 4. Clear L1 and VRAM again so only NVMe has the blocks
            console.print("  Clearing L1 + VRAM (NVMe only)...")
            await clear_all_caches(vllm_base, lmcache_http)
            await asyncio.sleep(2)

            # 5. Apply target read delay
            if dm_device:
                set_dm_delay(dm_device, delay)

            # 6. Measure
            metrics = await measure_point(
                client, model, specs, output_len, max_inflight, rounds
            )

            point = SweepPoint(delay_ms=delay, fill_fraction=fill, round_metrics=metrics)
            sweep_results.append(point)

            # Log summary
            for m in metrics:
                ttft_mean = m["ttft"]["mean"] if m["ttft"] else float("nan")
                e2e_mean = m["e2e"]["mean"] if m["e2e"] else float("nan")
                console.print(
                    f"  round {m['round']}: TTFT={ttft_mean:.3f}s  E2E={e2e_mean:.3f}s"
                )

    # Restore delay to 0
    if dm_device:
        set_dm_delay(dm_device, 0)

    return sweep_results
