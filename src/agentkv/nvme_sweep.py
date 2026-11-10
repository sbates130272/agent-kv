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
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import AsyncOpenAI

from agentkv.metrics import aggregate_round
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


async def reset_all_caches(vllm_base: str) -> None:
    """Clear vLLM VRAM + LMCache L1 + L2 (NVMe).

    Uses reset_external=true so vLLM cascades the clear into LMCache,
    wiping all tiers including NVMe L2.  Used once at sweep start to
    ensure no stale data from a previous run pollutes fill fractions.
    """
    url = vllm_base.rstrip("/").removesuffix("/v1") + "/reset_prefix_cache"
    async with httpx.AsyncClient() as hc:
        r = await hc.post(url, params={"reset_external": "true"}, timeout=60)
        if r.status_code not in (200, 204):
            raise RuntimeError(
                f"reset_prefix_cache (external) failed {r.status_code}: {r.text[:200]}"
            )
    console.print("  [dim]All caches cleared (VRAM + L1 + L2 NVMe)[/]")


async def clear_lmcache_l1(lmcache_http: str) -> None:
    """Clear LMCache L1 (CPU RAM) only, leaving L2 (NVMe) intact.

    POST /cache/clear with {"tier": "l1"} — the default body already targets L1,
    so an empty POST also works, but we're explicit.
    """
    base = lmcache_http.rstrip("/")
    async with httpx.AsyncClient() as hc:
        r = await hc.post(
            f"{base}/cache/clear",
            json={"tier": "l1"},
            timeout=30,
        )
        if r.status_code not in (200, 204):
            raise RuntimeError(f"LMCache L1 clear failed {r.status_code}: {r.text[:200]}")
    console.print("  [dim]LMCache L1 cleared[/]")


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
        capture_output=True,
        text=True,
        check=True,
    )
    fields = result.stdout.strip().split()
    # Field 5 is the read delay; replace it
    fields[5] = str(read_delay_ms)
    new_table = " ".join(fields)

    subprocess.run(
        ["sudo", "dmsetup", "load", device],
        input=new_table,
        text=True,
        check=True,
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
) -> tuple[list[RequestSpec], list[RequestSpec]]:
    """Run fill_fraction of specs so their KV blocks land on NVMe L2.

    Returns (warm_specs, cold_specs) so the caller knows which prompts have
    L2 blocks and which will recompute during measurement.
    """
    n = max(1, round(len(specs) * fill_fraction))
    warm_specs = specs[:n]
    cold_specs = specs[n:]
    console.print(
        f"  Warming NVMe: {n}/{len(specs)} prompts (fill={fill_fraction:.0%})"
        + (f", {len(cold_specs)} cold (recompute)" if cold_specs else "")
    )
    await run_round(
        client,
        model,
        warm_specs,
        output_len,
        max_inflight,
        round_idx=0,
        round_label="nvme-warm",
    )
    return warm_specs, cold_specs


async def measure_point(
    client: AsyncOpenAI,
    model: str,
    specs: list[RequestSpec],
    output_len: int,
    max_inflight: int,
    rounds: int,
    vllm_base: str,
    lmcache_http: str,
) -> list[dict[str, Any]]:
    """Run `rounds` passes over specs, clearing L1+VRAM before each round.

    L1 is cleared before every pass so each round reads from NVMe L2 rather
    than from L1 populated by the previous round.  L2 (NVMe) is never touched.
    """
    results = []
    for i in range(rounds):
        console.print(f"  Clearing L1 + VRAM before round {i + 1}...")
        await clear_all_caches(vllm_base, lmcache_http)
        await asyncio.sleep(2)
        label = f"measure-{i + 1}"
        raw = await run_round(
            client,
            model,
            specs,
            output_len,
            max_inflight,
            round_idx=i + 1,
            round_label=label,
        )
        m = aggregate_round(raw)
        results.append(
            {
                "round": i + 1,
                "ttft": m.ttft,
                "tpot": m.tpot,
                "e2e": m.e2e,
                "n_requests": len(raw),
            }
        )
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

    # Wipe all tiers (VRAM + L1 + L2/NVMe) before the first point so stale
    # data from a previous run doesn't pollute the fill-fraction isolation.
    console.print("[bold]Pre-sweep: clearing all caches (VRAM + L1 + L2 NVMe)...[/]")
    await reset_all_caches(vllm_base)
    await asyncio.sleep(2)

    total = len(delays_ms) * len(fill_fractions)
    done = 0

    for fill in fill_fractions:
        for delay in delays_ms:
            done += 1
            console.rule(f"[bold]Point {done}/{total}: fill={fill:.0%}, delay={delay} ms[/]")

            # 1. Reset dm-delay to 0 for the write (warm) phase
            if dm_device:
                set_dm_delay(dm_device, 0)

            # 2. Clear both caches before warming
            console.print("  Clearing caches...")
            await clear_all_caches(vllm_base, lmcache_http)
            await asyncio.sleep(2)

            # 3. Warm NVMe — issue fill_fraction of prompts; KV blocks go to L2
            warm_specs, _ = await warm_nvme(client, model, specs, fill, output_len, max_inflight)

            # 4. Apply target read delay before measurement
            if dm_device:
                set_dm_delay(dm_device, delay)

            # 5. Measure — only the warmed specs, so every request is an L2 hit.
            #    measure_point clears L1 + VRAM before each round so reads always
            #    come from NVMe L2 rather than L1 populated by the previous round.
            metrics = await measure_point(
                client,
                model,
                warm_specs,
                output_len,
                max_inflight,
                rounds,
                vllm_base,
                lmcache_http,
            )

            point = SweepPoint(delay_ms=delay, fill_fraction=fill, round_metrics=metrics)
            sweep_results.append(point)

            # Log summary
            for m in metrics:
                ttft_mean = m["ttft"]["mean"] if m["ttft"] else float("nan")
                e2e_mean = m["e2e"]["mean"] if m["e2e"] else float("nan")
                console.print(f"  round {m['round']}: TTFT={ttft_mean:.3f}s  E2E={e2e_mean:.3f}s")

    # Restore delay to 0
    if dm_device:
        set_dm_delay(dm_device, 0)

    return sweep_results
