#!/usr/bin/env python3
"""
Parse hsa-snoop NDJSON output captured during cold+warm LLM queries.

Usage:
    python3 scripts/parse-hsa-snoop.py <output-dir>

Expects <output-dir>/events.ndjson and <output-dir>/markers.txt
produced by hsa-snoop-capture.sh.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def parse_iso(ts: str) -> datetime:
    # Events end in 'Z' (UTC); markers carry a local offset like '-06:00'.
    # Normalise everything to UTC-aware so comparisons work across both.
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_markers(path: Path) -> dict[str, datetime]:
    markers = {}
    if not path.exists():
        return markers
    pattern = re.compile(r"=== (COLD QUERY START|COLD QUERY END|WARM QUERY START|WARM QUERY END) ===")
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        # Timestamp is on same line: "[<ts>] === ... ==="
        m = pattern.search(line)
        if m:
            ts_m = re.match(r"\[([^\]]+)\]", line)
            if ts_m:
                raw = ts_m.group(1)
                # date -Ins uses comma for sub-second separator on some locales
                raw = raw.replace(",", ".")
                # nanosecond precision — strip to microseconds
                raw = re.sub(r"(\.\d{6})\d+", r"\1", raw)
                markers[m.group(1)] = parse_iso(raw)
    return markers


def categorize_kernel(name: str) -> str:
    """Classify a demangled kernel name into a broad LLM workload bucket."""
    n = name.lower()
    if "gemm" in n or "matmul" in n or "mm_" in n or "hgemm" in n or "sgemm" in n:
        return "matmul/gemm"
    if "attention" in n or "flash" in n or "softmax" in n or "attn" in n:
        return "attention"
    if "layernorm" in n or "layer_norm" in n or "rms_norm" in n:
        return "layernorm"
    if "rotary" in n or "rope" in n:
        return "rope"
    if "elementwise" in n or "vectorized" in n or "unary" in n or "binary" in n:
        return "elementwise"
    if "copy" in n or "memcpy" in n or "fill" in n:
        return "copy/fill"
    if "gather" in n or "index" in n or "scatter" in n or "embedding" in n:
        return "gather/embedding"
    if "topk" in n or "argmax" in n or "sample" in n:
        return "sampling"
    if "reduce" in n or "sum" in n or "mean" in n:
        return "reduce"
    if "cache" in n or "paged" in n or "kv_" in n:
        return "kv-cache"
    return "other"


def analyze_window(events: list[dict], label: str) -> None:
    dispatches = [e for e in events if e["ev"] == "dispatch"]
    sdma_evts = [e for e in events if e["ev"] == "sdma"]

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Kernel dispatches : {len(dispatches)}")
    print(f"  SDMA events       : {len(sdma_evts)}")

    if not dispatches and not sdma_evts:
        print("  (no events in this window)")
        return

    # --- Kernel breakdown ---
    if dispatches:
        total_dur = sum(e.get("dur_us", 0) for e in dispatches)
        cat_counts: Counter = Counter()
        cat_dur: defaultdict = defaultdict(float)
        kernel_counts: Counter = Counter()

        for e in dispatches:
            cat = categorize_kernel(e.get("kernel", ""))
            cat_counts[cat] += 1
            cat_dur[cat] += e.get("dur_us", 0)
            kernel_counts[e.get("kernel", "?")] += 1

        print(f"\n  Total host-observed GPU time: {total_dur/1e3:.1f} ms")
        print("\n  By category (count | host-time ms | %time):")
        for cat, cnt in cat_counts.most_common():
            dur = cat_dur[cat]
            pct = 100 * dur / total_dur if total_dur else 0
            print(f"    {cat:<20} {cnt:5d} launches  {dur/1e3:8.1f} ms  {pct:5.1f}%")

        print("\n  Top 15 kernels by launch count:")
        for kname, cnt in kernel_counts.most_common(15):
            dur = sum(e.get("dur_us", 0) for e in dispatches if e.get("kernel") == kname)
            # Truncate long demangled names
            short = kname[:90] + ("…" if len(kname) > 90 else "")
            print(f"    {cnt:5d}x  {dur/1e3:7.1f}ms  {short}")

    # --- SDMA breakdown ---
    if sdma_evts:
        print("\n  SDMA copy activity:")
        for direction in ("h2d", "d2h", "d2d", "unknown"):
            subset = [e for e in sdma_evts if e.get("dir") == direction]
            if not subset:
                continue
            total_bytes = sum(e.get("bytes", 0) for e in subset)
            total_dur = sum(e.get("dur_us", 0) for e in subset)
            bw = (total_bytes / total_dur * 1e6 / 1e9) if total_dur else 0
            print(f"    {direction:7s}  {len(subset):4d} ops  "
                  f"{total_bytes/1e6:8.1f} MB  {bw:5.1f} GB/s est.")


def interpret(cold_events: list[dict], warm_events: list[dict]) -> None:
    """Print a human-readable interpretation of cold vs warm GPU activity."""
    cold_disp = [e for e in cold_events if e["ev"] == "dispatch"]
    warm_disp = [e for e in warm_events if e["ev"] == "dispatch"]
    cold_sdma = [e for e in cold_events if e["ev"] == "sdma"]
    warm_sdma = [e for e in warm_events if e["ev"] == "sdma"]

    print(f"\n{'='*70}")
    print("  INTERPRETATION: Cold vs Warm")
    print(f"{'='*70}")

    dispatch_ratio = len(warm_disp) / len(cold_disp) if cold_disp else float("nan")
    print(f"\n  Kernel dispatch count: cold={len(cold_disp)}, warm={len(warm_disp)}, ratio={dispatch_ratio:.2f}x")

    cold_sdma_bytes = sum(e.get("bytes", 0) for e in cold_sdma)
    warm_sdma_bytes = sum(e.get("bytes", 0) for e in warm_sdma)
    print(f"  SDMA bytes moved:      cold={cold_sdma_bytes/1e6:.1f} MB, warm={warm_sdma_bytes/1e6:.1f} MB")

    cold_dur = sum(e.get("dur_us", 0) for e in cold_disp)
    warm_dur = sum(e.get("dur_us", 0) for e in warm_disp)
    print(f"  Total host-GPU time:   cold={cold_dur/1e3:.1f} ms, warm={warm_dur/1e3:.1f} ms")

    print()
    if dispatch_ratio < 0.7:
        print("  ✓ Warm launched fewer kernels — prefix cache hit: prefill skipped for")
        print("    cached tokens.  Only new decode tokens needed compute.")
    elif dispatch_ratio < 0.95:
        print("  ~ Warm launched somewhat fewer kernels — partial prefix cache hit.")
    elif dispatch_ratio < 1.5:
        print("  ! Warm and cold dispatch counts are similar — cache may not have been")
        print("    hit, or the query is too short to cross the prefix cache threshold.")
        print("    Check vLLM logs for cache hit rate.")
    else:
        print("  ✓ Warm launched MORE kernels than cold.  This is the CUDAGraph boundary")
        print("    effect: cold prefill + decode runs within the graph-captured seq_len")
        print("    range so decode steps are submitted as batched graph replays (fewer")
        print("    visible AQL events per step).  Warm hit the prefix cache → decode")
        print("    starts at a longer initial seq_len that falls outside the capture")
        print("    range → vLLM falls back to individual kernel dispatch → more visible")
        print("    events per step, same (or shorter) wall-clock decode time.")
        print("    Both paths produce the same 300 output tokens; the difference is")
        print("    hsa-snoop visibility, not GPU work.")

    if warm_sdma_bytes < cold_sdma_bytes * 0.5:
        print("  ✓ Warm query moved substantially less data via SDMA — LMCache KV")
        print("    retrieval from CPU/NVMe L2 store replaced GPU re-computation,")
        print("    eliminating the KV transfer that would happen on a true cache miss.")
    elif warm_sdma_bytes > cold_sdma_bytes * 1.3:
        print("  ~ Warm query moved MORE data via SDMA — this suggests LMCache is")
        print("    actively copying KV tensors back to GPU from L1/L2 store (the")
        print("    retrieval path), which is expected when LMCacheMPConnector is active.")

    print()
    print("  Key kernel types to expect in a vLLM + LMCache-MP workload:")
    print("    matmul/gemm   : QKV projection, FFN up/gate/down — dominates prefill")
    print("    attention      : Flash-attention forward pass for new tokens")
    print("    kv-cache       : paged_copy_kernel, reshape_and_cache — KV staging")
    print("    layernorm/rope : Per-layer normalization and positional encoding")
    print("    elementwise    : Activation (SiLU/GeLU), residual adds, casting")
    print("    sampling       : Top-k/top-p sampling, argmax — decode phase")
    print("    copy/fill      : SDMA-side: LMCache KV block H↔D transfers")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: parse-hsa-snoop.py <output-dir>", file=sys.stderr)
        sys.exit(1)

    outdir = Path(sys.argv[1])
    events_file = outdir / "events.ndjson"
    markers_file = outdir / "markers.txt"

    if not events_file.exists():
        print(f"ERROR: {events_file} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading events from {events_file}...")
    all_events = []
    with events_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"Loaded {len(all_events)} events total")

    queue_evts = [e for e in all_events if e["ev"] == "queue"]
    print(f"  Queues discovered: {len(queue_evts)}")
    for q in queue_evts:
        print(f"    queue_uid={q.get('queue_uid')} type={q.get('qtype')} "
              f"pid={q.get('pid')} comm={q.get('comm')}")

    markers = load_markers(markers_file)
    if not markers:
        print("\nWARNING: could not parse markers — analyzing full event stream as one window")
        analyze_window(all_events, "All events")
        return

    t_cold_start = markers.get("COLD QUERY START")
    t_cold_end = markers.get("COLD QUERY END")
    t_warm_start = markers.get("WARM QUERY START")
    t_warm_end = markers.get("WARM QUERY END")

    def in_window(e: dict, t0, t1) -> bool:
        if t0 is None or t1 is None:
            return False
        try:
            ts = parse_iso(e["ts"])
            return t0 <= ts <= t1
        except Exception:
            return False

    cold_events = [e for e in all_events if in_window(e, t_cold_start, t_cold_end)]
    warm_events = [e for e in all_events if in_window(e, t_warm_start, t_warm_end)]

    print(f"\nTime windows:")
    if t_cold_start and t_cold_end:
        print(f"  Cold: {t_cold_start.isoformat()} → {t_cold_end.isoformat()}")
    if t_warm_start and t_warm_end:
        print(f"  Warm: {t_warm_start.isoformat()} → {t_warm_end.isoformat()}")

    analyze_window(cold_events, "COLD QUERY (first request, no cache)")
    analyze_window(warm_events, "WARM QUERY (same prompt, cache should hit)")

    if cold_events or warm_events:
        interpret(cold_events, warm_events)


if __name__ == "__main__":
    main()
