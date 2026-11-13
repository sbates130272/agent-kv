#!/usr/bin/env python3
"""
Time-based visualization of GPU kernel activity from hsa-snoop NDJSON output.

Blue  = compute kernels (GEMM, attention, SSM, norm, activation, sampling)
Green = movement kernels (blit copy, fill, tensor copy, elementwise type-cast)

Usage:
    python3 scripts/plot-hsa-snoop.py <output-dir>

Writes <output-dir>/timeline.png
"""

import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_iso(ts: str) -> float:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).timestamp()


def parse_markers(path: Path) -> dict[str, float]:
    markers = {}
    if not path.exists():
        return markers
    pattern = re.compile(
        r"=== (COLD QUERY START|COLD QUERY END|WARM QUERY START|WARM QUERY END) ==="
    )
    for line in path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            ts_m = re.match(r"\[([^\]]+)\]", line)
            if ts_m:
                raw = ts_m.group(1).replace(",", ".")
                raw = re.sub(r"(\.\d{6})\d+", r"\1", raw)
                markers[m.group(1)] = parse_iso(raw)
    return markers


MOVEMENT_PATTERNS = re.compile(
    r"rocclr_copy|rocclr_fill|rocclr_blit|rocclr_memset"
    r"|_copy_kernel|converting_copy|CatArrayBatched"
    r"|FillFunctor|fill_reverse|elementwise_kernel.*copy"
    r"|precopy_mamba",
    re.IGNORECASE,
)


def is_movement(name: str) -> bool:
    return bool(MOVEMENT_PATTERNS.search(name))


# ── Load ──────────────────────────────────────────────────────────────────────


def load(outdir: Path):
    events_file = outdir / "events.ndjson"
    markers_file = outdir / "markers.txt"

    print(f"Loading {events_file}...")
    rows = []
    with events_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e["ev"] != "dispatch":
                continue
            ts = parse_iso(e["ts"])
            dur = e.get("dur_us", 0.0)
            rows.append((ts, dur, e.get("kernel", ""), e.get("queue_uid", 0)))

    print(f"  {len(rows):,} dispatch events")
    markers = parse_markers(markers_file)
    return rows, markers


# ── Bin and plot ───────────────────────────────────────────────────────────────

COMPUTE_COLOR = "#2E86C1"  # strong blue
MOVEMENT_COLOR = "#1E8449"  # forest green
COMPUTE_ALPHA = 0.85
MOVEMENT_ALPHA = 0.85

BG = "#0F1117"
FG = "#E8E8E8"
GRID = "#2A2A35"
MARKER_COLD = "#E67E22"
MARKER_WARM = "#9B59B6"


def build_bins(rows, t_start, t_end, bin_s):
    n = max(1, int(np.ceil((t_end - t_start) / bin_s)))
    compute = np.zeros(n)
    movement = np.zeros(n)
    for ts, dur, name, _ in rows:
        if ts < t_start or ts > t_end:
            continue
        idx = min(n - 1, int((ts - t_start) / bin_s))
        if is_movement(name):
            movement[idx] += 1
        else:
            compute[idx] += 1
    t_axis = t_start + np.arange(n) * bin_s
    return t_axis, compute, movement


def fmt_secs(x, pos, t0):
    return f"{x - t0:.1f}s"


def draw_panel(
    ax,
    t_axis,
    compute,
    movement,
    t0,
    markers=None,
    cold_start=None,
    cold_end=None,
    warm_start=None,
    warm_end=None,
    title="",
    bin_s=0.1,
    shade_windows=False,
):

    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)

    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: fmt_secs(x, p, t0)))
    ax.set_xlim(t_axis[0], t_axis[-1] + bin_s)

    # shade query windows
    if shade_windows:
        if cold_start and cold_end:
            ax.axvspan(cold_start, cold_end, alpha=0.08, color=MARKER_COLD, zorder=0)
        if warm_start and warm_end:
            ax.axvspan(warm_start, warm_end, alpha=0.08, color=MARKER_WARM, zorder=0)

    # stacked area
    ax.fill_between(
        t_axis,
        0,
        compute,
        step="post",
        color=COMPUTE_COLOR,
        alpha=COMPUTE_ALPHA,
        linewidth=0,
        label="Compute",
    )
    ax.fill_between(
        t_axis,
        compute,
        compute + movement,
        step="post",
        color=MOVEMENT_COLOR,
        alpha=MOVEMENT_ALPHA,
        linewidth=0,
        label="Movement",
    )

    # grid
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f"{y:.0f}"))
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)

    # marker lines
    if cold_start:
        ax.axvline(cold_start, color=MARKER_COLD, linewidth=1.0, linestyle="--", alpha=0.9)
    if cold_end:
        ax.axvline(cold_end, color=MARKER_COLD, linewidth=1.0, linestyle=":", alpha=0.7)
    if warm_start:
        ax.axvline(warm_start, color=MARKER_WARM, linewidth=1.0, linestyle="--", alpha=0.9)
    if warm_end:
        ax.axvline(warm_end, color=MARKER_WARM, linewidth=1.0, linestyle=":", alpha=0.7)

    if title:
        ax.set_title(title, fontsize=9, color=FG, pad=4)

    ax.set_ylabel("Dispatches / bin", fontsize=8)


def main():
    if len(sys.argv) < 2:
        print("Usage: plot-hsa-snoop.py <output-dir> [out.png]", file=sys.stderr)
        sys.exit(1)

    outdir = Path(sys.argv[1])
    rows, markers = load(outdir)

    if not rows:
        print("No dispatch events found", file=sys.stderr)
        sys.exit(1)

    t_all_start = min(r[0] for r in rows)
    t_all_end = max(r[0] for r in rows)

    cold_start = markers.get("COLD QUERY START")
    cold_end = markers.get("COLD QUERY END")
    warm_start = markers.get("WARM QUERY START")
    warm_end = markers.get("WARM QUERY END")

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.subplots_adjust(hspace=0.45, left=0.07, right=0.97, top=0.92, bottom=0.07)

    gs = fig.add_gridspec(3, 1, height_ratios=[1.6, 1, 1])
    ax_full = fig.add_subplot(gs[0])
    ax_cold = fig.add_subplot(gs[1])
    ax_warm = fig.add_subplot(gs[2])

    # ── Full timeline (500 ms bins) ───────────────────────────────────────────
    BIN_FULL = 0.5
    t_axis_f, comp_f, move_f = build_bins(rows, t_all_start, t_all_end, BIN_FULL)
    draw_panel(
        ax_full,
        t_axis_f,
        comp_f,
        move_f,
        t_all_start,
        cold_start=cold_start,
        cold_end=cold_end,
        warm_start=warm_start,
        warm_end=warm_end,
        title=f"Full capture  ({t_all_end - t_all_start:.1f}s)  — 500 ms bins",
        bin_s=BIN_FULL,
        shade_windows=True,
    )

    # ── Cold query zoom (50 ms bins) ─────────────────────────────────────────
    if cold_start and cold_end:
        pad = (cold_end - cold_start) * 0.08
        t0c, t1c = cold_start - pad, cold_end + pad
        BIN_Q = 0.05
        t_axis_c, comp_c, move_c = build_bins(rows, t0c, t1c, BIN_Q)
        draw_panel(
            ax_cold,
            t_axis_c,
            comp_c,
            move_c,
            cold_start,
            cold_start=cold_start,
            cold_end=cold_end,
            title=f"Cold query  ({cold_end - cold_start:.2f}s)  — 50 ms bins",
            bin_s=BIN_Q,
        )
        ax_cold.set_xlim(t0c, t1c)
    else:
        ax_cold.set_visible(False)

    # ── Warm query zoom (50 ms bins) ─────────────────────────────────────────
    if warm_start and warm_end:
        pad = (warm_end - warm_start) * 0.08
        t0w, t1w = warm_start - pad, warm_end + pad
        BIN_Q = 0.05
        t_axis_w, comp_w, move_w = build_bins(rows, t0w, t1w, BIN_Q)
        draw_panel(
            ax_warm,
            t_axis_w,
            comp_w,
            move_w,
            warm_start,
            warm_start=warm_start,
            warm_end=warm_end,
            title=f"Warm query  ({warm_end - warm_start:.2f}s)  — 50 ms bins",
            bin_s=BIN_Q,
        )
        ax_warm.set_xlim(t0w, t1w)
    else:
        ax_warm.set_visible(False)

    # ── x-axis labels ─────────────────────────────────────────────────────────
    for ax in (ax_full, ax_cold, ax_warm):
        ax.set_xlabel("Time (s from window start)", fontsize=8)

    # ── Legend + title ────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(
            color=COMPUTE_COLOR,
            alpha=COMPUTE_ALPHA,
            label="Compute dispatches (GEMM · Attention · SSM · Norm · Activation · Sampling)",
        ),
        mpatches.Patch(
            color=MOVEMENT_COLOR,
            alpha=MOVEMENT_ALPHA,
            label="Movement dispatches (blit copy · fill · type-cast copy)",
        ),
        mpatches.Patch(color=MARKER_COLD, alpha=0.6, label="Cold query window"),
        mpatches.Patch(color=MARKER_WARM, alpha=0.6, label="Warm query window"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        fontsize=8,
        framealpha=0.15,
        labelcolor=FG,
        facecolor=BG,
        edgecolor=GRID,
        bbox_to_anchor=(0.5, 0.995),
    )

    fig.suptitle(
        "GPU Kernel Activity — vLLM + LMCache-MP  (Qwen3.5-2B · gfx1201)",
        fontsize=11,
        color=FG,
        y=1.0,
    )

    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2])
    else:
        out_path = outdir / "timeline.png"
        if not os.access(outdir, os.W_OK):
            out_path = Path.home() / "timeline.png"

    fig.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
