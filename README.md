# agent-kv

**agent-kv** benchmarks LLM inference servers by sweeping prefix KV-cache fill
rate from cold (0%) toward fully warm (~100%) and capturing how TTFT, TPOT, and
E2E latency evolve across rounds.

It uses public-domain books from [Project Gutenberg](https://www.gutenberg.org/)
as long-document prompts.  The same chunk text is re-sent verbatim each round,
so after the warmup round, any prefix-KV caching layer (vLLM + LMCache, SGLang
RadixAttention, etc.) should serve an increasing fraction from cache.

The design is inspired by [AgentX / InferenceX](https://inferencex.semianalysis.com/agentx)
from SemiAnalysis: measure a full latency trajectory, not a single operating
point, and store every result in a structured JSON file for later analysis and
visualization.

## Install

```bash
pip install agent-kv
# optional: accurate token counting
pip install "agent-kv[tiktoken]"
```

## Quick start

```bash
# Run against a local vLLM server
agent-kv run \
  --model Qwen/Qwen3.5-2B \
  --host localhost --port 8000 \
  --num-books 5 \
  --rounds 4 \
  --output results/my-run.json
```

## Docker stack (vLLM + LMCache-MP, AMD ROCm)

The `docker/` directory contains a production-ready serving stack targeting
AMD RDNA 4 GPUs (gfx1201, e.g. Radeon RX 9070 XT).  It runs:

- **vLLM** — inference engine with KV-transfer support
- **LMCache-MP** coordinator + server — two-tier KV cache (host RAM L1, NVMe L2)
- **Prometheus + Grafana** — live latency/throughput/VRAM dashboard

### Prerequisites

- AMD GPU with ROCm support (tested on gfx1201 / RX 9070 XT, 16 GiB VRAM)
- Docker with ROCm device access (`/dev/kfd`, `/dev/dri`)
- An NVMe drive mounted at `/mnt/kv-nvme` (see step 1 below)
- HuggingFace model weights pre-cached locally

### Setup and launch

```bash
# 1. Prepare the NVMe L2 tier (once, as root):
KV_DISK=/dev/disk/by-id/<your-nvme-disk-id>
mkfs.ext4 $KV_DISK
mkdir -p /mnt/kv-nvme && mount $KV_DISK /mnt/kv-nvme
mkdir -p /mnt/kv-nvme/agent-kv-l2
# Optional: add to /etc/fstab for persistence across reboots
echo "$KV_DISK /mnt/kv-nvme ext4 defaults,nofail 0 2" >> /etc/fstab

# 2. Build images (from repo root):
docker buildx bake -f docker/docker-bake.hcl

# 3. Start the stack with NVMe L2 KV offload:
MODEL=Qwen/Qwen3.5-2B \
KV_TIER=file KV_STORE=/mnt/kv-nvme/agent-kv-l2 \
  docker compose -f docker/docker-compose.yml --profile vllm-and-lmcache up -d

# 4. Wait for readiness and run a benchmark:
until curl -fsS http://localhost:8000/health; do sleep 5; done
agent-kv run --base-url http://localhost:8000/v1 --model Qwen/Qwen3.5-2B --rounds 4

# 5. Open the Grafana dashboard:
#    http://localhost:3000  (anonymous admin, no login required)
```

### Key environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3.5-2B` | HuggingFace model ID |
| `MAX_MODEL_LEN` | `4096` | Max context length in tokens |
| `GPU_UTIL` | `0.74` | Fraction of VRAM reserved for KV pool |
| `L1_SIZE_GB` | `8` | LMCache host-RAM L1 tier size |
| `CHUNK_SIZE` | `544` | KV chunk size in tokens (must be a multiple of the vLLM block size; Qwen3.5 requires 544) |
| `TRANSFER_MODE` | `lmcache_driven` | `lmcache_driven` or `engine_driven` |
| `KV_TIER` | `cpu` | `cpu` (RAM only) or `file` (RAM + NVMe) |
| `KV_STORE` | named volume | Host path for the NVMe L2 tier |
| `HF_CACHE` | `~/.cache/huggingface` | HuggingFace model cache |
| `GRAFANA_PORT` | `3000` | Host port for Grafana |
| `LMCACHE_HTTP_PORT` | `8080` | Host port for LMCache HTTP management API |

### Model notes

The stack is validated with **Qwen/Qwen3.5-2B**.  Qwen3.5 uses a hybrid
architecture (standard attention + Gated Delta Network layers) that requires:

- The hybrid KV cache manager enabled in vLLM (do **not** pass
  `--disable-hybrid-kv-cache-manager`)
- `CHUNK_SIZE=544` — Qwen3.5-2B's vLLM block size is 544 tokens (LCM of the
  attention block size and GDN state alignment); LMCache requires chunk size to
  be a multiple of the vLLM block size

Qwen3.5-4B does not fit alongside LMCache-MP on a 16 GiB card (8.87 GiB model
weights + ~4 GiB LMCache IPC overhead leaves insufficient VRAM for the KV pool).

## NVMe sweep benchmark

`agent-kv nvme-sweep` measures how NVMe read latency affects TTFT as the L2 KV
cache fills.  It uses Linux's `dm-delay` device mapper to inject a controlled
read latency, letting you sweep latency and fill fraction independently without
needing multiple drives.

For each `(delay_ms, fill_fraction)` point the sweep:

1. Clears VRAM and L1 caches
2. Warms the NVMe L2 tier by running `fill_fraction` of prompts with delay = 0 ms
3. Clears VRAM + L1 again so only the NVMe L2 holds the KV blocks
4. Sets the dm-delay read latency to `delay_ms`
5. Runs `--rounds` measurement passes, clearing L1 + VRAM before each round to
   force every measurement through the NVMe path

### dm-delay setup

```bash
# Set up the dm-delay device (as root, once per boot):
sudo scripts/setup-dm-delay.sh
```

The script wraps the NVMe partition in a `dm-delay` device named `kv-delay`.
Mount the kv-store from `/dev/mapper/kv-delay` instead of the raw NVMe device
so LMCache reads pass through the delay layer.

### Running a sweep

```bash
agent-kv nvme-sweep \
  --model Qwen/Qwen3.5-2B \
  --wait-for-health \
  --dm-device kv-delay \
  --lmcache-http-url http://localhost:8080 \
  --delays 0,100,1000 \
  --fill-fractions 1.0 \
  --num-books 16 \
  --rounds 3
```

Results are written to `nvme-sweep-results.json`.

### nvme-sweep options

```
Options:
  --host TEXT                 vLLM server host.  [default: localhost]
  --port INTEGER              vLLM server port.  [default: 8000]
  --base-url TEXT             Full base URL — overrides --host/--port.
  --model TEXT                [default: Qwen/Qwen3.5-2B]
  --lmcache-http-url TEXT     LMCache HTTP management API base URL.
                              [default: http://localhost:8080]
  --dm-device TEXT            dm-delay device name (e.g. kv-delay).
                              Omit to skip latency injection.
  --delays TEXT               Comma-separated read delays in ms.
                              [default: 0,10,50,100,200]
  --fill-fractions TEXT       Comma-separated NVMe fill fractions (0.0–1.0).
                              [default: 0.25,0.50,0.75,1.0]
  --num-books INTEGER         Max books from catalog.  [default: 2]
  --chunk-tokens INTEGER      Target tokens per chunk.  [default: 4096]
  --rounds INTEGER            Measurement rounds per (delay, fill) point.
                              [default: 2]
  -o, --output PATH           JSON output file.  [default: nvme-sweep-results.json]
  --wait-for-health           Poll /health before starting.
  --health-timeout INTEGER    Seconds to wait for /health.  [default: 300]
```

Requires `VLLM_SERVER_DEV_MODE=1` (for `POST /reset_prefix_cache`) and an
LMCache HTTP port (for cache clear).  `sudo dmsetup` must be passwordless if
`--dm-device` is set.

## What it measures

| Metric | Description |
|--------|-------------|
| **TTFT** | Time to first token — dominated by prompt processing and cache lookup |
| **TPOT** | Time per output token — `(E2E − TTFT) / (output_tokens − 1)` |
| **E2E** | Total end-to-end latency for the full response |
| **Cache hit rate** | Estimated fraction of requests whose TTFT < 50% of warmup TTFT |

Metrics are reported per round as `mean / p50 / p90 / p99` and written to the
output JSON alongside per-request rows for deeper analysis.

## Output format

Each run produces a single JSON file:

```json
{
  "run_id": "...",
  "timestamp": "2026-11-06T...",
  "schema_version": 1,
  "config": { ... },
  "rounds": [
    {
      "round": 0,
      "label": "warmup",
      "n_total": 40,
      "n_errors": 0,
      "estimated_cache_hit_rate": null,
      "ttft": { "mean": 2.14, "p50": 2.0, "p90": 3.1, "p99": 3.5, "min": 1.2, "max": 3.8 },
      "tpot": { ... },
      "e2e":  { ... }
    },
    {
      "round": 1,
      "label": "query-1",
      "estimated_cache_hit_rate": 0.85,
      ...
    }
  ],
  "summary": {
    "ttft_speedup_warmup_to_last_round": 3.2,
    "final_cache_hit_rate": 0.92,
    "n_rounds_total": 5
  },
  "per_request": [ ... ]
}
```

The schema is stable across runs of the same `schema_version`, making it
suitable for appending to a time-series database or plotting a family of runs
side-by-side (e.g. with pandas or DuckDB).

## CLI reference (`agent-kv run`)

```
agent-kv run [OPTIONS]

Options:
  --host TEXT               Inference server host.  [default: localhost]
  --port INTEGER            Inference server port.  [default: 8000]
  --base-url TEXT           Full base URL — overrides --host/--port.
  --model TEXT              [default: Qwen/Qwen3.5-2B]
  --catalog PATH            Custom JSONL book catalog.
  --book-ids TEXT           Comma-separated Gutenberg IDs.
  --num-books INTEGER       Max books (0 = all).  [default: 0]
  --chunk-tokens INTEGER    Target tokens per chunk.  [default: 4096]
  --token-align INTEGER     Token alignment for cache-friendly prefix offsets.
                            [default: 256]
  --max-chunks INTEGER      Max chunks per book.  [default: 4]
  --questions-per-chunk INT Questions per chunk per round.  [default: 2]
  --output-len INTEGER      Max output tokens per answer.  [default: 200]
  --max-inflight INTEGER    Concurrent in-flight requests.  [default: 4]
  --rounds INTEGER          Query rounds after warmup.  [default: 3]
  --sleep-between FLOAT     Seconds between rounds.  [default: 2.0]
  --shuffle / --no-shuffle  Shuffle request order each round.  [default: no]
  --shuffle-seed INTEGER    [default: 42]
  --cache-dir PATH          Local Gutenberg text cache.
  --output / -o PATH        JSON output file.  [default: agent-kv-results.json]
  --expected-ttft-gain FLT  Exit 1 if TTFT speedup < this factor.
```

## Book catalog

The built-in catalog (`src/agentkv/data/books.jsonl`) contains 100
public-domain Gutenberg books with pre-written comprehension questions.  Pass
`--catalog path/to/custom.jsonl` to use your own.  Each line must be valid JSON:

```json
{"id": 1342, "title": "Pride and Prejudice", "author": "Jane Austen",
 "questions": ["What is the main reason Mrs. Bennet is anxious?", ...]}
```

## Development

```bash
git clone https://github.com/sbates130272/agent-kv
cd agent-kv
uv pip install -e ".[tiktoken,dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
