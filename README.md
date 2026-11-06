# kv-sweep

**kv-sweep** benchmarks LLM inference servers by sweeping prefix KV-cache fill
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
pip install kv-sweep
# optional: accurate token counting
pip install "kv-sweep[tiktoken]"
```

## Quick start

```bash
# Run against a local vLLM server
kvsweep run \
  --model Qwen/Qwen2.5-3B-Instruct \
  --host localhost --port 8000 \
  --num-books 5 \
  --rounds 4 \
  --output results/my-run.json
```

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

## CLI reference

```
kvsweep run [OPTIONS]

Options:
  --host TEXT               Inference server host.  [default: localhost]
  --port INTEGER            Inference server port.  [default: 8000]
  --base-url TEXT           Full base URL — overrides --host/--port.
  --model TEXT              [default: Qwen/Qwen2.5-3B-Instruct]
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
  --output / -o PATH        JSON output file.  [default: kvsweep-results.json]
  --expected-ttft-gain FLT  Exit 1 if TTFT speedup < this factor.
```

## Book catalog

The built-in catalog (`src/kvsweep/data/books.jsonl`) contains 100
public-domain Gutenberg books with pre-written comprehension questions.  Pass
`--catalog path/to/custom.jsonl` to use your own.  Each line must be valid JSON:

```json
{"id": 1342, "title": "Pride and Prejudice", "author": "Jane Austen",
 "questions": ["What is the main reason Mrs. Bennet is anxious?", ...]}
```

## Development

```bash
git clone https://github.com/sbates130272/kv-sweep
cd kv-sweep
pip install -e ".[tiktoken,dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
