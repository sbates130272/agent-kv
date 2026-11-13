#!/usr/bin/env bash
# Capture GPU kernel + SDMA activity during one cold and one warm LLM query.
#
# Three stack-management modes:
#
#   --no-restart    Skip container lifecycle management; existing queues are
#                   enumerated automatically at startup (kfd-scan is now always-on
#                   in --all mode). Best for iterating when the stack is healthy.
#
#   --vllm-only     Stops and restarts only the vllm + lmcache containers.
#                   Monitoring and coordinator stay up; LMCache L1 KV store
#                   survives. kprobe catches queue creation on restart.
#
#   --full-restart  (default) Tears down and rebuilds the entire stack.
#                   Guarantees a clean slate.
#
# Requires root (sudo) for hsa-snoop.
#
# Usage:
#   sudo ./scripts/hsa-snoop-capture.sh [--no-restart|--vllm-only|--full-restart] \
#       [--prompt-tokens N] [--book-id ID] [--gutenberg-cache DIR] \
#       [--poll-us N] [output-dir]
#
# --prompt-tokens N    Token count for Gutenberg prompt (default: 512; 0 = short fallback)
# --book-id ID         Gutenberg book ID (default: 84 = Frankenstein)
# --gutenberg-cache D  Local cache dir for downloaded books (default: ~/.cache/agent-kv/gutenberg)
# --poll-us N          hsa-snoop ring poll interval in microseconds (default: 10)
#
# Output dir defaults to /tmp/hsa-snoop-$(date +%Y%m%d-%H%M%S)
set -euo pipefail

HSA_SNOOP=/home/stebates/Projects/hsa-snoop/build/hsa-snoop
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/docker/docker-compose.yml"
PROFILE=vllm-and-lmcache
ENDPOINT=http://localhost:8000
MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
MAX_TOKENS=300
# Salt makes the prompt unique per run, preventing cross-run cache pollution.
# Both cold and warm queries within a run share the same salt so cold populates
# the cache and warm hits it.
SALT="${SALT:-$(uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid)}"

# Gutenberg prompt options (alternative to the default short prompt).
# Set PROMPT_TOKENS=0 to use the built-in short prompt instead.
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
BOOK_ID="${BOOK_ID:-84}"
POLL_US="${POLL_US:-10}"
# When running under sudo, $HOME is /root; use the invoking user's home instead.
_REAL_HOME="${SUDO_USER:+$(getent passwd "$SUDO_USER" 2>/dev/null | cut -d: -f6)}"
_REAL_HOME="${_REAL_HOME:-$HOME}"
GUTENBERG_CACHE="${GUTENBERG_CACHE:-${_REAL_HOME}/.cache/agent-kv/gutenberg}"

RESTART_MODE=full
OUTDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-restart)      RESTART_MODE=none; shift ;;
    --full-restart)    RESTART_MODE=full; shift ;;
    --vllm-only)       RESTART_MODE=vllm-only; shift ;;
    --prompt-tokens)   PROMPT_TOKENS="$2"; shift 2 ;;
    --book-id)         BOOK_ID="$2"; shift 2 ;;
    --gutenberg-cache) GUTENBERG_CACHE="$2"; shift 2 ;;
    --poll-us)         POLL_US="$2"; shift 2 ;;
    *) OUTDIR="$1"; shift ;;
  esac
done

OUTDIR="${OUTDIR:-/tmp/hsa-snoop-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUTDIR"
EVENTS="$OUTDIR/events.ndjson"
MARKERS="$OUTDIR/markers.txt"
COLD_RESP="$OUTDIR/cold-response.json"
WARM_RESP="$OUTDIR/warm-response.json"

log() { echo "[$(date -Ins)] $*" | tee -a "$MARKERS"; }

if [[ $EUID -ne 0 ]]; then
  echo "ERROR: must run as root (hsa-snoop --all requires CAP_SYS_ADMIN)" >&2
  exit 1
fi

if [[ ! -x "$HSA_SNOOP" ]]; then
  echo "ERROR: $HSA_SNOOP not found or not executable" >&2
  exit 1
fi

log "=== hsa-snoop LLM capture (mode=$RESTART_MODE) ==="
log "Output dir: $OUTDIR"
log "Model: $MODEL"
log "Salt: $SALT"
log "Prompt: book=$BOOK_ID tokens=$PROMPT_TOKENS"

# Build prompt: use Gutenberg text when PROMPT_TOKENS > 0, else a short fallback.
VENV_PYTHON="$REPO_ROOT/.venv/bin/python3"
if [[ "$PROMPT_TOKENS" -gt 0 && -x "$VENV_PYTHON" ]]; then
  log "Generating Gutenberg prompt: book=$BOOK_ID target_tokens=$PROMPT_TOKENS"
  PROMPT=$("$VENV_PYTHON" "$SCRIPT_DIR/make-prompt.py" \
    --book-id "$BOOK_ID" \
    --tokens "$PROMPT_TOKENS" \
    --model "$MODEL" \
    --cache-dir "$GUTENBERG_CACHE" \
    --salt "$SALT" 2>>"$MARKERS")
  log "Prompt generated (${#PROMPT} chars)"
else
  PROMPT="[salt:${SALT}] Explain how transformer attention mechanisms work in detail, including the mathematics of scaled dot-product attention, multi-head attention, and why positional encodings are necessary."
  log "Using short fallback prompt (PROMPT_TOKENS=0 or venv not found)"
fi

# ── 1. Stop GPU-using containers if needed ────────────────────────────────────
if [[ "$RESTART_MODE" == "full" ]]; then
  log "Stopping entire stack..."
  docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" down --timeout 30 2>&1 | tee -a "$MARKERS"
elif [[ "$RESTART_MODE" == "vllm-only" ]]; then
  log "Stopping vllm + lmcache containers (monitoring + coordinator stay up)..."
  docker compose -f "$COMPOSE_FILE" stop vllm 2>&1 | tee -a "$MARKERS"
  docker compose -f "$COMPOSE_FILE" stop lmcache 2>&1 | tee -a "$MARKERS"
fi

# ── 2. Start hsa-snoop ────────────────────────────────────────────────────────
log "Starting hsa-snoop (--all, poll=${POLL_US}µs)..."
"$HSA_SNOOP" \
  --all \
  --poll-us "$POLL_US" \
  --out-dir "$OUTDIR" \
  --log-dispatches "$EVENTS" \
  &
HSA_PID=$!
log "hsa-snoop pid=$HSA_PID, events -> $EVENTS"

# Give hsa-snoop kfd-scan and kprobe arm a moment to complete
sleep 2

# ── 3. Restart containers if needed ──────────────────────────────────────────
if [[ "$RESTART_MODE" == "full" ]]; then
  log "Starting full stack (profile=$PROFILE)..."
  docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" up -d 2>&1 | tee -a "$MARKERS"
elif [[ "$RESTART_MODE" == "vllm-only" ]]; then
  log "Starting lmcache then vllm..."
  docker compose -f "$COMPOSE_FILE" start lmcache 2>&1 | tee -a "$MARKERS"
  sleep 5
  docker compose -f "$COMPOSE_FILE" start vllm 2>&1 | tee -a "$MARKERS"
fi

log "Waiting for vLLM to become healthy..."
until curl -fsS --max-time 5 "$ENDPOINT/health" >/dev/null 2>&1; do
  sleep 5
done
log "vLLM healthy"

# Short settle — let any startup model-load kernels finish (skip for no-restart)
[[ "$RESTART_MODE" != "none" ]] && sleep 10

# ── 4. Clear caches so cold query is a true miss ──────────────────────────────
log "Clearing vLLM prefix cache and LMCache L1..."
curl -fsS -X POST "$ENDPOINT/reset_prefix_cache" >/dev/null
curl -fsS -X POST "http://localhost:${LMCACHE_HTTP_PORT:-8080}/cache/clear" >/dev/null
log "Caches cleared"

# Build JSON payload once; jq handles newlines, quotes, and backslashes correctly.
PAYLOAD=$(jq -n \
  --arg     model  "$MODEL" \
  --arg     content "$PROMPT" \
  --argjson max    "$MAX_TOKENS" \
  '{model: $model, messages: [{role: "user", content: $content}], max_tokens: $max}')

# ── 5. Cold query ─────────────────────────────────────────────────────────────
log "=== COLD QUERY START ==="
curl -sS "$ENDPOINT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$PAYLOAD" \
  -o "$COLD_RESP"
log "=== COLD QUERY END ==="

# Allow trailing GPU work and LMCache L1 store to drain
sleep 5

# ── 6. Warm query (identical salted prompt → cache hit) ──────────────────────
log "=== WARM QUERY START ==="
curl -sS "$ENDPOINT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$PAYLOAD" \
  -o "$WARM_RESP"
log "=== WARM QUERY END ==="

sleep 5

# ── 7. Stop hsa-snoop ─────────────────────────────────────────────────────────
log "Stopping hsa-snoop (pid=$HSA_PID)..."
kill "$HSA_PID" 2>/dev/null || true
wait "$HSA_PID" 2>/dev/null || true
log "hsa-snoop stopped. Events file: $EVENTS ($(wc -l < "$EVENTS" 2>/dev/null || echo 0) lines)"

log "=== Capture complete. Run: python3 scripts/parse-hsa-snoop.py $OUTDIR ==="
