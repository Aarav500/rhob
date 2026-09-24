#!/usr/bin/env bash
# The register runs: one full pass of hvtb_hack_rate per model over all 89 HVTB tasks,
# then eval-retry until every sample has a label, then an acceptance check on the result.
#
#   HVTB_TASKS_DIR=/data/hv-terminal-bench-2-1 bash scripts/run_register_sweep.sh haiku-4-5
#
# Run from a clean checkout of the commit the listing pins. Environment:
#   HVTB_TASKS_DIR   the pinned dataset (required)
#   INSPECT          inspect executable (default: inspect)
#   PYTHON           python with this package installed (default: python)
#   CONCURRENCY      samples, sandboxes and connections in flight (default: 8)
#   LOG_ROOT         where logs go (default: logs/register)
#   AWS_REGION       Bedrock region (default: us-east-1)
#   MAX_RETRY_ROUNDS eval-retry rounds before giving up (default: 3)
#
# Why each piece is here:
# * One invocation per model with no --sample-id, so one log covers all 89 samples; the
#   register asks for "an evaluation run of all the samples" per model. Errored samples
#   are rerun with eval-retry, which writes a new log that carries the finished ones.
# * -M read_timeout=1200. Inspect's Bedrock client reads with a 60 s timeout on the
#   non-streaming Converse API, and a long generation (a heredoc writing a whole file)
#   exceeds it; the read timeout is not retried, so the sample would error and rerun.
# * AWS_DEFAULT_REGION as well as AWS_REGION: botocore reads the former.
# * A clean git tree: every log records the commit and a dirty flag.
# * The acceptance check fails the run unless the final log has status success, 89
#   samples, no sample errors, a label on all 89, and no sample whose tool calls mostly
#   failed to parse.
set -u

MODEL_SLUG=${1:?usage: run_register_sweep.sh <haiku-4-5|opus-4-6>}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "$HERE/.." && pwd)
cd "$EVAL_DIR" || exit 2
: "${HVTB_TASKS_DIR:?set HVTB_TASKS_DIR to the pinned hv-terminal-bench-2-1 directory}"
INSPECT=${INSPECT:-inspect}
PYTHON=${PYTHON:-python}
CONCURRENCY=${CONCURRENCY:-8}
MAX_RETRY_ROUNDS=${MAX_RETRY_ROUNDS:-3}
export AWS_REGION=${AWS_REGION:-us-east-1} PYTHONIOENCODING=utf-8 HVTB_TASKS_DIR
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-$AWS_REGION}
LOG_ROOT=${LOG_ROOT:-logs/register}
READ_TIMEOUT=1200

case $MODEL_SLUG in
  haiku-4-5) MODEL="bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0" ;;
  opus-4-6) MODEL="bedrock/global.anthropic.claude-opus-4-6-v1" ;;
  *) echo "unknown model slug: $MODEL_SLUG" >&2; exit 2 ;;
esac

DIR="$LOG_ROOT/$MODEL_SLUG"
mkdir -p "$DIR"
STATUS="$LOG_ROOT/status.log"
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [$MODEL_SLUG] $*" | tee -a "$STATUS"; }

if [ -n "$(git -C "$EVAL_DIR" status --porcelain)" ]; then
  say "ABORT: the git tree is dirty; the logs must name the commit that produced them"
  git -C "$EVAL_DIR" status --porcelain | head -20 | tee -a "$STATUS"
  exit 2
fi
say "commit $(git -C "$EVAL_DIR" rev-parse HEAD), model $MODEL, concurrency $CONCURRENCY"

until aws sts get-caller-identity >/dev/null 2>&1; do
  say "WAITING: AWS credentials are not valid"
  sleep 120
done

newest_log() { ls -t "$DIR"/*.eval 2>/dev/null | head -1; }

unfinished() { # samples in the newest log that errored or have no score; -1 if unreadable
  "$PYTHON" - "$(newest_log)" <<'PY'
import sys
from inspect_ai.log import read_eval_log
try:
    log = read_eval_log(sys.argv[1])
except Exception:
    print(-1)
    raise SystemExit
samples = log.samples or []
missing = 89 - len(samples)
bad = sum(1 for s in samples if s.error is not None or not s.scores)
print(bad + max(missing, 0))
PY
}

say "BEGIN full pass"
"$INSPECT" eval hvtb_hack_detection/tasks.py@hvtb_hack_rate \
  --model "$MODEL" -M read_timeout=$READ_TIMEOUT \
  --max-sandboxes "$CONCURRENCY" --max-connections "$CONCURRENCY" --max-samples "$CONCURRENCY" \
  --no-fail-on-error --retry-on-error 1 \
  --log-dir "$DIR" --display plain >> "$DIR/console.out" 2>&1
say "END full pass exit=$?"

for round in $(seq 1 "$MAX_RETRY_ROUNDS"); do
  left=$(unfinished)
  [ "$left" = "0" ] && break
  say "RETRY round $round: $left sample(s) unfinished"
  until aws sts get-caller-identity >/dev/null 2>&1; do sleep 120; done
  "$INSPECT" eval-retry "$(newest_log)" \
    --max-sandboxes "$CONCURRENCY" --max-connections "$CONCURRENCY" --max-samples "$CONCURRENCY" \
    --no-fail-on-error --log-dir "$DIR" --display plain >> "$DIR/console.out" 2>&1
  say "END retry $round exit=$?"
done

FINAL=$(newest_log)
if "$PYTHON" scripts/sweep_report.py "$FINAL" --accept --json "$DIR/report.json" | tee -a "$STATUS"; then
  say "ACCEPTED $(basename "$FINAL")"
else
  say "NOT ACCEPTED $(basename "$FINAL"); see report above"
  exit 1
fi
