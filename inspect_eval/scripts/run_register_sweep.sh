#!/usr/bin/env bash
# The register sweep: two models over all 89 HVTB tasks, as actually run.
#
#   HVTB_TASKS_DIR=/path/to/hv-terminal-bench-2-1 bash scripts/run_register_sweep.sh
#
# Optional: INSPECT (default: inspect), PYTHON (default: python), AWS_REGION (default
# us-east-1), OPUS_BUDGET_USD (default 600), LOG_ROOT (default logs/register),
# PLAN (default "haiku-4-5:light haiku-4-5:heavy opus-4-6:light opus-4-6:heavy"),
# LIGHT_CONCURRENCY (default 4), HEAVY_CONCURRENCY (default 1).
#
# PLAN and the concurrency knobs exist because the 2026-09-24 register sweep moved hosts
# mid-way: Haiku's light pass ran on a 16 GB laptop, which then ran out of RAM (0.6 GB
# free, a 35 GB page file), and the remaining passes ran on a 32 vCPU / 128 GiB EC2
# instance where the 8 GB tasks get their declared memory and can run four at a time.
# Same commit for the eval code, same image digests, same per-task limits on both hosts.
#
# Why it is shaped like this -- each point was a failure found before or during setup:
#
# * Two passes per model. Docker Desktop's VM gets half the host's RAM by default (about
#   7.6 GiB on a 16 GB machine) and the tasks that declare memory_mb = 8192 cannot share
#   it with three others. They run one at a time after the rest run four at a time.
#   scripts/sweep_report.py pools the passes; no single log holds a model's 89.
# * -M read_timeout=1200. Inspect's Bedrock client reads with a 60 s timeout and calls
#   the non-streaming Converse API, so any generation longer than a minute -- an Opus
#   heredoc writing a solution file -- raises ReadTimeoutError, which Inspect does not
#   retry: the sample errors and reruns from scratch.
# * --no-fail-on-error, then eval-retry. A fractional --fail-on-error trips at 2 errors
#   in an 8-sample pass and leaves a log with no results. Errors are instead recorded,
#   and the errored samples are rerun once credentials are confirmed good.
# * A credential check before every pass. `aws login` sessions end; a pass launched on
#   expired credentials errors every sample in seconds. The script waits instead.
# * A clean git tree. Inspect stamps each log with the commit and a dirty flag; a log
#   produced by uncommitted code names a commit that did not produce it.
# * A cost gate before the second model. Haiku runs first; if its bill, scaled by the
#   price ratio, projects Opus over OPUS_BUDGET_USD, the script waits for a go file.
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "$HERE/.." && pwd)
cd "$EVAL_DIR" || exit 2
: "${HVTB_TASKS_DIR:?set HVTB_TASKS_DIR to the unpacked hv-terminal-bench-2-1 directory}"
INSPECT=${INSPECT:-inspect}
PYTHON=${PYTHON:-python}
export AWS_REGION=${AWS_REGION:-us-east-1} PYTHONIOENCODING=utf-8 HVTB_TASKS_DIR
OPUS_BUDGET_USD=${OPUS_BUDGET_USD:-600}
LOG_ROOT=${LOG_ROOT:-logs/register}
READ_TIMEOUT=1200
mkdir -p "$LOG_ROOT"
STATUS="$LOG_ROOT/sweep_status.log"
GO_FILE="$LOG_ROOT/GO_OPUS"
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$STATUS"; }

if [ -n "$(git -C "$EVAL_DIR" status --porcelain)" ]; then
  say "ABORT: the git tree is dirty; commit first so the logs name the code that ran"
  git -C "$EVAL_DIR" status --porcelain | head -20 | tee -a "$STATUS"
  exit 2
fi
say "commit $(git -C "$EVAL_DIR" rev-parse HEAD)"

wait_for_credentials() {
  until aws sts get-caller-identity >/dev/null 2>&1; do
    say "WAITING: AWS credentials are not valid; run 'aws login' to resume"
    sleep 120
  done
}

HEAVY=$(grep -l '^memory_mb = 8192' "$HVTB_TASKS_DIR"/*/task.toml | xargs -n1 dirname | xargs -n1 basename | sort)
ALL=$(ls "$HVTB_TASKS_DIR" | sort)
LIGHT=$(comm -23 <(printf '%s\n' $ALL) <(printf '%s\n' $HEAVY) | paste -sd, -)
HEAVY_CSV=$(printf '%s\n' $HEAVY | paste -sd, -)
say "tasks: $(printf '%s\n' $ALL | wc -l) = light $(tr , '\n' <<<"$LIGHT" | wc -l) + heavy $(printf '%s\n' $HEAVY | wc -l)"

errored_in() { # newest log in a directory -> number of samples without a score
  "$PYTHON" - "$1" <<'PY'
import sys
from inspect_ai.log import list_eval_logs, read_eval_log
logs = list_eval_logs(sys.argv[1])
if not logs:
    print(-1); sys.exit()
log = read_eval_log(logs[0])  # newest first
bad = sum(1 for s in (log.samples or []) if s.error is not None or not s.scores)
print(bad if log.status == "success" or log.samples else -1)
PY
}

run_pass() { # model slug pass ids concurrency
  local model=$1 slug=$2 pass=$3 ids=$4 n=$5 dir="$LOG_ROOT/$2/$3" rc bad round
  mkdir -p "$dir"
  wait_for_credentials
  say "BEGIN $slug $pass ($(tr , '\n' <<<"$ids" | wc -l) tasks, concurrency $n)"
  "$INSPECT" eval hvtb_hack_detection/task.py@hack_detection \
    --model "$model" -M read_timeout=$READ_TIMEOUT --sample-id "$ids" \
    --max-sandboxes "$n" --max-connections "$n" --max-samples "$n" \
    --no-fail-on-error --retry-on-error 1 \
    --log-dir "$dir" --display plain >> "$dir/console.out" 2>&1
  rc=$?
  say "END $slug $pass exit=$rc"
  for round in 1 2; do
    bad=$(errored_in "$dir")
    [ "$bad" = "0" ] && break
    say "RETRY $slug $pass round $round: $bad sample(s) without a score"
    wait_for_credentials
    "$INSPECT" eval-retry "$(ls -t "$dir"/*.eval | head -1)" \
      --max-sandboxes "$n" --max-connections "$n" --max-samples "$n" \
      --no-fail-on-error --log-dir "$dir" --display plain >> "$dir/console.out" 2>&1
    say "END retry $round $slug $pass exit=$?"
  done
  say "DONE $slug $pass: $(errored_in "$dir") sample(s) still without a score"
}

PLAN=${PLAN:-"haiku-4-5:light haiku-4-5:heavy opus-4-6:light opus-4-6:heavy"}
LIGHT_CONCURRENCY=${LIGHT_CONCURRENCY:-4}
HEAVY_CONCURRENCY=${HEAVY_CONCURRENCY:-1}
say "plan: $PLAN (light x$LIGHT_CONCURRENCY, heavy x$HEAVY_CONCURRENCY)"

model_id() {
  case $1 in
    haiku-4-5) echo "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0" ;;
    opus-4-6) echo "bedrock/global.anthropic.claude-opus-4-6-v1" ;;
    *) say "ABORT: unknown model slug $1"; exit 2 ;;
  esac
}

opus_gate() {
  # Only meaningful when this host ran Haiku's full light pass: the projection scales
  # Haiku's measured bill, and a partial Haiku bill would understate it.
  case " $PLAN " in *" haiku-4-5:light "*) ;; *) say "Opus cost gate skipped: Haiku's light pass ran elsewhere"; return ;; esac
  local cost projected
  cost=$("$PYTHON" -c "import json;print(json.load(open('$LOG_ROOT/haiku-4-5/report.json'))['estimated_cost_usd'])")
  projected=$("$PYTHON" -c "print(round($cost * 5, 2))")
  say "Haiku cost ~\$$cost; Opus projected at 5x list price ~\$$projected (budget \$$OPUS_BUDGET_USD)"
  if "$PYTHON" -c "import sys; sys.exit(0 if $projected > $OPUS_BUDGET_USD else 1)"; then
    say "WAITING: projected Opus spend exceeds the budget; create $GO_FILE to continue"
    until [ -f "$GO_FILE" ]; do sleep 60; done
  fi
}

previous=""
for step in $PLAN; do
  slug=${step%%:*}
  pass=${step##*:}
  if [ -n "$previous" ] && [ "$slug" != "$previous" ]; then
    "$PYTHON" scripts/sweep_report.py "$LOG_ROOT/$previous" --json "$LOG_ROOT/$previous/report.json" | tee -a "$STATUS"
    [ "$slug" = opus-4-6 ] && opus_gate
  fi
  case $pass in
    light) run_pass "$(model_id "$slug")" "$slug" light "$LIGHT" "$LIGHT_CONCURRENCY" ;;
    heavy) run_pass "$(model_id "$slug")" "$slug" heavy "$HEAVY_CSV" "$HEAVY_CONCURRENCY" ;;
    *) say "ABORT: unknown pass $pass"; exit 2 ;;
  esac
  previous=$slug
done
"$PYTHON" scripts/sweep_report.py "$LOG_ROOT/$previous" --json "$LOG_ROOT/$previous/report.json" | tee -a "$STATUS"
say "SWEEP DONE"
