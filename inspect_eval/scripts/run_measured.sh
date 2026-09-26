#!/usr/bin/env bash
# The instrumented live re-run: one full pass of hvtb_hack_rate_measured for one Anthropic
# model over all 89 HVTB tasks, then eval-retry for samples that errored, then an
# acceptance check and the measurement report on the final log.
#
#   HVTB_TASKS_DIR=/data/hv-terminal-bench-2-1 uv run bash scripts/run_measured.sh claude-opus-5
#
# The model runs as anthropic/<model> on the Anthropic API, with ANTHROPIC_API_KEY taken
# from the environment; the script checks that it is set and never prints it.
#
# Run from a clean checkout, under `uv run` (or with the venv active) so `inspect` and
# `python` resolve to it, on a Docker host. Environment:
#   HVTB_TASKS_DIR       the pinned dataset (required)
#   ANTHROPIC_API_KEY    the Anthropic API key (required; never printed)
#   HVTB_BUILD_DIR       where the two QEMU task images are built (default: .hvtb-build
#                        beside HVTB_TASKS_DIR)
#   HVTB_CLONE_SLOTS_DIR the host-wide clone-test locks (default: a directory in /tmp)
#   INSPECT              inspect executable (default: inspect)
#   PYTHON               python with this package installed (default: python)
#   CONCURRENCY          samples in flight (default: 8)
#   MAX_SANDBOXES        live containers at once (default: CONCURRENCY)
#   MAX_CONNECTIONS      model connections at once (default: CONCURRENCY)
#   MAX_CLONE_TESTS      clone tests at once, host-wide (default: 3)
#   MAX_PENDING_SNAPSHOTS snapshot images on disk at once, host-wide (default: the task's, 32)
#   SNAPSHOT_MIN_FREE_GB free GiB under Docker's root directory below which a turn is not
#                        snapshotted (default: the task's, 30)
#   MEASUREMENT_WAIT_SEC how long a sample waits for its queued measurements once its
#                        verifier has run (default: the task's, 21600)
#   MAX_RETRY_ROUNDS     eval-retry rounds before the acceptance check (default: 3)
#   MIN_FREE_GB          free space required under Docker's root directory (default: 100)
#   ALLOW_MEMORY_OVERCOMMIT=1  start even when the host's memory cannot hold the live
#                        containers and the clone tests at the tasks' limits
#   HVTB_MEASURED_RUN_ID the run id in every snapshot and clone name (default: made here)
#   LOG_ROOT             where logs go (default: logs/measured)
#
# Why each piece is here:
# * One invocation per model with no --sample-id, so one log covers all 89 samples, as in
#   run_register_sweep.sh. --no-fail-on-error and --retry-on-error 1: one sample's error
#   must not cancel the others, and a transient error is retried once in place. Errored
#   samples are rerun with eval-retry (up to MAX_RETRY_ROUNDS), which writes a new log that
#   carries the finished ones, so the newest log is the final one and the only one to
#   convert.
# * A clean git tree, checked again before every inspect command: every log records the
#   commit and a dirty flag, and all of them must name the commit recorded here.
# * Enough inotify instances on the host for every watcher of every live container (the
#   clones run none).
# * Enough memory for the live containers and the clone tests beside them, each at its
#   task's limit (the largest tasks at once), or the kernel may kill processes in live
#   containers; ALLOW_MEMORY_OVERCOMMIT=1 overrides. Too few CPUs for the same is a
#   warning: clones run at a low CPU weight, so they slow down first.
# * Free disk: each snapshot waiting for its clone test holds a full copy of its
#   container's writable layer, gigabytes for the tasks that build toolchains. The task
#   also bounds that itself while it runs (MAX_PENDING_SNAPSHOTS, SNAPSHOT_MIN_FREE_GB).
# * Every snapshot image and clone is named with this run's id. After every inspect
#   command, once its process has exited, the ones of this run are removed: a killed
#   process never ran its own cleanup. Other runs' are only counted, before and after.
# * The acceptance check is sweep_report.py's, for this task and version: status success,
#   all 89 samples scored and labelled, no errors, no harness-degraded sample, one epoch,
#   no --sample-id or --limit, a clean tree. The measurement report follows it; runs with
#   an unmeasured turn are not a failure of the run, since the analysis counts them as N/A.
# Exit status: 0 when the final log is accepted, 1 when it is not, 2 when the script
# refused to start.
set -uo pipefail

MODEL_NAME=${1:?usage: run_measured.sh <anthropic model, e.g. claude-opus-5>}
case $MODEL_NAME in
  anthropic/*) MODEL=$MODEL_NAME ;;
  */*) echo "not an Anthropic API model: $MODEL_NAME" >&2; exit 2 ;;
  *) MODEL="anthropic/$MODEL_NAME" ;;
esac
SLUG=${MODEL#anthropic/}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "$HERE/.." && pwd)
cd "$EVAL_DIR" || exit 2
: "${HVTB_TASKS_DIR:?set HVTB_TASKS_DIR to the pinned hv-terminal-bench-2-1 directory}"
INSPECT=${INSPECT:-inspect}
PYTHON=${PYTHON:-python}
CONCURRENCY=${CONCURRENCY:-8}
MAX_SANDBOXES=${MAX_SANDBOXES:-$CONCURRENCY}
MAX_CONNECTIONS=${MAX_CONNECTIONS:-$CONCURRENCY}
MAX_CLONE_TESTS=${MAX_CLONE_TESTS:-3}
MAX_RETRY_ROUNDS=${MAX_RETRY_ROUNDS:-3}
MIN_FREE_GB=${MIN_FREE_GB:-100}
LOG_ROOT=${LOG_ROOT:-logs/measured}
TASK=hvtb_hack_rate_measured
VERSION=1-A-measured
# Lower-case letters, digits, '_', '.' and '-', as measured.run_id() requires.
RUN_ID=${HVTB_MEASURED_RUN_ID:-$(date -u +%Y%m%d%H%M%S)-$$}
if ! [[ $RUN_ID =~ ^[a-z0-9][a-z0-9_.-]{0,39}$ ]]; then
  echo "not a run id: HVTB_MEASURED_RUN_ID=$RUN_ID (at most 40 of a-z 0-9 _ . -)" >&2
  exit 2
fi
export PYTHONIOENCODING=utf-8 HVTB_TASKS_DIR HVTB_MEASURED_RUN_ID=$RUN_ID

TASK_ARGS=(-T "max_clone_tests=$MAX_CLONE_TESTS")
if [ -n "${MEASUREMENT_WAIT_SEC:-}" ]; then
  TASK_ARGS+=(-T "measurement_wait_sec=$MEASUREMENT_WAIT_SEC")
fi
if [ -n "${MAX_PENDING_SNAPSHOTS:-}" ]; then
  TASK_ARGS+=(-T "max_pending_snapshots=$MAX_PENDING_SNAPSHOTS")
fi
if [ -n "${SNAPSHOT_MIN_FREE_GB:-}" ]; then
  TASK_ARGS+=(-T "min_free_gb=$SNAPSHOT_MIN_FREE_GB")
fi
LIMITS=(--max-samples "$CONCURRENCY" --max-sandboxes "$MAX_SANDBOXES" --max-connections "$MAX_CONNECTIONS")

COMMIT=$(git -C "$EVAL_DIR" rev-parse HEAD) || {
  echo "ABORT: $EVAL_DIR is not a git checkout" >&2
  exit 2
}
# Inside the checkout, the logs must be ignored, or the first one would dirty the tree.
# git exits 1 for a path it does not ignore, and 128 for one outside the checkout.
git -C "$EVAL_DIR" check-ignore -q "$LOG_ROOT/probe.eval"
if [ "$?" = 1 ]; then
  echo "ABORT: git does not ignore $LOG_ROOT; put LOG_ROOT under logs/ or outside the checkout" >&2
  exit 2
fi
DIR="$LOG_ROOT/$SLUG"
mkdir -p "$DIR" || exit 2
STATUS="$LOG_ROOT/status.log"
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [$SLUG] $*" | tee -a "$STATUS"; }

check_tree() { # abort unless the tree is clean and still at the recorded commit
  if [ -n "$(git -C "$EVAL_DIR" status --porcelain)" ]; then
    say "ABORT: the git tree is dirty; the logs must name the commit that produced them"
    git -C "$EVAL_DIR" status --porcelain | head -20 | tee -a "$STATUS"
    exit 2
  fi
  if [ "$(git -C "$EVAL_DIR" rev-parse HEAD)" != "$COMMIT" ]; then
    say "ABORT: HEAD is no longer $COMMIT, the commit the earlier logs name"
    exit 2
  fi
}
check_tree
if [ -n "$(ls "$DIR"/*.eval 2>/dev/null)" ]; then
  say "ABORT: $DIR already holds a log; choose another LOG_ROOT, or eval-retry that log by hand"
  exit 2
fi
say "commit $COMMIT, model $MODEL, task $TASK $VERSION, run id $RUN_ID"
say "samples $CONCURRENCY, sandboxes $MAX_SANDBOXES, connections $MAX_CONNECTIONS, clone tests $MAX_CLONE_TESTS"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  say "ABORT: ANTHROPIC_API_KEY is not set"
  exit 2
fi
command -v "$INSPECT" >/dev/null \
  && "$PYTHON" -c "import inspect_ai, anthropic, hvtb_hack_detection.measured" 2>/dev/null || {
  say "ABORT: $INSPECT, or $PYTHON with inspect_ai, anthropic and this package, not found;"
  say "       run under 'uv run' or set INSPECT/PYTHON"
  exit 2
}
docker info >/dev/null 2>&1 || { say "ABORT: the Docker daemon is not reachable"; exit 2; }
# Each hack watcher is an inotify instance, and every container's root user shares the
# host's limit on them (128 by default). 14 per live container covers the largest task.
LIMIT_FILE=/proc/sys/fs/inotify/max_user_instances
LIVE=$((CONCURRENCY < MAX_SANDBOXES ? CONCURRENCY : MAX_SANDBOXES))
NEEDED=$((LIVE * 14))
if [ -r "$LIMIT_FILE" ] && [ "$(cat "$LIMIT_FILE")" -lt "$NEEDED" ]; then
  say "ABORT: fs.inotify.max_user_instances is $(cat "$LIMIT_FILE"), below $NEEDED for $LIVE containers;"
  say "       raise it (sudo sysctl -w fs.inotify.max_user_instances=8192), and add up all runs sharing this host"
  exit 2
fi
# Memory and CPUs: the LIVE largest tasks' containers beside MAX_CLONE_TESTS clones of the
# largest, each at its task.toml's limits, against what the Docker host has.
NEED=$("$PYTHON" - "$HVTB_TASKS_DIR" "$LIVE" "$MAX_CLONE_TESTS" <<'PY'
import sys
from hvtb_hack_detection.hvtb import load_hvtb_tasks
from hvtb_hack_detection.measured import worst_case_resources

tasks = load_hvtb_tasks(sys.argv[1], verify=False)
cpus, memory_mb = worst_case_resources(
    [(t.cpus, t.memory_mb) for t in tasks], int(sys.argv[2]), int(sys.argv[3])
)
print(f"{cpus:g} {memory_mb}")
PY
) || { say "ABORT: could not read the tasks' cpus and memory under $HVTB_TASKS_DIR"; exit 2; }
NEED_CPUS=${NEED% *}
NEED_MB=${NEED#* }
read -r HOST_CPUS HOST_MEM_BYTES <<<"$(docker info --format '{{.NCPU}} {{.MemTotal}}' 2>/dev/null)"
if [ -z "${HOST_MEM_BYTES:-}" ]; then
  say "WARNING: cannot read the Docker host's memory; memory headroom not checked"
elif [ "$HOST_MEM_BYTES" -lt $((NEED_MB * 1024 * 1024)) ]; then
  say "$LIVE live containers and $MAX_CLONE_TESTS clone tests can be granted $((NEED_MB / 1024)) GiB;" \
    "the Docker host has $((HOST_MEM_BYTES / 1024 / 1024 / 1024)) GiB"
  if [ "${ALLOW_MEMORY_OVERCOMMIT:-0}" = 1 ]; then
    say "WARNING: going on (ALLOW_MEMORY_OVERCOMMIT=1): clones are the OOM killer's first choice, live containers are not safe"
  else
    say "ABORT: lower CONCURRENCY or MAX_CLONE_TESTS, or set ALLOW_MEMORY_OVERCOMMIT=1"
    exit 2
  fi
fi
if [ -n "${HOST_CPUS:-}" ] && awk -v have="$HOST_CPUS" -v need="$NEED_CPUS" 'BEGIN { exit !(have < need) }'; then
  say "WARNING: the containers can be granted $NEED_CPUS CPUs and the Docker host has $HOST_CPUS;" \
    "clone tests run at a low CPU weight, but live commands may still slow down"
fi
DOCKER_ROOT=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)
if [ -n "$DOCKER_ROOT" ] && [ -d "$DOCKER_ROOT" ]; then
  FREE_KB=$(df -Pk "$DOCKER_ROOT" | awk 'NR == 2 { print $4 }')
  if [ -n "$FREE_KB" ] && [ "$FREE_KB" -lt $((MIN_FREE_GB * 1024 * 1024)) ]; then
    say "ABORT: $((FREE_KB / 1024 / 1024)) GB free under $DOCKER_ROOT, below MIN_FREE_GB=$MIN_FREE_GB;"
    say "       pending snapshots each hold a copy of their container's writable layer"
    exit 2
  fi
else
  say "WARNING: cannot read Docker's root directory from here; free disk not checked"
fi
leftovers() { # snapshot images and clones of any measured run, this one's included
  echo "$(docker ps -aq --filter name=hvtb-measured-clone | wc -l) clone(s)," \
    "$(docker images -q hvtb-measured-snapshot | wc -l) snapshot image(s)"
}
remove_ours() { # this run's snapshot images and clones; call only once its inspect has exited
  local clones images
  clones=$(docker ps -aq --filter "name=hvtb-measured-clone-$RUN_ID-")
  images=$(docker images --filter "reference=hvtb-measured-snapshot:$RUN_ID-*" \
    --format '{{.Repository}}:{{.Tag}}')
  # shellcheck disable=SC2086 # one id or name per word
  [ -n "$clones" ] && docker rm -f $clones >/dev/null
  # shellcheck disable=SC2086
  [ -n "$images" ] && docker rmi -f $images >/dev/null
  say "removed what run $RUN_ID left behind: $(printf '%s' "$clones" | grep -c .) clone(s)," \
    "$(printf '%s' "$images" | grep -c .) snapshot image(s)"
}
say "left over before the run (other runs'): $(leftovers)"

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

check_tree
# An interrupted runner still removes what the run left, once inspect has exited.
trap remove_ours EXIT
say "BEGIN full pass"
"$INSPECT" eval "hvtb_hack_detection/measured.py@$TASK" \
  --model "$MODEL" "${TASK_ARGS[@]}" "${LIMITS[@]}" \
  --no-fail-on-error --retry-on-error 1 \
  --log-dir "$DIR" --display plain >> "$DIR/console.out" 2>&1
say "END full pass exit=$?"
remove_ours
if [ -z "$(newest_log)" ]; then
  say "FAILED: no log written; see $DIR/console.out"
  exit 1
fi

for round in $(seq 1 "$MAX_RETRY_ROUNDS"); do
  left=$(unfinished)
  [ "$left" = "0" ] && break
  check_tree
  say "RETRY round $round: $left sample(s) unfinished"
  "$INSPECT" eval-retry "$(newest_log)" "${LIMITS[@]}" \
    --no-fail-on-error --retry-on-error 1 \
    --log-dir "$DIR" --display plain >> "$DIR/console.out" 2>&1
  say "END retry $round exit=$?"
  remove_ours
done

say "left over after the run (other runs'): $(leftovers)"
FINAL=$(newest_log)
"$PYTHON" scripts/measured_to_fixture.py "$FINAL" --out "$DIR/trajectories.json" | tee -a "$STATUS"
if "$PYTHON" scripts/sweep_report.py "$FINAL" --accept --task "$TASK" --version "$VERSION" \
  --json "$DIR/report.json" | tee -a "$STATUS"; then
  say "ACCEPTED $(basename "$FINAL")"
  say "analyse every model's final log together:"
  say "  $PYTHON scripts/measured_to_fixture.py <final logs> --out measured_trajectories.json"
  say "  $PYTHON scripts/replay_analysis.py measured_trajectories.json --measured-models <models>"
else
  say "NOT ACCEPTED $(basename "$FINAL"); see report above"
  exit 1
fi
