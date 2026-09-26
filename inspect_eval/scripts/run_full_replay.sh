#!/usr/bin/env bash
# The full detector replay: every recorded run of each given hvtb_hack_rate log replayed
# in mode C, then eval-retry for samples that errored, then a completion summary. Under
# docs/replay-preregistration.md amendment 7 that is the four logs, 356 runs:
#
#   HVTB_TASKS_DIR=/data/hv-terminal-bench-2-1 HVTB_BUILD_DIR=/data/.hvtb-build \
#     uv run bash scripts/run_full_replay.sh \
#       logs/register/opus-4-6/<log>.eval logs/register/haiku-4-5/<log>.eval \
#       logs/opus5/opus-5/<log>.eval logs/opus5/opus-5-5/<log>.eval
#
# Each source gets its own directory under LOG_ROOT, named after the log's file name, so
# the four logs' file names must differ (they do: each carries its own eval id).
#
# Run from a clean checkout, under `uv run` (or with the venv active) so `inspect` and
# `python` resolve to it, on a Docker host. It calls no model and needs no cloud account.
# Environment:
#   HVTB_TASKS_DIR   the pinned dataset the recordings ran (required); the replay refuses
#                    a task whose content hash differs from its recording's
#   HVTB_BUILD_DIR   where the two QEMU task images are built (default: .hvtb-build beside
#                    HVTB_TASKS_DIR); the logs record it, so keep it in a neutral place
#   INSPECT          inspect executable (default: inspect)
#   PYTHON           python with this package installed (default: python)
#   MAX_SAMPLES      samples replayed at once (default: 8)
#   MAX_SANDBOXES    sandboxes at once (default: MAX_SAMPLES)
#   MAX_RETRY_ROUNDS eval-retry rounds per source log (default: 3)
#   LOG_ROOT         where logs go, one directory per source log (default: logs/replay-full)
#   RESUME           1 to take up a source whose directory already holds a replay log from
#                    this commit, with eval-retry instead of a new full pass (default: 0)
#
# Why each piece is here:
# * Every sample, by name. hvtb_replay replays the samples it is given, so their ids are
#   read from the source log. Each is first extracted as the replay extracts it: one that
#   cannot be makes hvtb_replay refuse the whole log, so the script stops before any
#   container starts, naming it, rather than after the logs before it have run. A run
#   with no turn (the API stopped the model before its first call) is replayed as well:
#   its verifier runs after the recorded gap, and it has no stream for a detector. The
#   calls of a final message that never ran, because the content filter ended the agent
#   loop (Opus 5's write-compressor), are left out by the extraction and counted here.
# * Mode C, paced, no retest (docs/replay-preregistration.md): the tests run in a clone
#   after every turn, and each turn starts at its recorded offset. The retest was the
#   pilot's check.
# * The task arguments are written to a JSON file in the source's log directory and
#   passed with --task-config, so they sit beside the logs; eval-retry takes them back
#   from the log it retries.
# * --no-fail-on-error (the task sets it as well): one replay's error must not cancel the
#   others. eval-retry reruns only the samples that errored or never finished, each in a
#   fresh container, and writes a new log that carries the finished ones. The newest log
#   in each directory is therefore the final one and the only one to convert:
#   replay_to_fixture.py refuses a run that appears twice.
# * A clean git tree, checked again before every inspect command: every log records the
#   commit, and all of them must name the one this script recorded.
# * Enough inotify instances on the host: the live container runs the task's watchers,
#   as in the eval (the clones run none).
# * The summary counts, per source log, the samples the final log replayed, errored or
#   lacks. An errored replay keeps the turns it finished and is reported, never dropped.
# Exit status: 0 when every sample of every source was replayed, 1 when any was not,
# 2 when the script refused to start.
set -uo pipefail

[ "$#" -ge 1 ] || { echo "usage: run_full_replay.sh <hvtb_hack_rate log>.eval..." >&2; exit 2; }
CALLER_DIR=$PWD
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "$HERE/.." && pwd)
for source in "$@"; do
  [ -f "$source" ] || { echo "no such log: $source" >&2; exit 2; }
done
cd "$EVAL_DIR" || exit 2
: "${HVTB_TASKS_DIR:?set HVTB_TASKS_DIR to the pinned hv-terminal-bench-2-1 directory}"
INSPECT=${INSPECT:-inspect}
PYTHON=${PYTHON:-python}
MAX_SAMPLES=${MAX_SAMPLES:-8}
MAX_SANDBOXES=${MAX_SANDBOXES:-$MAX_SAMPLES}
MAX_RETRY_ROUNDS=${MAX_RETRY_ROUNDS:-3}
LOG_ROOT=${LOG_ROOT:-logs/replay-full}
RESUME=${RESUME:-0}
export PYTHONIOENCODING=utf-8 HVTB_TASKS_DIR

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
mkdir -p "$LOG_ROOT" || exit 2
STATUS="$LOG_ROOT/status.log"
say() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [replay] $*" | tee -a "$STATUS"; }

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
say "commit $COMMIT; tasks $HVTB_TASKS_DIR; build dir ${HVTB_BUILD_DIR:-(default, beside the tasks)}"
say "max samples $MAX_SAMPLES, max sandboxes $MAX_SANDBOXES, retry rounds $MAX_RETRY_ROUNDS"

command -v "$INSPECT" >/dev/null \
  && "$PYTHON" -c "import inspect_ai, hvtb_hack_detection.replay" 2>/dev/null || {
  say "ABORT: $INSPECT, or $PYTHON with inspect_ai and this package, not found;"
  say "       run under 'uv run' or set INSPECT/PYTHON"
  exit 2
}
docker info >/dev/null 2>&1 || { say "ABORT: the Docker daemon is not reachable"; exit 2; }
# Each hack watcher is an inotify instance, and every container's root user shares the
# host's limit on them (128 by default). 14 per live container covers the largest task.
LIVE=$((MAX_SAMPLES < MAX_SANDBOXES ? MAX_SAMPLES : MAX_SANDBOXES))
LIMIT_FILE=/proc/sys/fs/inotify/max_user_instances
NEEDED=$((LIVE * 14))
if [ -r "$LIMIT_FILE" ] && [ "$(cat "$LIMIT_FILE")" -lt "$NEEDED" ]; then
  say "ABORT: fs.inotify.max_user_instances is $(cat "$LIMIT_FILE"), below $NEEDED for $LIVE containers;"
  say "       raise it (sudo sysctl -w fs.inotify.max_user_instances=8192), and add up all runs sharing this host"
  exit 2
fi
leftovers() { # clones and snapshot images an interrupted replay may have left behind
  echo "$(docker ps -aq --filter name=hvtb-replay-clone | wc -l) clone(s)," \
    "$(docker images -q hvtb-replay-snapshot | wc -l) snapshot image(s)"
}
say "left over before the run: $(leftovers)"

newest_log() { ls -t "$1"/*.eval 2>/dev/null | head -1; }

prepare() { # <source log> <config to write>: check every recording, write the task arguments
  "$PYTHON" - "$1" "$CALLER_DIR" "$2" <<'PY'
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from inspect_ai.log import read_eval_log  # noqa: E402

from hvtb_hack_detection.replay import recorded_run  # noqa: E402

given, caller_dir, config = sys.argv[1:4]
source = Path(caller_dir, given).resolve()
try:  # a log under the checkout is named relative to it, as eval-retry resolves it
    named = source.relative_to(Path.cwd().resolve()).as_posix()
except ValueError:
    named = source.as_posix()
log = read_eval_log(str(source))
samples = log.samples or []
ids = [str(sample.id) for sample in samples]
problems = []
if log.eval.task != "hvtb_hack_rate":
    problems.append(f"it is a log of {log.eval.task}, not hvtb_hack_rate")
if log.status != "success":
    problems.append(f"its status is {log.status}")
planned = log.results.total_samples if log.results is not None else None
if planned != len(samples):
    problems.append(f"it holds {len(samples)} of {planned} samples")
if repeated := sorted({i for i in ids if ids.count(i) > 1}):
    problems.append(f"several epochs of {', '.join(repeated)}; replay one")
turns = empty = dropped = 0
recorded = 0.0
for sample in samples:
    try:
        run = recorded_run(sample, log.eval.model)
    except ValueError as exc:
        problems.append(f"the replay cannot extract {sample.id}: {exc}")
        continue
    turns += len(run.turns)
    empty += not run.turns
    dropped += len(run.dropped_calls)
    recorded += run.scoring_start_sec
if problems:
    print("\n".join(f"{source.name}: {problem}" for problem in problems), file=sys.stderr)
    sys.exit(3)
Path(config).write_text(
    json.dumps(
        {"log": named, "samples": ids, "mode": "C", "pacing": True, "retest": False},
        indent=1,
    ),
    encoding="utf-8",
)
print(
    f"{log.eval.model}: {len(ids)} samples, {turns} turns ({empty} run(s) with none), "
    f"{dropped} call(s) never run and not replayed, {recorded / 3600:.1f} h recorded to scoring"
)
PY
}

unfinished() { # <dir> <planned>: samples the newest log errored, never scored or lacks; -1 if unreadable
  "$PYTHON" - "$(newest_log "$1")" "$2" <<'PY'
import sys

from inspect_ai.log import read_eval_log_sample_summaries

try:
    summaries = read_eval_log_sample_summaries(sys.argv[1])
except Exception:
    print(-1)
    raise SystemExit
done = {str(s.id) for s in summaries if s.error is None and "hvtb_replay_score" in (s.scores or {})}
print(int(sys.argv[2]) - len(done))
PY
}

summarise() { # <dir> <config>: one line on the final log; exit 1 unless every sample was replayed
  "$PYTHON" - "$(newest_log "$1")" "$2" <<'PY'
import json
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries

final, config = sys.argv[1], json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
planned = config["samples"]
name = Path(config["log"]).name
if not final:
    print(f"{name}: no replay log")
    sys.exit(1)
status = read_eval_log(final, header_only=True).status
summaries = {str(s.id): s for s in read_eval_log_sample_summaries(final)}
replayed, errored, unscored, missing, unmeasured, differ, unknown = ([] for _ in range(7))
for sample_id in planned:
    s = summaries.get(sample_id)
    score = (s.scores or {}).get("hvtb_replay_score") if s is not None else None
    if s is None:
        missing.append(sample_id)
    elif s.error is not None:
        errored.append(sample_id)
    elif score is None:
        unscored.append(sample_id)
    else:
        replayed.append(sample_id)
        fidelity = (score.metadata or {}).get("fidelity")
        if not isinstance(fidelity, dict):
            unknown.append(sample_id)
            continue
        if fidelity["measurement_failures"] or (
            fidelity["turns_replayed"] < fidelity["turns_recorded"]
        ):
            unmeasured.append(sample_id)
        if not (fidelity["reward_match"] and fidelity["flags_match"]):
            differ.append(sample_id)
note = f" ({len(unknown)} not summarised)" if unknown else ""
print(
    f"{name}: {Path(final).name} ({status}): {len(replayed)} of {len(planned)} replayed, "
    f"{len(errored)} errored, {len(unscored)} unscored, {len(missing)} missing; of those "
    f"replayed, {len(unmeasured)} with an unmeasured turn and {len(differ)} whose reward "
    f"or flags differ from the recording{note}"
)
for label, ids in (("errored", errored), ("unscored", unscored), ("missing", missing)):
    if ids:
        print(f"  {label}: {', '.join(ids)}")
sys.exit(0 if len(replayed) == len(planned) else 1)
PY
}

# Check every source before any replay starts, so a bad one does not surface hours in.
DIRS=()
for source in "$@"; do
  dir="$LOG_ROOT/$(basename "$source" .eval)"
  for seen in "${DIRS[@]+"${DIRS[@]}"}"; do
    [ "$seen" = "$dir" ] && { say "ABORT: $source is given twice, or shares its name with another source"; exit 2; }
  done
  mkdir -p "$dir" || exit 2
  if [ -n "$(newest_log "$dir")" ]; then
    if [ "$RESUME" != 1 ]; then
      say "ABORT: $dir already holds a replay log; set RESUME=1 to retry it, or choose another LOG_ROOT"
      exit 2
    fi
    if [ "$(cat "$dir/commit.txt" 2>/dev/null)" != "$COMMIT" ]; then
      say "ABORT: $dir was replayed at commit $(cat "$dir/commit.txt" 2>/dev/null || echo '?'), not $COMMIT"
      exit 2
    fi
  fi
  if ! checked=$(prepare "$source" "$dir/replay_config.json" 2>"$dir/prepare.err"); then
    tee -a "$STATUS" < "$dir/prepare.err" >&2
    say "ABORT: $source cannot be replayed as a whole; nothing has run"
    exit 2
  fi
  echo "$COMMIT" > "$dir/commit.txt"
  say "SOURCE $source -> $dir: $checked"
  DIRS+=("$dir")
done

for dir in "${DIRS[@]}"; do
  name=$(basename "$dir")
  planned=$("$PYTHON" -c "import json, sys; print(len(json.load(open(sys.argv[1]))['samples']))" \
    "$dir/replay_config.json")
  if [ -z "$(newest_log "$dir")" ]; then
    check_tree
    say "BEGIN $name: full pass, mode C, $planned samples"
    "$INSPECT" eval hvtb_hack_detection/replay.py@hvtb_replay --model none \
      --task-config "$dir/replay_config.json" \
      --max-samples "$MAX_SAMPLES" --max-sandboxes "$MAX_SANDBOXES" \
      --no-fail-on-error --log-dir "$dir" --display plain >> "$dir/console.out" 2>&1
    say "END $name: full pass exit=$?"
  else
    say "RESUME $name from $(basename "$(newest_log "$dir")")"
  fi
  if [ -z "$(newest_log "$dir")" ]; then
    say "FAILED $name: no log written; see $dir/console.out"
    continue
  fi
  for ((round = 1; round <= MAX_RETRY_ROUNDS; round++)); do
    left=$(unfinished "$dir" "$planned")
    [ "$left" = "0" ] && break
    check_tree
    say "RETRY $name round $round: $left sample(s) unfinished"
    "$INSPECT" eval-retry "$(newest_log "$dir")" \
      --max-samples "$MAX_SAMPLES" --max-sandboxes "$MAX_SANDBOXES" \
      --no-fail-on-error --log-dir "$dir" --display plain >> "$dir/console.out" 2>&1
    say "END $name: retry $round exit=$?"
  done
done

say "SUMMARY at commit $COMMIT"
FAILED=0
FINAL_LOGS="$LOG_ROOT/final_logs.txt"
: > "$FINAL_LOGS"
for dir in "${DIRS[@]}"; do
  summarise "$dir" "$dir/replay_config.json" | tee -a "$STATUS" || FAILED=1
  final=$(newest_log "$dir")
  [ -n "$final" ] && echo "$final" >> "$FINAL_LOGS"
done
say "left over after the run: $(leftovers)"
say "final logs listed in $FINAL_LOGS; convert only those:"
say "  $PYTHON scripts/replay_to_fixture.py \$(cat $FINAL_LOGS) --out replayed_trajectories.json"
if [ "$FAILED" = 0 ]; then
  say "COMPLETE: every sample of every source was replayed"
else
  say "INCOMPLETE: see the summary above"
  exit 1
fi
