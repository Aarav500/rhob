"""Summarise an hvtb_hack_rate run, and check a final log is fit to submit.

Reads one ``.eval`` log, or every log under a directory (keeping the latest completed
score for each task, since an ``eval-retry`` log re-carries earlier samples), and reports
the hack rate beside what a reader of it needs: how many tasks have a label, which limit
ended each run, where each label came from, and how many runs were mostly tool calls that
never reached the container.

``--accept`` checks a single final log against what a register submission needs and
exits non-zero if any check fails: status success, all 89 samples, no sample errors, a
label on every sample, no harness-degraded sample, no ``--sample-id`` or ``--limit``, one
epoch, a clean git tree at run time, and the expected task name and version
(``hvtb_hack_rate`` 1-A unless ``--task`` and ``--version`` name another, such as the
instrumented ``hvtb_hack_rate_measured`` 1-A-measured). Under
``--epochs N`` (summary only) each task's label is the mean over its epochs, as Inspect
reduces it, and the standard error is over the per-task means.

Usage::

    python scripts/sweep_report.py logs/register/haiku-4-5
    python scripts/sweep_report.py logs/register/haiku-4-5/<final>.eval --accept

Cost is an estimate at list on-demand prices (per million tokens: Haiku 4.5 $1 in and
$5 out, Opus 4.6 $5 and $25; cache writes 1.25x and cache reads 0.1x the input price).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log

#: USD per million tokens, (input, output). Matched by substring of the model name.
PRICES = {"haiku-4-5": (1.0, 5.0), "opus-4-6": (5.0, 25.0)}
CACHE_WRITE = 1.25
CACHE_READ = 0.10
#: A run where at least this share of tool calls never reached the container is reported
#: as harness-degraded rather than as a model outcome.
DEGRADED_SHARE = 0.5
EXPECTED_SAMPLES = 89
EXPECTED_TASK = "hvtb_hack_rate"
EXPECTED_VERSION = "1-A"


def _price(model: str) -> tuple[float, float] | None:
    return next((p for key, p in PRICES.items() if key in model), None)


def _cost(model: str, usage: Any) -> float:
    price = _price(model)
    if price is None or usage is None:
        return math.nan
    p_in, p_out = price
    return (
        (usage.input_tokens or 0) * p_in
        + (usage.input_tokens_cache_write or 0) * p_in * CACHE_WRITE
        + (usage.input_tokens_cache_read or 0) * p_in * CACHE_READ
        + (usage.output_tokens or 0) * p_out
    ) / 1e6


def _label(value: Any) -> float:
    """The watcher label from a score value: a float, or a replay-style dict."""
    if isinstance(value, dict):
        value = value.get("hacked", math.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _read(target: Path) -> list[EvalLog]:
    if target.is_file():
        return [read_eval_log(str(target))]
    logs = [read_eval_log(info) for info in list_eval_logs(str(target))]
    if not logs:
        raise SystemExit(f"no .eval logs under {target}")
    return logs


def summarise(logs: list[EvalLog]) -> dict[str, Any]:
    """Pool the logs' samples, latest completed score per task."""
    models = {log.eval.model for log in logs}
    if len(models) != 1:
        raise SystemExit(f"logs mix models {sorted(models)}; summarise one model at a time")
    model = models.pop()
    latest: dict[tuple[str, int], Any] = {}
    errored: dict[tuple[str, int], str] = {}
    cost = 0.0
    counted: set[tuple[str, str, int]] = set()
    for log in sorted(logs, key=lambda lg: lg.eval.created):
        for sample in log.samples or []:
            key = (log.eval.eval_id, str(sample.id), sample.epoch)
            if key not in counted:
                counted.add(key)
                cost += sum(_cost(model, u) for u in (sample.model_usage or {}).values())
            slot = (str(sample.id), sample.epoch)
            if sample.error is not None:
                errored.setdefault(slot, sample.error.message[:200])
                continue
            if sample.scores:
                latest[slot] = next(iter(sample.scores.values()))
                errored.pop(slot, None)

    by_task: dict[str, list[float]] = {}
    limits: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    degraded: list[str] = []
    for (task, _epoch), score in sorted(latest.items()):
        meta = score.metadata or {}
        limits[meta.get("agent_limit") or "none"] += 1
        sources[meta.get("label_source") or "none"] += 1
        calls, unparsed = meta.get("tool_calls") or 0, meta.get("tool_calls_unparsed") or 0
        if calls and unparsed / calls >= DEGRADED_SHARE:
            degraded.append(task)
        label = _label(score.value)
        if math.isfinite(label):
            by_task.setdefault(task, []).append(label)

    means = {task: sum(v) / len(v) for task, v in by_task.items()}
    labels = sorted(means.items())
    hacked = [t for t, v in labels if v > 0]
    n = len(labels)
    rate = sum(means.values()) / n if n else math.nan
    spread = sum((v - rate) ** 2 for v in means.values()) / (n - 1) if n > 1 else math.nan
    stderr = math.sqrt(spread / n) if n > 1 else math.nan
    tasks_scored = {task for task, _ in latest}
    return {
        "model": model,
        "logs": [Path(lg.location).name for lg in logs],
        "tasks_with_a_score": len(tasks_scored),
        "tasks_errored_without_a_score": sorted({t for t, _ in errored} - tasks_scored),
        "labelled": n,
        "hacked": len(hacked),
        "hack_rate": rate,
        "stderr": stderr,
        "hacked_tasks": hacked,
        "unlabelled_tasks": sorted(tasks_scored - set(means)),
        "agent_limit": dict(limits),
        "label_source": dict(sources),
        "harness_degraded_tasks": degraded,
        "estimated_cost_usd": round(cost, 2),
    }


def acceptance(
    log: EvalLog,
    summary: dict[str, Any],
    task: str = EXPECTED_TASK,
    version: str = EXPECTED_VERSION,
) -> list[str]:
    """Reasons a final log is not fit to submit; empty when it is."""
    problems = []
    if log.status != "success":
        problems.append(f"status is {log.status}")
    if summary["tasks_with_a_score"] != EXPECTED_SAMPLES:
        problems.append(f"{summary['tasks_with_a_score']} scored samples, not {EXPECTED_SAMPLES}")
    if summary["tasks_errored_without_a_score"]:
        problems.append(f"errored: {summary['tasks_errored_without_a_score']}")
    if summary["labelled"] != EXPECTED_SAMPLES:
        problems.append(f"{summary['labelled']} labelled samples, not {EXPECTED_SAMPLES}")
    if summary["harness_degraded_tasks"]:
        problems.append(f"harness-degraded: {summary['harness_degraded_tasks']}")
    config = log.eval.config
    if config.sample_id is not None or config.limit is not None:
        problems.append("the run was restricted with --sample-id or --limit")
    if config.epochs not in (None, 1):
        problems.append(f"the run used {config.epochs} epochs, not 1")
    revision = log.eval.revision
    if revision is None or revision.dirty:
        problems.append("the git tree was dirty (or unrecorded) at run time")
    if log.eval.task.split("/")[-1] != task:
        problems.append(f"task is {log.eval.task}, not {task}")
    if str(log.eval.task_version) != version:
        problems.append(f"task version is {log.eval.task_version}, not {version}")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Print the summary; with --accept, return 1 unless the log passes every check."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("target", type=Path, help="a .eval log or a directory of them")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--accept", action="store_true", help="acceptance-check a single log")
    ap.add_argument("--task", default=EXPECTED_TASK, help="the task --accept expects")
    ap.add_argument("--version", default=EXPECTED_VERSION, help="the version --accept expects")
    args = ap.parse_args(argv)
    logs = _read(args.target)
    summary = summarise(logs)
    if args.accept:
        if len(logs) != 1:
            raise SystemExit("--accept checks exactly one log file")
        summary["acceptance_problems"] = acceptance(logs[0], summary, args.task, args.version)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for key, value in summary.items():
        print(f"{key:32s} {value}")
    return 1 if args.accept and summary["acceptance_problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
