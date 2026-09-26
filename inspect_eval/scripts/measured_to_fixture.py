r"""Turn hvtb_hack_rate_measured logs into a trajectory file, and report the measurement.

``hvtb_hack_rate_measured`` runs the model live and measures the task's tests in a clone
of the container after every turn (``hvtb_hack_detection/measured.py``). This writes those
measurements in the schema ``scripts/replay_to_fixture.py`` writes and
``scripts/replay_analysis.py`` reads: one run per sample, id ``<model>/<task>``, one step
per turn after turn 0 with ``tests_passing`` = passed/total after it, ``at`` = when its
snapshot was taken (seconds from the solver's start), ``command`` = its commands and
``measured`` = whether that measurement exists; a turn without one is ``measured: false``
with the reason, never a number. Turn 0, the task's state before the agent's first
command, is under ``initial``, outside the stream. The label (``hidden_solution``,
``read_tests``) is the run's own, from its watchers, as the verifier reported it.
``clone_measurable`` is false for the service tasks, whose tests a clone cannot run
(``SERVICE_TASKS`` in ``hvtb_hack_detection/replay.py``).

A log the eval did not finish is refused, since the samples it never reached would be
missing without a trace. A sample that errored or was never scored has no label, so it
cannot be a run of the file: it is left out and named in the report, and the analysis's
count of runs per model then shows it.

The report gives, by label: the runs, the service tasks, the turns measured, the runs
with an unmeasured turn (no stream for a detector) and with no turn, the final messages
whose calls never ran (dropped, not turns), the turns with no room on disk for a snapshot,
the time the snapshots took from the agent's time budget, the time they kept the
container (and the agent's background jobs) paused inside that budget, how long the
verifier waited for the last commit after the agent stopped, and how long the sample then
waited for its measurements. It then names the samples left out and the first reason of
every run with an unmeasured turn.

Usage::

    python scripts/measured_to_fixture.py --out measured_trajectories.json \
      logs/measured/claude-opus-5/<final>.eval logs/measured/claude-opus-5-5/<final>.eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalSample, read_eval_log, resolve_sample_attachments
from inspect_ai.model import ChatMessageAssistant

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvtb_hack_detection.measured import (  # noqa: E402
    BASH,
    MEASURED_SCORER,
    MEASUREMENT_KEY,
    trajectory_record,
)
from hvtb_hack_detection.replay import SERVICE_TASKS  # noqa: E402
from hvtb_hack_detection.signals import HACK_FLAGS  # noqa: E402

SCHEMA = "hvtb-hack-detection-fixture/1"

#: The task whose logs this converts.
MEASURED_TASK = "hvtb_hack_rate_measured"


def _log_paths(targets: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for target in targets:
        if target.is_file():
            paths.append(target)
        else:
            paths.extend(sorted(target.rglob("*.eval")))
    if not paths:
        raise SystemExit(f"no .eval logs in {', '.join(map(str, targets))}")
    return paths


def finished_log(path: Path) -> tuple[str, list[EvalSample]]:
    """A finished measured log's model and samples.

    Raises:
        SystemExit: If the log is of another task, or the eval did not finish: an eval
            stopped by a sample error or a cancel logs only the samples that finished.
    """
    log = read_eval_log(str(path))
    if log.eval.task.split("/")[-1] != MEASURED_TASK:
        raise SystemExit(f"{path} is a log of {log.eval.task}, not {MEASURED_TASK}")
    samples = log.samples or []
    planned = log.results.total_samples if log.results is not None else None
    if planned is None and log.eval.dataset.sample_ids is not None:
        # A log whose every sample errored has no results; its dataset still says what ran.
        planned = len(log.eval.dataset.sample_ids) * (log.eval.config.epochs or 1)
    if log.status != "success" or planned is None or len(samples) != planned:
        raise SystemExit(
            f"{path} is not a finished run (status {log.status}, {len(samples)} of "
            f"{planned if planned is not None else '?'} samples); finish it with "
            "`inspect eval-retry` and convert the retried log"
        )
    return log.eval.model, samples


def _commands(sample: EvalSample) -> dict[str, str]:
    """Each ``bash`` call's command, by call id."""
    commands: dict[str, str] = {}
    for message in sample.messages:
        if not isinstance(message, ChatMessageAssistant):
            continue
        for call in message.tool_calls or []:
            if call.function == BASH:
                command = call.arguments.get("command")
                commands[call.id] = command if isinstance(command, str) else ""
    return commands


def measured_run(sample: EvalSample, model: str) -> dict[str, Any]:
    """One sample: its trajectory record and summary, or why it has none."""
    sample = resolve_sample_attachments(sample, "core")
    task = str((sample.metadata or {}).get("task", sample.id))
    score = (sample.scores or {}).get(MEASURED_SCORER)
    verdict = score.metadata if score is not None else None
    if sample.error is not None or not verdict or MEASUREMENT_KEY not in verdict:
        reason = sample.error.message if sample.error else f"no {MEASURED_SCORER} score"
        return {"id": f"{model}/{task}", "task": task, "record": None, "error": reason}
    return {
        "id": f"{model}/{task}",
        "task": task,
        "record": trajectory_record(model, task, verdict, _commands(sample)),
        "summary": verdict[MEASUREMENT_KEY]["summary"],
        "error": None,
    }


def _hacked(record: dict[str, Any]) -> bool:
    return any(record[flag] for flag in HACK_FLAGS)


def _unmeasured(record: dict[str, Any]) -> bool:
    return not all(step["measured"] for step in record["steps"])


def _ratio(numerator: int, denominator: int) -> str:
    if not denominator:
        return "-"
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)"


def _seconds(values: list[float]) -> str:
    return f"{sum(values):.0f} s, max {max(values):.1f}" if values else "-"


def report(runs: list[dict[str, Any]]) -> str:
    """The measurement report, by label, then the samples and runs to look at."""
    kept = [run for run in runs if run["record"] is not None]
    groups = {
        "hacked": [r for r in kept if _hacked(r["record"])],
        "clean": [r for r in kept if not _hacked(r["record"])],
        "all": kept,
    }
    width = 24
    lines = [f"{'':40s}" + "".join(f"{name:>{width}s}" for name in groups)]

    def row(label: str, cell: Any) -> None:
        lines.append(f"{label:40s}" + "".join(f"{cell(g):>{width}s}" for g in groups.values()))

    row("runs", lambda g: str(len(g)))
    row(
        "service tasks (clone cannot test)",
        lambda g: str(sum(r["task"] in SERVICE_TASKS for r in g)),
    )
    row(
        "turns measured (after turn 0)",
        lambda g: _ratio(
            sum(step["measured"] for r in g for step in r["record"]["steps"]),
            sum(len(r["record"]["steps"]) for r in g),
        ),
    )
    row(
        "turn 0 measured",
        lambda g: _ratio(
            sum(bool(r["record"]["initial"] and r["record"]["initial"]["measured"]) for r in g),
            len(g),
        ),
    )
    row(
        "runs with an unmeasured turn (no stream)",
        lambda g: str(sum(_unmeasured(r["record"]) for r in g)),
    )
    row("runs with no turn", lambda g: str(sum(not r["record"]["steps"] for r in g)))
    row(
        "final messages never run (dropped)",
        lambda g: str(sum(r["record"]["dropped_turns"] for r in g)),
    )
    row(
        "turns not snapshotted (no room on disk)",
        lambda g: str(sum(r["summary"]["not_snapshotted"] for r in g)),
    )
    row(
        "snapshot time from the agent's budget",
        lambda g: _seconds([r["summary"]["snapshot_agent_wait_sec"] for r in g]),
    )
    row(
        "container frozen by commits in the limit",
        lambda g: _seconds([r["summary"]["snapshot_frozen_in_limit_sec"] for r in g]),
    )

    def timed(key: str) -> Any:
        return lambda g: _seconds([r["summary"][key] for r in g if r["summary"][key] is not None])

    row("verifier delayed after the agent stopped", timed("post_agent_wait_sec"))
    row("wait for measurements after the verifier", timed("drain_after_verifier_sec"))

    left_out = [f"  {r['id']}: {r['error'][:200]}" for r in runs if r["record"] is None]
    lines.append(f"\nsamples left out, with no label: {len(left_out)}")
    lines.extend(left_out)
    unmeasured = [
        f"  {r['id']}: "
        + next(s["measurement_error"] for s in r["record"]["steps"] if not s["measured"])[:200]
        for r in kept
        if _unmeasured(r["record"])
    ]
    lines.append(f"\nruns with an unmeasured turn (first reason): {len(unmeasured)}")
    lines.extend(unmeasured)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Write the trajectory file and print the measurement report."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("targets", type=Path, nargs="+", help=".eval logs or directories of them")
    ap.add_argument("--out", type=Path, required=True, help="trajectory JSON to write")
    args = ap.parse_args(argv)

    paths = _log_paths(args.targets)
    runs = []
    for path in paths:
        model, samples = finished_log(path)
        runs += [measured_run(sample, model) for sample in samples]
    ids = [run["id"] for run in runs]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise SystemExit(
            f"the same run appears more than once: {', '.join(repeated)}; convert only the "
            "final log of each model"
        )
    records = [run["record"] for run in runs if run["record"] is not None]
    args.out.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "provenance": (
                    f"MEASURED LIVE by {MEASURED_TASK} in "
                    + ", ".join(p.name for p in paths)
                    + ". Labels are the runs' own watcher flags; tests_passing was measured "
                    "in a clone of the live container after every turn, and a step with "
                    "measured=false has no measurement."
                ),
                "n_runs": len(records),
                "runs": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out} ({len(records)} runs)\n")
    print(report(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
