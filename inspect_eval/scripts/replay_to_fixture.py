"""Turn mode-C replay logs into a trajectory file, and report how faithful the replay was.

``hvtb_replay`` in mode C measures the task's tests after every turn of a recorded run.
This writes those measurements in the schema of
``hvtb_hack_detection/fixtures/synthetic_trajectories.json``, the one the post-hoc
detector scorer reads: one run per replayed sample, id ``<model>/<task>``, one step per
turn with ``tests_passing`` = passed/total after it, ``at`` = the turn's recorded start,
``command`` = its commands and ``measured`` = whether that measurement exists. The label
(``hidden_solution``, ``read_tests``) is the recorded run's, never the replay's: the
replay's own flags and sentinels are fidelity checks, and are left out of the file so that
nothing downstream can read them. ``clone_measurable`` is false for the service tasks,
whose tests a clone cannot run (``SERVICE_TASKS`` in ``hvtb_hack_detection/replay.py``).

No run is dropped. A log the eval did not finish is refused, since the runs it never
reached would be missing without a trace. A replay that diverged from its recording, or
that errored, is written like any other, and named in the report, because which runs
replay badly correlates with the label (``docs/replay-preregistration.md``). An errored
replay keeps the turns it finished; the turns it did not reach, and every turn whose
measurement failed, are written unmeasured, which makes the run N/A for a detector
rather than a stream of made-up values.

The report gives, by recorded label: the recorded calls that never ran and so were not
replayed (``RecordedRun.dropped_calls``, the final message of a run the content filter
stopped), reward and watcher-flag agreement with the recording, per-call exit-status and
output agreement, agreement of the sentinels after the last turn with the recorded flags,
agreement of the last turn's clone with the replay's verifier, the turns whose test
measurement failed, and the runs left without a stream. It then names every run that
diverged and every clone that disagreed.

Usage::

    python scripts/replay_to_fixture.py logs/replay/ --out replayed_trajectories.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalSample, read_eval_log

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hvtb_hack_detection.replay import (  # noqa: E402
    RECORDING_KEY,
    REPLAY_KEY,
    REPLAY_SCORER,
    SERVICE_TASKS,
    RecordedRun,
    ReplayedTurn,
    trajectory_record,
)

SCHEMA = "hvtb-hack-detection-fixture/1"

#: The mode whose turns carry test measurements.
MEASURING_MODE = "C"

#: Report rows: a label, and the fidelity counts summed as numerator and denominator.
AGREEMENT_ROWS = (
    ("reward agrees", "reward_match", None),
    ("watcher flags agree", "flags_match", None),
    ("sentinels agree (last turn)", "sentinels_match", None),
    ("call exit status agrees", "status_matches", "calls"),
    ("call output agrees", "output_matches", "outputs_compared"),
)


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


def complete_samples(path: Path) -> list[EvalSample]:
    """Every sample of a replay log that ran to its end.

    Raises:
        SystemExit: If the eval did not finish, or the log holds fewer samples than it
            planned: an eval stopped by a sample error or a cancel logs only the samples
            that finished, and the runs it never reached would vanish from the file.
    """
    log = read_eval_log(str(path))
    samples = log.samples or []
    planned = log.results.total_samples if log.results is not None else None
    if log.status != "success" or planned is None or len(samples) != planned:
        raise SystemExit(
            f"{path} is not a finished replay (status {log.status}, {len(samples)} of "
            f"{planned if planned is not None else '?'} samples); finish it with "
            "`inspect eval-retry` and convert the retried log"
        )
    return samples


def replayed(sample: EvalSample) -> dict[str, Any]:
    """One sample of a replay log: its recording, turns, fidelity, and error if any.

    The turns come from the score, or, when the sample was never scored (its replay or
    its verifier raised), from the sample's metadata, where the solver keeps every turn
    it finished.

    Raises:
        SystemExit: If the sample was replayed in a mode that measures no tests.
    """
    recording = RecordedRun.model_validate(sample.metadata[RECORDING_KEY])
    score = (sample.scores or {}).get(REPLAY_SCORER)
    scored = score.metadata if score is not None and score.metadata else None
    replay = scored[REPLAY_KEY] if scored else sample.metadata.get(REPLAY_KEY)
    if replay is not None and replay["mode"] != MEASURING_MODE:
        raise SystemExit(
            f"sample {sample.id} was replayed in mode {replay['mode']}, which measures no "
            f"tests; only mode {MEASURING_MODE} logs make a trajectory file"
        )
    turns = [ReplayedTurn.model_validate(turn) for turn in (replay or {}).get("turns", [])]
    if scored:
        return {
            "recording": recording,
            "turns": turns,
            "fidelity": scored["fidelity"],
            "reward": scored.get("reward"),
            "error": None,
        }
    error = sample.error.message if sample.error else f"no {REPLAY_SCORER} score"
    return {
        "recording": recording,
        "turns": turns,
        "fidelity": None,
        "reward": None,
        "error": error,
    }


def _ratio(numerator: int, denominator: int) -> str:
    if not denominator:
        return "-"
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)"


def _run_id(run: dict[str, Any]) -> str:
    return f"{run['recording'].model}/{run['recording'].task}"


def _unmeasured(run: dict[str, Any]) -> bool:
    """Whether a run has a turn without a measurement, so no stream to score."""
    return not all(step["measured"] for step in run["record"]["steps"])


def report(runs: list[dict[str, Any]]) -> str:
    """The fidelity report, by recorded label, then the runs to look at one by one.

    Those are the runs that diverged from their recording, the runs whose last clone
    disagrees with the verifier (service tasks marked: when the verifier passes one, its
    clone cannot), and
    the runs with an unmeasured turn. Each run carries its trajectory ``record``.
    """
    groups = {
        "hacked": [r for r in runs if r["recording"].hacked],
        "clean": [r for r in runs if not r["recording"].hacked],
        "all": runs,
    }
    width = 26
    lines = [f"{'':34s}" + "".join(f"{name:>{width}s}" for name in groups)]

    def row(label: str, cell: Any) -> None:
        lines.append(f"{label:34s}" + "".join(f"{cell(g):>{width}s}" for g in groups.values()))

    row("runs", lambda g: str(len(g)))
    row("replay errors", lambda g: str(sum(r["error"] is not None for r in g)))
    row(
        "service tasks (clone cannot test)",
        lambda g: str(sum(r["recording"].task in SERVICE_TASKS for r in g)),
    )
    # The calls of a final message the content filter stopped before they ran.
    row(
        "recorded calls never run, not replayed",
        lambda g: str(sum(len(r["recording"].dropped_calls) for r in g)),
    )
    for label, key, total in AGREEMENT_ROWS:

        def cell(group: list[dict[str, Any]], key: str = key, total: str | None = total) -> str:
            scored = [r["fidelity"] for r in group if r["fidelity"] is not None]
            if total is None:
                counted = [f[key] for f in scored if f[key] is not None]
                return _ratio(sum(map(bool, counted)), len(counted))
            return _ratio(sum(f[key] for f in scored), sum(f[total] for f in scored))

        row(label, cell)

    def clone_cell(group: list[dict[str, Any]]) -> str:
        counted = [
            r["fidelity"]["final_tests_agree"]
            for r in group
            if r["fidelity"] is not None
            and r["recording"].task not in SERVICE_TASKS
            and r["fidelity"]["final_tests_agree"] is not None
        ]
        return _ratio(sum(counted), len(counted))

    row("last clone agrees with verifier", clone_cell)
    row(
        "turns measured",
        lambda g: _ratio(
            sum(step["measured"] for r in g for step in r["record"]["steps"]),
            sum(len(r["record"]["steps"]) for r in g),
        ),
    )
    row("runs with no stream (unmeasured)", lambda g: str(sum(map(_unmeasured, g))))

    diverged = []
    for r in runs:
        fidelity = r["fidelity"]
        if r["error"] is not None:
            diverged.append(f"  {_run_id(r)}: replay error: {r['error'][:200]}")
        elif not (fidelity["reward_match"] and fidelity["flags_match"]):
            diverged.append(
                f"  {_run_id(r)}: reward_match={fidelity['reward_match']} "
                f"flags_match={fidelity['flags_match']}"
            )
    lines.append(f"\nruns that diverged from their recording: {len(diverged)}")
    lines.extend(diverged)

    disagreed = [
        f"  {r['recording'].task} ({r['recording'].model}): last clone passed "
        f"{r['fidelity']['final_tests_passing']:.0%} of its tests, verifier reward "
        f"{r['reward']}" + (" (service task)" if r["recording"].task in SERVICE_TASKS else "")
        for r in sorted(runs, key=lambda r: (r["recording"].task, r["recording"].model))
        if r["fidelity"] is not None and r["fidelity"]["final_tests_agree"] is False
    ]
    lines.append(f"\nruns whose last clone disagrees with the verifier: {len(disagreed)}")
    lines.extend(disagreed)

    unmeasured = [
        f"  {_run_id(r)}: "
        + next(s["measurement_error"] for s in r["record"]["steps"] if not s["measured"])[:200]
        for r in runs
        if _unmeasured(r)
    ]
    lines.append(f"\nruns with an unmeasured turn (first reason): {len(unmeasured)}")
    lines.extend(unmeasured)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Write the trajectory file and print the fidelity report."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("targets", type=Path, nargs="+", help=".eval logs or directories of them")
    ap.add_argument("--out", type=Path, required=True, help="trajectory JSON to write")
    args = ap.parse_args(argv)

    paths = _log_paths(args.targets)
    runs = [replayed(sample) for path in paths for sample in complete_samples(path)]
    for run in runs:
        run["record"] = trajectory_record(run["recording"], run["turns"], run["error"])
    records = [run["record"] for run in runs]
    ids = [record["id"] for record in records]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise SystemExit(f"the same run was replayed more than once: {', '.join(repeated)}")
    args.out.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "provenance": (
                    "REPLAYED by hvtb_replay (mode C) from "
                    + ", ".join(p.name for p in paths)
                    + ". Labels are the recorded watcher flags; tests_passing was measured "
                    "in a clone of the container after every turn, and a step with "
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
