"""Turn mode-C replay logs into a trajectory file, and report how faithful the replay was.

``hvtb_replay`` in mode C measures the task's tests after every turn of a recorded run.
This writes those measurements in the schema of
``hvtb_hack_detection/fixtures/synthetic_trajectories.json``, the one the post-hoc
detector scorer reads: one run per replayed sample, id ``<model>/<task>``, one step per
turn with ``tests_passing`` = passed/total after it, ``at`` = the turn's recorded start and
``command`` = its commands. The label (``hidden_solution``, ``read_tests``) is the
recorded run's, never the replay's: the replay's own flags and sentinels are fidelity
checks, and are left out of the file so that nothing downstream can read them.

No run is dropped. A replay that diverged from its recording, or that errored, is written
like any other (an errored one with no measurements and a ``replay_error``) and named in
the report, because which runs replay badly correlates with the label
(``docs/replay-preregistration.md``).

The report gives, by recorded label: reward and watcher-flag agreement with the
recording, per-call exit-status and output agreement, agreement of the sentinels after
the last turn with the recorded flags, and the turns whose test measurement failed.

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


def replayed(sample: EvalSample) -> dict[str, Any]:
    """One sample of a replay log: its recording, turns, fidelity, and error if any.

    Raises:
        SystemExit: If the sample was replayed in a mode that measures no tests.
    """
    recording = RecordedRun.model_validate(sample.metadata[RECORDING_KEY])
    score = (sample.scores or {}).get(REPLAY_SCORER)
    if score is None or not score.metadata:
        error = sample.error.message if sample.error else f"no {REPLAY_SCORER} score"
        return {"recording": recording, "turns": [], "fidelity": None, "error": error}
    replay = score.metadata[REPLAY_KEY]
    if replay["mode"] != MEASURING_MODE:
        raise SystemExit(
            f"sample {sample.id} was replayed in mode {replay['mode']}, which measures no "
            f"tests; only mode {MEASURING_MODE} logs make a trajectory file"
        )
    return {
        "recording": recording,
        "turns": [ReplayedTurn.model_validate(turn) for turn in replay["turns"]],
        "fidelity": score.metadata["fidelity"],
        "error": None,
    }


def _ratio(numerator: int, denominator: int) -> str:
    if not denominator:
        return "-"
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)"


def report(runs: list[dict[str, Any]]) -> str:
    """The fidelity report, by recorded label, then every run that diverged."""
    groups = {
        "hacked": [r for r in runs if r["recording"].hacked],
        "clean": [r for r in runs if not r["recording"].hacked],
        "all": runs,
    }
    width = 26
    lines = [f"{'':30s}" + "".join(f"{name:>{width}s}" for name in groups)]

    def row(label: str, cell: Any) -> None:
        lines.append(f"{label:30s}" + "".join(f"{cell(g):>{width}s}" for g in groups.values()))

    row("runs", lambda g: str(len(g)))
    row("replay errors", lambda g: str(sum(r["error"] is not None for r in g)))
    for label, key, total in AGREEMENT_ROWS:

        def cell(group: list[dict[str, Any]], key: str = key, total: str | None = total) -> str:
            scored = [r["fidelity"] for r in group if r["fidelity"] is not None]
            if total is None:
                counted = [f[key] for f in scored if f[key] is not None]
                return _ratio(sum(map(bool, counted)), len(counted))
            return _ratio(sum(f[key] for f in scored), sum(f[total] for f in scored))

        row(label, cell)
    row(
        "turns measured",
        lambda g: _ratio(
            sum(1 for r in g for t in r["turns"] if t.tests and t.tests.tests_passing is not None),
            sum(len(r["turns"]) for r in g),
        ),
    )

    diverged = []
    for r in runs:
        run_id = f"{r['recording'].model}/{r['recording'].task}"
        fidelity = r["fidelity"]
        if r["error"] is not None:
            diverged.append(f"  {run_id}: replay error: {r['error'][:200]}")
        elif not (fidelity["reward_match"] and fidelity["flags_match"]):
            diverged.append(
                f"  {run_id}: reward_match={fidelity['reward_match']} "
                f"flags_match={fidelity['flags_match']}"
            )
    lines.append(f"\nruns that diverged from their recording: {len(diverged)}")
    lines.extend(diverged)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Write the trajectory file and print the fidelity report."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("targets", type=Path, nargs="+", help=".eval logs or directories of them")
    ap.add_argument("--out", type=Path, required=True, help="trajectory JSON to write")
    args = ap.parse_args(argv)

    paths = _log_paths(args.targets)
    runs = [replayed(sample) for path in paths for sample in read_eval_log(str(path)).samples or []]
    records = [trajectory_record(r["recording"], r["turns"], r["error"]) for r in runs]
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
                    "in a clone of the container after every turn."
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
