"""Re-score named detectors on one committed HVTA pairs draw and merge them into its artifact.

When one detector is fixed, re-running ``score_detectors.py`` wholesale re-scores all 35
on whatever interpreter is at hand and moves numbers that have nothing to do with the fix.
The committed primary draw was scored on Windows (Python 3.13, numpy 2.4); on Linux with
numpy 2.5 the rank-deficient AR(p) Residual fit on degenerate cell 0 alone moves from
0.460 to 0.475. This script re-scores only the named detectors and leaves every other row
as committed.

It re-scores through ``score_detectors.score_cell`` itself, so each row gets the same
``_evaluate_cell`` call, the same bit-for-bit mirror check behind the onset diagnostics and
the same counterfactual-orientation pass as a full run. It then:

1. refuses unless the pairs file's metadata is exactly the artifact's ``generation`` block
   (the same draw), and unless recomputing ``per_detector`` and ``access_level_summary``
   from the committed ``per_cell`` reproduces the committed values exactly, so that
   recomputing them after the merge can only move the re-scored detectors;
2. replaces the named detectors' rows in ``per_cell`` in place, recomputes both aggregates
   from the merged rows and refuses if any other detector's aggregate moved;
3. appends a record to the artifact's ``rescored`` list: which detectors, why, when, the
   RHOB revision and the uncommitted ``src/`` files at the time, each detector's source
   (``score_detectors.source_sha256``: a hash of its code without docstrings or comments,
   and its git blob, with whether that blob is the one committed at the recorded head),
   the interpreter, the figures being replaced, and which replicate draws did not attest
   that source when the record was written (``replicates_not_rescored``, a snapshot);
4. if the artifact comes from a full ``score_detectors.py`` run and so carries a
   top-level ``detector_sources`` map, updates the re-scored detectors' entries in it,
   so the map keeps naming the source every row was scored with.

``make_results.py`` reads that record: it names the re-score in the report's provenance
line, and it marks a replicate's figures for a re-scored detector as pre-fix (excluded
from across-draw means and ranges) unless the replicate attests the same detector source,
in its own ``rescored`` record or the ``detector_sources`` map ``score_detectors.py``
writes. It works out which replicates are stale from the files it loads, not from the
snapshot, so re-scoring a replicate clears its markers and the provenance sentence alike.

An unknown ``--detector`` is refused through argparse (exit status 2) before anything is
read or written. Every other refusal exits with status 1 and writes nothing.

Run (from the repository root)::

    python -I external/hvta/rescore_detector.py --detector BOCPDDetector \\
        --reason "classify now returns the score its docstring specifies"

For a replicate draw, pass its frozen pairs with ``--pairs-stem`` and the replicate file
with ``--results``; the pairs of the committed replicates are not in the repository, so
that needs ``export_pairs.py`` in the HVTA venv first (see ``README.md``).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as _dt
import io
import json
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import score_detectors as sd  # noqa: E402  (puts the repository's src/ on sys.path)


def _aggregates(per_cell: dict, cells_meta: list) -> tuple[list, dict]:
    results = {int(k): v for k, v in per_cell.items()}
    agg = sd.aggregate(results, cells_meta)
    # Through JSON, so floats compare exactly as they will be written.
    agg = json.loads(json.dumps(agg, default=sd._json_safe))
    summary = json.loads(json.dumps(sd.access_level_summary(agg), default=sd._json_safe))
    return agg, summary


def rescore(artifact: dict, meta: dict, cells: list, factories: list) -> dict:
    """Return a copy of ``artifact`` with ``factories``' rows re-scored and merged."""
    if json.loads(json.dumps(meta)) != artifact["generation"]:
        raise SystemExit(
            "refusing: the pairs metadata is not this artifact's generation block, so "
            "these are not the runs it was scored on"
        )
    agg0, summary0 = _aggregates(artifact["per_cell"], meta["cells"])
    if agg0 != artifact["per_detector"] or summary0 != artifact["access_level_summary"]:
        raise SystemExit(
            "refusing: recomputing the aggregates from the committed per_cell does not "
            "reproduce the committed ones, so a merge would move other detectors too"
        )

    names = [f().name for f in factories]
    merged = copy.deepcopy(artifact)
    saved = sd.DETECTOR_FACTORIES
    sd.DETECTOR_FACTORIES = list(factories)
    try:
        for cell in cells:
            idx = str(cell["meta"]["index"])
            with contextlib.redirect_stdout(io.StringIO()):
                fresh = {r["detector"]: r for r in sd.score_cell(cell, verbose=False)}
            rows = merged["per_cell"][idx]
            for name in names:
                at = [i for i, r in enumerate(rows) if r["detector"] == name]
                if len(at) != 1:
                    raise SystemExit(f"refusing: cell {idx} has {len(at)} rows for {name}")
                if "error" in fresh[name]:
                    raise SystemExit(f"refusing: cell {idx} {name}: {fresh[name]['error']}")
                rows[at[0]] = json.loads(json.dumps(fresh[name], default=sd._json_safe))
    finally:
        sd.DETECTOR_FACTORIES = saved

    agg1, summary1 = _aggregates(merged["per_cell"], meta["cells"])
    before = {r["detector"]: r for r in agg0}
    after = {r["detector"]: r for r in agg1}
    moved = sorted(n for n in before if n not in names and before[n] != after.get(n))
    if moved or set(before) != set(after):
        raise SystemExit(f"refusing: the merge moved other detectors' aggregates: {moved}")
    merged["per_detector"] = agg1
    merged["access_level_summary"] = summary1
    if "detector_sources" in merged:
        for f in factories:
            merged["detector_sources"][f().name] = sd.source_sha256(f)
    return merged


def _previous(artifact: dict, name: str) -> dict:
    row = next(r for r in artifact["per_detector"] if r["detector"] == name)
    keys = ("mean_auroc", "min_auroc", "max_auroc", "mean_onset_mae_normalized",
            "mean_miss_rate_hacking", "mean_false_onset_rate_legit")
    return {k: row[k] for k in keys}


def _with_head(src: dict, head: str | None) -> dict:
    """``src`` plus whether its git blob is the file committed at ``head``.

    True means ``rhob_git_head`` alone reproduces the rows; False means the rows came from
    an uncommitted edit, which ``git_blob`` still names once it is committed.
    """
    at_head = sd._git("rev-parse", f"{head}:{src['path']}") if head else None
    blob = src.get("git_blob")
    return {**src, "matches_head": None if not (at_head and blob) else at_head.strip() == blob}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs-stem", default=str(HERE / "hvta_pairs"))
    ap.add_argument("--results", default=str(HERE / "hvta_results.json"))
    ap.add_argument("--out", default=None, help="defaults to --results (merge in place)")
    ap.add_argument("--detector", action="append", required=True,
                    help="class name from score_detectors.DETECTOR_FACTORIES; repeatable")
    ap.add_argument("--reason", required=True, help="why the rows are being re-scored")
    args = ap.parse_args(argv)

    by_class = {f.__name__: f for f in sd.DETECTOR_FACTORIES}
    unknown = [d for d in args.detector if d not in by_class]
    if unknown:
        ap.error(f"not in DETECTOR_FACTORIES: {unknown}")
    factories = [by_class[d] for d in args.detector]

    results_path = Path(args.results)
    artifact = json.loads(results_path.read_text(encoding="utf-8"))
    meta, cells = sd.load_cells(Path(args.pairs_stem))
    merged = rescore(artifact, meta, cells, factories)

    revision = sd.rhob_revision()
    sources = {
        f().name: _with_head(sd.source_sha256(f), revision["rhob_git_head"]) for f in factories
    }
    replicates = sorted((HERE / "replicates").glob("*.json"))
    stale = {}
    for name, src in sources.items():
        stale[name] = [
            p.relative_to(HERE).as_posix() for p in replicates
            if p.resolve() != results_path.resolve()
            and not sd.same_source(
                sd.current_source(json.loads(p.read_text(encoding="utf-8")), name), src
            )
        ]
    record = {
        "detectors": list(sources),
        "reason": args.reason,
        "date_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "external/hvta/rescore_detector.py",
        **revision,
        "detector_sources": sources,
        "scoring_env": {
            "python": sys.version.split()[0],
            "executable": sd._scrub_home(sys.executable),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "n_cells": len(cells),
        "previous": {name: _previous(artifact, name) for name in sources},
        "replicates_not_rescored": stale,
    }
    merged.setdefault("rescored", []).append(record)

    out = Path(args.out) if args.out else results_path
    out.write_text(json.dumps(merged, indent=2, default=sd._json_safe), encoding="utf-8")
    for name in sources:
        new = next(r for r in merged["per_detector"] if r["detector"] == name)
        old = record["previous"][name]
        print(f"{name}: mean AUROC {old['mean_auroc']} -> {new['mean_auroc']}, "
              f"onset MAE {old['mean_onset_mae_normalized']} -> "
              f"{new['mean_onset_mae_normalized']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
