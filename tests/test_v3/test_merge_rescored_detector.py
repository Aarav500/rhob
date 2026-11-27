"""scripts/merge_rescored_detector.py: refuse a moved cell, or keep the draw's value.

A horizon fix only removes cells, so a re-scored replicate must agree with the committed
one wherever both scored. Where it cannot (the family's code changed after the draw was
made), ``--keep-old-values`` keeps the committed value and applies only the re-score's
not-applicable set, so every detector in the replicate stays scored on the same runs.

And it merges only into a draw made since the layout-seed fix (``draw_schema`` 2): a
draw made before it was built at layout 0 whatever it records, and a re-score at its
recorded layout would mix two layouts into one replicate.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "merge_rescored_detector.py"
_spec = importlib.util.spec_from_file_location("merge_rescored_detector", _SCRIPT)
merge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merge)

_DET = "Reward Peak"


def _record(cells: dict[tuple[str, float], float | None], na_reason: str | None = None) -> dict:
    by_family: dict[str, list[float]] = {}
    for (fam, _), v in cells.items():
        by_family.setdefault(fam, [])
        if v is not None:
            by_family[fam].append(v)
    scored = [v for v in cells.values() if v is not None]
    return {
        "access_level": "L0",
        "overall_auroc": round(sum(scored) / len(scored), 3) if scored else None,
        "cells": len(cells),
        "cells_measured": len(scored),
        "cells_not_applicable": len(cells) - len(scored),
        "not_applicable_families": sorted(f for f, v in by_family.items() if not v),
        "not_applicable_reasons": {na_reason: len(cells) - len(scored)} if na_reason else {},
        "per_family": {f: (round(sum(v) / len(v), 3) if v else None) for f, v in sorted(by_family.items())},
        "cells_detail": [{"family": f, "difficulty": d, "auroc": v} for (f, d), v in cells.items()],
    }


def _replicate(cells, na_reason=None, draw_schema: int | None = 2) -> dict:
    rep = {
        "replicate_id": 0, "layout_seed": 0, "seed_base": 0, "n_seeds": 5,
        "provenance": {}, "results": {_DET: _record(cells, na_reason)},
    }
    if draw_schema is not None:
        rep["draw_schema"] = draw_schema
    return rep


@pytest.fixture
def dirs(tmp_path):
    into, src = tmp_path / "into", tmp_path / "from"
    into.mkdir()
    src.mkdir()
    old = {("short", 0.9): 0.5, ("short", 0.8): 0.5, ("long", 0.9): 0.7, ("drift", 0.9): 0.3}
    new = {("short", 0.9): None, ("short", 0.8): None, ("long", 0.9): 0.7, ("drift", 0.9): 0.6}
    (into / "replicate_000.json").write_text(json.dumps(_replicate(old)))
    (src / "replicate_000.json").write_text(json.dumps(_replicate(new, "horizon 40 is below 100")))
    return into, src


def _run(into, src, *extra):
    return merge.main(["--detector", _DET, "--from-dir", str(src), "--into-dir", str(into), *extra])


def test_a_moved_cell_is_refused_and_nothing_is_written(dirs, capsys):
    into, src = dirs
    before = (into / "replicate_000.json").read_text()
    assert _run(into, src) == 1
    assert "('drift', 0.9): 0.3 -> 0.6  NOT ALLOWED" in capsys.readouterr().out
    assert (into / "replicate_000.json").read_text() == before


def test_keep_old_values_applies_only_the_not_applicable_set(dirs):
    into, src = dirs
    assert _run(into, src, "--keep-old-values", "--allow-changed-family", "drift",
                "--source", "run 1") == 0
    out = json.loads((into / "replicate_000.json").read_text())
    rec = out["results"][_DET]
    assert rec["per_family"] == {"drift": 0.3, "long": 0.7, "short": None}
    assert rec["overall_auroc"] == 0.5
    assert (rec["cells_measured"], rec["cells_not_applicable"]) == (2, 2)
    assert rec["not_applicable_families"] == ["short"]
    entry = out["provenance"]["rescored"][-1]
    assert entry["source"] == "run 1"
    assert entry["cells_that_did_not_reproduce_old_value_kept"] == [
        {"family": "drift", "difficulty": 0.9, "old_auroc": 0.3, "new_auroc": 0.6}
    ]


def test_without_keep_old_values_the_allowed_new_value_is_taken(dirs):
    into, src = dirs
    assert _run(into, src, "--allow-changed", "drift:0.9") == 0
    rec = json.loads((into / "replicate_000.json").read_text())["results"][_DET]
    assert rec["per_family"]["drift"] == 0.6


def test_report_only_writes_nothing(dirs):
    into, src = dirs
    before = (into / "replicate_000.json").read_text()
    assert _run(into, src, "--report-only", "--allow-changed-family", "drift") == 0
    assert (into / "replicate_000.json").read_text() == before


# ------------------------------------------------------------------ the draw schema
@pytest.mark.parametrize("schema", [None, 1])
def test_a_draw_made_before_the_layout_fix_is_refused(dirs, schema):
    """The ``replicated`` mode's trap once the layout seed reaches the families.

    A committed draw without ``draw_schema`` was built at layout 0 whatever its
    ``layout_seed`` says. Re-scoring it at that seed now builds the recorded layout, so
    every cell moves, and ``--keep-old-values`` would keep the layout-0 values under the
    new not-applicable set. Both modes must refuse, before anything is written.
    """
    into, src = dirs
    old = json.loads((into / "replicate_000.json").read_text())
    if schema is None:
        old.pop("draw_schema")
    else:
        old["draw_schema"] = schema
    (into / "replicate_000.json").write_text(json.dumps(old))
    before = (into / "replicate_000.json").read_text()
    for extra in ((), ("--keep-old-values", "--allow-changed-family", "drift")):
        with pytest.raises(SystemExit, match="draw_schema"):
            _run(into, src, *extra)
    assert (into / "replicate_000.json").read_text() == before


def test_a_rescore_from_older_code_is_refused(dirs):
    """The re-score must be a draw of the same kind as the one it is merged into."""
    into, src = dirs
    new = json.loads((src / "replicate_000.json").read_text())
    new.pop("draw_schema")
    (src / "replicate_000.json").write_text(json.dumps(new))
    with pytest.raises(SystemExit, match="draw_schema differs"):
        _run(into, src, "--allow-changed-family", "drift")


def test_the_minimum_schema_is_the_replicate_scripts():
    """Restated in the merge script to avoid importing the detector suite; keep them equal."""
    spec = importlib.util.spec_from_file_location(
        "_replicate_leaderboard_under_test", _SCRIPT.parent / "replicate_leaderboard.py"
    )
    replicate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replicate)
    assert merge.MIN_DRAW_SCHEMA == replicate.DRAW_SCHEMA
