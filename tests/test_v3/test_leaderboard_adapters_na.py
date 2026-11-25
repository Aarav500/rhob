"""The ``results``-schema adapter on a board that records not-applicable cells.

Since the N/A schema, ``leaderboard/v5_leaderboard.json`` writes ``null`` for a family a
detector cannot measure and records ``cells_measured`` next to ``cells``. The adapter
used to copy ``per_family`` verbatim, so a ``None`` reached
``LeaderboardEntry.family_auroc`` (typed ``dict[str, float]``), and it reported
``cells`` (123 for every detector) next to an ``overall_auroc`` computed over 35.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rhob.v3.leaderboard.adapters import load_any_leaderboard_json

_V5 = Path(__file__).resolve().parents[2] / "leaderboard" / "v5_leaderboard.json"


def test_null_families_are_dropped_and_the_measured_denominator_is_used(tmp_path):
    path = tmp_path / "na_board.json"
    path.write_text(
        json.dumps(
            {
                "timestamp": "2026-11-19T00:00:00Z",
                "results": {
                    "L1 Detector": {
                        "access_level": "L1",
                        "overall_auroc": 0.9,
                        "cells": 123,
                        "cells_measured": 35,
                        "per_family": {"gridworld_camping": 0.9, "mujoco_camping": None},
                    },
                    "Old Schema": {
                        "access_level": "L0",
                        "overall_auroc": 0.5,
                        "cells": 40,
                        "per_family": {"gridworld_camping": 0.5},
                    },
                },
            }
        )
    )
    by_name = {e.detector_name: e for e in load_any_leaderboard_json(path).entries}

    assert by_name["L1 Detector"].family_auroc == {"gridworld_camping": 0.9}
    assert by_name["L1 Detector"].n_cells == 35
    # A board written before cells_measured existed still reports its cell count.
    assert by_name["Old Schema"].n_cells == 40


@pytest.mark.skipif(not _V5.exists(), reason="committed v5 board not present")
def test_committed_board_entries_carry_no_none_families():
    board = json.loads(_V5.read_text(encoding="utf-8"))
    entries = {e.detector_name: e for e in load_any_leaderboard_json(_V5).entries}

    for name, entry in entries.items():
        assert all(isinstance(v, float) for v in entry.family_auroc.values()), name
        raw = board["results"][name]
        measured = [f for f, v in raw["per_family"].items() if v is not None]
        assert sorted(entry.family_auroc) == sorted(measured), name
        if "cells_measured" in raw:
            assert entry.n_cells == raw["cells_measured"], name
