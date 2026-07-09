"""Tests for the leaderboard schema adapter."""

from __future__ import annotations

import json

from rhob.v3.leaderboard.adapters import load_any_leaderboard_json
from rhob.v3.leaderboard.board import Leaderboard, LeaderboardEntry


def test_loads_entries_schema(tmp_path):
    lb = Leaderboard(
        [
            LeaderboardEntry(
                detector_name="Foo",
                access_level="L2",
                author="alice",
                timestamp="2026-01-01T00:00:00Z",
                overall_auroc=0.8,
                n_cells=10,
            )
        ]
    )
    path = tmp_path / "entries_style.json"
    lb.save(path)

    loaded = load_any_leaderboard_json(path)
    assert len(loaded.entries) == 1
    assert loaded.entries[0].detector_name == "Foo"
    assert loaded.entries[0].overall_auroc == 0.8


def test_loads_results_schema(tmp_path):
    path = tmp_path / "results_style.json"
    path.write_text(
        json.dumps(
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "results": {
                    "Bar": {
                        "access_level": "L0",
                        "overall_auroc": 0.5,
                        "cells": 40,
                        "per_family": {"gridworld_camping": 0.5},
                    },
                    "Broken Detector": {"access_level": "L1", "error": "Input contains NaN."},
                },
            }
        )
    )

    loaded = load_any_leaderboard_json(path)
    # Broken detector (no overall_auroc) is skipped, not crashed on.
    assert len(loaded.entries) == 1
    assert loaded.entries[0].detector_name == "Bar"
    assert loaded.entries[0].family_auroc == {"gridworld_camping": 0.5}


def test_unrecognized_schema_raises(tmp_path):
    path = tmp_path / "garbage.json"
    path.write_text(json.dumps({"nonsense": True}))

    try:
        load_any_leaderboard_json(path)
        assert False, "expected ValueError"
    except ValueError:
        pass
