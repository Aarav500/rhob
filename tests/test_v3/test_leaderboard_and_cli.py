"""Tests for the v3 leaderboard and CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from rhob.v3.benchmark import BenchmarkResults, CellResult
from rhob.v3.leaderboard.board import Leaderboard, LeaderboardEntry


def _fake_results() -> BenchmarkResults:
    return BenchmarkResults(
        detector_name="fake_det",
        access_level="L2",
        cells=[
            CellResult("gridworld_camping", "camping", 1.0, 1.000, 0.1, 10),
            CellResult("continuous_camping", "camping", 0.90, 0.850, 0.2, 10),
            CellResult("continuous_camping", "camping", 0.70, 0.600, 0.3, 10),
        ],
    )


def test_entry_from_results_aggregates_correctly():
    entry = LeaderboardEntry.from_results(_fake_results(), author="tester")
    assert entry.detector_name == "fake_det"
    assert entry.access_level == "L2"
    assert entry.n_cells == 3
    assert entry.family_auroc["gridworld_camping"] == 1.0
    assert abs(entry.family_auroc["continuous_camping"] - 0.725) < 1e-6
    assert "camping" in entry.mechanism_auroc


def test_leaderboard_standings_sorted_descending():
    board = Leaderboard()
    board.submit(_fake_results(), author="a")
    weaker = BenchmarkResults("weak_det", "L0", [CellResult("gridworld_camping", "camping", 1.0, 0.5, 0.5, 10)])
    board.submit(weaker, author="b")
    standings = board.standings()
    assert standings[0].detector_name == "fake_det"
    assert standings[1].detector_name == "weak_det"


def test_leaderboard_save_and_load_roundtrip(tmp_path: Path):
    board = Leaderboard()
    board.submit(_fake_results(), author="tester")
    path = tmp_path / "leaderboard.json"
    board.save(path)

    reloaded = Leaderboard.load(path)
    assert len(reloaded.entries) == 1
    assert reloaded.entries[0].detector_name == "fake_det"


def test_leaderboard_load_missing_file_returns_empty(tmp_path: Path):
    board = Leaderboard.load(tmp_path / "does_not_exist.json")
    assert board.entries == []


def test_render_standings_md_contains_detector_name():
    board = Leaderboard()
    board.submit(_fake_results(), author="tester")
    md = board.render_standings_md()
    assert "fake_det" in md
    assert "tester" in md


def test_render_by_mechanism_and_difficulty():
    board = Leaderboard()
    board.submit(_fake_results(), author="tester")
    assert "camping" in board.render_by_mechanism_md()
    assert "TRIVIAL" in board.render_by_difficulty_md()


# --------------------------------------------------------------------- CLI (subprocess)


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "rhob", *args],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )


def test_cli_families_lists_builtin_families():
    result = _run_cli("families")
    assert result.returncode == 0
    assert "gridworld_camping" in result.stdout
    assert "continuous_camping" in result.stdout


def test_cli_validate_rejects_missing_keys(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"detector_name": "x"}))
    result = _run_cli("validate", str(bad))
    assert result.returncode == 1
    assert "INVALID" in result.stdout


def test_cli_validate_accepts_well_formed_submission(tmp_path: Path):
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps({"detector_name": "x", "access_level": "L2", "overall_auroc": 0.9, "n_cells": 4})
    )
    result = _run_cli("validate", str(good))
    assert result.returncode == 0
    assert "VALID" in result.stdout


def test_cli_evaluate_end_to_end_on_gridworld(tmp_path: Path):
    detector_file = tmp_path / "det.py"
    detector_file.write_text(
        "from rhob.detectors.posthoc import PosthocDetector\n"
        "class D(PosthocDetector):\n"
        "    @property\n"
        "    def access_level(self): return 'L2'\n"
        "    @property\n"
        "    def name(self): return 'toy'\n"
        "    def classify(self, run): return float(run.behav_trace[-1])\n"
        "    def detect_onset(self, run): return len(run.proxy_rewards)//2\n"
    )
    out = tmp_path / "submission.json"
    result = _run_cli(
        "evaluate", "--detector", str(detector_file), "--families", "gridworld_camping",
        "--n-seeds", "3", "--output", str(out),
    )
    assert result.returncode == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["detector_name"] == "toy"
    assert data["overall_auroc"] > 0.9
