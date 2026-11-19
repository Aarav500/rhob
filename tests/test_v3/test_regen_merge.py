"""``--merge-shards`` in the two regeneration scripts publishes only a complete, uniform set.

``.github/workflows/regenerate.yml`` certifies the ledger one cell per runner and scores
the board one detector per runner, then commits what the merge writes in place of the
published artifacts. A merge that accepted a missing cell, a duplicate, a shard from a
dirty tree or from different package versions would publish exactly the kind of
artifact the repository's provenance blocks exist to rule out, labelled complete. These
tests build shards by hand, so they need no rollouts, and check each refusal.

That a merged artifact equals an unsharded run of the same scope was checked by running
both on two families (2026-11-19, recorded in the commit that added the merge); that
needs rollouts and is not repeated here.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from rhob.v3.admission_gate import CRITERIA

_REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ledger_script():
    return _load("_admission_ledger_under_test", _REPO / "scripts" / "admission_ledger.py")


@pytest.fixture(scope="module")
def board_script():
    pytest.importorskip("torch", reason="the board script imports the two torch detectors")
    return _load("_v5_board_under_test", _REPO / "scripts" / "v5_leaderboard_and_transfer.py")


FAMILIES = ["eval_probe_sandbagging", "monitored_sandbagging"]


def _provenance(commit: str = "a" * 40) -> dict:
    return {
        "generated_utc": "2026-11-19T00:00:00Z",
        "git": {"commit": commit, "short_commit": commit[:12], "branch": "b",
                "dirty": False, "dirty_files": []},
        "python": "3.12.16",
        "python_implementation": "CPython",
        "platform": "Linux-x86_64",
        "packages": {"numpy": "2.5.4", "mujoco": "3.17.0"},
        "argv": [],
    }


def _ledger_shard(family: str, difficulty: float, status: str = "ADMITTED") -> dict:
    prov = _provenance()
    prov["scope"] = {"families_run": [family], "families_registered": 33,
                     "complete": False, "difficulty_override": [difficulty]}
    prov["gate"] = {"n_layouts": 12, "min_seeds_per_layout": 24, "n_bootstrap": 2000,
                    "equivalence_margin": 0.1, "informativeness_floor": 1e-4, "alpha": 0.05,
                    "target_power": 0.9, "behavioral_floor": 0.6, "onset_std_frac": 0.25}
    degenerate = status == "DEGENERATE"
    outcomes = {c: ("degenerate" if degenerate and c == "proxy_matched" else "pass")
                for c in CRITERIA}
    row = {
        "family": family, "difficulty": difficulty, "passed": status == "ADMITTED",
        "status": status, "criteria": {c: outcomes[c] == "pass" for c in CRITERIA},
        "outcomes": outcomes,
        "degenerate_criteria": ["proxy_matched"] if degenerate else [],
        "details": {}, "metrics": {}, "design": {"n_layouts": 12}, "error": None,
    }
    return {"schema": "rhob.admission_ledger/1", "provenance": prov,
            "design": {"n_layouts": 12}, "summary": {}, "results": [row],
            "timing_seconds": {family: 2.0}}


def _write_ledger_shards(root: Path, ledger_script, mutate=None) -> list[Path]:
    paths = []
    # Written in reverse so the merge has to restore grid order itself.
    for i, (family, difficulty) in enumerate(reversed(ledger_script.ledger_grid(FAMILIES))):
        shard = _ledger_shard(family, difficulty,
                              "DEGENERATE" if family == "monitored_sandbagging" else "ADMITTED")
        if mutate is not None:
            mutate(i, shard)
        path = root / f"ledger-{i:03d}" / "admission_ledger.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(shard))
        paths.append(path)
    return paths


def test_a_complete_set_merges_in_grid_order(tmp_path, ledger_script):
    _write_ledger_shards(tmp_path, ledger_script)
    payload = ledger_script.merge_shards(tmp_path, families=FAMILIES)
    grid = ledger_script.ledger_grid(FAMILIES)
    assert [(r["family"], r["difficulty"]) for r in payload["results"]] == grid
    s = payload["summary"]
    assert (s["n_cells"], s["n_admitted"], s["n_degenerate"]) == (10, 5, 5)
    assert s["degenerate_families"] == ["monitored_sandbagging"]
    assert payload["timing_seconds"] == {f: 10.0 for f in FAMILIES}
    scope = payload["provenance"]["scope"]
    assert scope["families_run"] == FAMILIES and scope["difficulty_override"] is None
    # Two of 33 families is not the full grid, and must not be labelled as one.
    assert scope["complete"] is False
    assert payload["provenance"]["sharding"]["n_shards"] == 10


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda i, s: s["provenance"]["git"].update(dirty=True), "dirty"),
        (lambda i, s: s["provenance"]["git"].update(dirty=None), "dirty"),
        (lambda i, s: i == 3 and s["provenance"]["git"].update(commit="b" * 40), "commit"),
        (lambda i, s: i == 3 and s["provenance"]["packages"].update(mujoco="4.0.0"), "packages"),
        (lambda i, s: i == 3 and s["provenance"]["gate"].update(n_layouts=6), "gate"),
        (lambda i, s: i == 3 and s["results"][0]["design"].update(n_layouts=6), "design"),
        (lambda i, s: i == 3 and s["provenance"]["scope"].update(difficulty_override=None),
         "scope"),
        (lambda i, s: i == 3 and s["results"].append(copy.deepcopy(s["results"][0])),
         "exactly one cell"),
    ],
)
def test_a_nonuniform_set_is_refused(tmp_path, ledger_script, mutate, message):
    _write_ledger_shards(tmp_path, ledger_script, mutate)
    with pytest.raises(ledger_script.ShardError, match=message):
        ledger_script.merge_shards(tmp_path, families=FAMILIES)


def test_a_missing_or_duplicated_cell_is_refused(tmp_path, ledger_script):
    paths = _write_ledger_shards(tmp_path, ledger_script)
    paths[0].unlink()
    with pytest.raises(ledger_script.ShardError, match="1 missing"):
        ledger_script.merge_shards(tmp_path, families=FAMILIES)
    paths[0].write_text(paths[1].read_text())
    with pytest.raises(ledger_script.ShardError, match="already supplied"):
        ledger_script.merge_shards(tmp_path, families=FAMILIES)


def test_a_cell_outside_the_grid_is_refused(tmp_path, ledger_script):
    _write_ledger_shards(tmp_path, ledger_script)
    stray = tmp_path / "stray" / "admission_ledger.json"
    stray.parent.mkdir()
    stray.write_text(json.dumps(_ledger_shard("orbit_chirality", 0.9)))
    with pytest.raises(ledger_script.ShardError, match="1 not in the grid"):
        ledger_script.merge_shards(tmp_path, families=FAMILIES)


def test_a_cell_that_raised_needs_an_explicit_allowance(tmp_path, ledger_script):
    def broken(i, shard):
        if i == 0:
            shard["results"][0].update(passed=False, status="NOT ADMITTED",
                                       error="ImportError: no mujoco")

    _write_ledger_shards(tmp_path, ledger_script, broken)
    with pytest.raises(ledger_script.ShardError, match="1 cell"):
        ledger_script.merge_shards(tmp_path, families=FAMILIES)
    payload = ledger_script.merge_shards(tmp_path, families=FAMILIES, allow_cell_errors=True)
    assert payload["summary"]["n_not_admitted"] == 1


def test_the_merged_ledger_passes_the_published_schema(tmp_path, ledger_script):
    """The same top-level keys the committed artifact is pinned to."""
    _write_ledger_shards(tmp_path, ledger_script)
    payload = ledger_script.merge_shards(tmp_path, families=FAMILIES)
    assert set(payload) == {"schema", "provenance", "design", "summary", "results",
                            "timing_seconds"}
    json_path, md_path = ledger_script.write_artifacts(payload, tmp_path / "out")
    assert json.loads(json_path.read_text()) == payload
    assert md_path.read_text().startswith("# RHOB Admission Ledger")


DETECTORS = ["Reward Threshold", "Reward Skewness", "State Frequency Anomaly"]


def _board_shard(board_script, name: str) -> dict:
    record = {"access_level": "L0", "overall_auroc": 0.5, "cells": 10, "cells_measured": 10,
              "cells_not_applicable": 0, "not_applicable_families": [],
              "not_applicable_reasons": {}, "per_family": {f: 0.5 for f in FAMILIES}}
    return board_script.build_board({name: record}, FAMILIES, 5, _provenance())


def _write_board_shards(root: Path, board_script, mutate=None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(reversed(DETECTORS)):
        shard = _board_shard(board_script, name)
        if mutate is not None:
            mutate(i, shard)
        (root / f"board-{i:02d}.json").write_text(json.dumps(shard))


def test_a_complete_board_set_merges_in_detector_order(tmp_path, board_script):
    _write_board_shards(tmp_path, board_script)
    board = board_script.merge_shards(tmp_path, detectors=DETECTORS, families=FAMILIES)
    assert list(board["results"]) == DETECTORS
    assert board["detectors"] == 3 and board["families_evaluated"] == FAMILIES
    assert board["sampling"] == _board_shard(board_script, DETECTORS[0])["sampling"]
    assert board["provenance"]["sharding"]["n_shards"] == 3


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda i, s: s["provenance"]["git"].update(dirty=True), "dirty"),
        (lambda i, s: i == 1 and s["provenance"]["git"].update(commit="b" * 40), "commit"),
        (lambda i, s: i == 1 and s["provenance"]["packages"].update(numpy="2.6.0"), "packages"),
        (lambda i, s: i == 1 and s["sampling"].update(n_seeds_per_variant=6), "sampling"),
        (lambda i, s: i == 1 and s.update(families_evaluated=FAMILIES[:1]), "families"),
        (lambda i, s: i == 1 and next(iter(s["results"].values())).update(error="boom"),
         "raised"),
        (lambda i, s: i == 1 and s.pop("provenance"), "provenance"),
    ],
)
def test_a_nonuniform_board_set_is_refused(tmp_path, board_script, mutate, message):
    _write_board_shards(tmp_path, board_script, mutate)
    with pytest.raises(board_script.ShardError, match=message):
        board_script.merge_shards(tmp_path, detectors=DETECTORS, families=FAMILIES)


def test_a_missing_or_duplicated_detector_is_refused(tmp_path, board_script):
    _write_board_shards(tmp_path, board_script)
    (tmp_path / "board-00.json").unlink()
    with pytest.raises(board_script.ShardError, match="missing"):
        board_script.merge_shards(tmp_path, detectors=DETECTORS, families=FAMILIES)
    (tmp_path / "board-00.json").write_text((tmp_path / "board-01.json").read_text())
    with pytest.raises(board_script.ShardError, match="already supplied"):
        board_script.merge_shards(tmp_path, detectors=DETECTORS, families=FAMILIES)


def test_the_cli_expects_every_detector_and_family(board_script, ledger_script):
    """The subset arguments are for tests; the defaults are the full artifact."""
    assert len(board_script.DETECTORS) == 30
    assert len(ledger_script.ledger_grid()) == 123
