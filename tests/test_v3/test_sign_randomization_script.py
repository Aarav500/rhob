"""``scripts/measure_sign_randomization.py`` end to end on stub families, with no real rollouts.

No test ran this script until 2026-11-27, and it did not run: ``board`` crashed on its
first cell reading ``CellResult.onset_mae``, a field removed on 2026-09-02. These tests
build the artifacts it merges by running it -- ``board``, ``transfer``, ``report`` and
``ci`` -- on the synthetic families of ``transfer_stubs``, and pin the refusals and the
computed claims that replaced hard-coded ones: the merges refuse mixed commits, dirty
trees and a family measured twice; the invariance check compares only cells scored in
both modes; the "reproduces exactly" sentence appears only when it is true; and the
``after`` RTS is the same measurement as ``scripts/cross_family_transfer.py``'s.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from transfer_stubs import TEST_STUBS, TRAIN_STUBS, one_torch_thread, register_stubs

from rhob.v3.benchmark import BenchmarkResults, CellResult
from rhob.v3.leaderboard.transfer import load_transfer_results

_REPO = Path(__file__).resolve().parents[2]
_HISTORICAL = _REPO / "leaderboard" / "cross_family_transfer.json"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", autouse=True)
def _one_torch_thread():
    with one_torch_thread():
        yield


@pytest.fixture(scope="module")
def msr():
    return _load("_msr_under_test", "scripts/measure_sign_randomization.py")


@pytest.fixture
def stubs(monkeypatch):
    register_stubs(monkeypatch, {**TRAIN_STUBS, **TEST_STUBS})
    return {"train": list(TRAIN_STUBS), "test": list(TEST_STUBS)}


def _no_nan(path: Path) -> dict:
    def refuse(token):
        raise AssertionError(f"{path} contains {token}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)


def _provenance(commit: str = "a" * 40, dirty=False, python="3.12.3") -> dict:
    return {
        "generated_utc": "2026-11-27T00:00:00Z",
        "git": {"commit": commit, "short_commit": commit[:12], "dirty": dirty},
        "python": python,
        "packages": {"numpy": "2.5.4"},
        "numerics": {"env": {"OPENBLAS_CORETYPE": "Haswell"}},
    }


def _board(families, n_seeds=3, provenance=None, cells=None) -> dict:
    cells = cells or [
        {"family": f, "difficulty": 0.9, "auroc": 0.7, "na_reason": None} for f in families
    ]
    block = {"overall_auroc": 0.7, "n_cells_scored": len(cells), "n_cells_na": 0,
             "per_family": {}, "cells": cells}
    return {
        "measurement": "board", "n_seeds": n_seeds, "families": list(families),
        "provenance": provenance or _provenance(),
        "detectors": {"Feature Magnitude": {
            "access_level": "L2", "direction_dependent": False,
            "by_config": {"before": copy.deepcopy(block), "after": copy.deepcopy(block)},
        }},
    }


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ------------------------------------------------------------------ the crash it had
def test_cell_records_reads_only_fields_a_cell_has(msr):
    """``onset_mae`` is gone from CellResult; reading it crashed the whole ``board`` run."""
    results = BenchmarkResults(detector_name="d", access_level="L2", cells=[
        CellResult("f", "camping", 0.9, 0.75, 3),
        CellResult("f", "camping", 0.8, float("nan"), 3, na_reason="horizon 40 ..."),
        CellResult("g", "camping", 0.9, float("nan"), 3),  # applicable, undefined
    ])
    records = msr.cell_records(results)
    assert records == [
        {"family": "f", "difficulty": 0.9, "auroc": 0.75, "na_reason": None},
        {"family": "f", "difficulty": 0.8, "auroc": None, "na_reason": "horizon 40 ..."},
        {"family": "g", "difficulty": 0.9, "auroc": None, "na_reason": None},
    ]


def test_derived_randomized_rollouts_are_the_simulated_ones_bit_for_bit(msr, stubs):
    """Half the rollout cost is skipped only because the result is identical."""
    from rhob.v3.benchmark import Benchmark, RolloutCacheKey
    from rhob.v3.registry import FamilyRegistry

    for name in stubs["test"]:
        pair = FamilyRegistry.get(name).generate_pair_at(0.9)
        key = RolloutCacheKey(name, pair.difficulty, 3, False, pair.layout_seed, 0)
        Benchmark._rollout_cache[key] = pair.rollout(3, randomize_sign=False)
    assert msr.derive_randomized_rollouts() == len(stubs["test"])
    assert msr.derive_randomized_rollouts() == 0  # idempotent
    for name in stubs["test"]:
        pair = FamilyRegistry.get(name).generate_pair_at(0.9)
        key = RolloutCacheKey(name, pair.difficulty, 3, True, pair.layout_seed, 0)
        derived_a, derived_b, derived_onsets = Benchmark._rollout_cache[key]
        direct_a, direct_b, direct_onsets = pair.rollout(3, randomize_sign=True)
        assert derived_onsets == direct_onsets
        for got, want in zip(derived_a + derived_b, direct_a + direct_b):
            for field in ("proxy_rewards", "true_rewards", "state_counts", "behav_trace"):
                g, w = getattr(got, field), getattr(want, field)
                assert (g is None and w is None) or np.array_equal(g, w), (name, field)


def test_board_runs_end_to_end_and_records_na_cells(msr, stubs, tmp_path):
    out = tmp_path / "board.json"
    msr.main([
        "board", "--families", "stub_train_long", "stub_train_short", "--n-seeds", "3",
        "--only", "Behavioral Threshold", "Feature Magnitude", "Reward CUSUM",
        "--out", str(out),
    ])
    payload = _no_nan(out)
    fm = payload["detectors"]["Feature Magnitude"]["by_config"]
    assert set(fm) == {"before", "after"}  # magnitude-only: once per randomization mode
    assert fm["after"]["n_cells_scored"] == 2 and fm["after"]["n_cells_na"] == 2
    assert fm["after"]["n_families_scored"] == 1
    na = [c for c in fm["after"]["cells"] if c["auroc"] is None]
    assert {c["family"] for c in na} == {"stub_train_short"}
    assert all("horizon 40" in c["na_reason"] for c in na)
    bt = payload["detectors"]["Behavioral Threshold"]["by_config"]
    assert set(bt) == set(msr.CONFIGS)


# ------------------------------------------------------------------- merge refusals
def test_boards_at_different_n_seeds_are_refused(msr, tmp_path):
    a = _write(tmp_path, "a.json", _board(["f1"], n_seeds=3))
    b = _write(tmp_path, "b.json", _board(["f2"], n_seeds=5))
    with pytest.raises(ValueError, match="n_seeds"):
        msr._merge_boards([a, b])


def test_boards_from_different_commits_are_refused(msr, tmp_path):
    a = _write(tmp_path, "a.json", _board(["f1"]))
    b = _write(tmp_path, "b.json", _board(["f2"], provenance=_provenance("b" * 40)))
    with pytest.raises(ValueError, match="not from the same run"):
        msr._merge_boards([a, b])
    with pytest.raises(ValueError, match="not from the same run"):
        msr._merge_boards([a, b], allow_dirty=True)  # never admits a mixture


def test_boards_from_different_stacks_are_refused(msr, tmp_path):
    a = _write(tmp_path, "a.json", _board(["f1"]))
    b = _write(tmp_path, "b.json", _board(["f2"], provenance=_provenance(python="3.13.14")))
    with pytest.raises(ValueError, match="python"):
        msr._merge_boards([a, b])


@pytest.mark.parametrize("dirty", [True, None])
def test_a_board_from_a_tree_not_known_clean_is_refused_unless_allowed(msr, tmp_path, dirty):
    a = _write(tmp_path, "a.json", _board(["f1"], provenance=_provenance(dirty=dirty)))
    with pytest.raises(ValueError, match="known to be clean"):
        msr._merge_boards([a])
    _, design = msr._merge_boards([a], allow_dirty=True)
    assert design["board_provenance"]["dirty"] is dirty


def test_a_family_in_two_boards_is_refused(msr, tmp_path):
    a = _write(tmp_path, "a.json", _board(["f1", "f2"]))
    b = _write(tmp_path, "b.json", _board(["f2"]))
    with pytest.raises(ValueError, match="counted twice"):
        msr._merge_boards([a, b])


def test_a_clean_uniform_merge_records_its_provenance(msr, tmp_path):
    a = _write(tmp_path, "a.json", _board(["f1"]))
    b = _write(tmp_path, "b.json", _board(["f2"]))
    detectors, design = msr._merge_boards([a, b])
    assert design["board_provenance"]["commit"] == "a" * 40
    assert design["board_provenance"]["dirty"] is False
    assert detectors["Feature Magnitude"]["by_config"]["after"]["n_cells_scored"] == 2


def test_a_configuration_measured_twice_is_refused(msr, tmp_path):
    payload = {
        "train_families": ["t"], "test_families": ["h"], "n_seeds_train": 2,
        "n_seeds_test": 2, "n_trials": 1, "provenance": _provenance(),
        "configs": {"after": {}}, "results": {"after": {}},
    }
    a = _write(tmp_path, "a.json", payload)
    b = _write(tmp_path, "b.json", payload)
    with pytest.raises(ValueError, match="second time"):
        msr._merge_transfers([a, b])


def test_a_randomized_published_transfer_file_is_refused(msr, tmp_path):
    v5 = _write(tmp_path, "v5.json", {
        "configs": {"shipped": {"randomize_behav_sign": True,
                                "detector_infers_orientation": True}},
        "results": {"Det": {"access_level": "L2", "avg_transfer_auroc_mean": 0.5}},
    })
    with pytest.raises(SystemExit, match="pre-randomization"):
        msr._check_published_transfer(v5)
    if _HISTORICAL.exists():
        msr._check_published_transfer(_HISTORICAL)  # the 2026-07-09 file is accepted


# ---------------------------------------------------------------- invariance, deltas
def _detectors(before_cells, after_cells, direction_dependent=False) -> dict:
    def block(cells):
        return {"cells": cells, "overall_auroc": None, "n_cells_scored": 0, "n_cells_na": 0}

    return {"Det": {"access_level": "L0", "direction_dependent": direction_dependent,
                    "by_config": {"before": block(before_cells), "after": block(after_cells)}}}


def _cell(fam, auroc, diff=0.9):
    return {"family": fam, "difficulty": diff, "auroc": auroc,
            "na_reason": None if auroc is not None else "n/a"}


def test_invariance_compares_only_cells_scored_in_both_modes(msr):
    report = msr._invariance(_detectors(
        [_cell("a", 0.6), _cell("b", None), _cell("c", 0.7)],
        [_cell("a", 0.6), _cell("b", None), _cell("c", 0.7)],
    ))["Det"]
    # Two scored, both identical; the N/A cell is not counted as an identical one.
    assert (report["n_cells"], report["n_scored"], report["n_identical"], report["n_na"]) == (3, 2, 2, 1)
    assert report["n_applicability_mismatches"] == 0


def test_invariance_flags_a_cell_scored_in_one_mode_only(msr):
    report = msr._invariance(_detectors(
        [_cell("a", 0.6), _cell("b", 0.55)],
        [_cell("a", 0.61), _cell("b", None)],
    ))["Det"]
    assert report["n_scored"] == 1 and report["n_identical"] == 0
    assert report["mismatched_cells"] == [["a", 0.9, 0.6, 0.61]]
    assert report["n_applicability_mismatches"] == 1


def test_config_deltas_count_only_cells_scored_on_both_sides(msr):
    dets = _detectors([_cell("a", 0.6), _cell("b", None)], [], direction_dependent=True)
    dets["Det"]["by_config"]["blind"] = {
        "cells": [_cell("a", 0.4), _cell("b", None)], "overall_auroc": 0.4,
    }
    row = msr._config_deltas(dets)["Det"]["before_vs_blind"]
    assert row["n_cells"] == 1  # the N/A cell is not a cell that agreed
    assert row["n_differing"] == 1


# ---------------------------------------------------------------- reproduction check
def test_the_reproduction_ladder_is_like_for_like(msr, tmp_path):
    """Same families on both sides, weighted by this run's scored cells per family."""
    cells = [_cell("f1", 0.9, 0.9), _cell("f1", 0.7, 0.8), _cell("f2", 0.6), _cell("f3", 0.5)]
    dets = {"Behavioral Threshold": {"access_level": "L2", "direction_dependent": True,
                                     "by_config": {"before": {
                                         "cells": cells,
                                         "per_family": {"f1": 0.8, "f2": 0.6, "f3": 0.5}}}}}
    board = _write(tmp_path, "board.json", {"results": {"Behavioral Threshold": {
        # f3 is null on the board: left out on both sides.
        "per_family": {"f1": 1.0, "f2": 0.4, "f3": None, "f4": 0.9},
    }}})
    check = msr._reproduction_check(dets, board, tmp_path / "missing.json")
    row = check["l2_ladder_like_for_like"]["per_detector"]["Behavioral Threshold"]
    assert row["n_families"] == 2 and row["n_cells"] == 3
    assert row["published"] == pytest.approx((2 * 1.0 + 1 * 0.4) / 3, abs=1e-4)
    assert row["this_run_before"] == pytest.approx(np.mean([0.9, 0.7, 0.6]), abs=1e-4)


# --------------------------------------------------------------------- the script, ci
_CI_ROWS = ("State Divergence", "Trajectory MLP", "Ensemble (Top 5)")


@pytest.fixture(scope="module")
def ci_run(msr, tmp_path_factory):
    """One reduced ``ci`` run on the stubs, shared by the tests that read its output.

    Reward MLP is left out of the transfer stage (``--transfer-rows``): it is the slowest
    row to fit and ``test_cross_family_transfer_script`` covers it.
    """
    if not _HISTORICAL.exists():
        pytest.skip("historical transfer artifact absent")
    tmp_path = tmp_path_factory.mktemp("ci")
    with pytest.MonkeyPatch.context() as mp:
        names = register_stubs(mp, {**TRAIN_STUBS, **TEST_STUBS})
        train, test = list(TRAIN_STUBS), list(TEST_STUBS)
        # Point the two companions at the stubs, so ci exercises them too.
        mp.setattr(msr, "COMPANIONS", {
            "n5_v5board_reproduction": (2, ("stub_train_long",)),
            "n20_doc_13family_subset": (4, ("stub_test_bins",)),
        })
        pre_audit = _write(tmp_path, "pre_audit.json", {"results": {
            name: {"per_family": {f: 0.8 for f in names}}
            for name in ("Behavioral Threshold", "Centroid Tracker", "Feature Magnitude")
        }})
        out_json, out_md = tmp_path / "impact.json", tmp_path / "impact.md"
        msr.main([
            "ci", "--out-json", str(out_json), "--out-md", str(out_md),
            "--work-dir", str(tmp_path / "work"),
            "--published-board", str(pre_audit),
            "--published-transfer", str(_HISTORICAL),
            "--families", *names,
            "--only", "Behavioral Threshold", "Centroid Tracker", "Feature Magnitude",
            "Reward CUSUM",
            "--n-seeds", "3", "--n-trials", "1", "--n-seeds-train", "3",
            "--n-seeds-test", "3", "--transfer-rows", *_CI_ROWS,
            "--train-families", *train, "--test-families", *test,
            "--allow-dirty",
        ])
    return {"json": out_json, "md": out_md}


def test_ci_runs_every_stage_and_writes_an_auditable_artifact(msr, ci_run):
    payload = _no_nan(ci_run["json"])
    md = ci_run["md"].read_text(encoding="utf-8")

    design = payload["design"]
    assert design["published_design"] is False
    assert design["ci_reduced_options"]  # names what was narrowed
    assert set(design["ci_stage_wall_clock_s"]) >= {"board", "report", "total"}
    assert design["input_provenance"]["commit"]

    # Every per-cell AUROC, as compact rows.
    assert payload["cells_columns"] == ["family", "difficulty", "auroc", "na_reason"]
    fm_after = payload["cells"]["Feature Magnitude"]["after"]
    assert len(fm_after) == 8  # 4 stub families x 2 tiers
    short = [r for r in fm_after if r[0] == "stub_train_short"]
    assert short and all(r[2] is None and "horizon 40" in r[3] for r in short)
    # The per-cell rows really are one per line in the file.
    assert '["stub_train_short", 0.9, null, ' in ci_run["json"].read_text(encoding="utf-8")

    fm = payload["per_detector"]["Feature Magnitude"]
    assert fm["n_cells_scored"]["after"] == 6 and fm["n_cells_na"]["after"] == 2

    inv = payload["l0_l1_and_magnitude_invariance"]["Feature Magnitude"]
    assert inv["n_scored"] == 6 and inv["n_na"] == 2 and inv["n_identical"] == 6

    assert set(payload["sample_size_sensitivity"]) == {
        "n5_v5board_reproduction", "n20_doc_13family_subset",
    }

    transfer = payload["transfer"]
    assert set(transfer["results"]) == {"before", "blind", "after"}
    assert set(transfer["results"]["before"]) == set(_CI_ROWS)
    assert set(transfer["results"]["after"]) == {"Trajectory MLP", "Ensemble (Top 5)"}
    sd = transfer["results"]["before"]["State Divergence"]
    assert sd["avg_transfer_auroc"] is None  # 6-bin centroids, 3-bin held-out histograms

    rts = payload["reproduction_check"]["rts_before_vs_published"]["Ensemble (Top 5)"]
    assert rts["published"] == pytest.approx(0.994)

    # The loader reads the aggregate's transfer block as before, with the new rules.
    board = load_transfer_results(ci_run["json"])
    assert board.shipped_config() == "after" and board.baseline_config() == "before"
    assert board.applies_na_rules is True

    # Computed claims, not hard-coded ones.
    assert "reproduces exactly" not in md
    assert "Reduced design" in md
    assert "| State Divergence | L1 | " in md and "**N/A**" in md
    assert "no held-out family scored" in md
    assert "identical / scored in both" in md


def test_the_rts_sentence_follows_the_data(msr, ci_run):
    payload = _no_nan(ci_run["json"])
    check = payload["reproduction_check"]["rts_before_vs_published"]["Ensemble (Top 5)"]

    payload["reproduction_check"]["rts_before_vs_published"]["Ensemble (Top 5)"] = {
        **check, "this_run_before": 0.994, "difference": 0.0,
    }
    assert "reproduces the published 0.994" in msr.render_markdown(payload)

    payload["reproduction_check"]["rts_before_vs_published"]["Ensemble (Top 5)"] = {
        **check, "this_run_before": 0.981, "difference": -0.013,
    }
    headline = msr.render_markdown(payload).split("## Design")[0]
    assert "reproduces the published" not in headline
    assert "does not reproduce exactly" in headline

    # A null RTS everywhere renders rather than crashing.
    for cfg in payload["transfer"]["results"].values():
        for row in cfg.values():
            row["avg_transfer_auroc_mean"] = row["avg_transfer_auroc"] = None
    payload["reproduction_check"]["rts_before_vs_published"] = {}
    assert "N/A" in msr.render_markdown(payload)


def test_the_after_rts_is_the_transfer_scripts_own_measurement(msr, stubs, tmp_path):
    """The CI gate's premise: signrand ``after`` trial k == cross_family_transfer trial k.

    The second run starts from an empty rollout cache, so it re-simulates rather than
    reading the first run's (derived) entries back.
    """
    from rhob.v3.benchmark import Benchmark

    out = tmp_path / "after.json"
    msr.main([
        "transfer", "--configs", "after", "--rows", "Ensemble (Top 5)",
        "--n-trials", "2", "--n-seeds-train", "3", "--n-seeds-test", "3",
        "--train-families", *stubs["train"], "--test-families", *stubs["test"],
        "--out", str(out),
    ])
    mine = _no_nan(out)["results"]["after"]
    Benchmark._rollout_cache.clear()
    cft = msr._load_transfer_module()
    theirs = cft.run_rows(
        ["Ensemble (Top 5)"], n_seeds_train=3, n_seeds_test=3, n_trials=2,
        train_families=stubs["train"], test_families=stubs["test"],
    )
    for row in ("Ensemble (Top 5)",):
        assert mine[row]["all_trial_transfer_aurocs"] == theirs[row]["all_trial_transfer_aurocs"]
        assert mine[row]["all_trial_train_aurocs"] == theirs[row]["all_trial_train_aurocs"]


def test_the_report_requires_the_pre_audit_board(msr, tmp_path):
    board = _write(tmp_path, "b.json", _board(["f1"]))
    with pytest.raises(SystemExit, match="published-board is required"):
        msr.main(["report", "--board", str(board),
                  "--out-json", str(tmp_path / "x.json"), "--out-md", str(tmp_path / "x.md")])
