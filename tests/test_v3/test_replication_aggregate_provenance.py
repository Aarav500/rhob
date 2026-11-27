"""What ``scripts/aggregate_replication.py`` says about its inputs, and how it words a rung.

Three things, each of which the artifact used to get wrong or leave out:

* **Where the draws came from.** The 20 committed replicates record no commit and were
  re-scored row by row afterwards; nothing in ``leaderboard/v5_replicated.json`` said so.
  The ``replicates`` block tallies the inputs' provenance, and
  ``--require-single-source`` refuses a set that is not one clean commit's, under one
  environment, unpatched.
* **What a draw resampled.** A replicate written before the layout-seed fix was built at
  layout 0, so "layout seed and rollout seeds resampled together" was false of every
  committed draw. The summary's first sentence is now written from ``draw_schema``.
* **A verdict needs both statistics** on the unsupervised partition too
  (``supervision.statistic_disagreement_unsupervised_only``): the summary said the
  unsupervised L0->L1 step "separates the other way" on the max alone while the mean did
  not separate it. It may say so only when the mean agrees.

Finally, the committed artifact and summary must be exactly what the aggregator now
writes from the committed replicates.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_AGG = _REPO / "scripts" / "aggregate_replication.py"
_spec = importlib.util.spec_from_file_location("_aggregate_replication_under_test", _AGG)
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)

_FAMILIES = ["fam_x", "fam_y"]  # not in the ledger's degenerate set
_LEVEL = {
    "Reward Threshold": "L0",
    "Reward Peak": "L0",
    "State Divergence": "L1",   # label-fitted
    "Bimodal Occupancy": "L1",  # unsupervised
    "Behavioral Threshold": "L2",
    "True Reward Oracle": "L3",
}


def _provenance(rid: int) -> dict:
    return {
        "generated_utc": f"2026-11-27T02:{rid:02d}:00Z",
        "git": {"commit": "f" * 40, "short_commit": "f" * 12, "branch": "b",
                "dirty": False, "dirty_files": [], "source": "git"},
        "python": "3.12.3",
        "python_implementation": "CPython",
        "platform": "Linux-6.8.0-x86_64",
        "packages": {"numpy": "2.3.5", "rhob": "1.8.0"},
        "numerics": {
            "env": {"OPENBLAS_CORETYPE": "Haswell"},
            "cpu": {"vendor": "AuthenticAMD", "model": "AMD EPYC 7763"},
            "torch_imported": True,
            "torch_cpu_capability": "AVX2",
            "blas": [{"internal_api": "openblas", "architecture": "Haswell"}],
        },
        "argv": [],
    }


def _replicate(rid: int, scores: dict[str, float], draw_schema: int | None = 2) -> dict:
    rep = {
        "replicate_id": rid,
        "layout_seed": rid,
        "seed_base": rid * 1_000_000,
        "n_seeds": 5,
        "families_evaluated": _FAMILIES,
        "provenance": _provenance(rid),
        "results": {
            name: {
                "access_level": _LEVEL[name],
                "overall_auroc": v,
                "per_family": {f: v for f in _FAMILIES},
                "cells": 2, "cells_measured": 2, "cells_not_applicable": 0,
                "not_applicable_families": [], "not_applicable_reasons": {},
            }
            for name, v in scores.items()
        },
    }
    if draw_schema is not None:
        rep["draw_schema"] = draw_schema
    return rep


# Reward Threshold, Reward Peak, Bimodal Occupancy per draw. The L0 max is always above
# the unsupervised L1, so the max separates the step below zero on every draw; the L0
# mean (0.49 to 0.51) straddles Bimodal Occupancy, so the mean does not separate it.
_STRADDLE = [(0.56, 0.44, 0.52), (0.57, 0.45, 0.49), (0.55, 0.43, 0.51),
             (0.58, 0.44, 0.50), (0.56, 0.46, 0.50)]


def _set(bimodal_shift: float = 0.0) -> list[dict]:
    reps = []
    for rid, (rt, rp, bo) in enumerate(_STRADDLE):
        reps.append(_replicate(rid, {
            "Reward Threshold": rt, "Reward Peak": rp,
            "State Divergence": 0.95 + 0.005 * rid, "Bimodal Occupancy": bo + bimodal_shift,
            "Behavioral Threshold": 0.90 + 0.01 * rid, "True Reward Oracle": 0.97 + 0.002 * rid,
        }))
    return reps


def _write(tmp_path: Path, reps: list[dict]) -> Path:
    in_dir = tmp_path / "reps"
    in_dir.mkdir(exist_ok=True)
    for r in reps:
        (in_dir / f"replicate_{r['replicate_id']:03d}.json").write_text(json.dumps(r))
    return in_dir


def _aggregate(tmp_path: Path, reps: list[dict], *extra: str) -> tuple[dict, str]:
    in_dir = _write(tmp_path, reps)
    agg.main(["--in-dir", str(in_dir), "--out", str(tmp_path / "agg.json"),
              "--out-md", str(tmp_path / "agg.md"), *extra])
    return (json.loads((tmp_path / "agg.json").read_text()),
            (tmp_path / "agg.md").read_text())


# ------------------------------------------------------- M1: both statistics, unsupervised
def test_a_max_only_reversal_is_reported_as_statistic_dependent(tmp_path):
    out, md = _aggregate(tmp_path, _set())
    unsup = out["supervision"]["adjacent_level_separation_unsupervised_only"]
    assert unsup["L1_minus_L0_max"]["excludes_zero"] is True
    assert unsup["L1_minus_L0_max"]["mean"] < 0
    assert unsup["L1_minus_L0_mean"]["excludes_zero"] is False

    sd = out["supervision"]["statistic_disagreement_unsupervised_only"]
    assert sd["rungs_where_max_and_mean_disagree_on_separation"] == ["L1_minus_L0"]
    assert sd["rungs_where_max_and_mean_disagree_on_direction"] == []
    assert sd["note"] == out["statistic_disagreement"]["note"]

    assert "separates the other way" not in md
    assert "**Statistic-dependent without the label-fitted detectors: L1 minus L0.**" in md
    assert ("Without the label-fitted detectors the step reverses sign under the max, and "
            "whether it separates depends on the statistic: the max puts L0 above L1 and "
            "the mean does not separate L1 from L0, so quote both.") in md


def test_separates_the_other_way_only_when_the_mean_agrees(tmp_path):
    out, md = _aggregate(tmp_path, _set(bimodal_shift=-0.10))
    unsup = out["supervision"]["adjacent_level_separation_unsupervised_only"]
    assert unsup["L1_minus_L0_max"]["excludes_zero"] and unsup["L1_minus_L0_mean"]["excludes_zero"]
    assert unsup["L1_minus_L0_mean"]["mean"] < 0
    sd = out["supervision"]["statistic_disagreement_unsupervised_only"]
    assert "L1_minus_L0" not in sd["rungs_where_max_and_mean_disagree_on_separation"]
    assert "L1_minus_L0" not in sd["rungs_where_max_and_mean_disagree_on_direction"]
    assert ("Without the label-fitted detectors the step reverses sign and separates the "
            "other way: unsupervised L1 sits below L0.") in md


def test_opposite_directions_are_a_direction_disagreement():
    mx = {"excludes_zero": True, "mean": -0.03}
    assert agg.disagreement_kind(mx, {"excludes_zero": True, "mean": +0.02}) == "direction"
    assert agg.disagreement_kind(mx, {"excludes_zero": False, "mean": -0.01}) == "separation"
    assert agg.disagreement_kind(mx, {"excludes_zero": True, "mean": -0.01}) == ""
    assert agg.disagreement_kind(
        {"excludes_zero": False, "mean": 0.01}, {"excludes_zero": False, "mean": -0.01}
    ) == ""


# ------------------------------------------------------------ provenance of the draws
def test_the_artifact_carries_its_own_provenance_and_the_inputs(tmp_path):
    out, md = _aggregate(tmp_path, _set())
    assert out["provenance"]["script"] == "scripts/aggregate_replication.py"
    assert "generated_utc" in out["provenance"] and "git" in out["provenance"]
    assert [d["draw_schema"] for d in out["draws"]] == [2] * 5

    reps = out["replicates"]
    assert reps["n_replicate_files"] == 5
    for key in ("commits", "dirty", "pythons", "package_sets", "platforms", "numerics",
                "cpus", "draw_schemas", "rescored_rows"):
        assert len(reps[key]) == 1 and reps[key][0]["n"] == 5, key
    assert reps["commits"][0]["value"] == "f" * 40
    assert reps["pythons"][0]["value"] == "CPython 3.12.3"
    assert reps["numerics"][0]["value"]["env"] == {"OPENBLAS_CORETYPE": "Haswell"}
    assert reps["rescored_rows"][0]["value"] == []
    assert reps["single_source"] is True and reps["single_source_problems"] == []
    assert reps["layouts_resampled"] is True

    assert md.startswith("# Leaderboard replication\n\n5 independent draws of the full "
                         "benchmark (layout seed and rollout seeds resampled together).")
    assert "## Provenance of the draws" in md
    assert "- Source commit: `ffffffffffff`, on all 5." in md
    assert "- Single source: **yes**" in md


def test_platform_and_cpu_are_reported_not_required(tmp_path):
    reps = _set()
    reps[3]["provenance"]["platform"] = "Linux-6.11.0-x86_64"
    reps[3]["provenance"]["numerics"]["cpu"] = {"vendor": "GenuineIntel", "model": "Xeon"}
    out, md = _aggregate(tmp_path, reps, "--require-single-source")
    assert out["replicates"]["single_source"] is True
    assert [g["replicate_ids"] for g in out["replicates"]["platforms"]] == [[0, 1, 2, 4], [3]]
    assert "GenuineIntel Xeon on replicate 3" in md


def _other_commit(reps):
    reps[2]["provenance"]["git"]["commit"] = "e" * 40


def _no_commit(reps):
    for r in reps:
        r["provenance"]["git"]["commit"] = None


def _dirty(reps):
    reps[1]["provenance"]["git"]["dirty"] = True


def _dirty_unknown(reps):
    reps[1]["provenance"]["git"]["dirty"] = None


def _rescored(reps):
    reps[4]["provenance"]["rescored"] = [{"detector": "Reward Peak", "commit": "6e8521a"}]


def _python(reps):
    reps[0]["provenance"]["python"] = "3.11.15"


def _packages(reps):
    reps[0]["provenance"]["packages"]["numpy"] = "2.4.6"


def _numerics(reps):
    reps[0]["provenance"]["numerics"]["env"]["OPENBLAS_CORETYPE"] = None


def _no_numerics(reps):
    for r in reps:
        r["provenance"].pop("numerics")


def _mixed_schema(reps):
    reps[0].pop("draw_schema")


@pytest.mark.parametrize(
    ("mutate", "says"),
    [
        (_other_commit, "2 source commits"),
        (_no_commit, "no replicate records the commit"),
        (_dirty, "not recorded as drawn from a clean tree"),
        (_dirty_unknown, "not recorded as drawn from a clean tree"),
        (_rescored, "re-scored after the draw"),
        (_python, "different Pythons"),
        (_packages, "different package sets"),
        (_numerics, "different numerics signatures"),
        (_no_numerics, "no replicate records its numerics signature"),
        (_mixed_schema, "different draw schemas"),
    ],
    ids=lambda v: v.__name__.strip("_") if callable(v) else "",
)
def test_require_single_source_refuses_and_writes_nothing(tmp_path, mutate, says):
    reps = _set()
    mutate(reps)
    with pytest.raises(SystemExit, match=says):
        _aggregate(tmp_path, reps, "--require-single-source")
    assert not (tmp_path / "agg.json").exists()
    assert not (tmp_path / "agg.md").exists()
    # Without the flag the same set is aggregated, and says why it is not single-source.
    out, md = _aggregate(tmp_path, copy.deepcopy(reps))
    assert out["replicates"]["single_source"] is False
    assert any(says in p for p in out["replicates"]["single_source_problems"])
    assert "- Single source: **no**:" in md


@pytest.mark.parametrize(
    ("schemas", "layouts", "opening"),
    [
        ([2, 2, 2, 2, 2], True, "(layout seed and rollout seeds resampled together)"),
        ([None] * 5, False, "that resample the rollout seeds only."),
        ([2, 2, None, None, None], False, "of two kinds: 2 resample the layout seed"),
    ],
)
def test_the_first_sentence_says_what_the_draws_resampled(tmp_path, schemas, layouts, opening):
    reps = _set()
    for r, schema in zip(reps, schemas):
        if schema is None:
            r.pop("draw_schema")
    out, md = _aggregate(tmp_path, reps)
    assert out["replicates"]["layouts_resampled"] is layouts
    assert opening in md.splitlines()[2]


def test_replicate_ids_are_written_as_ranges():
    assert agg._ids([0, 1, 2, 5, 7, 8]) == "replicates 0-2, 5, 7-8"
    assert agg._ids([3]) == "replicate 3"


# -------------------------------------------------- the committed artifact is current
_COMMITTED_REPLICATES = _REPO / "results" / "replication"
_COMMITTED_JSON = _REPO / "leaderboard" / "v5_replicated.json"
_COMMITTED_MD = _REPO / "docs" / "figures" / "replication_summary.md"


@pytest.mark.skipif(
    not any(_COMMITTED_REPLICATES.glob("replicate_*.json")),
    reason="committed replicates not present",
)
def test_the_committed_aggregate_is_what_the_aggregator_writes(tmp_path):
    """Re-aggregate the committed replicates; only the aggregation's own provenance may differ.

    So a change to the aggregator and its committed output land together, and a reader
    can trust that the summary describes the replicate files beside it.
    """
    agg.main(["--in-dir", str(_COMMITTED_REPLICATES), "--out", str(tmp_path / "agg.json"),
              "--out-md", str(tmp_path / "agg.md")])
    fresh = json.loads((tmp_path / "agg.json").read_text())
    committed = json.loads(_COMMITTED_JSON.read_text())
    fresh.pop("provenance")
    committed.pop("provenance", None)
    assert fresh == committed, (
        "leaderboard/v5_replicated.json is not what scripts/aggregate_replication.py writes "
        "from results/replication; re-run it"
    )
    assert (tmp_path / "agg.md").read_text() == _COMMITTED_MD.read_text(), (
        "docs/figures/replication_summary.md is stale; re-run scripts/aggregate_replication.py"
    )
