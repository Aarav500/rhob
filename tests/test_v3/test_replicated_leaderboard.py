"""Tests for the replicated-leaderboard loader.

The properties pinned here are the ones a renderer can silently violate: an unmeasured
family must stay unmeasured rather than becoming 0.5, the ``detectors`` key must not be
confused with the integer count of the same name in ``v5_leaderboard.json``, and both
supervision partitions must survive the parse.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rhob.v3.leaderboard.adapters import load_any_leaderboard_json
from rhob.v3.leaderboard.replicated import (
    NOT_MEASURED,
    Interval,
    is_replicated_schema,
    load_replicated_leaderboard,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPLICATED = _REPO_ROOT / "leaderboard" / "v5_replicated.json"
_V5 = _REPO_ROOT / "leaderboard" / "v5_leaderboard.json"


def _minimal(**overrides) -> dict:
    payload = {
        "n_replicates": 2,
        "draws": [
            {"replicate_id": 0, "layout_seed": 0, "seed_base": 0, "n_seeds": 5},
            {"replicate_id": 1, "layout_seed": 1, "seed_base": 1000000, "n_seeds": 5},
        ],
        "method": {"ci_level": 0.95, "bootstrap_resamples": 10, "bootstrap_seed": 7},
        "supervision": {
            "label_fitted_detectors": ["Fitted One"],
            "why_it_matters": "because",
            "access_levels_unsupervised_only": {
                "L1": {
                    "max_auroc": {"mean": 0.5, "sd": 0.0, "ci_lo": 0.5, "ci_hi": 0.5, "n_replicates": 2},
                    "mean_auroc": {"mean": 0.5, "sd": 0.0, "ci_lo": 0.5, "ci_hi": 0.5, "n_replicates": 2},
                    "best_detector_frequency": {"Plain One": 2},
                }
            },
            "adjacent_level_separation_unsupervised_only": {
                "L1_minus_L0_max": {
                    "mean": -0.01, "sd": 0.02, "ci_lo": -0.03, "ci_hi": 0.01,
                    "n_replicates": 2, "statistic": "max", "excludes_zero": False,
                    "replicates_with_hi_above_lo": 0, "n_paired": 2,
                    "denominators_equal": False,
                }
            },
        },
        "access_levels": {
            "L1": {
                "max_auroc": {"mean": 0.9, "sd": 0.01, "ci_lo": 0.88, "ci_hi": 0.92, "n_replicates": 2},
                "mean_auroc": {"mean": 0.7, "sd": 0.01, "ci_lo": 0.68, "ci_hi": 0.72, "n_replicates": 2},
                "best_detector_frequency": {"Fitted One": 2},
            }
        },
        "adjacent_level_separation": {
            "L1_minus_L0_max": {
                "mean": 0.39, "sd": 0.02, "ci_lo": 0.38, "ci_hi": 0.40,
                "n_replicates": 2, "statistic": "max", "excludes_zero": True,
                "replicates_with_hi_above_lo": 2, "n_paired": 2, "denominators_equal": False,
            }
        },
        "statistic_disagreement": {
            "rungs_where_max_and_mean_disagree_on_separation": ["L2_minus_L1"],
            "note": "quote both",
        },
        "degeneracy": {"detectors_with_zero_variance": ["Saturated One"], "note": "saturation"},
        "detectors": {
            "Fitted One": {
                "mean": 0.9, "sd": 0.01, "ci_lo": 0.88, "ci_hi": 0.92, "n_replicates": 2,
                "access_level": "L1", "cells_measured": 35,
                "per_family": {
                    "alpha": {"mean": 0.9, "sd": 0.01, "ci_lo": 0.88, "ci_hi": 0.92, "n_replicates": 2}
                },
            },
            "Plain One": {
                "mean": 0.5, "sd": 0.0, "ci_lo": 0.5, "ci_hi": 0.5, "n_replicates": 2,
                "access_level": "L1", "cells_measured": 123,
                "per_family": {
                    "alpha": {"mean": 0.5, "sd": 0.0, "ci_lo": 0.5, "ci_hi": 0.5, "n_replicates": 2},
                    "beta": {"mean": 0.6, "sd": 0.02, "ci_lo": 0.58, "ci_hi": 0.62, "n_replicates": 2},
                },
            },
        },
    }
    payload.update(overrides)
    return payload


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "replicated.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --------------------------------------------------------------- interval rendering
def test_interval_text_is_point_and_bounds():
    iv = Interval(mean=0.9430, sd=0.011, ci_lo=0.9379, ci_hi=0.9478, n_replicates=20)
    assert iv.text() == "0.943 [0.938, 0.948]"
    assert iv.text(4, signed=True) == "+0.9430 [+0.9379, +0.9478]"


def test_interval_without_bounds_is_not_decorated():
    """One replicate is the single-draw regime; it must not grow a fabricated range."""
    iv = Interval(mean=0.5, sd=None, ci_lo=None, ci_hi=None, n_replicates=1, note="single replicate")
    assert "[" not in iv.text()
    assert iv.text().startswith("0.500")


def test_missing_estimate_renders_as_not_measured():
    assert Interval(mean=None).text() == NOT_MEASURED


# ------------------------------------------------------------------- schema parsing
def test_unmeasured_family_is_none_not_chance(tmp_path):
    """The bug this whole artifact exists to remove: an absent family is not 0.5."""
    board = load_replicated_leaderboard(_write(tmp_path, _minimal()))
    fitted = board.detectors["Fitted One"]
    assert fitted.family("alpha") is not None
    assert fitted.family("beta") is None
    assert fitted.n_families_measured == 1
    assert board.families() == ["alpha", "beta"]


def test_standings_sorted_by_mean_descending(tmp_path):
    board = load_replicated_leaderboard(_write(tmp_path, _minimal()))
    assert [d.name for d in board.standings()] == ["Fitted One", "Plain One"]


def test_supervision_partition_survives_the_parse(tmp_path):
    board = load_replicated_leaderboard(_write(tmp_path, _minimal()))
    assert board.label_fitted_detectors == ("Fitted One",)
    assert board.detectors["Fitted One"].label_fitted is True
    assert board.detectors["Plain One"].label_fitted is False

    all_sep = board.all_detectors.separation("L0", "L1", "max")
    unsup_sep = board.unsupervised_only.separation("L0", "L1", "max")
    assert all_sep is not None and all_sep.separates is True
    assert unsup_sep is not None and unsup_sep.separates is False
    assert unsup_sep.label == "L1 - L0"


def test_statistic_disagreement_is_carried(tmp_path):
    board = load_replicated_leaderboard(_write(tmp_path, _minimal()))
    assert board.statistic_disagreement == ("L2_minus_L1",)
    assert board.zero_variance_detectors == ("Saturated One",)


def test_each_partition_carries_its_own_statistic_disagreement(tmp_path):
    payload = _minimal()
    payload["supervision"]["statistic_disagreement_unsupervised_only"] = {
        "rungs_where_max_and_mean_disagree_on_separation": ["L1_minus_L0"],
        "rungs_where_max_and_mean_disagree_on_direction": ["L3_minus_L2"],
        "note": "quote both",
    }
    board = load_replicated_leaderboard(_write(tmp_path, payload))
    assert board.unsupervised_only.statistic_disagreement == ("L1_minus_L0", "L3_minus_L2")
    assert board.all_detectors.statistic_disagreement == ("L2_minus_L1",)


def test_an_artifact_without_the_newer_blocks_still_loads_and_says_so(tmp_path):
    """Not recorded is ``None``, which a renderer must not read as "they agree"."""
    board = load_replicated_leaderboard(_write(tmp_path, _minimal()))
    assert board.unsupervised_only.statistic_disagreement is None
    assert board.layouts_resampled is None
    assert board.single_source is None
    assert board.single_source_problems == ()
    assert board.provenance == {} and board.replicates == {}


def test_the_replicates_block_is_read(tmp_path):
    payload = _minimal(
        provenance={"generated_utc": "2026-11-27T00:00:00Z", "script": "agg"},
        replicates={
            "n_replicate_files": 2,
            "layouts_resampled": False,
            "single_source": False,
            "single_source_problems": ["no replicate records the commit it was drawn from"],
        },
    )
    board = load_replicated_leaderboard(_write(tmp_path, payload))
    assert board.layouts_resampled is False
    assert board.single_source is False
    assert board.single_source_problems == ("no replicate records the commit it was drawn from",)
    assert board.provenance["script"] == "agg"


def test_method_block_is_exposed(tmp_path):
    board = load_replicated_leaderboard(_write(tmp_path, _minimal()))
    assert board.ci_percent == 95
    assert board.bootstrap_resamples == 10
    assert board.bootstrap_seed == 7
    assert board.seeds_per_variant == 5


def test_rejects_a_non_replicated_file(tmp_path):
    path = tmp_path / "results_style.json"
    path.write_text(json.dumps({"results": {"Bar": {"overall_auroc": 0.5}}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_replicated_leaderboard(path)


def test_integer_detectors_key_is_not_the_replicated_schema():
    """v5_leaderboard.json has "detectors": 30. Dispatching on the key name misreads it."""
    assert is_replicated_schema({"detectors": 30, "results": {}}) is False
    assert is_replicated_schema({"detectors": {"A": {}}}) is True


# ------------------------------------------------------- adapter interop, real files
@pytest.mark.skipif(not _REPLICATED.exists(), reason="committed replicated board not present")
def test_adapter_reads_the_committed_replicated_board():
    board = load_any_leaderboard_json(_REPLICATED)
    rich = load_replicated_leaderboard(_REPLICATED)
    assert len(board.entries) == len(rich.detectors)

    by_name = {e.detector_name: e for e in board.entries}
    l1 = by_name["State Divergence"]
    # 7 families, not 33: the flattened entry must not have gained imputed cells.
    # (25 families ship no state channel, and continuous_camping's 40 episodes are below
    # every L1 detector's min_episodes.)
    assert len(l1.family_auroc) == 7
    assert l1.n_cells == 31
    assert board.standings()[0].detector_name == rich.standings()[0].name


@pytest.mark.skipif(not _REPLICATED.exists(), reason="committed replicated board not present")
def test_the_committed_board_says_what_its_draws_resampled():
    """``layouts_resampled`` must be the committed replicate files' own answer.

    True only if every file records ``draw_schema`` 2 or later; the set drawn before the
    layout-seed fix records none and was built at layout 0.
    """
    board = load_replicated_leaderboard(_REPLICATED)
    files = sorted((_REPO_ROOT / "results" / "replication").glob("replicate_*.json"))
    if not files:
        pytest.skip("committed replicates not present")
    schemas = [json.loads(f.read_text(encoding="utf-8")).get("draw_schema") for f in files]
    assert board.replicates["n_replicate_files"] == len(files) == board.n_replicates
    assert board.layouts_resampled is all(isinstance(v, int) and v >= 2 for v in schemas)
    assert [d.get("draw_schema") for d in board.draws] == schemas


@pytest.mark.skipif(not _REPLICATED.exists(), reason="committed replicated board not present")
def test_the_committed_unsupervised_disagreement_matches_its_separations():
    """The recorded list is the max/mean rule applied to the partition's own rungs."""
    board = load_replicated_leaderboard(_REPLICATED)
    for partition in (board.all_detectors, board.unsupervised_only):
        expected = []
        for kind in ("separation", "direction"):
            for lo, hi in (("L0", "L1"), ("L1", "L2"), ("L2", "L3")):
                mx = partition.separation(lo, hi, "max")
                mn = partition.separation(lo, hi, "mean")
                if mx is None or mn is None:
                    continue
                separation = mx.separates != mn.separates
                direction = (mx.separates and mn.separates
                             and (mx.difference.mean > 0) != (mn.difference.mean > 0))
                if (separation if kind == "separation" else direction):
                    expected.append(mx.key)
        assert partition.statistic_disagreement == tuple(expected), partition.name


@pytest.mark.skipif(not _V5.exists(), reason="committed v5 board not present")
def test_adapter_still_reads_the_single_draw_board():
    board = load_any_leaderboard_json(_V5)
    assert len(board.entries) == 30
    assert board.standings()[0].overall_auroc is not None


# ------------------------------------------------------------------- the chance control
def test_interval_contains_needs_bounds():
    assert Interval(mean=0.5, ci_lo=0.49, ci_hi=0.51).contains(0.5) is True
    assert Interval(mean=0.6, ci_lo=0.55, ci_hi=0.65).contains(0.5) is False
    # No interval is not a silent "yes".
    assert Interval(mean=0.5).contains(0.5) is False


def test_control_check_reports_each_detector_not_the_level_max(tmp_path):
    """A level whose max clears chance while no detector does must say so both ways."""
    payload = _minimal(
        access_levels={
            "L0": {
                "max_auroc": {"mean": 0.548, "sd": 0.02, "ci_lo": 0.538, "ci_hi": 0.557, "n_replicates": 2},
                "mean_auroc": {"mean": 0.497, "sd": 0.01, "ci_lo": 0.493, "ci_hi": 0.4999, "n_replicates": 2},
                "best_detector_frequency": {"A": 1, "B": 1},
            }
        },
        detectors={
            "A": {
                "mean": 0.506, "sd": 0.01, "ci_lo": 0.495, "ci_hi": 0.517, "n_replicates": 2,
                "access_level": "L0", "cells_measured": 123, "per_family": {},
            },
            "B": {
                "mean": 0.488, "sd": 0.01, "ci_lo": 0.475, "ci_hi": 0.502, "n_replicates": 2,
                "access_level": "L0", "cells_measured": 123, "per_family": {},
            },
        },
    )
    control = load_replicated_leaderboard(_write(tmp_path, payload)).control_check("L0")

    assert control.n_detectors == 2
    assert control.holds is True and control.exceptions == []
    assert control.highest.name == "A"
    # The level max clears 0.5 and sits above every individual detector: selection, not signal.
    assert control.max_excludes_reference is True
    assert control.max_exceeds_every_detector is True


def test_control_check_names_a_detector_that_breaks_the_control(tmp_path):
    payload = _minimal(
        access_levels={},
        detectors={
            "Leaky": {
                "mean": 0.62, "sd": 0.01, "ci_lo": 0.60, "ci_hi": 0.64, "n_replicates": 2,
                "access_level": "L0", "cells_measured": 123, "per_family": {},
            }
        },
    )
    control = load_replicated_leaderboard(_write(tmp_path, payload)).control_check("L0")
    assert control.holds is False
    assert [d.name for d in control.exceptions] == ["Leaky"]


def test_the_level_mean_margin_is_reported_alongside_the_exclusion(tmp_path):
    """``mean_excludes_reference`` is bare arithmetic; the margin is what makes it readable."""
    payload = _minimal(
        access_levels={
            "L0": {
                "max_auroc": {"mean": 0.55, "sd": 0.01, "ci_lo": 0.54, "ci_hi": 0.56, "n_replicates": 2},
                "mean_auroc": {
                    "mean": 0.4966, "sd": 0.007,
                    "ci_lo": 0.49337, "ci_hi": 0.49992, "n_replicates": 2,
                },
                "best_detector_frequency": {},
            }
        },
        detectors={},
    )
    control = load_replicated_leaderboard(_write(tmp_path, payload)).control_check("L0")
    assert control.mean_excludes_reference is True
    # ...by 8e-5, which is the whole point: it is smaller than the endpoint's own scatter.
    assert control.mean_margin_from_reference == pytest.approx(0.00008, abs=1e-6)


@pytest.mark.skipif(not _REPLICATED.exists(), reason="committed replicated board not present")
def test_the_committed_l0_control_has_one_below_chance_exception_and_its_max_does_not_hold():
    """Regression guard for the reading that put a selection-biased 0.548 on the page.

    On the 2026-08-04 draws (layout 0 only) every one of the 13 L0 detectors had an
    interval containing 0.500. On the re-draw of 2026-11-27 (``b93c351``: real layouts,
    the numerics pins, the frozen ``continuous_camping`` camper) 12 do; Reward CUSUM's,
    0.486 [0.475, 0.497], lies wholly below chance. Below, not above: it separates the
    variants weakly in the direction that would not flag hacking.
    """
    control = load_replicated_leaderboard(_REPLICATED).control_check()

    assert control.level == "L0"
    assert control.n_detectors == 13
    assert control.holds is False
    assert [d.name for d in control.exceptions] == ["Reward CUSUM"]
    (cusum,) = control.exceptions
    assert cusum.overall.ci_hi < 0.5
    assert control.highest.overall.mean < 0.51

    # The level's "best detector" cell excludes 0.500 and beats every detector in it.
    assert control.max_excludes_reference is True
    assert control.max_exceeds_every_detector is True

    # And the suite mean's interval, [0.4918, 0.5036], contains 0.500 with room on both
    # sides. (On the 2026-08-04 draws it was [0.4899, 0.5000], reaching 0.500 only at its
    # upper end and within the bootstrap's own scatter there;
    # docs/figures/l0_control_robustness.md replays that.)
    assert control.mean_excludes_reference is False
    assert control.mean_margin_from_reference < 1e-4
