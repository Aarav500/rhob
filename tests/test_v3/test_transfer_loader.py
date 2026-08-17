"""Tests for the cross-family transfer loader.

The properties pinned here are the ones that let a stale artifact reach a deployed page
looking current: a pre-sign-randomization file must not claim to be post-randomization, a
row that predates the not-applicable fix must be identifiable as such, an absent family
must stay ``None`` instead of becoming 0.5, and an L2 row measured only with the flip off
must never be carried over as if it were still valid.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rhob.v3.leaderboard.transfer import (
    SIGN_INVARIANT_LEVELS,
    load_transfer_results,
    sign_randomization_invariant,
    transfer_under_sign_randomization,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PUBLISHED = _REPO_ROOT / "leaderboard" / "cross_family_transfer.json"
_MEASURED = _REPO_ROOT / "docs" / "figures" / "sign_randomization_impact.json"


def _write(tmp_path: Path, payload: dict, name: str = "transfer.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _multi_trial_row(level: str, avg: float, **extra) -> dict:
    row = {
        "access_level": level,
        "n_trials": 3,
        "train_auroc_mean": 0.8,
        "avg_transfer_auroc_mean": avg,
        "avg_transfer_auroc_std": 0.01,
        "per_family_transfer_mean": {"alpha": avg, "beta": avg},
        "test_families_not_applicable": [],
    }
    row.update(extra)
    return row


# ------------------------------------------------------------------ schema dispatch
def test_single_config_file_reads_as_one_unnamed_configuration(tmp_path):
    path = _write(tmp_path, {"results": {"Det": _multi_trial_row("L2", 0.9)}})
    board = load_transfer_results(path)
    assert board.config_names() == ["published"]
    assert board.rows()[0].name == "Det"


def test_config_keyed_file_keeps_its_configurations_apart(tmp_path):
    path = _write(
        tmp_path,
        {
            "configs": {
                "before": {"randomize_behav_sign": False, "detector_infers_orientation": False},
                "after": {"randomize_behav_sign": True, "detector_infers_orientation": True},
            },
            "results": {
                "before": {"Det": _multi_trial_row("L2", 0.99)},
                "after": {"Det": _multi_trial_row("L2", 0.51)},
            },
        },
    )
    board = load_transfer_results(path)
    assert board.shipped_config() == "after"
    assert board.baseline_config() == "before"
    assert board.row("Det", "after").avg_transfer_auroc == 0.51
    with pytest.raises(ValueError):
        board.rows()  # ambiguous: two configurations, caller must name one


def test_the_aggregate_artifacts_transfer_block_is_unwrapped(tmp_path):
    path = _write(
        tmp_path,
        {
            "measurement": "sign_randomization_impact",
            "provenance": {"generated_utc": "2026-08-03T00:00:00Z", "git": {"short_commit": "abc123"}},
            "transfer": {
                "configs": {"after": {"randomize_behav_sign": True, "detector_infers_orientation": True}},
                "results": {"after": {"Det": _multi_trial_row("L2", 0.5)}},
            },
        },
    )
    board = load_transfer_results(path)
    assert board.shipped_config() == "after"
    # Provenance lives on the enclosing document, not the transfer block.
    assert board.generated_utc.startswith("2026-08-03")
    assert board.git_commit == "abc123"


def test_a_file_with_no_results_block_raises(tmp_path):
    with pytest.raises(ValueError):
        load_transfer_results(_write(tmp_path, {"train_families": []}))


# ---------------------------------------------------------------- silence is not "off"
def test_an_artifact_without_the_flag_is_unknown_rather_than_randomized(tmp_path):
    """A file that never heard of the flag must not read as post-randomization."""
    board = load_transfer_results(
        _write(tmp_path, {"results": {"Det": _multi_trial_row("L2", 0.994)}})
    )
    cfg = board.configs["published"]
    assert cfg.randomize_behav_sign is None
    assert cfg.is_shipped_configuration is False
    assert cfg.predates_sign_randomization is True
    assert board.predates_sign_randomization is True
    assert board.shipped_config() is None


# ------------------------------------------------------------- not applicable vs chance
def test_an_absent_family_stays_none_and_is_out_of_the_denominator(tmp_path):
    row = _multi_trial_row("L1", 0.5)
    row["per_family_transfer_mean"] = {"alpha": 0.5, "beta": None, "gamma": None}
    row["test_families_not_applicable"] = ["beta", "gamma"]
    board = load_transfer_results(_write(tmp_path, {"results": {"L1 Det": row}}))
    parsed = board.rows()[0]
    assert parsed.per_family["beta"] is None  # not 0.5, not 0.0
    assert parsed.n_families_scored == 1
    assert parsed.n_families_not_applicable == 2
    assert parsed.families_not_applicable == ("beta", "gamma")


def test_a_row_predating_the_na_fix_is_flagged_untrustworthy(tmp_path):
    """No ``test_families_not_applicable`` key means absent cells were imputed at 0.5."""
    old = {
        "access_level": "L1",
        "train_auroc": 0.52,
        "per_family_transfer": {"alpha": 0.5, "beta": 0.5},
        "avg_transfer_auroc": 0.5,
    }
    board = load_transfer_results(_write(tmp_path, {"results": {"Old": old}}))
    parsed = board.rows()[0]
    assert parsed.records_not_applicable is False
    assert parsed.n_trials == 1  # a single fit, not "never run"


# -------------------------------------------------------------------- the carry-over rule
def test_l0_and_l1_are_the_sign_invariant_levels():
    assert SIGN_INVARIANT_LEVELS == ("L0", "L1")
    assert sign_randomization_invariant("L1") is True
    assert sign_randomization_invariant("L2") is False
    assert sign_randomization_invariant("L3") is False


def test_an_l2_row_measured_only_before_randomization_is_not_carried_over(tmp_path):
    """The whole point: a pre-randomization L2 figure must not reappear as current."""
    path = _write(
        tmp_path,
        {
            "configs": {
                "before": {"randomize_behav_sign": False, "detector_infers_orientation": False},
                "after": {"randomize_behav_sign": True, "detector_infers_orientation": True},
            },
            "results": {
                "before": {
                    "L0 Det": _multi_trial_row("L0", 0.48),
                    "Stale L2": _multi_trial_row("L2", 0.994),
                },
                "after": {"Fresh L2": _multi_trial_row("L2", 0.51)},
            },
        },
    )
    rows = transfer_under_sign_randomization(load_transfer_results(path))
    names = [c.name for c in rows]
    assert "Stale L2" not in names  # L2, measured with the flip off -> dropped
    assert "L0 Det" in names  # L0 sees no behavioural trace -> carried over
    assert "Fresh L2" in names
    carried = {c.name: c.carried_over_as_invariant for c in rows}
    assert carried["L0 Det"] is True
    assert carried["Fresh L2"] is False


def test_the_shipped_configuration_wins_over_the_baseline(tmp_path):
    path = _write(
        tmp_path,
        {
            "configs": {
                "before": {"randomize_behav_sign": False, "detector_infers_orientation": False},
                "after": {"randomize_behav_sign": True, "detector_infers_orientation": True},
            },
            "results": {
                "before": {"Det": _multi_trial_row("L2", 0.994)},
                "after": {"Det": _multi_trial_row("L2", 0.508)},
            },
        },
    )
    (row,) = transfer_under_sign_randomization(load_transfer_results(path))
    assert row.rts == 0.508
    assert row.measured_config == "after"


def test_supersession_is_measured_against_the_trial_spread(tmp_path):
    """A move smaller than the fit's own wobble has not been contradicted."""
    measured = _write(
        tmp_path,
        {
            "configs": {"after": {"randomize_behav_sign": True, "detector_infers_orientation": True}},
            "results": {
                "after": {
                    "Noisy": _multi_trial_row("L2", 0.500, avg_transfer_auroc_std=0.05),
                    "Broken": _multi_trial_row("L2", 0.508, avg_transfer_auroc_std=0.002),
                }
            },
        },
        name="measured.json",
    )
    published = _write(
        tmp_path,
        {
            "results": {
                "Noisy": _multi_trial_row("L2", 0.520),  # 0.020 move, spread 0.05
                "Broken": _multi_trial_row("L2", 0.994),  # 0.486 move, spread 0.002
            }
        },
        name="published.json",
    )
    rows = {
        c.name: c
        for c in transfer_under_sign_randomization(
            load_transfer_results(measured), load_transfer_results(published)
        )
    }
    assert rows["Noisy"].published_differs is False
    assert rows["Broken"].published_differs is True
    assert rows["Broken"].published_delta == pytest.approx(-0.486)


def test_a_flat_row_is_reported_as_flat(tmp_path):
    flat = _multi_trial_row("L1", 0.5)
    flat["per_family_transfer_mean"] = {"a": 0.5, "b": 0.5, "c": None}
    flat["test_families_not_applicable"] = ["c"]
    varied = _multi_trial_row("L2", 0.7)
    varied["per_family_transfer_mean"] = {"a": 0.6, "b": 0.8}
    path = _write(
        tmp_path,
        {
            "configs": {"after": {"randomize_behav_sign": True, "detector_infers_orientation": True}},
            "results": {"after": {"Flat": flat, "Varied": varied}},
        },
    )
    rows = {c.name: c for c in transfer_under_sign_randomization(load_transfer_results(path))}
    assert rows["Flat"].all_scored_families_at == 0.5
    assert rows["Varied"].all_scored_families_at is None


# --------------------------------------------------------------------- the real artifacts
@pytest.mark.skipif(not _PUBLISHED.exists(), reason="published transfer artifact not present")
def test_the_committed_published_artifact_is_pre_randomization_and_pre_na_fix():
    """Regression guard for the two defects that put 0.994 and an imputed 0.500 on the page."""
    board = load_transfer_results(_PUBLISHED)
    assert board.predates_sign_randomization is True
    assert board.shipped_config() is None
    rows = {r.name: r for r in board.rows()}
    assert rows["Ensemble (Top 5)"].avg_transfer_auroc == pytest.approx(0.994)
    # Its L1 row reports all eight test families as numbers, including the two that ship
    # no state channel -- which is exactly why records_not_applicable must be False.
    state_div = rows["State Divergence"]
    assert state_div.records_not_applicable is False
    assert state_div.n_families_not_applicable == 0


@pytest.mark.skipif(not _MEASURED.exists(), reason="sign-randomization measurement not present")
def test_the_measured_artifact_supersedes_the_published_l2_rows():
    measured = load_transfer_results(_MEASURED)
    published = load_transfer_results(_PUBLISHED) if _PUBLISHED.exists() else None
    rows = {c.name: c for c in transfer_under_sign_randomization(measured, published)}

    assert measured.shipped_config() == "after"
    # L2 re-measured, and far enough from the published figure to count as superseded.
    ensemble = rows["Ensemble (Top 5)"]
    assert ensemble.carried_over_as_invariant is False
    assert ensemble.rts == pytest.approx(0.508)
    assert ensemble.published == pytest.approx(0.994)
    assert ensemble.published_differs is True

    # L1 carried over, and now with honest not-applicable bookkeeping.
    state_div = rows["State Divergence"]
    assert state_div.carried_over_as_invariant is True
    assert state_div.row.records_not_applicable is True
    assert state_div.row.n_families_scored == 6
    assert set(state_div.row.families_not_applicable) == {
        "monitored_sandbagging",
        "eval_probe_sandbagging",
    }
