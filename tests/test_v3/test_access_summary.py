"""Tests for per-access-level aggregation, including the duplicate-detector exclusion.

The numbers asserted against ``leaderboard/v5_leaderboard.json`` are the committed,
paper-cited board. They are pinned here so that a regeneration that changes the
access-level ladder cannot slip through unnoticed, and so that the specific defect these
tests were written for -- an L3 "oracle ceiling" that was half a copy of an L2
detector -- cannot silently return.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rhob.v3.leaderboard import (
    ACCESS_LEVELS,
    degenerate_families_from_ledger,
    render_access_summary_md,
    summarize_access_levels,
)

_BOARD = Path(__file__).resolve().parents[2] / "leaderboard" / "v5_leaderboard.json"

#: The families the admission ledger cannot certify a matched proxy for, and which are
#: therefore held out of the L0 negative control. Duplicated here rather than read from
#: the ledger so that a ledger regeneration which changes the set has to be acknowledged
#: in a test instead of silently moving every number below.
DEGENERATE_FAMILIES = [
    "distributional_shift",
    "monitored_sandbagging",
    "orbit_chirality",
    "physics_exploitation",
    "rlhf_reward_model_overopt",
    "shortcut_exploitation",
]


@pytest.fixture(scope="module")
def v5_results() -> dict:
    return json.loads(_BOARD.read_text())["results"]


def test_all_levels_present_even_when_empty():
    summaries = summarize_access_levels({})
    assert set(summaries) == set(ACCESS_LEVELS)
    for s in summaries.values():
        assert s.n_detectors == 0
        assert s.mean_auroc is None
        assert s.max_auroc is None
        assert s.best_detector is None


def test_duplicate_is_excluded_and_reported_separately():
    results = {
        "Behavioral Threshold": {"access_level": "L2", "overall_auroc": 0.9},
        "True Reward Oracle": {"access_level": "L3", "overall_auroc": 0.98},
        "Perfect Feature Oracle": {"access_level": "L3", "overall_auroc": 0.9},
    }
    summaries = summarize_access_levels(results)

    assert summaries["L3"].n_detectors == 1
    assert summaries["L3"].best_detector == "True Reward Oracle"
    assert summaries["L3"].mean_auroc == pytest.approx(0.98)
    assert summaries["L3"].excluded_duplicates == [
        ("Perfect Feature Oracle", "Behavioral Threshold")
    ]
    # The duplicate is not quietly re-homed under L2 either -- it belongs to no level.
    assert summaries["L2"].n_detectors == 1
    assert summaries["L2"].mean_auroc == pytest.approx(0.9)


def test_exclude_duplicates_false_reproduces_the_pre_fix_number():
    """The opt-out exists only to reproduce the old, wrong aggregate for comparison."""
    results = {
        "True Reward Oracle": {"access_level": "L3", "overall_auroc": 0.98},
        "Perfect Feature Oracle": {"access_level": "L3", "overall_auroc": 0.9},
    }
    summaries = summarize_access_levels(results, exclude_duplicates=False)
    assert summaries["L3"].n_detectors == 2
    assert summaries["L3"].mean_auroc == pytest.approx(0.94)
    assert summaries["L3"].excluded_duplicates == []


def test_errored_and_malformed_entries_are_skipped_not_zeroed():
    results = {
        "Good": {"access_level": "L0", "overall_auroc": 0.6},
        "Errored": {"access_level": "L0", "error": "NaN input"},
        "Null": {"access_level": "L0", "overall_auroc": None},
        "Unknown level": {"access_level": "L9", "overall_auroc": 0.99},
    }
    summaries = summarize_access_levels(results)
    assert summaries["L0"].n_detectors == 1
    assert summaries["L0"].mean_auroc == pytest.approx(0.6)


def test_max_is_the_best_detector_not_the_mean():
    results = {
        "Weak A": {"access_level": "L2", "overall_auroc": 0.50},
        "Weak B": {"access_level": "L2", "overall_auroc": 0.52},
        "Strong": {"access_level": "L2", "overall_auroc": 0.96},
    }
    s = summarize_access_levels(results)["L2"]
    assert s.max_auroc == pytest.approx(0.96)
    assert s.best_detector == "Strong"
    assert s.mean_auroc == pytest.approx(0.66)
    assert s.scored[0][0] == "Strong"  # sorted best-first


def test_committed_v5_board_ladder(v5_results):
    """Pin the published ladder on the real, committed board.

    Means and maxima as measured 2026-08 on ``leaderboard/v5_leaderboard.json``
    (33 families, 123 cells per detector), with the six degenerate-proxy families held
    out of L0 -- which is what makes L0 the negative control the docs describe rather
    than a mean padded with cells that could not have come out any other way. L3 has
    exactly one member after the Perfect Feature Oracle duplicate is removed.

    L0 moved from 0.4898 to 0.4866 when the exclusion was implemented, and its best
    detector's figure from 0.5180 to 0.5311 (still Reward CUSUM). Of that L0 shift,
    0.0003 is the re-weighting the exclusion forces (family-weighted over the retained
    families, against the cell-weighted ``overall_auroc`` the other rungs use -- see
    ``rhob.v3.leaderboard.access_summary``) and the remaining 0.0029 is the exclusion
    itself; ``test_the_l0_exclusion_is_worth_this_much`` separates the two.

    These are pins on a committed artifact, not on library behaviour: regenerating
    ``v5_leaderboard.json`` or ``admission/admission_ledger.json`` will move them. That
    is the point -- these exact figures are quoted in the docs, so a regeneration must
    update the pins and the prose in the same change rather than letting the two drift
    apart silently.
    """
    s = summarize_access_levels(v5_results, degenerate_families=DEGENERATE_FAMILIES)

    assert [s[lv].n_detectors for lv in ACCESS_LEVELS] == [13, 8, 7, 1]

    assert s["L0"].mean_auroc == pytest.approx(0.4866, abs=5e-4)
    assert s["L1"].mean_auroc == pytest.approx(0.5200, abs=5e-4)
    assert s["L2"].mean_auroc == pytest.approx(0.7213, abs=5e-4)
    assert s["L3"].mean_auroc == pytest.approx(0.9830, abs=5e-4)

    assert s["L0"].n_families_scored == 27
    assert s["L0"].excluded_families == sorted(DEGENERATE_FAMILIES)

    assert (s["L0"].best_detector, s["L0"].max_auroc) == (
        "Reward CUSUM",
        pytest.approx(0.5311, abs=5e-4),
    )
    assert (s["L1"].best_detector, s["L1"].max_auroc) == ("State Divergence", pytest.approx(0.621))
    assert (s["L2"].best_detector, s["L2"].max_auroc) == (
        "Behavioral Threshold",
        pytest.approx(0.975),
    )
    assert (s["L3"].best_detector, s["L3"].max_auroc) == (
        "True Reward Oracle",
        pytest.approx(0.983),
    )

    # The audit's headline: best-L3 exceeds best-L2 by 0.008 AUROC. Whatever else the
    # access ladder shows, it does not show a meaningful oracle gap above L2.
    assert s["L3"].max_auroc - s["L2"].max_auroc == pytest.approx(0.008, abs=5e-4)

    assert s["L3"].excluded_duplicates == [("Perfect Feature Oracle", "Behavioral Threshold")]


def test_the_l0_exclusion_is_worth_this_much(v5_results):
    """Separate the two things that moved L0, so neither can hide behind the other.

    Holding families out forces the L0 rung to be computed family-weighted (the artifact
    records AUROCs per family, not per cell), and a re-weighting is not a finding. Both
    figures below are family-weighted, so the difference between them is the exclusion
    and nothing else: **0.4895 with the degenerate families, 0.4866 without.** For
    scale, the re-weighting on its own is worth 0.0003 -- family-weighted over all 33
    families is 0.4895 against the cell-weighted 0.4898 the un-excluded ladder reports.
    """
    excluded = summarize_access_levels(
        v5_results, degenerate_families=DEGENERATE_FAMILIES
    )["L0"]
    assert excluded.mean_auroc == pytest.approx(0.4866, abs=5e-4)
    assert excluded.mean_auroc_including_degenerate == pytest.approx(0.4895, abs=5e-4)

    cell_weighted = summarize_access_levels(v5_results)["L0"]
    assert cell_weighted.mean_auroc == pytest.approx(0.4898, abs=5e-4)
    assert cell_weighted.excluded_families == []
    assert cell_weighted.mean_auroc_including_degenerate is None

    # The exclusion moves L0 ten times further than the re-weighting does.
    exclusion = abs(excluded.mean_auroc - excluded.mean_auroc_including_degenerate)
    reweighting = abs(excluded.mean_auroc_including_degenerate - cell_weighted.mean_auroc)
    assert exclusion > 5 * reweighting


def test_the_exclusion_applies_to_l0_and_only_to_l0(v5_results):
    """Degeneracy is a property of the proxy reward, which only L0 reads.

    L1/L2/L3 score the same families through state visitation, behavior and true reward,
    where nothing is degenerate, so dropping those families there would throw away real
    measurements. If this ever starts failing, the ladder has quietly become
    incomparable between rungs.
    """
    plain = summarize_access_levels(v5_results)
    excluded = summarize_access_levels(v5_results, degenerate_families=DEGENERATE_FAMILIES)
    for level in ("L1", "L2", "L3"):
        assert excluded[level].mean_auroc == plain[level].mean_auroc
        assert excluded[level].excluded_families == []
        assert excluded[level].n_families_scored is None
    assert excluded["L0"].mean_auroc != plain["L0"].mean_auroc


def test_excluding_families_from_a_board_without_per_family_data_is_an_error():
    """Silently leaving them in is the defect; refusing is the only safe alternative."""
    results = {"Reward Threshold": {"access_level": "L0", "overall_auroc": 0.5}}
    with pytest.raises(ValueError, match="no per_family block"):
        summarize_access_levels(results, degenerate_families=["distributional_shift"])


def test_the_excluded_set_comes_from_the_committed_ledger():
    """The list is read from the artifact, not maintained by hand in two places.

    A family that is given an informative-but-matched proxy rejoins the L0 control by
    being re-certified in ``scripts/admission_ledger.py``, and nothing else has to be
    remembered.
    """
    assert degenerate_families_from_ledger() == sorted(DEGENERATE_FAMILIES)


def test_committed_board_duplicate_really_is_identical(v5_results):
    """The exclusion is only justified while the two detectors still agree everywhere."""
    baseline = v5_results["Behavioral Threshold"]
    duplicate = v5_results["Perfect Feature Oracle"]
    assert duplicate["per_family"] == baseline["per_family"]
    assert duplicate["overall_auroc"] == baseline["overall_auroc"]
    assert len(baseline["per_family"]) == 33


def test_markdown_reports_mean_max_and_the_exclusion(v5_results):
    md = render_access_summary_md(summarize_access_levels(v5_results))
    assert "| Access level | Detectors | Mean AUROC | SD across detectors | Best AUROC | Best detector |" in md
    # Mean and max are both present on the same row, and they differ at L2.
    assert "| L2 | 7 | 0.7213 | 0.2052 | 0.9750 | Behavioral Threshold |" in md
    assert "| L3 | 1 | 0.9830 | 0.0000 | 0.9830 | True Reward Oracle |" in md
    assert "**Perfect Feature Oracle** is a relabelled duplicate of **Behavioral Threshold**" in md
    assert "not a confidence interval" in md
    # Nothing was held out of L0, so nothing claims to have been.
    assert "held out" not in md


def test_markdown_states_the_l0_exclusion_and_what_it_cost(v5_results):
    """An exclusion nobody can see is indistinguishable from a number that drifted.

    The rendered table has to name every withheld family and print the figure they would
    have produced, so a reader can check the adjustment instead of taking it.
    """
    md = render_access_summary_md(
        summarize_access_levels(v5_results, degenerate_families=DEGENERATE_FAMILIES)
    )
    assert "| L0 | 13 | 0.4866 |" in md
    assert "L0 is computed over 27 families" in md
    assert "6 are held out" in md
    for family in DEGENERATE_FAMILIES:
        assert f"**{family}**" in md
    assert "Putting them back would read 0.4895 instead of 0.4866" in md
    assert "held out of L0 only" in md


#: The generated ladder that the docs cite, written by ``scripts/plot_v5_results.py``.
_ACCESS_SUMMARY_MD = (
    Path(__file__).resolve().parents[2] / "docs" / "figures" / "v5_access_summary.md"
)


def test_the_published_ladder_was_generated_with_the_exclusion(v5_results):
    """The artifact on disk, against what the library produces from the same inputs.

    Everything above tests the library, and the library was never the problem: it honoured
    ``degenerate_families`` and these tests pinned it, while **neither generator passed
    it** -- ``scripts/plot_v5_results.py`` and ``scripts/v5_leaderboard_and_transfer.py``
    both called ``summarize_access_levels`` with the argument omitted. So the exclusion
    was implemented, tested, documented and absent from every published number: the
    committed ``v5_access_summary.md`` carried the un-excluded ``0.4898 | 0.0156 | 0.5180``
    L0 row, and the docs quoted it.

    A test on the library cannot catch that, because the library was right. This one
    regenerates the table from the same two committed artifacts the generator reads and
    demands the file match byte for byte, so an unwired generator and a stale figure fail
    identically -- which is correct, since they are the same defect to a reader.
    """
    expected = render_access_summary_md(
        summarize_access_levels(
            v5_results, degenerate_families=degenerate_families_from_ledger()
        )
    )
    published = _ACCESS_SUMMARY_MD.read_text(encoding="utf-8")

    assert published.strip() == expected.strip(), (
        f"{_ACCESS_SUMMARY_MD.name} does not match what the library produces from "
        "leaderboard/v5_leaderboard.json + admission/admission_ledger.json. Either a "
        "generator dropped the degenerate_families argument, or the figure is stale -- "
        "rerun `python scripts/plot_v5_results.py`."
    )
    # Named explicitly as well, because this row is the one the audit found wrong and
    # an equality assertion alone would not say which number matters.
    assert "| L0 | 13 | 0.4866 | 0.0250 | 0.5311 | Reward CUSUM |" in published
