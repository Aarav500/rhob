"""Tests for per-access-level aggregation, including the duplicate-detector exclusion.

The numbers asserted against ``leaderboard/v5_leaderboard.json`` are the committed
board's, as re-scored on CI on 2026-11-26 once every detector declared the horizon it
needs, and the docs quote them. They
are pinned here so that a regeneration that changes the access-level ladder cannot slip
through unnoticed, and so that the specific defect these tests were written for -- an L3
"oracle ceiling" that was half a copy of an L2 detector -- cannot silently return.
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
import rhob.detectors as detectors
from rhob.detectors.posthoc import PosthocDetector

_BOARD = Path(__file__).resolve().parents[2] / "leaderboard" / "v5_leaderboard.json"

#: The families whose proxy the admission ledger marks DEGENERATE (could not measure),
#: and which are therefore held out of the L0 negative control. Duplicated here rather than read from
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

    Means and maxima on ``leaderboard/v5_leaderboard.json`` as re-drawn on CI on
    2026-11-27 (``b93c351``: under the numerics pins, with the frozen
    ``continuous_camping`` camper; 33 families, 123 cells). Every detector is ``null`` on a family that does
    not emit its channel (each L1 detector has 25) and on a family that runs fewer
    episodes than the ``min_episodes`` it declares
    (``test_no_detector_is_scored_below_its_declared_horizon``). The six degenerate-proxy
    families are held out of L0, which is what makes L0 the negative control the docs
    describe rather than a mean padded with cells that could not have come out any other
    way. L3 has exactly one member after the Perfect Feature Oracle duplicate is removed.

    The board of 2026-11-26 (``f83a152``, unpinned, its camper trained on the runner)
    read L0 0.4834 and L2 0.5771 (best 0.943, Trajectory MLP); six detectors' overall
    AUROCs moved in the re-draw, none by more than 0.013 (Reward KDE, 0.495 to 0.482).

    The board of 2026-11-19 (``52656cb``) read L0 0.4876 (best 0.5351, Reward CUSUM), L1
    0.5701 (best 0.927, State Divergence) and L2 0.5716 (best 0.943, Trajectory MLP). It
    recorded eight L0 detectors' fallback constant as 0.5 on the families shorter than
    their windows (104 per-family entries), every L1 detector's on ``continuous_camping``,
    and three L2 detectors' on the short-horizon families. Those entries are ``null`` now,
    and that is all that moved: every other per-family value is unchanged.

    The board before that (2026-07-17; ``git show acd35c2:leaderboard/v5_leaderboard.json``)
    read L0 0.4866, L1 0.5200, L2 0.7213 (best 0.975, Behavioral Threshold) and L3 0.9830.
    The 2026-11-19 regeneration scored the behavioural feature with its sign randomized
    and left cells without the channel out instead of imputing 0.5.

    These are pins on a committed artifact, not on library behaviour: regenerating
    ``v5_leaderboard.json`` or ``admission/admission_ledger.json`` will move them. That
    is the point -- these exact figures are quoted in the docs, so a regeneration must
    update the pins and the prose in the same change rather than letting the two drift
    apart silently.
    """
    s = summarize_access_levels(v5_results, degenerate_families=DEGENERATE_FAMILIES)

    assert [s[lv].n_detectors for lv in ACCESS_LEVELS] == [13, 8, 7, 1]

    assert s["L0"].mean_auroc == pytest.approx(0.4835, abs=5e-4)
    assert s["L1"].mean_auroc == pytest.approx(0.5789, abs=5e-4)
    assert s["L2"].mean_auroc == pytest.approx(0.5771, abs=5e-4)
    assert s["L3"].mean_auroc == pytest.approx(0.9830, abs=5e-4)

    assert s["L0"].n_families_scored == 27
    assert s["L0"].excluded_families == sorted(DEGENERATE_FAMILIES)
    # Nine L0 detectors need more episodes than some families run. The five that need
    # 100 are null on the 19 families with horizon 40 or 60 and averaged over 8 of the
    # 27; the four that need 50, 51 or 60 are null on the 7 with horizon 40 and averaged
    # over 20. The other four have a value on all 27.
    by_detector = s["L0"].n_families_scored_by_detector
    assert {n: k for n, k in by_detector.items() if k != 27} == {
        "Reward Autocorrelation": 8,
        "Reward Peak": 8,
        "Reward Skewness": 8,
        "Reward Trend": 8,
        "Variance Window": 8,
        "Gradient Reversal": 20,
        "Max Plateau": 20,
        "Reward CUSUM": 20,
        "Reward MLP": 20,
    }
    assert len(by_detector) == 13

    assert (s["L0"].best_detector, s["L0"].max_auroc) == (
        "Reward CUSUM",
        pytest.approx(0.5474, abs=5e-4),
    )
    # State Divergence's 0.5 on continuous_camping was its untrained fallback: below its
    # 50-episode baseline ``fit`` sees no histogram. Over the 7 families it can score it
    # reads 0.982.
    assert (s["L1"].best_detector, s["L1"].max_auroc) == ("State Divergence", pytest.approx(0.982))
    assert (s["L2"].best_detector, s["L2"].max_auroc) == (
        "Trajectory MLP",
        pytest.approx(0.944),
    )
    assert (s["L3"].best_detector, s["L3"].max_auroc) == (
        "True Reward Oracle",
        pytest.approx(0.983),
    )

    # The oracle gap above L2. The audit's headline, on the board before the 2026-11-19
    # regeneration, was 0.008 (0.983 against Behavioral Threshold's 0.975, a figure that
    # depended on the behavioural feature's fixed sign). With the sign randomized,
    # Behavioral Threshold reads 0.512, the best L2 detector is the label-fitted
    # Trajectory MLP, and the gap is 0.039 on this single draw (0.040 on the 2026-11-26
    # board, whose Trajectory MLP read 0.943 before the pins). The best L1 detector is
    # within 0.001 of the oracle, but over the 31 cells of the 7 families it can score,
    # against the oracle's 123, so the two maxima are not like for like.
    assert s["L3"].max_auroc - s["L2"].max_auroc == pytest.approx(0.039, abs=5e-4)
    assert s["L3"].max_auroc - s["L1"].max_auroc == pytest.approx(0.001, abs=5e-4)

    assert s["L3"].excluded_duplicates == [("Perfect Feature Oracle", "Behavioral Threshold")]


def test_the_l0_exclusion_is_worth_this_much(v5_results):
    """Separate the two things that moved L0, so neither can hide behind the other.

    Holding families out forces the L0 rung to be computed family-weighted (the artifact
    records AUROCs per family, not per cell), and a re-weighting is not a finding. Both
    figures below are family-weighted, so the difference between them is the exclusion
    and nothing else: **0.4871 with the degenerate families, 0.4835 without**, an
    exclusion worth 0.0036 (0.003590 from unrounded values). For scale, the re-weighting
    on its own is worth 0.0001 -- family-weighted over every family each detector has a
    value on is 0.4871 against the cell-weighted 0.4872 the un-excluded ladder reports
    (0.000090 unrounded).

    On the board of 2026-11-26 (``f83a152``, before the re-draw under the numerics pins)
    the same three figures were 0.4877, 0.4834 and 0.4881: an exclusion of 0.0043 against
    a re-weighting of 0.00036. On the board of 2026-11-19 the same three figures were 0.4876, 0.4898 and 0.4903: an
    exclusion of 0.0023 against a re-weighting of 0.0005. On the board before that they
    were 0.4866, 0.4895 and 0.4898: 0.0029 against 0.00036.
    """
    excluded = summarize_access_levels(
        v5_results, degenerate_families=DEGENERATE_FAMILIES
    )["L0"]
    assert excluded.mean_auroc == pytest.approx(0.4835, abs=5e-4)
    assert excluded.mean_auroc_including_degenerate == pytest.approx(0.4871, abs=5e-4)

    cell_weighted = summarize_access_levels(v5_results)["L0"]
    assert cell_weighted.mean_auroc == pytest.approx(0.4872, abs=5e-4)
    assert cell_weighted.excluded_families == []
    assert cell_weighted.mean_auroc_including_degenerate is None

    # The exclusion moves L0 further than the re-weighting does: 0.0036 against 0.0001,
    # about forty times.
    exclusion = abs(excluded.mean_auroc - excluded.mean_auroc_including_degenerate)
    reweighting = abs(excluded.mean_auroc_including_degenerate - cell_weighted.mean_auroc)
    assert exclusion == pytest.approx(0.0036, abs=1e-4)
    assert reweighting == pytest.approx(0.0001, abs=1e-4)
    assert exclusion > 10 * reweighting


_LEDGER = Path(__file__).resolve().parents[2] / "admission" / "admission_ledger.json"


def _declared_minimum() -> dict[str, int]:
    """Board name -> ``min_episodes``, for every detector buildable with no arguments."""
    out = {}
    for attr in detectors.__all__:
        cls = getattr(detectors, attr)
        if isinstance(cls, type) and issubclass(cls, PosthocDetector):
            try:
                det = cls()
            except TypeError:  # EnsembleDetector needs members
                continue
            out[det.name] = int(det.min_episodes)
    return out


def test_no_detector_is_scored_below_its_declared_horizon(v5_results):
    """A family shorter than a detector's ``min_episodes`` is ``null``, with a reason.

    Until 2026-11-26 only Reward Skewness declared one, and the board recorded the
    fallback constant of the others on the families too short for them as an AUROC of
    0.5: 104 per-family entries behind the L0 figures, 8 behind L1 (every L1 detector on
    ``continuous_camping``) and 33 behind L2 (Feature Consistency on the 19 families with
    horizon 40 or 60, Feature Magnitude and Reward-Feature Correlation on the 7 with
    horizon 40). Family horizons come from the committed ledger
    (``onset_localizable.horizon``, one per family).
    """
    horizon: dict[str, float] = {}
    cells: dict[str, int] = {}
    for row in json.loads(_LEDGER.read_text(encoding="utf-8"))["results"]:
        horizon[row["family"]] = row["metrics"]["onset_localizable"]["horizon"]
        cells[row["family"]] = cells.get(row["family"], 0) + 1

    # L1 reads state_counts, which these 8 families emit; every other level reads a
    # channel all 33 emit. A family without the channel is null for that reason first.
    state_families = {
        "continuous_camping",
        "distributional_shift",
        "goal_misgeneralization",
        "gridworld_camping",
        "physics_exploitation",
        "reward_channel_tampering",
        "rlhf_reward_model_overopt",
        "sensor_calibration_tampering",
    }
    needed = _declared_minimum()
    assert set(v5_results) <= set(needed)
    short: dict[str, list[str]] = {}
    for name, info in v5_results.items():
        families = sorted(
            f
            for f in info["per_family"]
            if horizon[f] < needed[name]
            and (info["access_level"] != "L1" or f in state_families)
        )
        short[name] = families
        for f in families:
            assert info["per_family"][f] is None, (name, f)
            assert f in info["not_applicable_families"], (name, f)
        reasons = info["not_applicable_reasons"]
        assert sum(
            k for r, k in reasons.items() if r.startswith("horizon ")
        ) == sum(cells[f] for f in families), name

    level = {name: info["access_level"] for name, info in v5_results.items()}
    by_level = {
        lv: sum(len(f) for n, f in short.items() if level[n] == lv and n != "Reward Skewness")
        for lv in ACCESS_LEVELS
    }
    # Reward Skewness's 19 have been null since 2026-09-03 and are not counted. At L1,
    # continuous_camping (40 episodes) is the only state_counts family shorter than an L1
    # detector needs, so each of the 8 is null there and measured on the other 7.
    assert by_level == {"L0": 104, "L1": 8, "L2": 33, "L3": 0}
    for name in (n for n in v5_results if level[n] == "L1"):
        assert short[name] == ["continuous_camping"], name
        assert v5_results[name]["cells_measured"] == 31, name
    assert v5_results["Reward Peak"]["cells_measured"] == 123 - 59
    assert v5_results["Reward CUSUM"]["cells_measured"] == 123 - 22


def test_the_exclusion_applies_to_l0_and_only_to_l0(v5_results):
    """Degeneracy is a property of the proxy reward, which only L0 reads.

    L1/L2/L3 read state visitation, behavior and true reward, not the proxy, so nothing
    they read is degenerate, and dropping those families there would throw away real
    measurements. (L1 is not applicable on the 25 families that emit no
    ``state_counts``; those cells are ``null`` on the board, which is a different thing
    and is handled per detector.) If this ever starts failing, the exclusion has leaked
    into rungs that do not read the proxy.
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
    assert "| L2 | 7 | 0.5771 | 0.1644 | 0.9440 | Trajectory MLP |" in md
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
    assert "| L0 | 13 | 0.4835 |" in md
    assert "L0 is computed over 27 families" in md
    assert (
        "Gradient Reversal, Max Plateau, Reward CUSUM and Reward MLP over 20 of them; "
        "Reward Autocorrelation, Reward Peak, Reward Skewness, Reward Trend and Variance "
        "Window over 8 of them: the other families run fewer episodes than the detector "
        "declares it needs"
    ) in md
    assert "6 are held out" in md
    for family in DEGENERATE_FAMILIES:
        assert f"**{family}**" in md
    assert "Putting them back would read 0.4871 instead of 0.4835" in md
    assert "held out of L0 only" in md


def test_markdown_has_no_fallback_left_to_name(v5_results):
    """Until 2026-11-26 the table named eight L0 detectors whose short-run 0.5 was a
    fallback. They declare their horizons now, so there is nothing to name."""
    for degenerate in ([], DEGENERATE_FAMILIES):
        md = render_access_summary_md(
            summarize_access_levels(v5_results, degenerate_families=degenerate)
        )
        assert "fallback" not in md


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
    L0 row (cell-weighted, on the board before the 2026-11-19 regeneration), and the docs
    quoted it.

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
    assert "| L0 | 13 | 0.4835 | 0.0328 | 0.5474 | Reward CUSUM |" in published
