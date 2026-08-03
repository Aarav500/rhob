"""A detector's missing input channel must produce N/A, not a fabricated 0.5.

The audit found that 8 L1 detectors were being scored on the 25 of 33 families that
ship ``state_counts=None`` -- 88 of the 123 leaderboard cells. Every one of them returns
a hardcoded 0.5 in that case, so the published "L1 access level" number was 71% constant
averaged in as if it were a measurement. These tests pin the corrected behaviour: an
absent channel is excluded from every aggregate, and is distinguishable in the artifact
from a real 0.5.

Also covers the provenance block that every result JSON now carries.
"""

from __future__ import annotations

import numpy as np
import pytest

from rhob.detectors.posthoc import PosthocDetector, RunData
from rhob.v3.benchmark import (
    BenchmarkResults,
    CellResult,
    _evaluate_cell,
    missing_channels,
    required_channels,
)
from rhob.v3.provenance import _git, git_provenance, provenance_block, sampling_block


class _FakeL1(PosthocDetector):
    """An L1 detector with the same 0.5 no-signal fallback every real one has."""

    @property
    def access_level(self) -> str:
        return "L1"

    @property
    def name(self) -> str:
        return "Fake L1"

    def classify(self, run: RunData) -> float:
        if run.state_counts is None:
            return 0.5
        return float(run.state_counts[-1].sum())

    def detect_onset(self, run: RunData) -> int:
        return -1


def _run(state_counts) -> RunData:
    return RunData(
        proxy_rewards=np.linspace(0.0, 1.0, 20),
        true_rewards=np.linspace(0.0, 1.0, 20),
        state_counts=state_counts,
        behav_trace=np.zeros(20),
    )


def test_required_channels_defaults_to_the_access_level_signal():
    assert required_channels(_FakeL1()) == ("state_counts",)


def test_required_channels_respects_a_detector_declared_override():
    detector = _FakeL1()
    detector.required_channels = ("behav_trace",)
    assert required_channels(detector) == ("behav_trace",)


def test_missing_channels_flags_none_and_empty():
    present = [_run(np.ones((20, 3)))]
    absent = [_run(None)]
    empty = [_run(np.array([]))]
    assert missing_channels(present, ("state_counts",)) == []
    assert missing_channels(absent, ("state_counts",)) == ["state_counts"]
    assert missing_channels(empty, ("state_counts",)) == ["state_counts"]


def test_absent_channel_yields_nan_cell_with_a_reason_not_a_half():
    runs_a = [_run(None) for _ in range(4)]
    runs_b = [_run(None) for _ in range(4)]
    auroc, mae, reason = _evaluate_cell(_FakeL1(), runs_a, runs_b, [5] * 4, "L1", 20)
    assert np.isnan(auroc)
    assert np.isnan(mae)
    assert reason is not None and "state_counts" in reason


def test_present_channel_is_scored_normally():
    runs_a = [_run(np.ones((20, 3)) * 2) for _ in range(4)]
    runs_b = [_run(np.ones((20, 3))) for _ in range(4)]
    auroc, _, reason = _evaluate_cell(_FakeL1(), runs_a, runs_b, [5] * 4, "L1", 20)
    assert reason is None
    assert auroc == pytest.approx(1.0)


def test_na_cells_are_excluded_from_overall_auroc_not_averaged_as_chance():
    """The whole point: a 0.9 measurement plus an N/A must not report 0.7."""
    results = BenchmarkResults(
        detector_name="Fake L1",
        access_level="L1",
        cells=[
            CellResult("fam_with_counts", "camping", 0.9, 0.9, 0.1, 5),
            CellResult(
                "fam_without_counts", "camping", 0.9, float("nan"), float("nan"), 5,
                na_reason="family does not provide state_counts",
            ),
        ],
    )
    assert results.overall_auroc == pytest.approx(0.9)
    assert [c.family for c in results.scored_cells] == ["fam_with_counts"]
    assert results.na_families == ["fam_without_counts"]
    assert len(results.na_cells) == 1


def test_overall_auroc_is_nan_when_nothing_was_measurable():
    results = BenchmarkResults(
        detector_name="Fake L1",
        access_level="L1",
        cells=[
            CellResult(
                "fam", "camping", 0.9, float("nan"), float("nan"), 5,
                na_reason="family does not provide state_counts",
            )
        ],
    )
    assert np.isnan(results.overall_auroc)
    assert results.scored_cells == []


def test_summary_renders_na_rather_than_a_number():
    results = BenchmarkResults(
        detector_name="Fake L1",
        access_level="L1",
        cells=[
            CellResult(
                "fam", "camping", 0.9, float("nan"), float("nan"), 5,
                na_reason="family does not provide state_counts",
            )
        ],
    )
    text = results.summary()
    assert "N/A" in text
    assert "0.500" not in text


def test_provenance_block_carries_commit_versions_and_timestamp():
    block = provenance_block(script="test")
    assert block["script"] == "test"
    assert block["generated_utc"].endswith("Z")
    assert block["python"].count(".") == 2
    assert set(block["packages"]) >= {"numpy", "scipy", "scikit-learn", "rhob"}
    assert block["packages"]["numpy"] is not None
    # git may legitimately be unavailable (sdist install); the keys must exist either way.
    assert set(block["git"]) == {"commit", "short_commit", "branch", "dirty", "dirty_files"}


def test_dirty_file_paths_are_not_truncated():
    """Regression: ``git status --porcelain`` starts an unstaged line with a space.

    Stripping the whole stdout ate that space and therefore the first character of the
    first path, silently writing ``ocs/figures/...`` into the provenance block.
    """
    info = git_provenance()
    if not info["dirty_files"]:
        pytest.skip("clean working tree: nothing to check")
    raw = _git("status", "--porcelain")
    assert raw is not None
    # The invariant that the bug broke: every porcelain line keeps both status columns,
    # so column 2 is always the separating space. Under the old whole-output ``.strip()``
    # the first line arrived as "M docs/..." instead of " M docs/...", column 2 was "d",
    # and slicing [3:] produced "ocs/figures/...".
    for line in raw.splitlines():
        assert line[2] == " ", f"porcelain line lost a leading column: {line!r}"
    assert all(not path.startswith(" ") for path in info["dirty_files"])


def test_sampling_block_marks_a_single_draw_as_such():
    block = sampling_block(
        n_seeds=5,
        n_layouts=1,
        layout_seeds=[0],
        rollout_seeds_hacking=[0, 1, 2, 3, 4],
        rollout_seeds_legit=[1000, 1001, 1002, 1003, 1004],
        n_replicates=1,
    )
    assert block["single_draw"] is True
    assert block["confidence_intervals"] is None
    assert block["n_seeds_per_variant"] == 5
    assert block["rollout_seeds_legit"] == [1000, 1001, 1002, 1003, 1004]
    assert "Single unreplicated draw" in block["note"]


def test_sampling_block_replicated_run_is_not_marked_single_draw():
    block = sampling_block(
        n_seeds=10,
        n_layouts=1,
        layout_seeds=[0],
        rollout_seeds_hacking=list(range(10)),
        rollout_seeds_legit=[1000 + s for s in range(10)],
        n_replicates=5,
    )
    assert block["single_draw"] is False
    assert block["note"] is None
