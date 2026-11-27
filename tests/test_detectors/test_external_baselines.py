"""Tests for classical external-baseline detectors (Phase 5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import rhob.detectors.external_baselines.bocpd as bocpd_module
from rhob.detectors import (
    ARResidualDetector,
    BOCPDDetector,
    IsolationForestDetector,
    PageHinkleyDetector,
    PCAReconstructionDetector,
    RunData,
)
from rhob.v3.benchmark import _onset_mae

_HVTA = Path(__file__).resolve().parents[2] / "external" / "hvta"
_BOCPD = "Bayesian Online Changepoint Detection"


def _behav_run(late_mean: float, n_episodes: int = 150, noise: float = 0.05, seed: int = 0) -> RunData:
    rng = np.random.default_rng(seed)
    half = n_episodes // 2
    trace = np.concatenate(
        [rng.normal(0.0, noise, half), rng.normal(late_mean, noise, n_episodes - half)]
    )
    return RunData(
        proxy_rewards=np.ones(n_episodes),
        true_rewards=np.ones(n_episodes),
        state_counts=None,
        behav_trace=trace,
    )


def _hist_run(bin_hi: int, n_episodes: int = 150, n_bins: int = 10, seed: int = 0) -> RunData:
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 3, size=(n_episodes, n_bins)).astype(np.int64)
    counts[:, bin_hi] += 20
    return RunData(
        proxy_rewards=np.ones(n_episodes),
        true_rewards=np.ones(n_episodes),
        state_counts=counts,
        behav_trace=None,
    )


def _reward_run(shift_at: int | None, n_episodes: int = 150, seed: int = 0) -> RunData:
    """A reward stream that shifts mean at ``shift_at``, or stays flat if None."""
    rng = np.random.default_rng(seed)
    if shift_at is None or shift_at >= n_episodes:
        rewards = rng.normal(0.0, 0.05, n_episodes)
    else:
        rewards = np.concatenate(
            [rng.normal(0.0, 0.05, shift_at), rng.normal(2.0, 0.05, n_episodes - shift_at)]
        )
    return RunData(
        proxy_rewards=rewards,
        true_rewards=rewards,
        state_counts=None,
        behav_trace=None,
    )


class TestPageHinkley:
    def test_flags_shift(self):
        detector = PageHinkleyDetector()
        shifted = _reward_run(shift_at=75)
        stable = _reward_run(shift_at=None)  # no shift within run
        assert detector.classify(shifted) > detector.classify(stable)

    def test_onset_near_shift(self):
        detector = PageHinkleyDetector()
        run = _reward_run(shift_at=75)
        onset = detector.detect_onset(run)
        assert onset == -1 or 60 < onset < 100

    def test_short_run_neutral(self):
        detector = PageHinkleyDetector()
        run = RunData(np.ones(2), np.ones(2), None, None)
        assert detector.classify(run) == 0.5
        assert detector.detect_onset(run) == -1


class TestIsolationForest:
    def test_untrained_returns_neutral(self):
        detector = IsolationForestDetector()
        run = _behav_run(late_mean=1.0)
        assert detector.classify(run) == 0.5

    def test_fit_flags_anomalous_trace(self):
        detector = IsolationForestDetector(window=50)
        runs_b = [_behav_run(late_mean=0.0, seed=s) for s in range(20)]  # legitimate: no shift
        runs_a = [_behav_run(late_mean=3.0, seed=100 + s) for s in range(20)]  # hacking: shifted
        detector.fit(runs_a, runs_b)
        assert detector.is_trained

        score_hacking = detector.classify(_behav_run(late_mean=3.0, seed=999))
        score_legit = detector.classify(_behav_run(late_mean=0.0, seed=998))
        assert score_hacking > score_legit


class TestARResidual:
    def test_untrained_returns_neutral(self):
        detector = ARResidualDetector()
        run = _behav_run(late_mean=1.0)
        assert detector.classify(run) == 0.5

    def test_fit_flags_deviation_from_learned_dynamics(self):
        detector = ARResidualDetector(order=3, window=50)
        runs_b = [_behav_run(late_mean=0.0, seed=s) for s in range(20)]
        runs_a = [_behav_run(late_mean=3.0, seed=100 + s) for s in range(20)]
        detector.fit(runs_a, runs_b)
        assert detector.is_trained

        score_hacking = detector.classify(_behav_run(late_mean=3.0, seed=999))
        score_legit = detector.classify(_behav_run(late_mean=0.0, seed=998))
        assert score_hacking > score_legit


class TestPCAReconstruction:
    def test_untrained_returns_neutral(self):
        detector = PCAReconstructionDetector()
        run = _hist_run(bin_hi=3)
        assert detector.classify(run) == 0.5

    def test_fit_flags_novel_occupancy_pattern(self):
        detector = PCAReconstructionDetector(n_components=2, steady_window=50)
        runs_b = [_hist_run(bin_hi=2, seed=s) for s in range(10)]  # legitimate: bin 2
        runs_a = [_hist_run(bin_hi=8, seed=100 + s) for s in range(10)]  # hacking: bin 8
        detector.fit(runs_a, runs_b)
        assert detector.is_trained

        score_hacking = detector.classify(_hist_run(bin_hi=8, seed=999))
        score_legit = detector.classify(_hist_run(bin_hi=2, seed=998))
        assert score_hacking > score_legit


class TestBOCPD:
    """The score is ``1 - P(no changepoint over the whole run)``, as the class documents.

    Until 2026-11-27 it was ``max_t P(r_t = 0)``, which a constant hazard pins to the
    hazard at every step: every run scored 0.004 (to within 4e-18), and the onset rule
    could never fire. These tests are strict where the old ones were not.
    """

    def test_flags_shift(self):
        detector = BOCPDDetector()
        shifted = detector.classify(_reward_run(shift_at=75))
        stable = detector.classify(_reward_run(shift_at=None))
        assert shifted > stable
        assert shifted > 0.99
        assert stable < 0.05

    @pytest.mark.parametrize("shift_at", [30, 75, 120])
    def test_onset_near_shift(self, shift_at):
        detector = BOCPDDetector()
        onset = detector.detect_onset(_reward_run(shift_at=shift_at))
        assert abs(onset - shift_at) <= 3, onset

    def test_no_onset_on_a_stable_run(self):
        detector = BOCPDDetector()
        assert detector.detect_onset(_reward_run(shift_at=None)) == -1

    def test_score_varies_across_runs_of_one_length(self):
        """Not the hazard, and not one constant for every run."""
        detector = BOCPDDetector()
        scores = [detector.classify(_reward_run(shift_at=None, seed=s)) for s in range(8)]
        assert len({round(x, 12) for x in scores}) == len(scores)
        assert all(abs(x - detector.hazard) > 1e-6 for x in scores)

    def test_short_run_neutral(self):
        detector = BOCPDDetector()
        for n in (2, detector.min_episodes - 1):
            run = RunData(np.ones(n), np.ones(n), None, None)
            assert detector.classify(run) == 0.5
            assert detector.detect_onset(run) == -1

    def test_scores_bounded(self):
        detector = BOCPDDetector()
        for run in (_reward_run(shift_at=75), _reward_run(shift_at=None)):
            assert 0.0 <= detector.classify(run) <= 1.0

    @pytest.mark.parametrize("hazard", [0.0, 1.0, -0.1, 1.5])
    def test_hazard_must_be_a_probability_strictly_inside_0_1(self, hazard):
        with pytest.raises(ValueError):
            BOCPDDetector(hazard=hazard)

    def test_a_stable_run_keeps_its_full_length_hypothesis_through_truncation(
        self, monkeypatch
    ):
        """700 episodes, past the 500 run lengths the recursion keeps.

        The full-length hypothesis is the most probable one on a stable run, so it is
        never truncated, and the score matches the untruncated recursion.
        """
        detector = BOCPDDetector()
        rewards = np.random.default_rng(3).normal(0.0, 0.05, 700)
        log_no_change = detector._log_no_change_probabilities(rewards)
        assert np.all(np.isfinite(log_no_change))
        score = detector.classify(RunData(rewards, rewards, None, None))
        assert score < 0.05

        monkeypatch.setattr(bocpd_module, "MAX_RUN_LENGTHS", 10**6)
        untruncated = detector.classify(RunData(rewards, rewards, None, None))
        assert score == pytest.approx(untruncated, abs=1e-9)

    def test_a_dropped_full_length_hypothesis_stays_dropped(self, monkeypatch):
        """Once truncation drops the segment that began at the first reward, no change is 0.

        Three rewards near the start sit 3.0 above the rest, so every hypothesis that
        holds one of them is less probable than any other, and truncation drops them
        oldest first from step 499. Reading whichever hypothesis is last after that would
        report the clean segment that began after them, i.e. no changepoint (about 0.03);
        the run did change, and the untruncated recursion agrees.
        """
        detector = BOCPDDetector()
        rewards = np.random.default_rng(0).normal(0.0, 0.05, 600)
        rewards[1:4] += 3.0
        run = RunData(rewards, rewards, None, None)
        log_no_change = detector._log_no_change_probabilities(rewards)

        dropped = np.flatnonzero(~np.isfinite(log_no_change))
        assert len(dropped) > 0
        first = int(dropped[0])
        assert first >= bocpd_module.MAX_RUN_LENGTHS - 1  # nothing is truncated earlier
        assert np.all(np.isneginf(log_no_change[first:]))  # sticky
        assert detector.classify(run) == 1.0
        assert 0 < detector.detect_onset(run) < 20  # found on the exact posterior

        monkeypatch.setattr(bocpd_module, "MAX_RUN_LENGTHS", 10**6)
        reference = detector._log_no_change_probabilities(rewards)
        assert np.array_equal(reference[:first], log_no_change[:first])
        assert reference[-1] < -100.0
        assert detector.classify(run) == 1.0


def _bocpd_cell(runs_a, runs_b, onsets_a, n_episodes):
    """BOCPD's AUROC, onset MAE and hacking-run onsets on one cell, as ``_evaluate_cell``
    computes them for an unfitted detector, with the recursion run once per run."""
    detector = BOCPDDetector()
    changed = [detector._changed_probabilities(np.asarray(r.proxy_rewards)) for r in runs_a + runs_b]
    scores = [float(c[-1]) for c in changed]
    onsets = [int(np.flatnonzero(c > 0.5)[0]) if np.any(c > 0.5) else -1 for c in changed]
    # The shortcut is the class's own classify / detect_onset, not a re-implementation.
    assert scores[0] == detector.classify(runs_a[0])
    assert onsets[0] == detector.detect_onset(runs_a[0])
    labels = [1] * len(runs_a) + [0] * len(runs_b)
    auroc = float(roc_auc_score(labels, scores))
    onsets_a_pred = onsets[: len(runs_a)]
    return auroc, _onset_mae(onsets_a_pred, onsets_a, n_episodes), onsets_a_pred


class TestBOCPDOnHVTA:
    """The committed HVTA artifacts' BOCPD rows are what this ``bocpd.py`` computes.

    They were re-scored on 2026-11-27 after the fix above
    (``external/hvta/rescore_detector.py`` for the primary pairs draw, a full
    ``score_channels.py`` run for the channels). A later change to the detector that moves
    its scores has to re-score them too, or this fails.
    """

    def test_the_pairs_draw(self):
        meta = json.loads((_HVTA / "hvta_pairs.meta.json").read_text(encoding="utf-8"))
        arrays = np.load(_HVTA / "hvta_pairs.npz")
        artifact = json.loads((_HVTA / "hvta_results.json").read_text(encoding="utf-8"))
        assert any(_BOCPD in rec["detectors"] for rec in artifact.get("rescored", []))
        for cell in meta["cells"]:
            i = cell["index"]

            def runs(tag, i=i):
                proxy = arrays[f"c{i}_{tag}_proxy_rewards"]
                return [RunData(proxy[k], proxy[k], None, None) for k in range(proxy.shape[0])]

            onsets_a = [int(x) for x in arrays[f"c{i}_onsets_a"]]
            auroc, mae, onsets = _bocpd_cell(runs("a"), runs("b"), onsets_a, cell["n_steps"])
            row = next(r for r in artifact["per_cell"][str(i)] if r["detector"] == _BOCPD)
            assert auroc == pytest.approx(row["auroc"], abs=1e-12), i
            assert mae == pytest.approx(row["onset_mae_normalized"], abs=1e-12), i
            assert onsets == row["onsets_pred_hacking"], i

    def test_the_other_three_channels(self):
        meta = json.loads((_HVTA / "channels_pairs.meta.json").read_text(encoding="utf-8"))
        arrays = np.load(_HVTA / "channels_pairs.npz")
        artifact = json.loads((_HVTA / "channels_results.json").read_text(encoding="utf-8"))
        scored = 0
        for cell, result in zip(meta["cells"], artifact["cells"]):
            if not cell["n_pairs_usable"]:
                continue
            i, n = cell["index"], cell["n_pairs_usable"]

            def runs(tag, i=i, n=n):
                return [
                    RunData(arrays[f"c{i}_{tag}{j}_proxy"], arrays[f"c{i}_{tag}{j}_proxy"],
                            None, None)
                    for j in range(n)
                ]

            auroc, mae, _ = _bocpd_cell(runs("a"), runs("b"), cell["onsets"], cell["n_episodes"])
            row = next(r for r in result["detectors"] if r["detector"] == _BOCPD)
            assert round(auroc, 4) == row["auroc"], (cell["channel"], cell["params"])
            assert round(mae, 4) == row["onset_mae_normalized"], (cell["channel"], cell["params"])
            scored += 1
        assert scored == 12
