"""Tests for classical external-baseline detectors (Phase 5)."""

from __future__ import annotations

import numpy as np

from rhob.detectors import (
    ARResidualDetector,
    BOCPDDetector,
    IsolationForestDetector,
    PageHinkleyDetector,
    PCAReconstructionDetector,
    RunData,
)


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
    def test_flags_shift(self):
        detector = BOCPDDetector()
        shifted = _reward_run(shift_at=75)
        stable = _reward_run(shift_at=None)
        assert detector.classify(shifted) >= detector.classify(stable)

    def test_short_run_neutral(self):
        detector = BOCPDDetector()
        run = RunData(np.ones(2), np.ones(2), None, None)
        assert detector.classify(run) == 0.5
        assert detector.detect_onset(run) == -1

    def test_scores_bounded(self):
        detector = BOCPDDetector()
        run = _reward_run(shift_at=75)
        score = detector.classify(run)
        assert 0.0 <= score <= 1.0
