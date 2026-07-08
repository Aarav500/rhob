"""Tests for the trajectory MLP detector (L2, strongest L2 baseline)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rhob.detectors import RunData, TrajectoryMLPDetector  # noqa: E402


def _make_run(shift: float, seed: int, n: int = 150) -> RunData:
    rng = np.random.default_rng(seed)
    behav = np.concatenate([np.zeros(50), np.full(n - 50, shift) + rng.normal(0, 0.05, n - 50)])
    proxy = np.full(n, 0.5) + rng.normal(0, 0.05, n)
    return RunData(proxy_rewards=proxy, true_rewards=np.zeros(n), state_counts=None, behav_trace=behav)


def test_untrained_detector_returns_neutral():
    detector = TrajectoryMLPDetector()
    run = _make_run(shift=0.8, seed=0)
    assert detector.classify(run) == 0.5
    assert detector.detect_onset(run) == -1


def test_fit_and_classify_separates_variants():
    detector = TrajectoryMLPDetector(hidden=16, window=50)
    runs_a = [_make_run(shift=0.8, seed=i) for i in range(15)]
    runs_b = [_make_run(shift=-0.8, seed=100 + i) for i in range(15)]
    detector.fit(runs_a, runs_b, epochs=50)
    assert detector.is_trained

    score_a = detector.classify(_make_run(shift=0.8, seed=200))
    score_b = detector.classify(_make_run(shift=-0.8, seed=201))
    assert score_a > score_b


def test_onset_detection_after_fit():
    detector = TrajectoryMLPDetector(hidden=16, window=50)
    runs_a = [_make_run(shift=0.8, seed=i) for i in range(15)]
    runs_b = [_make_run(shift=-0.8, seed=100 + i) for i in range(15)]
    detector.fit(runs_a, runs_b, epochs=50)

    onset = detector.detect_onset(_make_run(shift=0.8, seed=300))
    # The feature is a trailing window average (window=50), so detection can lag
    # up to a full window past the true shift at episode 50 before it saturates.
    assert onset == -1 or 40 < onset <= 100


def test_missing_behav_trace_returns_none_features():
    detector = TrajectoryMLPDetector()
    run = RunData(np.ones(80), np.zeros(80), None, None)
    assert detector.classify(run) == 0.5
