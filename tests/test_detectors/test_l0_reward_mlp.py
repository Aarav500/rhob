"""Tests for the reward MLP detector (L0, strongest L0 baseline)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rhob.detectors import RewardMLPDetector, RunData  # noqa: E402


def _make_runs(n: int, shift: float, seed: int) -> list[RunData]:
    rng = np.random.default_rng(seed)
    return [
        RunData(np.full(80, shift) + rng.normal(0, 0.05, 80), np.zeros(80), None, None)
        for _ in range(n)
    ]


def test_untrained_detector_returns_neutral():
    detector = RewardMLPDetector(window_size=50)
    run = RunData(np.ones(80) * 0.5, np.zeros(80), None, None)
    assert detector.classify(run) == 0.5
    assert detector.detect_onset(run) == -1


def test_fit_and_classify_separable_data():
    """A trivially separable reward-history difference should be learnable."""
    detector = RewardMLPDetector(window_size=50, hidden=16)
    runs_a = _make_runs(20, shift=0.8, seed=0)
    runs_b = _make_runs(20, shift=0.2, seed=1)
    detector.fit(runs_a, runs_b, epochs=50)
    assert detector.is_trained

    score_a = detector.classify(runs_a[0])
    score_b = detector.classify(runs_b[0])
    assert score_a > score_b


def test_short_run_returns_neutral_after_fit():
    detector = RewardMLPDetector(window_size=50, hidden=16)
    runs_a = _make_runs(20, shift=0.8, seed=0)
    runs_b = _make_runs(20, shift=0.2, seed=1)
    detector.fit(runs_a, runs_b, epochs=10)
    short_run = RunData(np.ones(10) * 0.5, np.zeros(10), None, None)
    assert detector.classify(short_run) == 0.5
