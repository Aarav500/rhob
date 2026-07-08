"""Tests for state-visitation divergence detector (L1)."""

from __future__ import annotations

import numpy as np

from rhob.detectors import RunData, StateDivergenceDetector


def _make_run(bin_a: int, bin_b: int, n_episodes: int = 100, n_bins: int = 10) -> RunData:
    """A run that visits ``bin_a`` pre-onset (first half) and ``bin_b`` post-onset."""
    counts = np.zeros((n_episodes, n_bins), dtype=np.int64)
    half = n_episodes // 2
    counts[:half, bin_a] = 20
    counts[half:, bin_b] = 20
    return RunData(
        proxy_rewards=np.ones(n_episodes),
        true_rewards=np.ones(n_episodes),
        state_counts=counts,
        behav_trace=None,
    )


def test_untrained_detector_returns_neutral():
    detector = StateDivergenceDetector(baseline_episodes=20)
    run = _make_run(bin_a=2, bin_b=8)
    assert detector.classify(run) == 0.5


def test_fit_and_classify_separates_variants():
    """After fitting on labeled runs, a new A-like run should score high, B-like low."""
    detector = StateDivergenceDetector(baseline_episodes=20, steady_window=50)
    runs_a = [_make_run(bin_a=2, bin_b=8) for _ in range(5)]  # switches to bin 8
    runs_b = [_make_run(bin_a=2, bin_b=1) for _ in range(5)]  # switches to bin 1
    detector.fit(runs_a, runs_b)
    assert detector.is_trained

    score_a = detector.classify(_make_run(bin_a=2, bin_b=8))
    score_b = detector.classify(_make_run(bin_a=2, bin_b=1))
    assert score_a > 0.5 > score_b


def test_onset_detection_near_switch():
    """Onset detection is oracle-free and doesn't require fit()."""
    detector = StateDivergenceDetector(baseline_episodes=20)
    run = _make_run(bin_a=2, bin_b=8, n_episodes=100)
    onset = detector.detect_onset(run)
    assert 45 < onset < 55  # near episode 50 (the switch point)


def test_missing_state_counts_returns_neutral():
    detector = StateDivergenceDetector()
    run = RunData(np.ones(10), np.ones(10), None, None)
    assert detector.classify(run) == 0.5
    assert detector.detect_onset(run) == -1


def test_short_run_returns_neutral():
    detector = StateDivergenceDetector(baseline_episodes=50)
    run = _make_run(bin_a=2, bin_b=8, n_episodes=10)
    assert detector.classify(run) == 0.5
    assert detector.detect_onset(run) == -1
