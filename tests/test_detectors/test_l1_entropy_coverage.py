"""Tests for the new L1 baselines: visitation entropy trend, state coverage rate."""

from __future__ import annotations

import numpy as np

from rhob.detectors import RunData, StateCoverageRateDetector, VisitationEntropyTrendDetector


def _spread_run(n_episodes: int, n_bins: int, spread_after: int) -> RunData:
    """Visits one bin pre-spread, then spreads across many bins post-spread."""
    counts = np.zeros((n_episodes, n_bins), dtype=np.int64)
    counts[:spread_after, 0] = 20
    rng = np.random.default_rng(0)
    for t in range(spread_after, n_episodes):
        bins = rng.integers(0, n_bins, size=20)
        for b in bins:
            counts[t, b] += 1
    return RunData(np.ones(n_episodes), np.ones(n_episodes), counts, None)


def _focused_run(n_episodes: int, n_bins: int, bin_idx: int) -> RunData:
    counts = np.zeros((n_episodes, n_bins), dtype=np.int64)
    counts[:, bin_idx] = 20
    return RunData(np.ones(n_episodes), np.ones(n_episodes), counts, None)


def test_entropy_trend_fit_and_classify():
    detector = VisitationEntropyTrendDetector(baseline_episodes=20, late_window=30)
    runs_a = [_spread_run(80, 10, spread_after=20) for _ in range(5)]  # A: spreads out
    runs_b = [_focused_run(80, 10, bin_idx=3) for _ in range(5)]       # B: stays focused
    detector.fit(runs_a, runs_b)
    assert detector.is_trained

    score_a = detector.classify(_spread_run(80, 10, spread_after=20))
    score_b = detector.classify(_focused_run(80, 10, bin_idx=3))
    assert score_a > score_b


def test_entropy_trend_untrained_returns_neutral():
    detector = VisitationEntropyTrendDetector()
    run = _focused_run(80, 10, bin_idx=0)
    assert detector.classify(run) == 0.5


def test_entropy_trend_onset_detection():
    detector = VisitationEntropyTrendDetector(baseline_episodes=20)
    run = _spread_run(80, 10, spread_after=20)
    onset = detector.detect_onset(run)
    assert 15 < onset < 30


def test_coverage_rate_fit_and_classify():
    # Many bins (slow coupon-collector) so the "spread" run is still discovering new
    # bins inside the late window, rather than saturating within a couple of episodes.
    n_bins = 500
    detector = StateCoverageRateDetector(baseline_episodes=20, late_window=30)
    runs_a = [_spread_run(80, n_bins, spread_after=20) for _ in range(5)]  # keeps discovering
    runs_b = [_focused_run(80, n_bins, bin_idx=3) for _ in range(5)]       # discovers nothing new
    detector.fit(runs_a, runs_b)
    assert detector.is_trained

    score_a = detector.classify(_spread_run(80, n_bins, spread_after=20))
    score_b = detector.classify(_focused_run(80, n_bins, bin_idx=3))
    assert score_a > score_b


def test_coverage_rate_missing_counts_returns_neutral():
    detector = StateCoverageRateDetector()
    run = RunData(np.ones(10), np.ones(10), None, None)
    assert detector.classify(run) == 0.5
    assert detector.detect_onset(run) == -1
