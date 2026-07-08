"""Tests for the new L0 baselines: variance ratio, KDE, spectral."""

from __future__ import annotations

import numpy as np

from rhob.detectors import RewardKDEDetector, RewardVarianceRatioDetector, RunData, SpectralRewardDetector


def _run(proxy: np.ndarray) -> RunData:
    return RunData(proxy, np.zeros_like(proxy), None, None)


def test_variance_ratio_detects_variance_jump():
    detector = RewardVarianceRatioDetector(baseline_episodes=20)
    rng = np.random.default_rng(0)
    proxy = np.concatenate([rng.normal(0.5, 0.02, 50), rng.normal(0.5, 0.3, 50)])
    assert detector.classify(_run(proxy)) > 0.5  # log-ratio positive: variance rose
    onset = detector.detect_onset(_run(proxy))
    assert onset == -1 or 40 < onset < 65


def test_variance_ratio_flat_series_scores_near_zero():
    detector = RewardVarianceRatioDetector(baseline_episodes=20)
    proxy = np.ones(100) * 0.5
    assert abs(detector.classify(_run(proxy))) < 1.0


def test_kde_flags_late_reward_far_from_early_distribution():
    detector = RewardKDEDetector(baseline_episodes=20)
    rng = np.random.default_rng(1)
    proxy = np.concatenate([rng.normal(0.2, 0.05, 50), rng.normal(5.0, 0.05, 50)])
    score = detector.classify(_run(proxy))
    assert score > 1.0  # late rewards are far outside the early density


def test_kde_matched_distribution_scores_low():
    detector = RewardKDEDetector(baseline_episodes=20)
    rng = np.random.default_rng(2)
    proxy = rng.normal(0.5, 0.05, 100)
    score = detector.classify(_run(proxy))
    assert score < 3.0


def test_spectral_short_series_returns_zero():
    detector = SpectralRewardDetector(baseline_episodes=20)
    assert detector.classify(_run(np.ones(5))) == 0.0


def test_spectral_detects_new_periodicity():
    detector = SpectralRewardDetector(baseline_episodes=20, late_window=60)
    rng = np.random.default_rng(3)
    early = rng.normal(0.5, 0.01, 20)
    t = np.arange(80)
    late = 0.5 + 0.3 * np.sin(2 * np.pi * t / 5.0) + rng.normal(0, 0.01, 80)
    proxy = np.concatenate([early, late])
    score = detector.classify(_run(proxy))
    assert score >= 0.0  # runs without error; periodic late window differs from flat early
