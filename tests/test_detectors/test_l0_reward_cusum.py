"""Tests for CUSUM detector (L0)."""

from __future__ import annotations

import numpy as np

from rhob.detectors import RewardCUSUMDetector, RunData


def test_cusum_detector_creates():
    """Test detector instantiation."""
    detector = RewardCUSUMDetector()
    assert detector.access_level == "L0"
    assert detector.name == "Reward CUSUM"


def test_cusum_detects_jump():
    """Test CUSUM detects an abrupt change in mean amid realistic noise."""
    detector = RewardCUSUMDetector(drift_std=0.5, threshold_std=5.0, baseline_window=50)
    rng = np.random.default_rng(0)
    rewards = np.concatenate(
        [
            0.2 + rng.normal(0, 0.05, 50),  # noisy baseline
            0.9 + rng.normal(0, 0.05, 100),  # jump
        ]
    )
    run = RunData(
        proxy_rewards=rewards,
        true_rewards=np.zeros(150),
        state_counts=None,
        behav_trace=None,
    )
    onset = detector.detect_onset(run)
    assert 48 < onset < 85  # detected within a reasonable window of the jump


def test_cusum_no_jump():
    """Test CUSUM returns -1 when no change is detected (high threshold)."""
    detector = RewardCUSUMDetector(threshold_std=1000.0)
    run = RunData(
        proxy_rewards=np.ones(150) * 0.5,
        true_rewards=np.zeros(150),
        state_counts=None,
        behav_trace=None,
    )
    assert detector.detect_onset(run) == -1


def test_cusum_short_trajectory():
    """Test CUSUM on a trajectory shorter than the baseline window."""
    detector = RewardCUSUMDetector(baseline_window=50)
    run = RunData(
        proxy_rewards=np.ones(30) * 0.5,
        true_rewards=np.zeros(30),
        state_counts=None,
        behav_trace=None,
    )
    assert detector.classify(run) == 0.0
    assert detector.detect_onset(run) == -1


def test_cusum_scale_invariant():
    """Detector must behave identically regardless of the proxy's absolute scale."""
    detector = RewardCUSUMDetector(baseline_window=50)
    rng = np.random.default_rng(1)
    small = np.concatenate([0.2 + rng.normal(0, 0.05, 50), 0.9 + rng.normal(0, 0.05, 100)])
    large = small * 200.0
    run_small = RunData(small, np.zeros(150), None, None)
    run_large = RunData(large, np.zeros(150), None, None)
    assert detector.detect_onset(run_small) == detector.detect_onset(run_large)
