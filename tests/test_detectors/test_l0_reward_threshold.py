"""Tests for reward threshold detector (L0)."""

from __future__ import annotations

import numpy as np

from rhob.detectors import RewardThresholdDetector, RunData


def test_threshold_classifier():
    """Test classification by late-training mean reward (raw score, rank-order only)."""
    detector = RewardThresholdDetector(late_window=50)
    run = RunData(
        proxy_rewards=np.concatenate([np.zeros(100), np.ones(100)]),
        true_rewards=np.zeros(200),
        state_counts=None,
        behav_trace=None,
    )
    score = detector.classify(run)
    assert score == 1.0


def test_threshold_onset_detection():
    """Test onset detection at the baseline/steady-state midpoint crossing."""
    detector = RewardThresholdDetector(baseline_episodes=20)
    rewards = np.concatenate([np.ones(50) * 0.2, np.ones(100) * 0.8])
    run = RunData(
        proxy_rewards=rewards,
        true_rewards=np.zeros(150),
        state_counts=None,
        behav_trace=None,
    )
    onset = detector.detect_onset(run)
    assert 48 < onset < 52  # near episode 50


def test_threshold_no_onset_when_flat():
    """Test onset returns -1 when reward never shifts from baseline."""
    detector = RewardThresholdDetector()
    run = RunData(
        proxy_rewards=np.ones(100) * 0.5,
        true_rewards=np.zeros(100),
        state_counts=None,
        behav_trace=None,
    )
    assert detector.detect_onset(run) == -1


def test_threshold_scale_invariant():
    """Detector must behave identically regardless of the proxy's absolute scale."""
    detector = RewardThresholdDetector(baseline_episodes=20)
    small = np.concatenate([np.ones(50) * 0.2, np.ones(100) * 0.8])
    large = small * 200.0  # gridworld-scale magnitudes
    run_small = RunData(small, np.zeros(150), None, None)
    run_large = RunData(large, np.zeros(150), None, None)
    assert detector.detect_onset(run_small) == detector.detect_onset(run_large)
