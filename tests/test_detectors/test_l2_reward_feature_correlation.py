"""Tests for RewardFeatureCorrelationDetector."""

import numpy as np
from rhob.detectors import RewardFeatureCorrelationDetector
from rhob.detectors.posthoc import RunData


def test_access_level():
    """Detector claims L2 access."""
    det = RewardFeatureCorrelationDetector()
    assert det.access_level == "L2"


def test_classify_returns_probability():
    """Score is in [0, 1]."""
    det = RewardFeatureCorrelationDetector()
    rewards = np.random.rand(100)
    behav = np.random.randn(100)
    run = RunData(rewards, np.zeros(100), None, behav)

    score = det.classify(run)
    assert 0.0 <= score <= 1.0


def test_correlated_reward_feature():
    """Detects strong correlation between reward and feature."""
    det = RewardFeatureCorrelationDetector(test_window=50)
    # Create highly correlated reward and feature
    feature = np.random.randn(100)
    rewards = np.concatenate([
        np.random.rand(50),  # Early: uncorrelated
        feature[-50:] + np.random.normal(0, 0.1, 50)  # Late: correlated
    ])
    behav = feature
    run = RunData(rewards, np.zeros(100), None, behav)

    score = det.classify(run)
    assert score > 0.6  # Should detect correlation


def test_short_run():
    """Short runs return 0.5."""
    det = RewardFeatureCorrelationDetector()
    run = RunData(np.random.rand(5), np.zeros(5), None, np.random.randn(5))

    score = det.classify(run)
    assert score == 0.5


def test_no_behav_trace():
    """Handles None behav_trace gracefully."""
    det = RewardFeatureCorrelationDetector()
    run = RunData(np.random.rand(100), np.zeros(100), None, None)

    score = det.classify(run)
    assert score == 0.5

    onset = det.detect_onset(run)
    assert onset == -1


def test_detect_onset():
    """Onset detection returns -1 or valid episode."""
    det = RewardFeatureCorrelationDetector()
    run = RunData(np.random.rand(100), np.zeros(100), None, np.random.randn(100))

    onset = det.detect_onset(run)
    assert onset == -1 or 0 <= onset < 100
