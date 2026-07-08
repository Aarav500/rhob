"""Tests for FeatureConsistencyDetector."""

import numpy as np
from rhob.detectors import FeatureConsistencyDetector
from rhob.detectors.posthoc import RunData


def test_access_level():
    """Detector claims L2 access."""
    det = FeatureConsistencyDetector()
    assert det.access_level == "L2"


def test_classify_returns_probability():
    """Score is in [0, 1]."""
    det = FeatureConsistencyDetector()
    rewards = np.random.rand(100)
    behav = np.random.randn(100)
    run = RunData(rewards, np.zeros(100), None, behav)

    score = det.classify(run)
    assert 0.0 <= score <= 1.0


def test_feature_consistency_detection():
    """Detects reduced feature variance (hacking signal)."""
    det = FeatureConsistencyDetector(baseline_window=50, test_window=50)
    rewards = np.random.rand(100)
    # Early: high variance; late: low variance (committed to sign)
    behav = np.concatenate([
        np.random.normal(0, 1.0, 50),  # High variance early
        np.ones(50) * 0.5 + np.random.normal(0, 0.05, 50)  # Low variance late
    ])
    run = RunData(rewards, np.zeros(100), None, behav)

    score = det.classify(run)
    assert score > 0.6  # Should detect consistency


def test_short_run():
    """Short runs return 0.5."""
    det = FeatureConsistencyDetector()
    behav = np.random.randn(5)
    run = RunData(np.random.rand(5), np.zeros(5), None, behav)

    score = det.classify(run)
    assert score == 0.5


def test_detect_onset():
    """Onset detection returns -1 or valid episode."""
    det = FeatureConsistencyDetector()
    behav = np.random.randn(100)
    run = RunData(np.random.rand(100), np.zeros(100), None, behav)

    onset = det.detect_onset(run)
    assert onset == -1 or 0 <= onset < 100


def test_no_behav_trace():
    """Handles None behav_trace gracefully."""
    det = FeatureConsistencyDetector()
    run = RunData(np.random.rand(100), np.zeros(100), None, None)

    score = det.classify(run)
    assert score == 0.5

    onset = det.detect_onset(run)
    assert onset == -1
