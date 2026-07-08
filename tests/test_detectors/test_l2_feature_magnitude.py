"""Tests for FeatureMagnitudeDetector."""

import numpy as np
from rhob.detectors import FeatureMagnitudeDetector
from rhob.detectors.posthoc import RunData


def test_access_level():
    """Detector claims L2 access."""
    det = FeatureMagnitudeDetector()
    assert det.access_level == "L2"


def test_classify_returns_probability():
    """Score is in [0, 1]."""
    det = FeatureMagnitudeDetector()
    rewards = np.random.rand(100)
    behav = np.random.randn(100)
    run = RunData(rewards, np.zeros(100), None, behav)

    score = det.classify(run)
    assert 0.0 <= score <= 1.0


def test_feature_magnitude_detection():
    """Detects strong behavioral feature (hacking signal)."""
    det = FeatureMagnitudeDetector(threshold=0.2)
    rewards = np.random.rand(100)
    # Strong feature signal in late window
    behav = np.concatenate([
        np.random.uniform(-0.05, 0.05, 50),  # Weak early
        np.ones(50) * 0.5  # Strong late (hacking)
    ])
    run = RunData(rewards, np.zeros(100), None, behav)

    score = det.classify(run)
    assert score > 0.8  # Should detect strong feature


def test_short_run():
    """Short runs return 0.5."""
    det = FeatureMagnitudeDetector()
    behav = np.random.randn(5)
    run = RunData(np.random.rand(5), np.zeros(5), None, behav)

    score = det.classify(run)
    assert score == 0.5


def test_detect_onset():
    """Onset detection returns -1 or valid episode."""
    det = FeatureMagnitudeDetector()
    behav = np.random.randn(100)
    run = RunData(np.random.rand(100), np.zeros(100), None, behav)

    onset = det.detect_onset(run)
    assert onset == -1 or 0 <= onset < 100


def test_no_behav_trace():
    """Handles None behav_trace gracefully."""
    det = FeatureMagnitudeDetector()
    run = RunData(np.random.rand(100), np.zeros(100), None, None)

    score = det.classify(run)
    assert score == 0.5

    onset = det.detect_onset(run)
    assert onset == -1
