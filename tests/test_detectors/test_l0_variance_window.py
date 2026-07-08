"""Tests for VarianceWindowDetector."""

import numpy as np
from rhob.detectors import VarianceWindowDetector
from rhob.detectors.posthoc import RunData


def test_access_level():
    """Detector claims L0 access."""
    det = VarianceWindowDetector()
    assert det.access_level == "L0"


def test_classify_returns_probability():
    """Score is in [0, 1]."""
    det = VarianceWindowDetector()
    rewards = np.random.rand(100)
    run = RunData(rewards, np.zeros(100), None, None)

    score = det.classify(run)
    assert 0.0 <= score <= 1.0


def test_short_run_neutral():
    """Short runs return 0.5 (neutral)."""
    det = VarianceWindowDetector()
    rewards = np.random.rand(5)
    run = RunData(rewards, np.zeros(5), None, None)

    score = det.classify(run)
    assert score == 0.5


def test_variance_drop():
    """Detects when variance drops (hacking signal)."""
    det = VarianceWindowDetector()
    # Early: high variance; late: low variance (hacking)
    early = np.random.normal(0, 1.0, 50)
    late = np.ones(50) * 5.0 + np.random.normal(0, 0.1, 50)
    rewards = np.concatenate([early, late])
    run = RunData(rewards, np.zeros(100), None, None)

    score = det.classify(run)
    assert score > 0.6  # Should detect hacking


def test_detect_onset():
    """Onset detection returns -1 or valid episode."""
    det = VarianceWindowDetector()
    rewards = np.random.rand(100)
    run = RunData(rewards, np.zeros(100), None, None)

    onset = det.detect_onset(run)
    assert onset == -1 or 0 <= onset < 100
