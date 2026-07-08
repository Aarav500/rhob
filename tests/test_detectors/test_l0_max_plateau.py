"""Tests for MaxPlateauDetector."""

import numpy as np
from rhob.detectors import MaxPlateauDetector
from rhob.detectors.posthoc import RunData


def test_access_level():
    """Detector claims L0 access."""
    det = MaxPlateauDetector()
    assert det.access_level == "L0"


def test_classify_returns_probability():
    """Score is in [0, 1]."""
    det = MaxPlateauDetector()
    rewards = np.random.rand(100)
    run = RunData(rewards, np.zeros(100), None, None)

    score = det.classify(run)
    assert 0.0 <= score <= 1.0


def test_plateau_detection():
    """Detects when rewards plateau at maximum."""
    det = MaxPlateauDetector(threshold=0.9)
    # Early: varied; late: stuck at max
    early = np.random.uniform(0, 5, 50)
    late = np.ones(50) * 10.0  # All at maximum
    rewards = np.concatenate([early, late])
    run = RunData(rewards, np.zeros(100), None, None)

    score = det.classify(run)
    assert score > 0.8  # Should detect plateau


def test_short_run():
    """Short runs return 0.5."""
    det = MaxPlateauDetector()
    rewards = np.random.rand(5)
    run = RunData(rewards, np.zeros(5), None, None)

    score = det.classify(run)
    assert score == 0.5


def test_detect_onset():
    """Onset detection returns -1 or valid episode."""
    det = MaxPlateauDetector()
    rewards = np.random.rand(100)
    run = RunData(rewards, np.zeros(100), None, None)

    onset = det.detect_onset(run)
    assert onset == -1 or 0 <= onset < 100
