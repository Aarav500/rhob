"""Tests for GradientReversalDetector."""

import numpy as np
from rhob.detectors import GradientReversalDetector
from rhob.detectors.posthoc import RunData


def test_access_level():
    """Detector claims L0 access."""
    det = GradientReversalDetector()
    assert det.access_level == "L0"


def test_classify_returns_probability():
    """Score is in [0, 1]."""
    det = GradientReversalDetector()
    rewards = np.random.rand(100)
    run = RunData(rewards, np.zeros(100), None, None)

    score = det.classify(run)
    assert 0.0 <= score <= 1.0


def test_gradient_flip():
    """Detects slope reversal (negative to positive)."""
    det = GradientReversalDetector(window_size=30)
    # Early: decreasing trend (exploration)
    early = np.linspace(5, 1, 30)
    # Late: increasing trend (exploit found)
    late = np.linspace(1, 5, 30)
    rewards = np.concatenate([early, late])
    run = RunData(rewards, np.zeros(60), None, None)

    score = det.classify(run)
    assert score > 0.6  # Should detect gradient flip


def test_short_run():
    """Short runs return 0.5."""
    det = GradientReversalDetector()
    rewards = np.random.rand(5)
    run = RunData(rewards, np.zeros(5), None, None)

    score = det.classify(run)
    assert score == 0.5


def test_detect_onset():
    """Onset detection returns -1 or valid episode."""
    det = GradientReversalDetector()
    rewards = np.random.rand(100)
    run = RunData(rewards, np.zeros(100), None, None)

    onset = det.detect_onset(run)
    assert onset == -1 or 0 <= onset < 100
