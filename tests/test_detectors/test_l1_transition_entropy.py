"""Tests for TransitionEntropyDetector."""

import numpy as np
from rhob.detectors import TransitionEntropyDetector
from rhob.detectors.posthoc import RunData


def test_access_level():
    """Detector claims L1 access."""
    det = TransitionEntropyDetector()
    assert det.access_level == "L1"


def test_classify_returns_probability():
    """Score is in [0, 1]."""
    det = TransitionEntropyDetector()
    n_episodes = 100
    n_states = 10
    counts = np.random.poisson(2, (n_episodes, n_states))
    run = RunData(np.zeros(n_episodes), np.zeros(n_episodes), counts, None)

    score = det.classify(run)
    assert 0.0 <= score <= 1.0


def test_short_run():
    """Short runs return 0.5."""
    det = TransitionEntropyDetector()
    counts = np.random.poisson(2, (5, 10))
    run = RunData(np.zeros(5), np.zeros(5), counts, None)

    score = det.classify(run)
    assert score == 0.5


def test_detect_onset():
    """Onset detection returns -1 or valid episode."""
    det = TransitionEntropyDetector()
    counts = np.random.poisson(2, (100, 10))
    run = RunData(np.zeros(100), np.zeros(100), counts, None)

    onset = det.detect_onset(run)
    assert onset == -1 or 0 <= onset < 100
