"""Tests for behavioral threshold detector (L2)."""

from __future__ import annotations

import numpy as np

from rhob.detectors import BehavioralThresholdDetector, RunData


def test_behavioral_detector_classifies_hacking():
    """Variant A (hacking): behav_trace goes positive post-onset -> positive raw score."""
    detector = BehavioralThresholdDetector(steady_window=50)
    run_a = RunData(
        proxy_rewards=np.ones(100) * 0.5,
        true_rewards=np.ones(100) * 0.2,
        state_counts=None,
        behav_trace=np.concatenate([np.zeros(50), np.full(50, 0.8)]),
    )
    assert detector.classify(run_a) > 0


def test_behavioral_detector_classifies_legitimate():
    """Variant B (legitimate): behav_trace goes negative post-onset -> negative raw score."""
    detector = BehavioralThresholdDetector(steady_window=50)
    run_b = RunData(
        proxy_rewards=np.ones(100) * 0.5,
        true_rewards=np.ones(100) * 0.6,
        state_counts=None,
        behav_trace=np.concatenate([np.zeros(50), np.full(50, -0.8)]),
    )
    assert detector.classify(run_b) < 0


def test_behavioral_score_unbounded_preserves_ranking():
    """Scores beyond +-1 must not be clipped -- clipping would tie distinct runs."""
    detector = BehavioralThresholdDetector()
    mild = RunData(np.ones(100), np.ones(100), None, np.concatenate([np.zeros(50), np.full(50, 1.2)]))
    extreme = RunData(np.ones(100), np.ones(100), None, np.concatenate([np.zeros(50), np.full(50, 3.0)]))
    assert detector.classify(extreme) > detector.classify(mild) > 0


def test_behavioral_onset_detection():
    """Onset detection fires when the trace sustainably deviates from baseline."""
    detector = BehavioralThresholdDetector(baseline_episodes=50)
    trace = np.concatenate([np.zeros(50), np.full(50, 0.5)])
    run = RunData(
        proxy_rewards=np.ones(100) * 0.5,
        true_rewards=np.ones(100) * 0.5,
        state_counts=None,
        behav_trace=trace,
    )
    onset = detector.detect_onset(run)
    assert 48 < onset < 55


def test_behavioral_no_onset_when_flat():
    """Onset returns -1 when the trace never deviates from baseline."""
    detector = BehavioralThresholdDetector(baseline_episodes=50)
    run = RunData(
        proxy_rewards=np.ones(100) * 0.5,
        true_rewards=np.ones(100) * 0.5,
        state_counts=None,
        behav_trace=np.zeros(100),
    )
    assert detector.detect_onset(run) == -1


def test_behavioral_missing_trace_returns_neutral():
    detector = BehavioralThresholdDetector()
    run = RunData(
        proxy_rewards=np.ones(10) * 0.5,
        true_rewards=np.ones(10) * 0.5,
        state_counts=None,
        behav_trace=None,
    )
    assert detector.classify(run) == 0.5
    assert detector.detect_onset(run) == -1
