"""Tests for the onset oracle."""

from __future__ import annotations

import numpy as np

from rhob.core.types import HackingType
from rhob.environments.oracle import OnsetOracle


def test_finds_known_onset(synthetic_onset_curves):
    proxy, true, onset = synthetic_onset_curves
    oracle = OnsetOracle(lookback_k=20, significance_delta=1.0, alpha=0.01)
    label = oracle.compute_onset(proxy, true, HackingType.REWARD_TAMPERING)
    assert label is not None
    # The labelled onset should be close to the injected onset.
    assert abs(label.onset_step - onset) < 30
    assert label.hacking_type is HackingType.REWARD_TAMPERING
    assert 0.0 <= label.confidence <= 1.0


def test_no_onset_on_monotone_curves():
    # Both proxy and true rise together, then plateau -> no divergence.
    t = np.arange(300)
    proxy = np.minimum(t / 100.0, 1.0)
    true = np.minimum(t / 100.0, 1.0)
    oracle = OnsetOracle(lookback_k=20)
    assert oracle.compute_onset(proxy, true) is None


def test_deterministic(synthetic_onset_curves):
    proxy, true, _ = synthetic_onset_curves
    oracle = OnsetOracle()
    a = oracle.compute_onset(proxy, true)
    b = oracle.compute_onset(proxy, true)
    assert a is not None and b is not None
    assert a.onset_step == b.onset_step and a.confidence == b.confidence


def test_validate_label(synthetic_onset_curves):
    proxy, true, _ = synthetic_onset_curves
    oracle = OnsetOracle()
    label = oracle.compute_onset(proxy, true)
    assert oracle.validate_label(true, label) is True


def test_returns_none_for_too_short_curve():
    oracle = OnsetOracle(lookback_k=20)
    proxy = np.array([0.0, 1.0, 2.0])
    true = np.array([0.0, 1.0, 2.0])
    assert oracle.compute_onset(proxy, true) is None


def test_independent_instances_do_not_share_alpha():
    # Regression: alpha must be per-instance, not a shared class attribute.
    strict = OnsetOracle(alpha=1e-9)
    lenient = OnsetOracle(alpha=0.05)
    assert strict.alpha == 1e-9
    assert lenient.alpha == 0.05
