"""Tests for the baseline detectors and the detector contract."""

from __future__ import annotations

import numpy as np
import pytest

from rhob.core.types import AccessLevel
from rhob.detectors.base import AbstractDetector
from rhob.detectors.baselines.cusum import CUSUMDetector
from rhob.detectors.baselines.random_detector import RandomDetector


def _scores(detector, traj):
    detector.reset()
    return np.array([detector.step(o) for o in traj.iter_observations(detector.access_level)])


def test_abstract_detector_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AbstractDetector()  # abstract methods not implemented


@pytest.mark.parametrize("factory", [RandomDetector, CUSUMDetector])
def test_scores_are_bounded(factory, hacking_runs):
    scores = _scores(factory(), hacking_runs[0])
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize("factory", [RandomDetector, CUSUMDetector])
def test_deterministic_after_reset(factory, hacking_runs):
    det = factory()
    first = _scores(det, hacking_runs[0])
    second = _scores(det, hacking_runs[0])
    assert np.array_equal(first, second)


def test_reset_isolation(hacking_runs):
    """Scores after reset must not depend on a previously seen trajectory."""
    det = CUSUMDetector()
    _scores(det, hacking_runs[1])  # prime with a different trajectory
    after = _scores(det, hacking_runs[0])
    fresh = _scores(CUSUMDetector(), hacking_runs[0])
    assert np.array_equal(after, fresh)


def test_access_levels_declared():
    assert RandomDetector().access_level == AccessLevel.L1
    assert CUSUMDetector().access_level == AccessLevel.L1
    assert RandomDetector().is_oracle_free is True
    assert CUSUMDetector().is_oracle_free is True


def test_cusum_warmup_scores_zero():
    det = CUSUMDetector(warmup=10)
    from rhob.core.trajectory import Observation

    for t in range(5):
        obs = Observation(t=t, access_level=AccessLevel.L1, reward=float(t))
        assert det.step(obs) == 0.0


def test_hyperparameters_reported():
    hp = CUSUMDetector(slack_k=0.7).hyperparameters()
    assert hp["slack_k"] == 0.7
    assert set(hp) == {"slack_k", "threshold_h", "warmup"}
