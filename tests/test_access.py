"""Tests for access-level enforcement (information-leakage prevention)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rhob.core.access import AccessFilter
from rhob.core.trajectory import Observation
from rhob.core.types import AccessLevel


def test_l1_hides_policy_features():
    filt = AccessFilter(AccessLevel.L1)
    obs = filt.filter(t=5, reward_proxy=1.5, policy_features=np.ones(4))
    assert obs.reward == 1.5
    assert obs.policy_features is None
    assert obs.has_features is False


def test_l2_exposes_policy_features():
    filt = AccessFilter(AccessLevel.L2)
    obs = filt.filter(t=5, reward_proxy=1.5, policy_features=np.ones(4))
    assert obs.policy_features is not None
    assert np.array_equal(obs.policy_features, np.ones(4))


def test_observation_is_immutable():
    obs = Observation(t=0, access_level=AccessLevel.L1, reward=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.reward = 2.0  # type: ignore[misc]


def test_defensive_copy_prevents_source_mutation_leaking_in():
    source = np.ones(3)
    filt = AccessFilter(AccessLevel.L2)
    obs = filt.filter(t=0, reward_proxy=0.0, policy_features=source)
    source[0] = 999.0
    assert obs.policy_features[0] == 1.0  # unaffected by later source mutation


def test_filtered_features_are_read_only():
    filt = AccessFilter(AccessLevel.L2)
    obs = filt.filter(t=0, reward_proxy=0.0, policy_features=np.ones(3))
    with pytest.raises(ValueError):
        obs.policy_features[0] = 5.0  # read-only array


def test_l1_detector_never_sees_l2_fields_via_trajectory(hacking_runs):
    """End-to-end leakage check: an L1 stream exposes no behavioural features."""
    traj = hacking_runs[0]
    assert traj.policy_features is not None  # data exists
    for obs in traj.iter_observations(AccessLevel.L1):
        assert obs.policy_features is None
