"""Tests for the reflection-symmetrisation of the DQN camper.

The symmetrised greedy policy is exactly reflection-equivariant for ANY network
weights, so these tests need no training.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from rhob.agents.dqn import DQNCamper, _QNet  # noqa: E402
from rhob.environments.continuous.point_mass import MIRROR_ACTION, reflect_obs  # noqa: E402


def test_mirror_action_is_an_involution():
    for a in range(9):
        assert MIRROR_ACTION[MIRROR_ACTION[a]] == a


def test_reflect_obs_negates_x_components_only():
    o = np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.6], dtype=np.float32)
    r = reflect_obs(o)
    assert np.allclose(r, [-0.3, -0.2, -0.5, 0.1, 0.4, 0.6])


def test_policy_is_reflection_equivariant():
    camper = DQNCamper(_QNet(hidden=32), box=10.0)  # untrained: equivariance is exact regardless
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = rng.normal(0, 1, 6).astype(np.float32)
        assert camper.act(reflect_obs(s)) == MIRROR_ACTION[camper.act(s)]
