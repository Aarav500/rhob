"""Tests for the shared MuJoCo rollout infrastructure."""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

import numpy as np

from rhob.environments.mujoco.config import MuJoCoConfig
from rhob.environments.mujoco.rollout import calibrate_scale, generate_mujoco_rundata


def test_calibrate_scale_converges():
    # measure_fn: a simple monotonic linear function, target=5.0 at param=2.5
    def measure_fn(param: float) -> float:
        return param * 2.0

    result = calibrate_scale(measure_fn, target=5.0, lo=0.0, hi=10.0, tol=0.01)
    assert abs(result - 2.5) < 0.05


def test_generate_mujoco_rundata_shapes():
    config = MuJoCoConfig(env_id="Reacher-v5", n_episodes=3, horizon=10)

    def action_fn(t, horizon, rng):
        return np.zeros(2, dtype=np.float32)

    def proxy_fn(env, info, reward):
        return 1.0

    def true_fn(env, info, reward):
        return 2.0

    def behav_fn(env, info, reward):
        return 0.0

    run = generate_mujoco_rundata(config, action_fn, proxy_fn, true_fn, behav_fn, seed=0)
    assert run.proxy_rewards.shape == (3,)
    assert run.true_rewards.shape == (3,)
    assert run.state_counts is None
    assert run.behav_trace.shape == (3,)
    assert np.allclose(run.proxy_rewards, 1.0)
    assert np.allclose(run.true_rewards, 2.0)
