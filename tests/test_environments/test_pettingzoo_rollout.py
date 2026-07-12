# tests/test_environments/test_pettingzoo_rollout.py
"""Tests for shared PettingZoo (mpe2) rollout infrastructure."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mpe2")

from rhob.environments.pettingzoo.config import PettingZooConfig
from rhob.environments.pettingzoo.rollout import generate_pettingzoo_rundata


def _make_simple_spread_env():
    from mpe2 import simple_spread_v3

    return simple_spread_v3.parallel_env(N=3, continuous_actions=True)


def _zero_action_fn(agent, t, horizon, obs, rng):
    return np.zeros(5, dtype=np.float32)


def _sum_agent_rewards(env, obs, rewards, infos) -> float:
    return float(sum(rewards.values()))


def test_generate_pettingzoo_rundata_runs_and_produces_expected_shapes():
    config = PettingZooConfig(env_factory=_make_simple_spread_env, n_episodes=3, horizon=10)
    action_fns = {f"agent_{i}": _zero_action_fn for i in range(3)}
    run = generate_pettingzoo_rundata(
        config, action_fns, _sum_agent_rewards, _sum_agent_rewards, _sum_agent_rewards, seed=0
    )
    assert run.proxy_rewards.shape == (3,)
    assert run.true_rewards.shape == (3,)
    assert run.behav_trace.shape == (3,)
    assert run.state_counts is None


def test_generate_pettingzoo_rundata_is_deterministic_given_seed():
    config = PettingZooConfig(env_factory=_make_simple_spread_env, n_episodes=2, horizon=10)
    action_fns = {f"agent_{i}": _zero_action_fn for i in range(3)}
    run_a = generate_pettingzoo_rundata(
        config, action_fns, _sum_agent_rewards, _sum_agent_rewards, _sum_agent_rewards, seed=42
    )
    run_b = generate_pettingzoo_rundata(
        config, action_fns, _sum_agent_rewards, _sum_agent_rewards, _sum_agent_rewards, seed=42
    )
    np.testing.assert_allclose(run_a.proxy_rewards, run_b.proxy_rewards)


def test_different_agents_can_follow_different_action_fns():
    def _moving_action_fn(agent, t, horizon, obs, rng):
        return np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    config = PettingZooConfig(env_factory=_make_simple_spread_env, n_episodes=2, horizon=10)
    action_fns = {"agent_0": _moving_action_fn, "agent_1": _zero_action_fn, "agent_2": _zero_action_fn}
    run = generate_pettingzoo_rundata(
        config, action_fns, _sum_agent_rewards, _sum_agent_rewards, _sum_agent_rewards, seed=0
    )
    assert run.proxy_rewards.shape == (2,)
