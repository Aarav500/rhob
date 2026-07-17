# tests/test_environments/test_rlhf_rm_rollout.py
"""Tests for the shared RLHF-RM rollout infrastructure."""

from __future__ import annotations

import numpy as np

from rhob.environments.rlhf_rm.config import RLHFConfig
from rhob.environments.rlhf_rm.preference import (
    fit_reward_model,
    generate_preference_data,
    true_reward,
)
from rhob.environments.rlhf_rm.rollout import (
    default_proxy_fn,
    default_true_fn,
    generate_rlhf_rundata,
)


def _uniform_sample_fn(rng, n, d):
    return rng.normal(0.0, 1.0, size=(n, d))


def test_true_reward_shape():
    x = np.zeros((5, 8))
    r = true_reward(x)
    assert r.shape == (5,)


def test_fit_reward_model_recovers_positive_weights_for_positively_weighted_dims():
    rng = np.random.default_rng(0)
    x, y = generate_preference_data(rng, 500, 8, _uniform_sample_fn, label_noise_std=0.3)
    weights = fit_reward_model(x, y)
    assert weights.shape == (8,)
    # Dim 0 has the largest true linear weight (1.0); the fit should recover a
    # positive coefficient for it given enough pairs.
    assert weights[0] > 0


def test_generate_rlhf_rundata_runs_and_produces_expected_shapes():
    rng = np.random.default_rng(0)
    x, y = generate_preference_data(rng, 500, 8, _uniform_sample_fn, label_noise_std=0.3)
    rm_weights = fit_reward_model(x, y)
    config = RLHFConfig(n_episodes=5, n_steps=10)
    mu_0 = np.zeros(config.response_dim)

    def null_behav_fn(mu, batch, w):
        return 0.0

    run = generate_rlhf_rundata(
        config, rm_weights, mu_0, default_proxy_fn, default_true_fn, null_behav_fn, seed=1
    )
    assert run.proxy_rewards.shape == (5,)
    assert run.true_rewards.shape == (5,)
    assert run.state_counts is None


def test_policy_ascent_increases_mean_proxy_over_episodes():
    """Sanity check: policy-gradient ascent against the RM should raise proxy over
    the course of an episode's n_steps -- not asserting hacking, just that the
    optimization loop is actually optimizing."""
    rng = np.random.default_rng(0)
    x, y = generate_preference_data(rng, 500, 8, _uniform_sample_fn, label_noise_std=0.1)
    rm_weights = fit_reward_model(x, y)
    config = RLHFConfig(n_episodes=1, n_steps=60, step_size=0.1, beta=0.0)
    mu_0 = np.zeros(config.response_dim)

    def proxy_fn(mu, batch, w):
        return float((batch @ w).mean())

    # Re-implement a short manual loop to capture early vs late mu @ rm_weights,
    # since generate_rlhf_rundata only returns the episode mean.
    from rhob.environments.rlhf_rm.rollout import run_rlhf_episode

    p_first, _, _ = run_rlhf_episode(
        RLHFConfig(n_episodes=1, n_steps=5, step_size=0.1, beta=0.0),
        rm_weights, mu_0, proxy_fn, proxy_fn, proxy_fn, np.random.default_rng(1),
    )
    p_full, _, _ = run_rlhf_episode(
        config, rm_weights, mu_0, proxy_fn, proxy_fn, proxy_fn, np.random.default_rng(1),
    )
    assert p_full >= p_first
