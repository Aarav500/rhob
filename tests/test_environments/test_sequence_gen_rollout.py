"""Tests for the shared sequence-generation rollout primitive."""

from __future__ import annotations

import numpy as np

from rhob.environments.sequence_gen.config import SequenceGenConfig
from rhob.environments.sequence_gen.grammar import true_grammar, start_distribution, grammar_log_prob_step
from rhob.environments.sequence_gen.rollout import generate_sequence_rundata


def _grammar_action_fn(t, horizon, tokens_so_far, rng):
    """Samples the next token from the true grammar -- the 'legit' policy shape,
    used directly in these infra-level tests."""
    if t == 0:
        probs = start_distribution()
    else:
        probs = true_grammar()[int(tokens_so_far[-1])]
    return int(rng.choice(len(probs), p=probs))


def _flat_proxy_fn(tokens_so_far, t, horizon) -> float:
    return 1.0  # trivial constant proxy, just exercises the plumbing


def _flat_behav_fn(tokens_so_far, t, horizon) -> float:
    return 0.0


def test_generate_sequence_rundata_shapes():
    config = SequenceGenConfig(n_episodes=6, horizon=10)
    run = generate_sequence_rundata(
        config, _grammar_action_fn, _flat_proxy_fn, grammar_log_prob_step, _flat_behav_fn, seed=0
    )
    assert run.proxy_rewards.shape == (6,)
    assert run.true_rewards.shape == (6,)
    assert run.behav_trace.shape == (6,)
    assert run.state_counts is None


def test_generate_sequence_rundata_true_reward_is_finite():
    """Every transition has positive probability (grammar floor), so no -inf."""
    config = SequenceGenConfig(n_episodes=10, horizon=20)
    run = generate_sequence_rundata(
        config, _grammar_action_fn, _flat_proxy_fn, grammar_log_prob_step, _flat_behav_fn, seed=1
    )
    assert np.isfinite(run.true_rewards).all()


def test_generate_sequence_rundata_deterministic_given_seed():
    config = SequenceGenConfig(n_episodes=4, horizon=8)
    run_a = generate_sequence_rundata(
        config, _grammar_action_fn, _flat_proxy_fn, grammar_log_prob_step, _flat_behav_fn, seed=42
    )
    run_b = generate_sequence_rundata(
        config, _grammar_action_fn, _flat_proxy_fn, grammar_log_prob_step, _flat_behav_fn, seed=42
    )
    assert np.array_equal(run_a.true_rewards, run_b.true_rewards)


def test_grammar_following_true_reward_exceeds_uniform_random():
    """Sanity check on the true-reward construction itself: a policy that
    follows the grammar should score higher (less negative log-prob) than one
    that ignores it entirely (uniform random token choice)."""

    def _uniform_action_fn(t, horizon, tokens_so_far, rng):
        return int(rng.integers(0, 24))

    config = SequenceGenConfig(n_episodes=15, horizon=25)
    grammar_run = generate_sequence_rundata(
        config, _grammar_action_fn, _flat_proxy_fn, grammar_log_prob_step, _flat_behav_fn, seed=2
    )
    uniform_run = generate_sequence_rundata(
        config, _uniform_action_fn, _flat_proxy_fn, grammar_log_prob_step, _flat_behav_fn, seed=2
    )
    assert grammar_run.true_rewards.mean() > uniform_run.true_rewards.mean()
