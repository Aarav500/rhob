"""Tests for the shared sequence-generation rollout primitive."""

from __future__ import annotations

import numpy as np

import pytest

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


def test_step_functions_receive_correctly_sliced_token_arrays():
    """Locks in the rollout loop's slicing contract: action_fn sees tokens
    BEFORE the current step's token is chosen (length t); proxy_fn/behav_fn see
    tokens AFTER (length t+1); true_fn sees the full fixed-size array plus t."""
    seen_action_lengths = []
    seen_proxy_lengths = []
    seen_behav_lengths = []
    seen_true_calls = []

    def _recording_action_fn(t, horizon, tokens_so_far, rng):
        seen_action_lengths.append((t, len(tokens_so_far)))
        return 0

    def _recording_proxy_fn(tokens_so_far, t, horizon):
        seen_proxy_lengths.append((t, len(tokens_so_far)))
        return 0.0

    def _recording_true_fn(tokens, t):
        seen_true_calls.append((t, len(tokens)))
        return 0.0

    def _recording_behav_fn(tokens_so_far, t, horizon):
        seen_behav_lengths.append((t, len(tokens_so_far)))
        return 0.0

    config = SequenceGenConfig(n_episodes=1, horizon=4)
    generate_sequence_rundata(
        config, _recording_action_fn, _recording_proxy_fn, _recording_true_fn, _recording_behav_fn, seed=0
    )

    assert seen_action_lengths == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert seen_proxy_lengths == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert seen_behav_lengths == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert seen_true_calls == [(0, 4), (1, 4), (2, 4), (3, 4)]  # full fixed-size array every time


def test_run_sequence_episode_rejects_nonpositive_horizon():
    config = SequenceGenConfig(n_episodes=1, horizon=0)
    with pytest.raises(ValueError):
        generate_sequence_rundata(
            config, _grammar_action_fn, _flat_proxy_fn, grammar_log_prob_step, _flat_behav_fn, seed=0
        )
