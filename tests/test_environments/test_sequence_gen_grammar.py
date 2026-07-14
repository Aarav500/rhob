"""Tests for the shared hidden Markov 'true grammar'."""

from __future__ import annotations

import numpy as np

from rhob.environments.sequence_gen.grammar import (
    true_grammar,
    start_distribution,
    grammar_log_prob_step,
    VOCAB_SIZE,
)


def test_true_grammar_is_row_stochastic():
    P = true_grammar()
    assert P.shape == (VOCAB_SIZE, VOCAB_SIZE)
    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9)


def test_true_grammar_is_deterministic_across_calls():
    """The grammar is fixed ground truth -- must be identical every call, not
    re-randomized. Bust the lru_cache between calls so this forces genuine
    recomputation from the seeded RNG, rather than just verifying memoization
    returns the same cached object."""
    P1 = true_grammar()
    true_grammar.cache_clear()
    P2 = true_grammar()
    assert np.array_equal(P1, P2)


def test_true_grammar_has_no_zero_probabilities():
    """Every transition must have positive probability (a floor), so log-prob
    true reward is always well-defined -- never -inf."""
    P = true_grammar()
    assert (P > 0).all()


def test_true_grammar_has_preferred_transitions():
    """Each row should have a small number of much-more-likely successors
    (modeling coherent phrase structure), not be uniform."""
    P = true_grammar()
    for row in P:
        assert row.max() > 3 * row.min()


def test_start_distribution_is_a_valid_distribution():
    start = start_distribution()
    assert start.shape == (VOCAB_SIZE,)
    assert np.isclose(start.sum(), 1.0, atol=1e-9)
    assert (start > 0).all()


def test_grammar_log_prob_step_at_t0_matches_start_distribution():
    """At t=0 there is no previous token -- the log-prob should come from the
    start distribution for the token actually placed at index 0."""
    start = start_distribution()
    token0 = int(np.argmax(start))
    tokens_so_far = np.array([token0, 5, 5])
    result = grammar_log_prob_step(tokens_so_far, 0)
    assert np.isclose(result, np.log(start[token0]))


def test_grammar_log_prob_step_at_t_gt_0_matches_true_grammar_transition():
    """At t>0 the log-prob should come from the transition matrix, conditioned
    on the previous token."""
    P = true_grammar()
    prev_token = 3
    next_token = int(np.argmax(P[prev_token]))
    tokens_so_far = np.array([7, 7, prev_token, next_token, 2])
    result = grammar_log_prob_step(tokens_so_far, 3)
    assert np.isclose(result, np.log(P[prev_token, next_token]))


def test_grammar_log_prob_step_threads_vocab_size_parameter():
    """Passing an explicit non-default vocab_size must actually change the
    result -- proving the parameter is threaded through to true_grammar() and
    start_distribution(), not silently ignored."""
    tokens_so_far = np.array([0, 1])
    result_default = grammar_log_prob_step(tokens_so_far, 1, vocab_size=VOCAB_SIZE)
    result_small = grammar_log_prob_step(tokens_so_far, 1, vocab_size=10)
    assert result_default != result_small
