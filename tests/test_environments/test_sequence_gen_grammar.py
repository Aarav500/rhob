"""Tests for the shared hidden Markov 'true grammar'."""

from __future__ import annotations

import numpy as np

from rhob.environments.sequence_gen.grammar import true_grammar, start_distribution, VOCAB_SIZE


def test_true_grammar_is_row_stochastic():
    P = true_grammar()
    assert P.shape == (VOCAB_SIZE, VOCAB_SIZE)
    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9)


def test_true_grammar_is_deterministic_across_calls():
    """The grammar is fixed ground truth -- must be identical every call, not
    re-randomized (functools.lru_cache should guarantee this, but verify the
    actual returned values match, not just object identity)."""
    P1 = true_grammar()
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
