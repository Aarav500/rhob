"""The shared hidden Markov 'true grammar' for the sequence-generation setting.

This is the single, fixed ground truth shared across all 5 families in this
sub-project -- constructed once (not per-episode, not per-family) with a fixed
seed baked in, so it never changes between runs or families. It is oracle-only:
exposed solely via each family's ``_true_fn`` (as a log-probability), never an
input to any proxy scorer, policy, or detector.

Design: rather than a uniform-random transition matrix (which would make every
sequence equally "grammatical", giving no genuine true-reward signal to diverge
from), each token has a small number of strongly-preferred successors -- modeling
coherent local phrase structure -- with a low-probability floor on every other
transition so no transition is literally impossible (log-probability stays
well-defined, never -inf).
"""

from __future__ import annotations

import functools

import numpy as np

VOCAB_SIZE = 24
_GRAMMAR_SEED = 20260714  # fixed -- this is ground truth, never re-randomized
_N_PREFERRED_SUCCESSORS = 3
_FLOOR_PROB = 0.02
_PREFERRED_WEIGHT_RANGE = (2.0, 4.0)


@functools.lru_cache(maxsize=1)
def true_grammar(vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    """The fixed VxV row-stochastic transition matrix. Row i = P(next token | token i)."""
    rng = np.random.default_rng(_GRAMMAR_SEED)
    P = np.full((vocab_size, vocab_size), _FLOOR_PROB, dtype=np.float64)
    for i in range(vocab_size):
        preferred = rng.choice(vocab_size, size=_N_PREFERRED_SUCCESSORS, replace=False)
        P[i, preferred] += rng.uniform(*_PREFERRED_WEIGHT_RANGE, size=_N_PREFERRED_SUCCESSORS)
    P = P / P.sum(axis=1, keepdims=True)
    return P


@functools.lru_cache(maxsize=1)
def start_distribution(vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    """Fixed distribution for the first token of an episode (t=0 has no
    previous token to condition on)."""
    rng = np.random.default_rng(_GRAMMAR_SEED + 1)
    weights = rng.uniform(0.5, 2.0, size=vocab_size)
    return weights / weights.sum()


def grammar_log_prob_step(tokens_so_far: np.ndarray, t: int, vocab_size: int = VOCAB_SIZE) -> float:
    """Log-probability of the token just emitted (``tokens_so_far[t]``) under the
    true grammar, given the previous token (or the start distribution if t=0).
    Shared by every family's ``_true_fn`` -- the grammar itself never varies
    between families or between the legit/hacking variants of any one family.

    ``vocab_size`` is threaded through to both ``true_grammar()`` and
    ``start_distribution()`` so a caller can use a different vocabulary than
    the shared default (``VOCAB_SIZE``) without changing this primitive --
    though every family in this sub-project uses the default."""
    P = true_grammar(vocab_size)
    token = int(tokens_so_far[t])
    if t == 0:
        return float(np.log(start_distribution(vocab_size)[token]))
    prev = int(tokens_so_far[t - 1])
    return float(np.log(P[prev, token]))
