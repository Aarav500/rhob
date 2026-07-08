"""Tabular Q-learning agent (validation only).

Deliberately minimal and self-contained (numpy only). The one non-obvious design
choice is **random tie-breaking** in the greedy step: with deterministic
``argmax`` tie-breaking, low-index actions are systematically preferred, which
breaks the transpose symmetry between the two Pair 01 variants and leaks a proxy
signal (empirically verified). Random tie-breaking restores the symmetry so the
two variants' proxy dynamics match.
"""

from __future__ import annotations

import numpy as np


class TabularQLearner:
    """A tabular Q-learning agent with epsilon-greedy, random-tie-break action.

    Args:
        n_states: Number of discrete states.
        n_actions: Number of discrete actions.
        alpha: Learning rate.
        gamma: Discount factor.
        q_init: Initial Q-value for every state-action (optimistic if > 0).
        seed: Seed for the agent's private RNG (exploration + tie-breaking).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        *,
        alpha: float,
        gamma: float,
        q_init: float,
        seed: int,
    ) -> None:
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self._rng = np.random.default_rng(seed)
        self.q_table = np.full((n_states, n_actions), float(q_init), dtype=np.float64)

    def act(self, state: int, epsilon: float) -> int:
        """Select an action epsilon-greedily with random tie-breaking.

        Args:
            state: Current state index.
            epsilon: Probability of a uniform-random exploratory action.

        Returns:
            The chosen action index.
        """
        if self._rng.random() < epsilon:
            return int(self._rng.integers(self.q_table.shape[1]))
        row = self.q_table[state]
        best = np.flatnonzero(row == row.max())
        return int(self._rng.choice(best))

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """Apply the Q-learning update for one transition (no terminal state)."""
        target = reward + self.gamma * float(self.q_table[next_state].max())
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])
