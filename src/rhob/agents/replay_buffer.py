"""Fixed-size experience replay buffer (numpy-backed)."""

from __future__ import annotations

import random
from collections import deque
from typing import Sequence

import numpy as np


class ReplayBuffer:
    """Ring buffer of ``(state, action, reward, next_state)`` transitions.

    Args:
        capacity: Maximum number of transitions retained.
        seed: Seed for sampling reproducibility.
    """

    def __init__(self, capacity: int, seed: int = 0) -> None:
        self._buf: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buf)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        """Append one transition."""
        self._buf.append((state, action, reward, next_state))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch as stacked arrays ``(states, actions, rewards, next_states)``."""
        batch: Sequence = self._rng.sample(self._buf, batch_size)
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        return states, actions, rewards, next_states
