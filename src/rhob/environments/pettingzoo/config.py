# src/rhob/environments/pettingzoo/config.py
"""Shared configuration for PettingZoo-based (mpe2) multi-agent families."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class PettingZooConfig:
    """Configuration for one rollout of an mpe2 parallel environment.

    ``env_factory`` is a zero-argument callable returning a fresh ``parallel_env``
    instance (not the env itself -- a fresh env is created per rollout so state never
    leaks between episodes, mirroring MuJoCoConfig's ``env_id`` + kwargs pattern, but
    mpe2 environments need per-family constructor kwargs (e.g. ``N=3`` for
    simple_spread, ``num_adversaries=3`` for simple_tag) that don't fit a single
    ``env_id`` string cleanly, so a factory callable is used instead.
    """

    env_factory: Callable[[], object]
    n_episodes: int
    horizon: int
    action_noise_std: float = 0.0
    calibration_seed: int = 0
