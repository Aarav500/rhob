"""Shared config for the sequence-generation SEQUENTIAL setting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SequenceGenConfig:
    """One rollout configuration: how many episodes, how long each sequence is,
    and which calibration seed (if any) this config represents.

    ``vocab_size`` defaults to the shared grammar's own vocabulary size (see
    ``grammar.py``) -- families should not need to override it, but it's a real
    field (not hardcoded into the rollout loop) so a future family could use a
    different vocabulary without changing the rollout primitive.
    """

    n_episodes: int
    horizon: int
    vocab_size: int = 24
    calibration_seed: int = 0
