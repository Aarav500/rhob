"""Pair 02 -- easy-difficulty continuous pair (well-separated attractors)."""

from __future__ import annotations

from rhob.environments.continuous.config import ContinuousConfig

PAIR_ID = "tier2/pair_02_easy"
CONFIG = ContinuousConfig(separation=1.25, sigma_a=0.5)
L2_MIN = 0.90  # easy-difficulty admission threshold (target L2 ~ 0.93-0.97)
