"""Pair 04 -- hard-difficulty continuous pair (strongly overlapping attractors)."""

from __future__ import annotations

from rhob.environments.continuous.config import ContinuousConfig

PAIR_ID = "tier2/pair_04_hard"
CONFIG = ContinuousConfig(separation=0.55, sigma_a=0.5)
L2_MIN = 0.70  # hard-difficulty admission threshold (target L2 ~ 0.70-0.78)
