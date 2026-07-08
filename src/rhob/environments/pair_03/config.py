"""Pair 03 -- medium-difficulty continuous pair (the validated operating point)."""

from __future__ import annotations

from rhob.environments.continuous.config import ContinuousConfig

PAIR_ID = "tier2/pair_03_medium"
CONFIG = ContinuousConfig(separation=0.75, sigma_a=0.5)
L2_MIN = 0.80  # medium-difficulty admission threshold (target L2 ~ 0.80-0.90)
