"""Pair 03 Environment B (legitimate): attractor left of centre, and it IS the true goal."""

from __future__ import annotations

from rhob.environments.continuous import ContinuousRunLog, generate_run
from rhob.environments.pair_03.config import CONFIG


def generate_legitimate_run(camper, seed: int) -> ContinuousRunLog:
    """Generate one legitimate-variant (B) run for Pair 03.

    Args:
        camper: A trained camping policy (``rhob.agents.dqn.train_camper``).
        seed: Determines the per-run attractor and noise.
    """
    return generate_run(camper, "B", CONFIG, seed)
