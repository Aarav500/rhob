"""Pair 03 Environment A (hacking): attractor right of centre, true objective centre."""

from __future__ import annotations

from rhob.environments.continuous import ContinuousRunLog, generate_run
from rhob.environments.pair_03.config import CONFIG


def generate_hacking_run(camper, seed: int) -> ContinuousRunLog:
    """Generate one hacking-variant (A) run for Pair 03.

    Args:
        camper: A trained camping policy (``rhob.agents.dqn.train_camper``).
        seed: Determines the per-run attractor and noise.
    """
    return generate_run(camper, "A", CONFIG, seed)
