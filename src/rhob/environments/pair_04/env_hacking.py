"""pair_04 Environment A (hacking): attractor right of centre, true objective centre."""

from __future__ import annotations

from rhob.environments.continuous import ContinuousRunLog, generate_run
from rhob.environments.pair_04.config import CONFIG


def generate_hacking_run(camper, seed: int) -> ContinuousRunLog:
    """Generate one hacking-variant (A) run for pair_04."""
    return generate_run(camper, "A", CONFIG, seed)
