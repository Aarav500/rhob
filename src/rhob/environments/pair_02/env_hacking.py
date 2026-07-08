"""pair_02 Environment A (hacking): attractor right of centre, true objective centre."""

from __future__ import annotations

from rhob.environments.continuous import ContinuousRunLog, generate_run
from rhob.environments.pair_02.config import CONFIG


def generate_hacking_run(camper, seed: int) -> ContinuousRunLog:
    """Generate one hacking-variant (A) run for pair_02."""
    return generate_run(camper, "A", CONFIG, seed)
