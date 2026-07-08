"""pair_04 Environment B (legitimate): attractor left of centre, and it IS the true goal."""

from __future__ import annotations

from rhob.environments.continuous import ContinuousRunLog, generate_run
from rhob.environments.pair_04.config import CONFIG


def generate_legitimate_run(camper, seed: int) -> ContinuousRunLog:
    """Generate one legitimate-variant (B) run for pair_04."""
    return generate_run(camper, "B", CONFIG, seed)
