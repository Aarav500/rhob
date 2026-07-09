"""Shared helper for per-family admission tests.

Before ``AdmissionGate`` existed, each family's admission test re-implemented all 5
criteria by hand (~100-190 lines). Use ``assert_admitted`` for the standard checks and
keep only family-specific extras (sign conventions, mechanism-specific behavior) as
hand-written tests alongside it.
"""

from __future__ import annotations

from rhob.v3.admission_gate import AdmissionGate
from rhob.v3.base_family import BaseFamily


def assert_admitted(
    family: BaseFamily,
    difficulty: float | None = None,
    n_seeds_per_variant: int = 30,
    **gate_kwargs,
) -> None:
    """Certify ``family`` and assert it passes all 5 admission criteria.

    Raises a descriptive ``AssertionError`` (via the certificate's own summary) on
    failure, so a failing test tells you exactly which criterion failed and why.
    """
    gate = AdmissionGate(**gate_kwargs)
    cert = gate.certify(family, difficulty=difficulty, n_seeds_per_variant=n_seeds_per_variant)
    assert cert.passed, "\n" + cert.summary()
