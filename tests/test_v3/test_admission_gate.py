"""Tests for the AdmissionGate module itself.

The gate's diagnostic value is only real if it (a) admits genuinely-matched
families and (b) actually fails families with a real proxy leak or dead
behavioral signal. Both are exercised here against real families rather than
synthetic mocks, since the real families are where the real bugs were found.
"""

from __future__ import annotations

import numpy as np

from rhob.v3.admission_gate import AdmissionGate
from rhob.v3.registry import FamilyRegistry


def test_admits_a_correctly_matched_family():
    gate = AdmissionGate()
    fam = FamilyRegistry.get("orbit_chirality")
    cert = gate.certify(fam, n_seeds_per_variant=24)
    assert cert.passed, cert.summary()


def test_certificate_reports_all_five_criteria():
    gate = AdmissionGate()
    fam = FamilyRegistry.get("distributional_shift")
    cert = gate.certify(fam, n_seeds_per_variant=24)
    for field in (
        "proxy_matched",
        "behavioral_separated",
        "true_reward_diverges",
        "onset_localizable",
        "camping_quality",
    ):
        assert field in cert.details
        assert isinstance(getattr(cert, field), bool)


def test_catches_a_dead_behavioral_signal():
    """A family with a behav_trace that's identically zero for both variants
    must fail behavioral_separated (and would previously fail proxy_matched
    too, but this isolates the specific check)."""
    from rhob.v3.base_family import BaseFamily
    from rhob.v3.base_pair import MatchedPair
    from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism
    from rhob.detectors.posthoc import RunData

    class _DeadSignalFamily(BaseFamily):
        @property
        def name(self) -> str:
            return "dead_signal_test_family"

        @property
        def mechanism(self) -> HackingMechanism:
            return HackingMechanism.CAMPING_EXPLOIT

        @property
        def complexity(self) -> EnvironmentComplexity:
            return EnvironmentComplexity.TABULAR

        def difficulty_range(self) -> tuple[float, float]:
            return (0.90, 0.90)

        def generate_pair(self, difficulty: float, seed: int = 0) -> MatchedPair:
            n = 100

            def _rollout(true_offset: float):
                def _inner(s: int):
                    rng = np.random.default_rng(s)
                    proxy = rng.normal(0.5, 0.01, n)
                    true_r = np.full(n, 0.5 + true_offset)
                    behav = np.zeros(n)  # dead signal: no separation at all
                    return RunData(proxy, true_r, None, behav), 0

                return _inner

            return MatchedPair(
                family=self.name,
                mechanism=self.mechanism,
                complexity=self.complexity,
                difficulty=difficulty,
                n_episodes=n,
                rollout_hacking=_rollout(-0.2),
                rollout_legit=_rollout(0.2),
                params={},
            )

    gate = AdmissionGate()
    cert = gate.certify(_DeadSignalFamily(), difficulty=0.90, n_seeds_per_variant=20)
    assert not cert.behavioral_separated, cert.summary()
    assert not cert.passed
