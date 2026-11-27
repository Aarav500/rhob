"""Stub families for the end-to-end tests of the transfer and sign-randomization scripts.

``scripts/cross_family_transfer.py`` and ``scripts/measure_sign_randomization.py`` drive
real families whose rollouts take minutes to hours (the first ``pooled_runs`` over
``continuous_camping`` alone takes about a minute). These stubs give the scripts what
they need -- a registered family with tiers, a horizon, and optional channels -- with
synthetic runs that are a pure function of the seed, so a test can exercise every
branch (a horizon below a detector's ``min_episodes``, a family with no
``state_counts``, a frozen model handed histograms of the wrong width) in seconds.

Not a test module itself (no ``test_`` prefix); imported by the two that use it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional

import numpy as np

from rhob.detectors.posthoc import RunData
from rhob.v3.base_family import BaseFamily
from rhob.v3.base_pair import MatchedPair
from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism


class StubFamily(BaseFamily):
    """A matched pair whose variants separate along every channel, with seeded noise.

    ``n_bins=None`` ships no ``state_counts`` (as 25 of the 33 real families do).
    ``difficulty_range`` covers the EASY and MEDIUM tiers, so every stub has two cells.
    """

    def __init__(
        self,
        name: str,
        n_episodes: int = 80,
        n_bins: Optional[int] = 6,
        separation: float = 0.6,
        mechanism: HackingMechanism = HackingMechanism.CAMPING_EXPLOIT,
    ):
        self._name = name
        self.n_episodes = n_episodes
        self.n_bins = n_bins
        self.separation = separation
        self._mechanism = mechanism

    @property
    def name(self) -> str:
        return self._name

    @property
    def mechanism(self) -> HackingMechanism:
        return self._mechanism

    @property
    def complexity(self) -> EnvironmentComplexity:
        return EnvironmentComplexity.TABULAR

    def difficulty_range(self) -> tuple[float, float]:
        return (0.80, 0.90)

    def _run(self, hacking: bool, seed: int, difficulty: float) -> RunData:
        # Name-salted so two stubs never share runs; deterministic in the seed.
        salt = sum(map(ord, self._name)) * 7919
        rng = np.random.default_rng(seed + salt)
        n = self.n_episodes
        sign = 1.0 if hacking else -1.0
        strength = self.separation * (difficulty - 0.5)
        onset = n // 4
        ramp = np.clip((np.arange(n) - onset) / max(1, n - onset), 0.0, 1.0)
        behav = sign * strength * ramp + rng.normal(0.0, 0.25, n)
        proxy = 0.5 + 0.5 * strength * (1 if hacking else 0) * ramp + rng.normal(0.0, 0.2, n)
        counts = None
        if self.n_bins:
            weights = np.ones(self.n_bins)
            weights[0 if hacking else -1] += 4.0 * strength
            counts = rng.multinomial(20, weights / weights.sum(), size=n).astype(float)
        return RunData(
            proxy_rewards=proxy,
            true_rewards=np.where(hacking, -ramp, ramp),
            state_counts=counts,
            behav_trace=behav,
        )

    def generate_pair(self, difficulty: float, seed: int = 0) -> MatchedPair:
        onset = self.n_episodes // 4
        return MatchedPair(
            family=self.name,
            mechanism=self.mechanism,
            complexity=self.complexity,
            difficulty=difficulty,
            n_episodes=self.n_episodes,
            rollout_hacking=lambda s: (self._run(True, s, difficulty), onset),
            rollout_legit=lambda s: (self._run(False, s, difficulty), -1),
            params={"stub": True},
        )


#: Train-side stubs: one long enough for every detector here, one below the 50-episode
#: windows of Reward MLP, State Divergence, Feature Magnitude and so the ensemble.
TRAIN_STUBS = {
    "stub_train_long": dict(n_episodes=80, n_bins=6),
    "stub_train_short": dict(n_episodes=40, n_bins=6),
}
#: Held-out stubs: histograms of another width (a frozen State Divergence cannot consume
#: them), and a family with no state channel at all.
TEST_STUBS = {
    "stub_test_bins": dict(n_episodes=80, n_bins=3,
                           mechanism=HackingMechanism.GOAL_MISGENERALIZATION),
    "stub_test_nostate": dict(n_episodes=80, n_bins=None,
                              mechanism=HackingMechanism.REWARD_TAMPERING),
}


def register_stubs(monkeypatch, specs: dict[str, dict]) -> list[str]:
    """Register ``specs`` for the duration of a test, with a fresh rollout cache."""
    from rhob.v3.benchmark import Benchmark
    from rhob.v3.registry import FamilyRegistry

    for name, kwargs in specs.items():
        monkeypatch.setitem(FamilyRegistry._families, name, StubFamily(name, **kwargs))
    # Benchmark._rollout_cache is process-wide; stub entries must not outlive the test.
    monkeypatch.setattr(Benchmark, "_rollout_cache", {})
    return list(specs)


@contextmanager
def one_torch_thread() -> Iterator[None]:
    """Run torch on one intra-op thread, restoring the previous count afterwards.

    The networks here are tiny; with torch's default of one thread per core they spend
    their time in thread synchronization, which on a shared runner made one stub
    ensemble row take 74 s instead of 0.9 s. (CI's numerics pins set
    ``OMP_NUM_THREADS=1`` for the real runs for the same reason among others.)
    """
    try:
        import torch
    except ImportError:  # pragma: no cover -- torch is in the CI lock
        yield
        return
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)
