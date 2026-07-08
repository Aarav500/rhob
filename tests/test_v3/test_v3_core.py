"""Fast unit tests for the v3 benchmark layer (mock family; no agent training)."""

from __future__ import annotations

import numpy as np
import pytest

from rhob.detectors.posthoc import PosthocDetector, RunData
from rhob.v3 import Benchmark, FamilyRegistry
from rhob.v3.access import restrict
from rhob.v3.base_family import BaseFamily
from rhob.v3.base_pair import MatchedPair
from rhob.v3.families.continuous_camping import difficulty_to_separation
from rhob.v3.taxonomy import DifficultyTier, EnvironmentComplexity, HackingMechanism


def _mock_run(sign: float, n: int = 40, seed: int = 0) -> RunData:
    rng = np.random.default_rng(seed)
    proxy = np.full(n, 0.5) + rng.normal(0, 0.02, n)  # matched across variants
    behav = np.concatenate([np.zeros(10), np.full(n - 10, sign)]) + rng.normal(0, 0.05, n)
    return RunData(proxy_rewards=proxy, true_rewards=np.zeros(n), state_counts=None, behav_trace=behav)


if "mock_camping" not in FamilyRegistry.list_families():

    @FamilyRegistry.register("mock_camping")
    class _MockFamily(BaseFamily):
        @property
        def name(self):
            return "mock_camping"

        @property
        def mechanism(self):
            return HackingMechanism.CAMPING_EXPLOIT

        @property
        def complexity(self):
            return EnvironmentComplexity.TABULAR

        def difficulty_range(self):
            return (0.70, 0.98)

        def generate_pair(self, difficulty, seed=0):
            return MatchedPair(
                family=self.name, mechanism=self.mechanism, complexity=self.complexity,
                difficulty=difficulty, n_episodes=40,
                rollout_hacking=lambda s: (_mock_run(+1.0, seed=s), 10),
                rollout_legit=lambda s: (_mock_run(-1.0, seed=s), 10),
                params={},
            )


class _L2Feature(PosthocDetector):
    access_level = "L2"
    name = "mock_l2"

    @property
    def access_level(self):  # noqa: F811 - satisfy abstract property
        return "L2"

    @property
    def name(self):  # noqa: F811
        return "mock_l2"

    def classify(self, run):
        return float(run.behav_trace[-15:].mean())

    def detect_onset(self, run):
        return 10


def test_taxonomy_tiers():
    assert len(DifficultyTier.all()) == 5
    assert DifficultyTier.MEDIUM.value == 0.80


def test_registry_lists_builtin_families():
    fams = FamilyRegistry.list_families()
    assert "gridworld_camping" in fams
    assert "continuous_camping" in fams


def test_registry_resolve_all_and_unknown():
    assert len(FamilyRegistry.resolve("all")) >= 2
    with pytest.raises(KeyError):
        FamilyRegistry.get("does_not_exist")


def test_difficulty_to_separation_monotone_and_inverts_phi():
    from scipy.special import ndtr  # standard-normal CDF

    d_easy = difficulty_to_separation(0.95)
    d_hard = difficulty_to_separation(0.75)
    assert d_easy > d_hard > 0
    # round-trip: L2 = Phi(d / (sqrt2 * sigma_a)) recovers the target
    recovered = ndtr(d_hard / (np.sqrt(2) * 0.5))
    assert abs(recovered - 0.75) < 1e-6


def test_access_restriction_nulls_disallowed_fields():
    run = RunData(np.ones(5), np.ones(5), np.ones((5, 3)), np.ones(5))
    l0 = restrict(run, "L0")
    assert l0.state_counts is None and l0.behav_trace is None
    assert l0.true_rewards.size == 0  # oracle channel never exposed
    l1 = restrict(run, "L1")
    assert l1.state_counts is not None and l1.behav_trace is None
    l2 = restrict(run, "L2")
    assert l2.behav_trace is not None and l2.true_rewards.size == 0


def test_benchmark_evaluate_on_mock_family_reproduces_separation():
    res = Benchmark.evaluate(_L2Feature(), families="mock_camping", n_seeds=10, verbose=False)
    assert len(res.cells) >= 1
    assert res.overall_auroc > 0.9  # L2 feature separates the matched-proxy variants


def test_benchmark_evaluate_cross_validates_trainable_detectors():
    """A detector with .fit() must be scored out-of-fold, not fit on the test data."""
    from rhob.detectors import StateDivergenceDetector

    detector = StateDivergenceDetector(baseline_episodes=5, steady_window=10)
    assert not detector.is_trained  # untrained before evaluation
    res = Benchmark.evaluate(detector, families="mock_camping", n_seeds=10, verbose=False)
    assert len(res.cells) >= 1
    # StateDivergenceDetector needs state_counts; mock_camping's RunData has none, so
    # it should gracefully report chance rather than crash or silently score itself.
    assert not detector.is_trained  # the ORIGINAL instance is untouched (CV forks copies)
