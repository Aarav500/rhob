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
    """The evaluation path carries the pair's separation through to the cell AUROC.

    Scored with the behavioral sign randomization off, because ``_L2Feature`` is a
    deliberately naive detector that reads the raw signed feature and never orients
    itself: it measures the pair's separation *in the family's own coordinate*, which is
    what this test is about. See ``test_sign_randomization.py`` for the complementary
    assertion -- that with randomization on, this same detector loses the direction.
    """
    res = Benchmark.evaluate(
        _L2Feature(), families="mock_camping", n_seeds=10, verbose=False,
        randomize_behav_sign=False,
    )
    assert len(res.cells) >= 1
    assert res.overall_auroc > 0.9  # L2 feature separates the matched-proxy variants


def test_benchmark_evaluate_cross_validates_trainable_detectors():
    """A trainable detector on a family without its channel: N/A, and the original untouched.

    This end-to-end run stops at the missing-channel guard (mock_camping ships no
    ``state_counts``), which ``_evaluate_cell`` checks before its fit/CV branch, so it
    cannot see cross-validation itself. That path -- scored out-of-fold, never fit on
    the test data -- is exercised by
    ``test_evaluate_cell_scores_a_trainable_detector_out_of_fold`` below.
    """
    from rhob.detectors import StateDivergenceDetector

    detector = StateDivergenceDetector(baseline_episodes=5, steady_window=10)
    assert not detector.is_trained  # untrained before evaluation
    res = Benchmark.evaluate(detector, families="mock_camping", n_seeds=10, verbose=False)
    assert len(res.cells) >= 1
    # StateDivergenceDetector needs state_counts; mock_camping's RunData has none, so
    # its cells come back not applicable rather than crashing or silently scoring itself.
    assert all(c.na_reason and "state_counts" in c.na_reason for c in res.cells)
    assert not detector.is_trained  # the ORIGINAL instance is untouched


def _state_run(hacking: bool, seed: int, n: int = 30, n_bins: int = 4) -> RunData:
    """A run with a visitation histogram: after episode 10 the hacking variant camps
    on bin 0 and the legitimate one on bin 1; the proxy reward is matched."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(2.0, size=(n, n_bins)).astype(float)
    counts[10:, 0 if hacking else 1] += 6.0
    proxy = np.full(n, 0.5) + rng.normal(0, 0.02, n)
    return RunData(proxy_rewards=proxy, true_rewards=np.zeros(n), state_counts=counts, behav_trace=None)


def test_evaluate_cell_scores_a_trainable_detector_out_of_fold():
    """A detector with .fit() must be scored out-of-fold, not fit on the test data.

    Every run here carries ``state_counts`` and more rows than ``min_episodes``, so the
    cell passes both guards and reaches the fit/CV branch: ``_evaluate_cell`` must fork
    the detector once per fold, fit each fork on its training fold only, score every run
    with a fork that never saw it, and leave the caller's instance unfitted.
    """
    from rhob.detectors import StateDivergenceDetector
    from rhob.v3.benchmark import _evaluate_cell

    class _Spy(StateDivergenceDetector):
        # Class-level, so every deepcopied fold fork reports into the same lists.
        fit_sizes: list = []
        leaks: list = []

        def fit(self, runs_a, runs_b):
            self._seen = {id(r) for r in (*runs_a, *runs_b)}
            type(self).fit_sizes.append(len(self._seen))
            super().fit(runs_a, runs_b)

        def classify(self, run):
            if id(run) in getattr(self, "_seen", ()):
                type(self).leaks.append(id(run))
            return super().classify(run)

    detector = _Spy(baseline_episodes=5, steady_window=10)
    assert detector.min_episodes == 5
    runs_a = [_state_run(True, seed=s) for s in range(10)]
    runs_b = [_state_run(False, seed=100 + s) for s in range(10)]

    auroc, _, reason = _evaluate_cell(detector, runs_a, runs_b, [10] * 10, "L1", 30)

    assert reason is None, reason  # scored, not routed to a not-applicable guard
    assert _Spy.fit_sizes == [16] * 5, "expected five folds, each fit on 16 of the 20 runs"
    assert not _Spy.leaks, "a run was scored by a fork that had been fit on it"
    assert auroc > 0.9  # out-of-fold scores still recover the planted visitation split
    assert not detector.is_trained and not hasattr(detector, "_seen")  # caller's copy unfit
