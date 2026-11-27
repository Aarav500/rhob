"""A replicate that cannot differ from its predecessor is not a replicate.

RHOB's leaderboard was, until 2026-08, a single draw: one environment layout and one
seed sequence per cell, five runs per variant. At ``n_seeds=5`` the Mann-Whitney null
standard error is ``sqrt((n+m+1)/(12nm))`` = 0.19, so most differences the leaderboard
reported sat inside the noise of the one draw that produced them. The fix is to run the
suite over many independent draws and publish an interval.

That fix has a failure mode which is worse than the disease. ``Benchmark._rollout_cache``
memoizes rollouts, and its key originally omitted the seeds -- correctly, because both
were pinned at 0 and nothing could observe the omission. Run R replicates in one process
against that cache and every replicate after the first is a cache hit on the layout-0
draw. All R AUROCs come back bit-identical, the bootstrap over them has zero width, and
the result is a "95% CI" of [x, x] that would be published as evidence of precision
while being a measurement of the cache.

This module fails if that can happen. It is deliberately not a test of *how much*
replicates vary -- that is the experiment's output and must be free to be small. It
tests only that variation is *possible*: that the seeds reach the simulation and that
the cache does not collapse distinct draws onto one entry.

The four tests that roll out ``gridworld_camping`` four times cost minutes, so they carry
``@pytest.mark.slow`` and run in CI's ``admission-slow`` job rather than the default
suite. ``test_cache_key_carries_both_seed_coordinates`` is their fast counterpart.

Those draws vary both coordinates at once, which is why they could not see the second
version of the bug. Until 2026-11-27 ``FamilyRegistry.generate_suite`` accepted
``layout_seed`` and dropped it, building and orienting every cell at layout 0; a draw at
``(k, seed_base)`` still differed from ``(0, 0)`` through ``seed_base`` alone, and every
test here passed. The layout tests below hold ``seed_base`` fixed and vary only the
layout, on families cheap enough (milliseconds) to stay in the default run.

Mutation-checked against four ways to reintroduce the bug:
  * drop ``layout_seed`` from the cache key  -> test_distinct_draws_are_not_cache_hits
  * drop ``seed_base`` from the cache key    -> test_distinct_draws_are_not_cache_hits
  * stop forwarding either into ``rollout``  -> test_replicate_aurocs_are_not_all_equal
  * ``generate_suite`` calls ``generate_pair_at(d)`` without ``seed=layout_seed``
    -> test_generate_suite_builds_and_stamps_the_requested_layout, and every
    ``layout_seed`` test after it
"""

from __future__ import annotations

import numpy as np
import pytest

import rhob.v3.families  # noqa: F401  -- registers the families with FamilyRegistry
from rhob.detectors.l2_behavioral_threshold import BehavioralThresholdDetector
from rhob.detectors.posthoc import RunData
from rhob.v3.base_family import BaseFamily
from rhob.v3.base_pair import MatchedPair
from rhob.v3.benchmark import Benchmark, RolloutCacheKey
from rhob.v3.registry import FamilyRegistry
from rhob.v3.sign_randomization import behav_sign
from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism

# A real tabular family. Four draws x 10 runs costs ~200s per test, which is why the
# tests using it are marked slow -- the "cheap tabular family" it was chosen for is only
# cheap relative to MuJoCo. The property under test (cache keying, seed forwarding) is
# family-independent, so a synthetic family would be faster; a real one is used anyway
# because the bug being guarded against lives in the path real evaluations take.
_FAMILY = "gridworld_camping"
_DIFFICULTY = 1.0  # gridworld_camping has a single-point difficulty range
_N_SEEDS = 5

# The draws a real replication would use: (layout_seed, seed_base) pairs, distinct in
# both coordinates so neither can be the sole source of variation.
_DRAWS = [(0, 0), (1, 1000), (2, 2000), (3, 3000)]


@pytest.fixture(autouse=True)
def _clear_cache():
    """Each test starts from an empty cache, or ordering decides the outcome."""
    Benchmark._rollout_cache.clear()
    yield
    Benchmark._rollout_cache.clear()


def _rollout_for(draw: tuple[int, int]):
    layout_seed, seed_base = draw
    pair = FamilyRegistry.generate_suite(_FAMILY, [_DIFFICULTY], layout_seed=layout_seed)[0]
    return pair.rollout(_N_SEEDS, seed_base=seed_base, randomize_sign=True)


def test_cache_key_carries_both_seed_coordinates():
    """The structural guard, and the only one of these that is fast.

    The three tests below roll out a real family four times and cost minutes each, so
    they are marked ``slow`` and deselected from the default suite (see pyproject.toml).
    That is the right home for them -- but it would leave the regression they exist to
    catch invisible to anyone running ``pytest`` locally, which is precisely the
    situation that produced the bug in the first place.

    This one costs microseconds. Dropping either field from the key is the whole bug:
    with them absent, R replicates collapse onto one cached draw and the study reports
    zero-width intervals. Asserting on the field names catches that in the default run;
    the slow tests below prove the seeds also reach the simulation, which a field-name
    check cannot.
    """
    fields = set(RolloutCacheKey._fields)
    assert {"layout_seed", "seed_base"} <= fields, (
        f"RolloutCacheKey lost a seed coordinate: {sorted(fields)}. Without both, "
        f"Benchmark.evaluate serves one draw's rollouts to every replicate and any "
        f"replication study built on it reports intervals of zero width."
    )
    assert {"family", "difficulty", "n_seeds", "randomize_behav_sign"} <= fields


# ---------------------------------------------------------------- the layout coordinate

# Families whose rollouts cost milliseconds and that calibrate nothing, so these tests
# can stay in the default run. The first two build their layout from the seed:
# ``proxy_correlation_gaming`` shuffles its red region and item masks with it, and
# ``rlhf_reward_model_overopt`` fits its reward model from it. The rest have a fixed
# layout, so on them the layout seed moves only the orientation.
_LAYOUT_FAMILIES = (
    "proxy_correlation_gaming",
    "rlhf_reward_model_overopt",
    "distributional_shift",
    "shortcut_exploitation",
    "reward_channel_tampering",
    "sensor_calibration_tampering",
)


def _suite_pair(family: str, layout_seed: int) -> MatchedPair:
    difficulty = FamilyRegistry.get(family).default_difficulties()[0]
    pairs = FamilyRegistry.generate_suite(family, [difficulty], layout_seed=layout_seed)
    assert len(pairs) == 1
    return pairs[0]


class _RecordingFamily(BaseFamily):
    """A stub that records the ``seed`` each ``generate_pair`` call received."""

    def __init__(self) -> None:
        self.seeds: list[int] = []

    name = "_layout_seed_probe"
    mechanism = HackingMechanism.CAMPING_EXPLOIT
    complexity = EnvironmentComplexity.TABULAR

    def difficulty_range(self) -> tuple[float, float]:
        return (0.8, 0.8)

    def generate_pair(self, difficulty: float, seed: int = 0) -> MatchedPair:
        self.seeds.append(seed)

        def rollout(s: int):
            run = RunData(
                proxy_rewards=np.zeros(3), true_rewards=np.zeros(3),
                state_counts=None, behav_trace=np.ones(3),
            )
            return run, 0

        return MatchedPair(
            family=self.name, mechanism=self.mechanism, complexity=self.complexity,
            difficulty=difficulty, n_episodes=3, rollout_hacking=rollout,
            rollout_legit=rollout, params={},
        )


@pytest.mark.parametrize("layout_seed", [0, 1, 7, 19])
def test_generate_suite_builds_and_stamps_the_requested_layout(monkeypatch, layout_seed):
    """The family must be *built* at the layout, not merely labelled with it.

    The stub records the seed its ``generate_pair`` was called with, which a check of
    the stamped ``layout_seed`` alone cannot see: ``generate_pair_at`` stamps whatever
    seed it receives, so a ``generate_suite`` that dropped the argument stamped 0 and a
    future one that stamped ``layout_seed`` without building at it would stamp ``k``.
    """
    probe = _RecordingFamily()
    monkeypatch.setitem(FamilyRegistry._families, probe.name, probe)
    (pair,) = FamilyRegistry.generate_suite(probe.name, [0.8], layout_seed=layout_seed)
    assert probe.seeds == [layout_seed], (
        f"generate_suite(layout_seed={layout_seed}) built the family at seed "
        f"{probe.seeds}; the layout coordinate of a replicate never reached the family"
    )
    assert pair.layout_seed == layout_seed


@pytest.mark.parametrize("family", _LAYOUT_FAMILIES)
def test_a_real_suite_cell_carries_the_requested_layout(family):
    for layout_seed in (0, 3, 11):
        assert _suite_pair(family, layout_seed).layout_seed == layout_seed, family


def test_the_suite_orients_each_layout_by_its_own_sign():
    """A replicate at layout ``k`` is scored under layout ``k``'s orientation.

    With the layout dropped, every replicate was scored under layout 0's sign, so an L2
    detector met each family in one orientation across all 20 draws -- the convention
    sign randomization exists to remove. Fifteen of the 33 families flip between layouts
    0 and 1; this finds one among the cheap families rather than hard-coding it.
    """
    flipping = [
        (family, k)
        for family in _LAYOUT_FAMILIES
        for k in range(1, 20)
        if behav_sign(family, k) != behav_sign(family, 0)
    ]
    assert flipping, "no cheap family changes orientation across 20 layouts"
    family, k = flipping[0]
    for layout_seed in (0, k):
        pair = _suite_pair(family, layout_seed)
        raw = np.asarray(pair.rollout(1, seed_base=0, randomize_sign=False)[0][0].behav_trace)
        shown = np.asarray(pair.rollout(1, seed_base=0, randomize_sign=True)[0][0].behav_trace)
        np.testing.assert_array_equal(shown, behav_sign(family, layout_seed) * raw)


@pytest.mark.parametrize("family", ["proxy_correlation_gaming", "rlhf_reward_model_overopt"])
def test_varying_only_the_layout_changes_the_rollouts(family):
    """``seed_base`` held fixed: a different layout alone must give different runs.

    ``randomize_sign=False``, so what differs is the environment the family built and
    not just the orientation it is shown in.
    """
    def runs(layout_seed: int) -> np.ndarray:
        runs_a, runs_b, _ = _suite_pair(family, layout_seed).rollout(
            2, seed_base=0, randomize_sign=False
        )
        return np.concatenate(
            [np.asarray(r.proxy_rewards, dtype=float).ravel() for r in runs_a + runs_b]
            + [np.asarray(r.behav_trace, dtype=float).ravel() for r in runs_a + runs_b]
        )

    base = runs(0)
    assert np.array_equal(base, runs(0)), "precondition: a layout rolls out reproducibly"
    assert any(not np.array_equal(base, runs(k)) for k in (1, 2, 3)), (
        f"{family}: layouts 1-3 rolled out bit-identically to layout 0 at the same "
        f"seed_base, so a replicate's layout_seed does not reach the environment"
    )


@pytest.mark.slow
def test_distinct_draws_are_not_cache_hits():
    """R distinct draws must occupy R distinct cache entries, not one."""
    for layout_seed, seed_base in _DRAWS:
        Benchmark.evaluate(
            BehavioralThresholdDetector(),
            families=_FAMILY,
            difficulties=[_DIFFICULTY],
            n_seeds=_N_SEEDS,
            verbose=False,
            layout_seed=layout_seed,
            seed_base=seed_base,
        )

    assert len(Benchmark._rollout_cache) == len(_DRAWS), (
        f"{len(_DRAWS)} distinct draws produced {len(Benchmark._rollout_cache)} cache "
        f"entries. Every draw after the first was served another draw's rollouts, so a "
        f"replication study run through this cache would report a zero-width interval. "
        f"Both layout_seed and seed_base must be in Benchmark._rollout_cache's key."
    )


@pytest.mark.slow
def test_draws_produce_different_rollout_data():
    """The seeds must reach the simulation, not merely the cache key.

    Keying the cache correctly while dropping the seeds on the way into ``rollout``
    would give R distinct entries holding R identical arrays -- the same zero-width
    interval, one layer down.
    """
    behav_by_draw = []
    for draw in _DRAWS:
        runs_a, _, _ = _rollout_for(draw)
        behav_by_draw.append(np.concatenate([np.asarray(r.behav_trace).ravel() for r in runs_a]))

    first = behav_by_draw[0]
    assert any(
        other.shape != first.shape or not np.array_equal(first, other)
        for other in behav_by_draw[1:]
    ), (
        "Every draw produced a bit-identical behavioral trace. (layout_seed, seed_base) "
        "are not reaching MatchedPair.rollout, so replicates cannot differ."
    )


@pytest.mark.slow
def test_replicate_aurocs_are_not_all_equal():
    """End-to-end: the number a replication study actually bootstraps must vary.

    This is the test that would have caught the bug in its published form -- it looks at
    the AUROC, which is what lands in the artifact, rather than at the machinery.
    """
    aurocs = []
    for layout_seed, seed_base in _DRAWS:
        res = Benchmark.evaluate(
            BehavioralThresholdDetector(),
            families=_FAMILY,
            difficulties=[_DIFFICULTY],
            n_seeds=_N_SEEDS,
            verbose=False,
            layout_seed=layout_seed,
            seed_base=seed_base,
        )
        aurocs.append(res.cells[0].discrimination_auroc)

    finite = [a for a in aurocs if a == a]  # drop NaN
    assert len(finite) == len(_DRAWS), f"replicate produced a non-finite AUROC: {aurocs}"
    assert len(set(finite)) > 1, (
        f"All {len(_DRAWS)} replicates returned AUROC={finite[0]}. A bootstrap over "
        f"these would report a zero-width confidence interval as a precise result. "
        f"Replicates must be independent draws."
    )
