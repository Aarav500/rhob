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

All but the structural check roll out a real family four times and cost minutes, so
they carry ``@pytest.mark.slow`` and run in CI's ``admission-slow`` job rather than the
default suite. ``test_cache_key_carries_both_seed_coordinates`` is the fast counterpart
and stays in the default run.

Mutation-checked against three ways to reintroduce the bug:
  * drop ``layout_seed`` from the cache key  -> test_distinct_draws_are_not_cache_hits
  * drop ``seed_base`` from the cache key    -> test_distinct_draws_are_not_cache_hits
  * stop forwarding either into ``rollout``  -> test_replicate_aurocs_are_not_all_equal
"""

from __future__ import annotations

import numpy as np
import pytest

import rhob.v3.families  # noqa: F401  -- registers the families with FamilyRegistry
from rhob.detectors.l2_behavioral_threshold import BehavioralThresholdDetector
from rhob.v3.benchmark import Benchmark, RolloutCacheKey
from rhob.v3.registry import FamilyRegistry

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
