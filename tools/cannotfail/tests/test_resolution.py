"""D1 must equal the brute-force tie count exactly, and say plainly when a statistic cannot move.

The reference below is the n x m comparison, written from the tie rule in the module
docstring. The RHOB repository separately holds this package equal to its admission
gate's own implementation (tests/test_cannotfail_parity.py there).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from cannotfail import (
    dispersion,
    n_distinct,
    resolution,
    resolution_by_group,
    summarize,
    tied_pair_count,
)
from cannotfail.resolution import ABS_TOL, REL_TOL


def brute_force_ties(pos, neg, rel_tol=REL_TOL, abs_tol=ABS_TOL) -> int:
    a = np.asarray(pos, dtype=float)[:, None]
    b = np.asarray(neg, dtype=float)[None, :]
    with np.errstate(invalid="ignore"):  # inf - inf and 0 * inf: those pairs are tied anyway
        diff = np.abs(a - b)
        tol = np.maximum(abs_tol, rel_tol * np.maximum(np.abs(a), np.abs(b)))
        return int((~np.isfinite(diff) | (diff <= tol)).sum())


def _cases(rng):
    """Score vectors chosen to stress the tie rule, not to look like real data."""
    yield rng.normal(size=40), rng.normal(size=33)                     # no ties
    yield np.round(rng.normal(size=60), 1), np.round(rng.normal(size=50), 1)  # heavy exact ties
    yield np.full(25, 3.0), np.full(17, 3.0)                           # one value
    base = rng.normal(scale=100.0, size=30)
    # Partners sitting just inside, on, and just outside the relative tolerance.
    for k in (0.5, 0.999999, 1.0, 1.000001, 1.0 / (1.0 - REL_TOL), 1.5, 3.0):
        yield base, base * (1.0 + REL_TOL * k)
    tiny = rng.normal(scale=1e-12, size=30)                            # the absolute floor
    yield tiny, tiny + rng.choice([-1, 1], size=30) * ABS_TOL * rng.uniform(0.5, 1.5, size=30)
    yield np.concatenate([rng.normal(size=10), [np.nan, np.inf]]), np.concatenate([rng.normal(size=8), [-np.inf]])
    yield rng.normal(scale=1e12, size=20), rng.normal(scale=1e12, size=20)
    yield np.zeros(10), np.zeros(10)
    yield -np.abs(rng.normal(size=15)), np.abs(rng.normal(size=15))


@pytest.mark.parametrize("seed", range(25))
def test_tie_count_equals_brute_force(seed):
    rng = np.random.default_rng(seed)
    for pos, neg in _cases(rng):
        assert tied_pair_count(pos, neg) == brute_force_ties(pos, neg)
        assert tied_pair_count(neg, pos) == brute_force_ties(neg, pos)


@pytest.mark.parametrize("rel_tol,abs_tol", [(0.0, 0.0), (0.0, 1e-3), (1e-3, 0.0), (0.2, 1e-6)])
def test_tie_count_equals_brute_force_at_other_tolerances(rel_tol, abs_tol):
    """Including tolerances so small the certain window vanishes and everything is checked."""
    rng = np.random.default_rng(7)
    for pos, neg in _cases(rng):
        got = tied_pair_count(pos, neg, rel_tol=rel_tol, abs_tol=abs_tol)
        assert got == brute_force_ties(pos, neg, rel_tol=rel_tol, abs_tol=abs_tol)


def test_bad_tolerances_are_refused():
    with pytest.raises(ValueError):
        tied_pair_count([1.0], [1.0], rel_tol=1.0)
    with pytest.raises(ValueError):
        tied_pair_count([1.0], [1.0], abs_tol=-1.0)


def test_a_constant_score_vector_is_fixed():
    r = resolution([2.0] * 12, [1] * 6 + [0] * 6)
    assert r.n_distinct == 1 and r.single_valued
    assert r.resolution == 0.0
    assert r.auroc_range == (0.5, 0.5)
    assert r.verdict == "fixed"


def test_perfectly_separated_scores_can_land_anywhere():
    r = resolution(list(range(10)), [0] * 5 + [1] * 5)
    assert r.resolution == 1.0
    assert r.auroc_range == (0.0, 1.0)
    assert r.verdict == "ok"


def test_mostly_tied_scores_are_pinned_inside_the_margin():
    # 10 vs 10, every score 0 except one positive: 90 of 100 pairs tied, r = 0.1, so the
    # AUROC cannot leave 0.5 +/- 0.05 and an equivalence test at 0.10 cannot fail.
    scores = [0.0] * 9 + [1.0] + [0.0] * 10
    labels = [1] * 10 + [0] * 10
    r = resolution(scores, labels)
    assert r.resolution == pytest.approx(0.1)
    assert r.auroc_range == pytest.approx((0.45, 0.55))
    assert r.verdict == "pinned"
    assert resolution(scores, labels, margin=0.02).verdict == "ok"


def test_an_empty_class_is_fixed_not_ok():
    r = resolution([0.1, 0.2, 0.3], [1, 1, 1])
    assert r.n_neg == 0
    assert r.resolution != r.resolution  # nan
    assert r.verdict == "fixed"


def test_non_finite_scores_count_as_ties():
    r = resolution([np.nan, 1.0, 2.0, 3.0], [1, 1, 0, 0])
    # pairs: (nan,2) (nan,3) tied; (1,2) (1,3) ordered -> r = 0.5
    assert r.resolution == 0.5
    assert r.n_nonfinite == 1


def test_without_labels_only_distinct_values_are_reported():
    r = resolution([1.0, 1.0, 1.0])
    assert r.resolution is None and r.auroc_range is None
    assert r.verdict == "fixed"
    assert resolution([1.0, 2.0]).verdict == "ok"


def test_labels_must_be_binary():
    with pytest.raises(ValueError, match="0/1"):
        resolution([1.0, 2.0], [1, 2])
    with pytest.raises(ValueError):
        resolution([1.0, 2.0], [1])
    assert resolution([1.0, 2.0], [True, False]).resolution == 1.0


def test_dispersion_is_reported_and_only_flags_with_a_floor():
    jitter = 5.0 + np.random.default_rng(0).normal(scale=1e-7, size=20)
    labels = [1] * 10 + [0] * 10
    r = resolution(jitter, labels)
    assert r.dispersion < 1e-6
    assert r.verdict != "dust", "dust needs an explicit floor"
    assert resolution(jitter, labels, dispersion_floor=1e-4).verdict == "dust"


def test_dispersion_edge_cases():
    assert dispersion([0.0, 0.0, 0.0]) == 0.0
    assert dispersion([1.0]) != dispersion([1.0])  # nan
    assert n_distinct([1.0, 1.0, np.nan, 2.0]) == 2
    assert n_distinct(x for x in (1.0, 1.0)) == 1


def test_grouped_summary_counts_the_class_b_statistic():
    scores = [1.0, 1.0, 1.0, 1.0] + [0.1, 0.2, 0.3, 0.4] + [5.0, 5.0]
    groups = ["a"] * 4 + ["b"] * 4 + ["c"] * 2
    labels = [1, 0, 1, 0] + [1, 1, 0, 0] + [1, 0]
    results = resolution_by_group(scores, groups, labels)
    assert list(results) == ["a", "b", "c"]
    s = summarize(results)
    assert s.n_groups == 3
    assert s.n_single_valued == 2
    assert s.by_verdict == {"fixed": 2, "ok": 1}
    assert s.single_valued_rate == pytest.approx(2 / 3)


def test_large_vectors_run_without_the_n_by_m_matrix():
    """20,000 a side would need 3.2 GB as a comparison matrix."""
    rng = np.random.default_rng(1)
    n = 20_000
    for pos, neg in ((rng.normal(size=n), rng.normal(size=n)), (np.zeros(n), np.zeros(n))):
        start = time.perf_counter()
        tied = tied_pair_count(pos, neg)
        assert time.perf_counter() - start < 10.0
        assert 0 <= tied <= n * n
    assert tied == n * n
