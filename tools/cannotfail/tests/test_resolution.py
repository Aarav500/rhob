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
    # The stated rule: a pair ties when a score is non-finite, or when both are finite and
    # within tolerance. A finite pair whose difference overflows is NOT a tie.
    with np.errstate(invalid="ignore", over="ignore"):
        diff = np.abs(a - b)
        tol = np.maximum(abs_tol, rel_tol * np.maximum(np.abs(a), np.abs(b)))
        nonfinite = ~(np.isfinite(a) & np.isfinite(b))
        return int((nonfinite | (diff <= tol)).sum())


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
    yield np.array([1e308, 1.0]), np.array([-1e308, 2.0])               # difference overflows
    yield np.array([0.0, -0.0]), np.array([-0.0, 5e-324])


def _boundary_cases(rng, rel_tol):
    """Partners within a few ulps of b = a / (1 - rel_tol), the far edge of the tie window."""
    a = rng.uniform(1e-3, 1e3, size=200) * rng.choice([-1.0, 1.0], size=200)
    edge = a / (1.0 - rel_tol)
    for k in range(-6, 7):
        yield a, edge + k * np.spacing(edge)


def _subnormal_cases():
    tiny = np.finfo(float).smallest_subnormal
    grid = np.arange(0, 120) * tiny
    yield grid, grid[::-1] + tiny


@pytest.mark.parametrize("seed", range(25))
def test_tie_count_equals_brute_force(seed):
    rng = np.random.default_rng(seed)
    for pos, neg in _cases(rng):
        assert tied_pair_count(pos, neg) == brute_force_ties(pos, neg)
        assert tied_pair_count(neg, pos) == brute_force_ties(neg, pos)


@pytest.mark.parametrize(
    "rel_tol,abs_tol",
    [(0.0, 0.0), (0.0, 1e-3), (1e-3, 0.0), (0.2, 1e-6), (0.5, 0.0), (0.9, 0.0),
     (0.95, 0.0), (0.99, 1e-12), (0.999, 0.0)],
)
def test_tie_count_equals_brute_force_at_other_tolerances(rel_tol, abs_tol):
    """Including tolerances so small the certain window vanishes, and so large that rounding
    at the far edge of the window is amplified by 1 / (1 - rel_tol) -- the review's case."""
    rng = np.random.default_rng(7)
    cases = list(_cases(rng)) + list(_boundary_cases(rng, rel_tol)) + list(_subnormal_cases())
    for pos, neg in cases:
        got = tied_pair_count(pos, neg, rel_tol=rel_tol, abs_tol=abs_tol)
        assert got == brute_force_ties(pos, neg, rel_tol=rel_tol, abs_tol=abs_tol)
        assert tied_pair_count(neg, pos, rel_tol=rel_tol, abs_tol=abs_tol) == got


def test_the_reviewed_undercount_is_fixed():
    """Found by review: at rel_tol 0.999 the far-edge partner was missed."""
    a, b = 0.07336229951221185, 73.3622995122175
    assert tied_pair_count([a], [b], rel_tol=0.999, abs_tol=1e-12) == brute_force_ties([a], [b], 0.999, 1e-12) == 1
    assert tied_pair_count([6.4e-323], [1.3e-322], rel_tol=0.5, abs_tol=0.0) == 1


def test_an_overflowing_difference_is_ordered_not_tied():
    """RHOB's gate counts this pair tied, because its matrix turns the difference into inf."""
    assert tied_pair_count([1e308], [-1e308]) == 0
    assert resolution([1e308, -1e308], [1, 0]).tolerance_resolution == 1.0


def test_bad_tolerances_are_refused():
    with pytest.raises(ValueError):
        tied_pair_count([1.0], [1.0], rel_tol=1.0)
    with pytest.raises(ValueError):
        tied_pair_count([1.0], [1.0], abs_tol=-1.0)


def test_a_constant_score_vector_is_fixed():
    r = resolution([2.0] * 12, [1] * 6 + [0] * 6)
    assert r.n_distinct == 1 and r.single_valued
    assert r.exact_resolution == 0.0 and r.tolerance_resolution == 0.0
    assert r.auroc_range == (0.5, 0.5)
    assert r.verdict == "fixed"


def test_perfectly_separated_scores_can_land_anywhere():
    r = resolution(list(range(10)), [0] * 5 + [1] * 5)
    assert r.exact_resolution == 1.0
    assert r.auroc_range == (0.0, 1.0)
    assert r.verdict == "ok"


def test_mostly_tied_scores_are_pinned_inside_the_margin():
    # 10 vs 10, every score 0 except one positive: 90 of 100 pairs tied, r = 0.1, so the
    # AUROC cannot leave 0.5 +/- 0.05 and an equivalence test at 0.10 cannot fail.
    scores = [0.0] * 9 + [1.0] + [0.0] * 10
    labels = [1] * 10 + [0] * 10
    r = resolution(scores, labels)
    assert r.exact_resolution == pytest.approx(0.1)
    assert r.auroc_range == pytest.approx((0.45, 0.55))
    assert r.verdict == "pinned"
    assert resolution(scores, labels, margin=0.02).verdict == "ok"


def test_an_empty_class_is_fixed_not_ok():
    r = resolution([0.1, 0.2, 0.3], [1, 1, 1])
    assert r.n_neg == 0
    assert r.exact_resolution != r.exact_resolution  # nan
    assert r.verdict == "fixed"


def test_non_finite_scores_tie_under_the_gate_rule_and_drop_from_the_exact_one():
    r = resolution([np.nan, 1.0, 2.0, 3.0], [1, 1, 0, 0])
    # Gate rule: (nan,2) (nan,3) tied, (1,2) (1,3) ordered -> 0.5.
    assert r.tolerance_resolution == 0.5
    # A standard AUROC over the finite scores: 1 vs (2, 3), all ordered -> 1.0.
    assert r.exact_resolution == 1.0
    assert r.n_nonfinite == 1
    assert r.verdict == "ok"


def test_large_magnitude_scores_separated_below_the_tolerance_are_dust_not_fixed():
    """Found by review: summed log-probs near -12345, perfectly separated by 5e-6.

    The relative tolerance there is 1.2e-5, so the gate's rule ties every pair, but a
    standard AUROC over these scores is 1.0. Calling that 'fixed' would be false.
    """
    rng = np.random.default_rng(3)
    pos = -12345.0 + 5e-6 + rng.uniform(0, 1e-6, 100)
    neg = -12345.0 + rng.uniform(0, 1e-6, 100)
    r = resolution(np.concatenate([pos, neg]), [1] * 100 + [0] * 100)
    assert r.tolerance_resolution == 0.0
    assert r.exact_resolution == 1.0
    assert r.auroc_range == (0.0, 1.0)
    assert r.tolerance_auroc_range == (0.5, 0.5)
    assert r.verdict == "dust"
    assert r.why == "differences only below the tie tolerance"


def test_without_labels_only_distinct_values_are_reported():
    r = resolution([1.0, 1.0, 1.0])
    assert r.exact_resolution is None and r.auroc_range is None
    assert r.verdict == "fixed"
    assert resolution([1.0, 2.0]).verdict == "ok"


def test_labels_must_be_binary():
    with pytest.raises(ValueError, match="0/1"):
        resolution([1.0, 2.0], [1, 2])
    with pytest.raises(ValueError):
        resolution([1.0, 2.0], [1])
    assert resolution([1.0, 2.0], [True, False]).exact_resolution == 1.0


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


@pytest.mark.parametrize("rel_tol,abs_tol", [(1e-9, 1e-12), (0.0, 0.0)])
def test_large_vectors_run_without_the_n_by_m_matrix(rel_tol, abs_tol):
    """20,000 a side would need 3.2 GB as a comparison matrix.

    Exact equality on fully tied data was the review's worst case: every pair used to fall
    into the pair-by-pair band. Equal scores now count as certain, by binary search.
    """
    rng = np.random.default_rng(1)
    n = 20_000
    for pos, neg in ((rng.normal(size=n), rng.normal(size=n)), (np.zeros(n), np.zeros(n)),
                     (np.round(rng.normal(size=n), 1), np.round(rng.normal(size=n), 1))):
        start = time.perf_counter()
        tied = tied_pair_count(pos, neg, rel_tol=rel_tol, abs_tol=abs_tol)
        assert time.perf_counter() - start < 10.0
        assert 0 <= tied <= n * n
    assert tied_pair_count(np.zeros(n), np.zeros(n), rel_tol=rel_tol, abs_tol=abs_tol) == n * n
