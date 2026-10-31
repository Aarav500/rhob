"""D2 must reproduce the numbers RHOB's gate documents, and its difference power must hold up against simulation."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

from cannotfail import (
    difference_power,
    equivalence_power,
    mann_whitney_null_sd,
    required_seeds_per_layout,
)
from cannotfail.power import ci_multiplier, min_two_sided_p

pytest.importorskip("scipy")


def test_null_sd_at_five_a_side_is_the_documented_0_19():
    assert mann_whitney_null_sd(5) == pytest.approx(math.sqrt(11 / 300))
    assert round(mann_whitney_null_sd(5), 2) == 0.19
    assert mann_whitney_null_sd(5, 5) == mann_whitney_null_sd(5)
    with pytest.raises(ValueError):
        mann_whitney_null_sd(0)


def test_ci_multiplier_at_twelve_layouts():
    assert ci_multiplier(12) == pytest.approx(1.876, abs=5e-4)
    assert ci_multiplier(1) == float("inf")


def test_the_pre_audit_design_cannot_certify():
    """4 a side over 12 layouts: SE 0.0625, half-width 0.117, wider than the margin."""
    p = equivalence_power(4, 12)
    assert p.se == pytest.approx(0.0625, abs=1e-4)
    assert p.half_width == pytest.approx(0.117, abs=1e-3)
    assert not p.can_certify
    assert p.power == 0.0


def test_the_shipped_design_meets_its_target():
    """RHOB's gate needs 24 a side for four tests at 90% combined power."""
    assert required_seeds_per_layout(n_tests=4) == 24
    p = equivalence_power(24, 12)
    assert p.can_certify
    assert p.power >= 0.9 ** (1 / 4) - 1e-12
    assert equivalence_power(23, 12).power < 0.9 ** (1 / 4)


def test_equivalence_power_falls_as_the_truth_leaves_the_band():
    at_null = equivalence_power(24, 12).power
    near = equivalence_power(24, 12, true_auroc=0.55).power
    far = equivalence_power(24, 12, true_auroc=0.62).power
    assert at_null > near > far
    assert equivalence_power(24, 12, true_auroc=0.45).power == pytest.approx(near)


def test_difference_power_is_alpha_at_the_null_and_grows_with_effect_and_size():
    assert difference_power(10, 10, 0.5) == pytest.approx(0.05)
    assert difference_power(10, 10, 0.6) < difference_power(10, 10, 0.7) < difference_power(10, 10, 0.8)
    assert difference_power(10, 10, 0.7) < difference_power(40, 40, 0.7)
    assert difference_power(10, 10, 0.3) == pytest.approx(difference_power(10, 10, 0.7))
    with pytest.raises(ValueError):
        difference_power(5, 5, 1.2)


def test_five_a_side_cannot_reliably_see_a_real_violation():
    """The survey's 'underpowered' case: at 5 vs 5, AUROC 0.7 is caught well under half the time."""
    assert difference_power(5, 5, 0.7) < 0.5


@pytest.mark.parametrize("n,theta", [(30, 0.65), (30, 0.75), (60, 0.6)])
def test_difference_power_matches_simulation(n, theta):
    rng = np.random.default_rng(1234)
    sims = 4000
    shift = math.sqrt(2.0) * NormalDist().inv_cdf(theta)  # AUROC of N(shift,1) vs N(0,1)
    crit = NormalDist().inv_cdf(0.975) * mann_whitney_null_sd(n)
    pos = rng.normal(loc=shift, size=(sims, n))
    neg = rng.normal(size=(sims, n))
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg], axis=1), axis=1), axis=1) + 1
    u = ranks[:, :n].sum(axis=1) - n * (n + 1) / 2
    auroc = u / (n * n)
    simulated = float(np.mean(np.abs(auroc - 0.5) > crit))
    assert difference_power(n, n, theta) == pytest.approx(simulated, abs=0.035)


def test_no_test_can_reject_when_the_design_is_too_small():
    """Found by review: 3 vs 3 used to report power 1.0, where the exact test never rejects."""
    assert min_two_sided_p(3, 3) == pytest.approx(0.1)
    assert min_two_sided_p(3, 4) == pytest.approx(2 / 35)
    assert difference_power(3, 3, 1.0) == 0.0
    assert difference_power(3, 4, 0.95) == 0.0
    assert difference_power(4, 4, 0.9) > 0.0
