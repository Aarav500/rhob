"""D2, power: at the design that ships, could this precondition have failed?

Two questions, because benchmarks state preconditions in two ways.

**An equivalence precondition** ("the two arms' AUROC is within 0.5 +/- margin") is
certified by an equivalence test, and the question is whether the test can *pass*: a
design whose confidence half-width is wider than the margin cannot certify at any
observed mean, so a benchmark that reports it as passed is not reporting a test. The
functions here are lifted from RHOB's admission gate (``mann_whitney_null_sd``,
``ci_multiplier``, ``required_seeds_per_layout``), and a parity test in the RHOB
repository holds them equal.

**A precondition asserted from a non-significant difference test** ("we found no
significant difference, so the arms are matched") is the shape the paper's survey found
in 15 of 77 preconditions. There the question is how often the test would *reject* if the
precondition were violated by a stated amount. A difference test with 20% power at the
smallest violation anyone would care about is evidence of nothing when it fails to
reject. :func:`difference_power` answers that; it is not in RHOB's gate.

Both power figures assume no heterogeneity between layouts or clusters, so they are a
ceiling: the power a real design achieves is at most this.

Only :func:`ci_multiplier` and what calls it need SciPy, for a Student-t quantile.
Everything else uses the standard library's normal distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

__all__ = [
    "EquivalencePower",
    "TOST_ALPHA",
    "TARGET_POWER",
    "ci_multiplier",
    "difference_power",
    "equivalence_power",
    "mann_whitney_null_sd",
    "required_seeds_per_layout",
]

TOST_ALPHA = 0.05
TARGET_POWER = 0.90
_NORMAL = NormalDist()


def mann_whitney_null_sd(n_pos: int, n_neg: int | None = None) -> float:
    """Exact SD of an AUROC when the two classes are exchangeable.

    ``sqrt((n + m + 1) / (12 n m))``. At 5 runs a side it is 0.19: one 5-vs-5 AUROC
    cannot distinguish 0.6 from 0.5.
    """
    n = int(n_pos)
    m = n if n_neg is None else int(n_neg)
    if n < 1 or m < 1:
        raise ValueError(f"class sizes must be at least 1, got {n} and {m}")
    return math.sqrt((n + m + 1) / (12.0 * n * m))


def ci_multiplier(n_layouts: int, alpha: float = TOST_ALPHA) -> float:
    """Half-width of a one-sided ``1 - alpha`` bound, in units of the bootstrap SE.

    A Student-t quantile on ``n_layouts - 1`` degrees of freedom, times
    ``sqrt(L / (L - 1))`` to undo the downward bias of a bootstrap SE over ``L``
    clusters. At ``L = 12`` it is about 1.876, against a naive 1.645.
    """
    if n_layouts < 2:
        return float("inf")
    try:
        from scipy.stats import t as student_t
    except ImportError:  # pragma: no cover - exercised only without scipy
        raise ImportError(
            "equivalence power needs SciPy for a Student-t quantile: "
            "pip install 'cannotfail[power]'"
        ) from None
    return float(student_t.ppf(1.0 - alpha, n_layouts - 1)) * math.sqrt(
        n_layouts / (n_layouts - 1.0)
    )


@dataclass(frozen=True)
class EquivalencePower:
    """Whether an equivalence test at this design can pass, and how often it would."""

    se: float
    half_width: float
    margin: float
    power: float

    @property
    def can_certify(self) -> bool:
        """False when the half-width alone is at least the margin: no mean passes."""
        return self.half_width < self.margin


def equivalence_power(
    n_per_side: int,
    n_layouts: int,
    *,
    margin: float = 0.10,
    alpha: float = TOST_ALPHA,
    true_auroc: float = 0.5,
) -> EquivalencePower:
    """Probability that TOST certifies ``|AUROC - 0.5| < margin`` at this design.

    The test certifies when ``|mean - 0.5| + mult * SE < margin``, with SE the exact
    null SD of one layout's AUROC over ``sqrt(n_layouts)``. With the mean distributed
    ``N(true_auroc, SE)``, that probability is
    ``Phi((0.5 + h - mu) / SE) - Phi((0.5 - h - mu) / SE)`` where ``h = margin - mult*SE``.
    At ``true_auroc = 0.5`` it reduces to the gate's ``2 * Phi(h / SE) - 1``.
    """
    se = mann_whitney_null_sd(n_per_side) / math.sqrt(n_layouts)
    half_width = ci_multiplier(n_layouts, alpha) * se
    h = margin - half_width
    if h <= 0.0:
        power = 0.0
    else:
        power = _NORMAL.cdf((0.5 + h - true_auroc) / se) - _NORMAL.cdf((0.5 - h - true_auroc) / se)
    return EquivalencePower(se=se, half_width=half_width, margin=margin, power=float(power))


def required_seeds_per_layout(
    *,
    margin: float = 0.10,
    n_layouts: int = 12,
    alpha: float = TOST_ALPHA,
    power: float = TARGET_POWER,
    n_tests: int = 1,
) -> int:
    """Smallest runs-per-side-per-layout at which TOST can pass at the target power.

    ``n_tests`` is how many equivalence tests must *all* pass (an intersection-union
    test); each is sized at ``power ** (1 / n_tests)`` so the combined false-reject rate
    on a genuinely matched design stays at ``1 - power``. RHOB's gate runs four, and at
    its defaults this returns 24 -- against the 4 the pre-audit gate used, a design whose
    half-width (0.117) was wider than the whole margin.
    """
    mult = ci_multiplier(n_layouts, alpha)
    per_test = power ** (1.0 / max(1, int(n_tests)))
    z = _NORMAL.inv_cdf(0.5 * (1.0 + per_test))
    max_se = margin / (mult + z)
    if max_se <= 0:
        raise ValueError(f"margin {margin} too small to ever certify at power {power}")
    for seeds in range(2, 100_000):
        if mann_whitney_null_sd(seeds) / math.sqrt(n_layouts) <= max_se:
            return seeds
    raise ValueError(  # pragma: no cover - unreachable for sane margins
        f"no feasible seed count for margin={margin}, n_layouts={n_layouts}, power={power}"
    )


def difference_power(
    n_pos: int,
    n_neg: int,
    true_auroc: float,
    *,
    alpha: float = 0.05,
) -> float:
    """Power of a two-sided Mann-Whitney test to reject when the true AUROC is ``true_auroc``.

    Use it on a precondition that was asserted from a *non*-significant difference test:
    set ``true_auroc`` to the smallest violation that would matter, and the result is how
    often the test would have caught it. At ``true_auroc = 0.5`` it returns ``alpha``.

    Normal approximation: the rejection threshold uses the exact null SD, and the
    alternative uses the Hanley-McNeil variance of the AUROC. It is close at 20 or more a
    side and indicative below 10, where the exact permutation distribution is lumpy.
    """
    theta = float(true_auroc)
    if not 0.0 <= theta <= 1.0:
        raise ValueError(f"true_auroc must be in [0, 1], got {theta}")
    n, m = int(n_pos), int(n_neg)
    sd0 = mann_whitney_null_sd(n, m)
    crit = _NORMAL.inv_cdf(1.0 - alpha / 2.0) * sd0

    q1 = theta / (2.0 - theta)
    q2 = 2.0 * theta * theta / (1.0 + theta)
    var1 = (theta * (1.0 - theta) + (n - 1) * (q1 - theta * theta) + (m - 1) * (q2 - theta * theta)) / (n * m)
    if theta == 0.5:
        return float(alpha)
    if var1 <= 0.0:
        # theta at 0 or 1: the AUROC is certain to be extreme, so the test rejects
        # whenever the extreme lies beyond the threshold at all.
        return 1.0 if abs(theta - 0.5) > crit else 0.0
    sd1 = math.sqrt(var1)
    upper = 1.0 - _NORMAL.cdf((0.5 + crit - theta) / sd1)
    lower = _NORMAL.cdf((0.5 - crit - theta) / sd1)
    return float(upper + lower)
