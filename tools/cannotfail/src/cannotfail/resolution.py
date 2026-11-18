"""D1, resolution: what range could the statistic have taken?

A rank statistic cannot move further than its ties allow. In a standard AUROC every
cross-class pair of equal scores contributes exactly 0.5, whatever the data are, so when a
fraction ``1 - r`` of the pairs is tied the AUROC is confined to ``0.5 +/- r/2``. That is an
identity, not a bound with slack. Its cheapest form is a count of distinct values: a score
vector with one distinct value gives an AUROC of exactly 0.5, and any check built on that
AUROC returns the same verdict on every dataset the design can produce.

Two resolutions, and why both
-----------------------------
:attr:`Resolution.exact_resolution` counts ties by exact equality over the finite scores.
That is what ``sklearn.metrics.roc_auc_score`` and most harnesses compute, so the
:attr:`Resolution.auroc_range` it implies holds for the AUROC a maintainer actually reports.

:attr:`Resolution.tolerance_resolution` counts ties the way RHOB's admission gate does
(``rhob.v3.admission_gate._score_resolution``): two scores tie when
``|a - b| <= max(ABS_TOL, REL_TOL * max(|a|, |b|))``, and any pair with a non-finite score
ties. The tolerance exists because rounding error in a scorer's own arithmetic can make
identical inputs score differently in the 11th significant digit, and a rank statistic
will happily rank that dust. When the tolerance confines the AUROC and exact equality does
not, the verdict is ``dust``: the scores differ, but only at a level the gate treats as
noise, and whether those differences mean anything is a judgement about the scorer. It is
not one this package can make. At large magnitudes the relative tolerance is not small
(1e-9 of 12345 is 1.2e-5), which is why it is never allowed to turn a cell ``fixed``.

The tolerance resolution is held to the gate's by a parity test in the RHOB repository. It
is bit-identical except for pairs of finite scores whose difference overflows float64
(``|a - b|`` above about 1.8e308): the gate's ``n x m`` matrix turns those into ``inf`` and
counts them tied, and this counts them ordered, which is what they are. What differs in
general is cost. The gate's matrix needs about 800 MB at 10,000 scores a side; this sorts
once and counts.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ABS_TOL",
    "EQUIVALENCE_MARGIN",
    "REL_TOL",
    "Resolution",
    "ResolutionSummary",
    "dispersion",
    "n_distinct",
    "resolution",
    "resolution_by_group",
    "summarize",
    "tied_pair_count",
]

#: Relative tie tolerance, from RHOB's admission gate.
REL_TOL = 1e-9
#: Absolute tie tolerance, for scores near zero, from RHOB's admission gate.
ABS_TOL = 1e-12
#: The equivalence margin RHOB's gate ships with.
EQUIVALENCE_MARGIN = 0.10

_EPS = float(np.finfo(float).eps)
_TINY = float(np.finfo(float).smallest_subnormal)


def _floats(values: Iterable[float] | np.ndarray) -> np.ndarray:
    if not isinstance(values, (np.ndarray, list, tuple)):
        values = list(values)
    return np.asarray(values, dtype=float).ravel()


def n_distinct(values: Iterable[float]) -> int:
    """Number of distinct finite values, by exact equality.

    The cheapest form of D1, and the statistic the paper's survey counts per cell.
    """
    arr = _floats(values)
    return int(np.unique(arr[np.isfinite(arr)]).size)


def dispersion(values: Iterable[float]) -> float:
    """``SD / mean(|x|)`` over the finite values: a scale-free spread.

    RHOB's gate uses this on proxy-reward streams to catch ``constant + tiny noise``,
    which differs everywhere and means nothing. On arbitrary score vectors it cannot tell
    rounding noise from a real but small spread around a large offset (scores near 1000
    that differ by 0.001 read 1e-6), so it is reported, and only becomes a verdict when
    the caller supplies a floor. ``nan`` with fewer than two finite values; ``0.0`` for an
    all-zero vector.
    """
    arr = _floats(values)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore"):  # near 1e308 the sums overflow
        scale = float(np.abs(arr).mean())
        if scale <= 0.0:
            return 0.0
        return float(arr.std(ddof=0)) / scale


def tied_pair_count(
    pos: Sequence[float] | np.ndarray,
    neg: Sequence[float] | np.ndarray,
    *,
    rel_tol: float = REL_TOL,
    abs_tol: float = ABS_TOL,
) -> int:
    """How many of the ``len(pos) * len(neg)`` cross-class pairs are tied.

    A pair is tied when either score is non-finite, or when both are finite and
    ``|a - b| <= max(abs_tol, rel_tol * max(|a|, |b|))`` evaluated in float64. With
    ``rel_tol = abs_tol = 0`` that is exact equality. The count equals the brute-force
    ``n x m`` comparison of that rule.

    How it stays exact with a tolerance that depends on both scores: for each ``a``, every
    ``b`` equal to it, or within ``max(abs_tol, rel_tol * |a|)`` of it, is certainly tied;
    every tied ``b`` lies within ``max(abs_tol, rel_tol * |a| / (1 - rel_tol))``. The first
    set is counted with a binary search. Scores between the two are checked one by one with
    the rule itself. Both windows are padded beyond what float64 rounding can move them,
    including the ``1 / (1 - rel_tol)`` amplification at large tolerances and the absolute
    spacing of subnormals. Cost is ``O((n + m) log m)`` plus one check per pair that falls
    in that thin band, which is empty unless scores sit within a few rounding units of a
    tolerance boundary.
    """
    if not (0.0 <= rel_tol < 1.0):
        raise ValueError(f"rel_tol must be in [0, 1), got {rel_tol}")
    if abs_tol < 0.0:
        raise ValueError(f"abs_tol must be non-negative, got {abs_tol}")

    a = np.asarray(pos, dtype=float).ravel()
    b = np.asarray(neg, dtype=float).ravel()
    n, m = a.size, b.size
    if n == 0 or m == 0:
        return 0

    finite_a, finite_b = np.isfinite(a), np.isfinite(b)
    bad_a = int(n - finite_a.sum())
    bad_b = int(m - finite_b.sum())
    tied = bad_a * m + (n - bad_a) * bad_b  # every pair that touches a non-finite score

    a = a[finite_a]
    b = np.sort(b[finite_b])
    if a.size == 0 or b.size == 0:
        return int(tied)

    absa = np.abs(a)
    # Near 1e308 with a large rel_tol the windows overflow to inf. That is safe: an infinite
    # candidate window sends every pair to the exact check, it just costs time.
    with np.errstate(over="ignore"):
        inner = np.maximum(abs_tol, rel_tol * absa)
        outer = np.maximum(abs_tol, rel_tol * absa / (1.0 - rel_tol))
        guard = 8.0 * _EPS * (absa + outer)
        # Candidate window: contains every b that could be tied to a.
        reach = outer + (guard + 4.0 * _TINY) / (1.0 - rel_tol)
    c_lo = np.searchsorted(b, a - reach, side="left")
    c_hi = np.searchsorted(b, a + reach, side="right")

    # Certain window: every b inside it is tied. Equal scores always are, so when the
    # tolerance window is too narrow to trust it shrinks to exactly the equal ones.
    half = inner - guard
    wide = half > 0.0
    s_lo = np.searchsorted(b, a, side="left")
    s_hi = np.searchsorted(b, a, side="right")
    if wide.any():
        s_lo[wide] = np.searchsorted(b, a[wide] - half[wide], side="left")
        s_hi[wide] = np.searchsorted(b, a[wide] + half[wide], side="right")
    tied += int((s_hi - s_lo).sum())

    band = (s_lo - c_lo) + (c_hi - s_hi)
    for i in np.nonzero(band)[0]:
        ai = a[i]
        cand = np.concatenate((b[c_lo[i]:s_lo[i]], b[s_hi[i]:c_hi[i]]))
        with np.errstate(over="ignore"):  # an overflowing difference is inf, so not a tie
            diff = np.abs(ai - cand)
        tol = np.maximum(abs_tol, rel_tol * np.maximum(np.abs(ai), np.abs(cand)))
        tied += int(np.count_nonzero(diff <= tol))

    return int(tied)


def _range(r: float | None) -> tuple[float, float] | None:
    if r is None or r != r:
        return None
    return (0.5 - 0.5 * r, 0.5 + 0.5 * r)


@dataclass(frozen=True)
class Resolution:
    """D1 for one score vector, split by class when labels are given.

    Attributes:
        n: Number of scores, finite or not.
        n_distinct: Distinct finite values, by exact equality.
        n_nonfinite: NaN or infinite scores.
        n_pos, n_neg: Class sizes, finite or not, or ``None`` without labels.
        exact_resolution: Fraction of cross-class pairs of *finite* scores that are not
            exactly equal -- what a standard AUROC can order. ``None`` without labels,
            ``nan`` when a class has no finite score. When it is unknown but the gate's
            tolerance resolution is (as for RHOB's ledger), set only that.
        tolerance_resolution: The same fraction under RHOB's gate rule: tolerance ties,
            and non-finite scores count as ties.
        dispersion: See :func:`dispersion`.
        margin: The equivalence margin ``pinned`` is read against.
        dispersion_floor: If set, a vector whose dispersion is below it is ``dust``.
    """

    n: int
    n_distinct: int
    n_nonfinite: int
    n_pos: int | None
    n_neg: int | None
    exact_resolution: float | None
    tolerance_resolution: float | None
    dispersion: float
    margin: float = EQUIVALENCE_MARGIN
    dispersion_floor: float | None = None

    @property
    def single_valued(self) -> bool:
        """At most one distinct finite score."""
        return self.n_distinct <= 1

    @property
    def _r(self) -> float | None:
        return self.exact_resolution if self.exact_resolution is not None else self.tolerance_resolution

    @property
    def auroc_range(self) -> tuple[float, float] | None:
        """Where any standard AUROC over the finite scores must lie, or ``None``."""
        return _range(self._r)

    @property
    def tolerance_auroc_range(self) -> tuple[float, float] | None:
        """The same interval for an AUROC that ties scores the way RHOB's gate does."""
        return _range(self.tolerance_resolution)

    @property
    def verdict(self) -> str:
        """``fixed``, ``dust``, ``pinned`` or ``ok``.

        - ``fixed``: a standard AUROC over these scores is 0.5 whatever the data are -- at
          most one distinct value, no pair of unequal scores across the classes, or a class
          with no finite score.
        - ``dust``: the scores do differ, but either only by less than the gate's tie
          tolerance (a standard AUROC ranks those differences; whether they are real is a
          question about the scorer), or by less than a dispersion floor you set.
        - ``pinned``: a standard AUROC cannot get further from 0.5 than ``margin``, so its
          point estimate can never leave an equivalence band of that width. A test that
          also needs its confidence interval inside the band can still fail on the
          interval's width; a pass says the scores tie, not that the classes match.
        - ``ok``: none of these. That is not evidence the check is sound; it says only that
          D1 found no reason it cannot fail.
        """
        r = self._r
        if self.single_valued or (r is not None and (r != r or r == 0.0)):
            return "fixed"
        if (
            self.dispersion_floor is not None
            and self.dispersion == self.dispersion
            and self.dispersion < self.dispersion_floor
        ):
            return "dust"
        t = self.tolerance_resolution
        if r is not None and t is not None and t == t and 0.5 * t <= self.margin < 0.5 * r:
            return "dust"
        if r is not None and 0.5 * r <= self.margin:
            return "pinned"
        return "ok"

    @property
    def why(self) -> str:
        """One line saying what put the vector in its verdict."""
        v, r, t = self.verdict, self._r, self.tolerance_resolution
        if v == "fixed":
            if self.n_distinct == 0:
                return "no finite score"
            if self.single_valued:
                return "one distinct value"
            if r != r:
                return "a class has no finite score"
            return "no pair of unequal scores across the classes"
        if v == "dust":
            if t is not None and r is not None and t == t and 0.5 * t <= self.margin < 0.5 * r:
                return "differences only below the tie tolerance"
            return "spread below the dispersion floor"
        if v == "pinned":
            return f"AUROC confined within 0.5 +/- {0.5 * r:.3g}"
        return ""

    def as_dict(self) -> dict:
        def num(x):
            return None if x is None or x != x else x

        rng, trng = self.auroc_range, self.tolerance_auroc_range
        return {
            "n": self.n,
            "n_distinct": self.n_distinct,
            "n_nonfinite": self.n_nonfinite,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "exact_resolution": num(self.exact_resolution),
            "tolerance_resolution": num(self.tolerance_resolution),
            "auroc_range": list(rng) if rng is not None else None,
            "tolerance_auroc_range": list(trng) if trng is not None else None,
            "dispersion": num(self.dispersion),
            "verdict": self.verdict,
            "why": self.why,
        }


def _split(scores: np.ndarray, labels: Sequence | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = np.asarray(labels)
    if lab.shape != scores.shape:
        raise ValueError(f"{lab.size} labels for {scores.size} scores")
    if lab.dtype == bool:
        mask = lab
    else:
        try:
            num = lab.astype(float)
        except (TypeError, ValueError):
            raise ValueError("labels must be 0/1 or boolean; map other labels first") from None
        bad = ~np.isin(num, (0.0, 1.0))
        if bad.any():
            raise ValueError(
                f"labels must be 0/1 or boolean; found {sorted(set(num[bad].tolist()))[:5]}"
            )
        mask = num == 1.0
    return scores[mask], scores[~mask]


def resolution(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence | np.ndarray | None = None,
    *,
    margin: float = EQUIVALENCE_MARGIN,
    rel_tol: float = REL_TOL,
    abs_tol: float = ABS_TOL,
    dispersion_floor: float | None = None,
) -> Resolution:
    """D1 for one score vector.

    Without labels this reports only what needs none: the distinct-value count and the
    dispersion. With 0/1 labels it also reports both resolutions and the AUROC ranges they
    allow. ``rel_tol`` and ``abs_tol`` affect only the tolerance resolution.
    """
    s = np.asarray(scores, dtype=float).ravel()
    common = dict(
        n=int(s.size),
        n_distinct=n_distinct(s),
        n_nonfinite=int(s.size - np.isfinite(s).sum()),
        dispersion=dispersion(s),
        margin=margin,
        dispersion_floor=dispersion_floor,
    )
    if labels is None:
        return Resolution(n_pos=None, n_neg=None, exact_resolution=None, tolerance_resolution=None, **common)

    pos, neg = _split(s, labels)
    if pos.size == 0 or neg.size == 0:
        tol_r = float("nan")
    else:
        tol_r = 1.0 - tied_pair_count(pos, neg, rel_tol=rel_tol, abs_tol=abs_tol) / (pos.size * neg.size)

    fpos, fneg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if fpos.size == 0 or fneg.size == 0:
        exact_r = float("nan")
    else:
        exact_r = 1.0 - tied_pair_count(fpos, fneg, rel_tol=0.0, abs_tol=0.0) / (fpos.size * fneg.size)

    return Resolution(
        n_pos=int(pos.size), n_neg=int(neg.size),
        exact_resolution=exact_r, tolerance_resolution=tol_r, **common,
    )


def resolution_by_group(
    scores: Sequence[float] | np.ndarray,
    groups: Sequence[Hashable],
    labels: Sequence | np.ndarray | None = None,
    **kwargs,
) -> dict[Hashable, Resolution]:
    """D1 per group (a benchmark cell, a detector-by-dataset pair), in first-seen order."""
    s = np.asarray(scores, dtype=float).ravel()
    g = list(groups)
    if len(g) != s.size:
        raise ValueError(f"{len(g)} group keys for {s.size} scores")
    lab = None if labels is None else np.asarray(labels)
    if lab is not None and lab.size != s.size:
        raise ValueError(f"{lab.size} labels for {s.size} scores")

    order: dict[Hashable, list[int]] = {}
    for i, key in enumerate(g):
        order.setdefault(key, []).append(i)
    out: dict[Hashable, Resolution] = {}
    for key, idx in order.items():
        ix = np.asarray(idx)
        out[key] = resolution(s[ix], None if lab is None else lab[ix], **kwargs)
    return out


@dataclass(frozen=True)
class ResolutionSummary:
    """Counts over groups, in the form the paper's survey reports them."""

    n_groups: int
    n_single_valued: int
    by_verdict: dict[str, int] = field(default_factory=dict)

    @property
    def single_valued_rate(self) -> float:
        return self.n_single_valued / self.n_groups if self.n_groups else float("nan")


def summarize(results: dict[Hashable, Resolution] | Iterable[Resolution]) -> ResolutionSummary:
    """Totals over groups. ``n_single_valued`` (at most one distinct value) is the paper's
    class-B statistic."""
    items = list(results.values()) if isinstance(results, dict) else list(results)
    verdicts: dict[str, int] = {}
    for r in items:
        verdicts[r.verdict] = verdicts.get(r.verdict, 0) + 1
    return ResolutionSummary(
        n_groups=len(items),
        n_single_valued=sum(1 for r in items if r.single_valued),
        by_verdict=dict(sorted(verdicts.items())),
    )
