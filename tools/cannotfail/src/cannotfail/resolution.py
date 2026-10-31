"""D1, resolution: what range could the statistic have taken?

A rank statistic cannot move further than its ties allow. Every tied cross-class pair
contributes exactly 0.5 to an AUROC whatever the data are, so when a fraction ``1 - r``
of the pairs is tied, the AUROC is pinned inside ``0.5 +/- r/2``. That is an identity,
not a bound with slack. Its cheapest form is a count of distinct values: a score vector
with one distinct value gives an AUROC of exactly 0.5, and any check built on that AUROC
returns the same verdict on every dataset the design can produce.

This module computes both forms. The tie rule and its tolerances are the ones RHOB's
admission gate uses (``rhob.v3.admission_gate._score_resolution``), and a parity test in
the RHOB repository holds the two to bit-identical results. What differs is cost: the
gate builds the full ``n x m`` comparison matrix, which is fine at 24 runs a side and
needs about 800 MB at 10,000. This version sorts once and counts, so a maintainer can run
it on a real benchmark's score vectors.

Why a tolerance, not exact equality
-----------------------------------
Rounding error in a detector's own arithmetic can make identical inputs score
differently in the 11th significant digit, and a rank statistic will happily rank that
dust. Two scores count as tied when ``|a - b| <= max(ABS_TOL, REL_TOL * max(|a|, |b|))``.
The tolerance is deliberately tiny: it exists to absorb arithmetic noise, not to decide
whether a weak signal is real. Adding jitter one order of magnitude above it defeats it,
which is why dispersion is reported separately rather than folded in.
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
#: The equivalence margin RHOB's gate ships with. An AUROC confined to ``0.5 +/- margin``
#: cannot fail an equivalence test at that margin.
EQUIVALENCE_MARGIN = 0.10


def _floats(values: Iterable[float] | np.ndarray) -> np.ndarray:
    if not isinstance(values, (np.ndarray, list, tuple)):
        values = list(values)
    return np.asarray(values, dtype=float).ravel()


def n_distinct(values: Iterable[float]) -> int:
    """Number of distinct finite values, by exact equality.

    The cheapest form of D1, and the statistic the paper's survey counts per cell. It
    uses exact equality on purpose: "one distinct value" should mean what it says, and
    the tolerance-aware question is :func:`tied_pair_count`'s job.
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

    A pair is tied when ``|a - b| <= max(abs_tol, rel_tol * max(|a|, |b|))``, or when
    either score is non-finite -- a NaN is not an ordering. The count is exact and equal
    to the brute-force ``n x m`` comparison, in ``O((n + m) log m)`` time and ``O(n + m)``
    memory.

    How it stays exact with a tolerance that depends on both scores: for each ``a``, every
    ``b`` with ``|a - b| <= max(abs_tol, rel_tol * |a|)`` is certainly tied, and every tied
    ``b`` satisfies ``|a - b| <= max(abs_tol, rel_tol * |a| / (1 - rel_tol))``. Scores in
    the first window are counted with a binary search; the thin band between the two
    windows is checked with the exact rule. Both windows are widened or shrunk by a margin
    well above floating-point rounding, so a pair is never counted on the wrong side.
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
    inner = np.maximum(abs_tol, rel_tol * absa)
    outer = np.maximum(abs_tol, rel_tol * absa / (1.0 - rel_tol))
    guard = 8.0 * np.finfo(float).eps * (absa + outer)

    # Candidate window: contains every b that could be tied to a.
    reach = outer + guard
    c_lo = np.searchsorted(b, a - reach, side="left")
    c_hi = np.searchsorted(b, a + reach, side="right")

    # Certain window: every b inside it is tied under the exact rule.
    half = inner - guard
    has_certain = half > 0.0
    s_lo = c_lo.copy()
    s_hi = c_lo.copy()
    if has_certain.any():
        s_lo[has_certain] = np.searchsorted(b, a[has_certain] - half[has_certain], side="left")
        s_hi[has_certain] = np.searchsorted(b, a[has_certain] + half[has_certain], side="right")
    tied += int((s_hi - s_lo).sum())

    # The band between the windows is checked pair by pair. It is empty except for
    # scores within a few ulps of a tolerance boundary, or when the tolerance is set so
    # small that the certain window vanishes.
    band = (s_lo - c_lo) + (c_hi - s_hi)
    for i in np.nonzero(band)[0]:
        ai = a[i]
        cand = np.concatenate((b[c_lo[i]:s_lo[i]], b[s_hi[i]:c_hi[i]]))
        diff = np.abs(ai - cand)
        tol = np.maximum(abs_tol, rel_tol * np.maximum(np.abs(ai), np.abs(cand)))
        tied += int(np.count_nonzero(diff <= tol))

    return int(tied)


@dataclass(frozen=True)
class Resolution:
    """D1 for one score vector, split by class when labels are given.

    Attributes:
        n: Number of scores, finite or not.
        n_distinct: Distinct finite values, by exact equality.
        n_nonfinite: NaN or infinite scores.
        n_pos, n_neg: Class sizes, or ``None`` when no labels were given.
        resolution: Fraction of cross-class pairs that are strictly ordered, or ``None``
            without labels. ``nan`` when a class is empty: there is nothing to order.
        dispersion: See :func:`dispersion`.
        margin: The equivalence margin the ``pinned`` verdict is read against.
        dispersion_floor: If set, a vector whose dispersion is below it is ``dust``.
    """

    n: int
    n_distinct: int
    n_nonfinite: int
    n_pos: int | None
    n_neg: int | None
    resolution: float | None
    dispersion: float
    margin: float = EQUIVALENCE_MARGIN
    dispersion_floor: float | None = None

    @property
    def single_valued(self) -> bool:
        """Every finite score is the same number, or there is no finite score."""
        return self.n_distinct <= 1

    @property
    def auroc_range(self) -> tuple[float, float] | None:
        """The interval any AUROC over these scores must lie in, or ``None``."""
        if self.resolution is None or self.resolution != self.resolution:
            return None
        half = 0.5 * self.resolution
        return (0.5 - half, 0.5 + half)

    @property
    def verdict(self) -> str:
        """``fixed``, ``dust``, ``pinned`` or ``ok``.

        - ``fixed``: the statistic cannot move at all -- one distinct value, no ordered
          pair, or an empty class.
        - ``dust``: only when a dispersion floor was given and the spread is below it.
        - ``pinned``: the AUROC cannot leave ``0.5 +/- margin``, so an equivalence test at
          that margin passes whatever the data are.
        - ``ok``: none of the above. This is not evidence the check is sound; it says only
          that D1 found no reason it cannot fail.
        """
        r = self.resolution
        if self.single_valued or (r is not None and (r != r or r == 0.0)):
            return "fixed"
        if (
            self.dispersion_floor is not None
            and self.dispersion == self.dispersion
            and self.dispersion < self.dispersion_floor
        ):
            return "dust"
        if r is not None and 0.5 * r <= self.margin:
            return "pinned"
        return "ok"

    def as_dict(self) -> dict:
        rng = self.auroc_range
        return {
            "n": self.n,
            "n_distinct": self.n_distinct,
            "n_nonfinite": self.n_nonfinite,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "resolution": self.resolution,
            "auroc_range": list(rng) if rng is not None else None,
            "dispersion": self.dispersion,
            "verdict": self.verdict,
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
    dispersion. With 0/1 labels it also reports the fraction of cross-class pairs that
    are strictly ordered, and the AUROC range that fraction allows.
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
        return Resolution(n_pos=None, n_neg=None, resolution=None, **common)

    pos, neg = _split(s, labels)
    if pos.size == 0 or neg.size == 0:
        r = float("nan")
    else:
        tied = tied_pair_count(pos, neg, rel_tol=rel_tol, abs_tol=abs_tol)
        r = 1.0 - tied / (pos.size * neg.size)
    return Resolution(n_pos=int(pos.size), n_neg=int(neg.size), resolution=r, **common)


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
    """Totals over groups: the single-distinct-value count is the paper's class-B statistic."""
    items = list(results.values()) if isinstance(results, dict) else list(results)
    verdicts: dict[str, int] = {}
    for r in items:
        verdicts[r.verdict] = verdicts.get(r.verdict, 0) + 1
    return ResolutionSummary(
        n_groups=len(items),
        n_single_valued=sum(1 for r in items if r.single_valued),
        by_verdict=dict(sorted(verdicts.items())),
    )
