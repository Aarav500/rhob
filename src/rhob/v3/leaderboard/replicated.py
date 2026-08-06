"""Reader for the replicated leaderboard artifact (``leaderboard/v5_replicated.json``).

``scripts/aggregate_replication.py`` writes a third leaderboard schema, keyed
``"detectors"``, in which every number is an interval over independent draws of the whole
benchmark rather than a point from one draw. :mod:`rhob.v3.leaderboard.adapters` can
flatten it into a :class:`~rhob.v3.leaderboard.board.Leaderboard`, but that shape has
nowhere to put an interval, a replicate count, or the supervision partition -- and a
consumer that renders the flattened point estimates alone reproduces exactly the
single-draw presentation the replication exists to replace. This module is the loader for
consumers that need the whole artifact: the Space leaderboard is the current one.

Three properties of the artifact are load-bearing and are preserved here rather than
smoothed over:

**A family appears only where the detector measured it.** The L1 (state-visitation)
detectors carry 8 families, not 33, because 25 families ship no state channel. The
aggregator drops those cells instead of imputing them; earlier boards averaged in a
hard-coded ``0.500`` for each, which made a mostly-unmeasured row read as a measurement.
:meth:`ReplicatedDetector.family` therefore returns ``None`` -- not ``0.5``, and not
``0.0`` -- for a family the detector never scored, and a renderer must show that as
"not measured" rather than as a number.

**Two partitions, both reportable.** Five detectors are fitted on labels under 5-fold
cross-validation (:mod:`rhob.detectors.supervision`). The artifact carries the
access-level ladder computed with and without them, because on this board the published
L0->L1 step is carried entirely by one of them and reverses sign without it. Both are
exposed as :class:`LadderPartition` and neither is the default.

**Zero variance is saturation, not precision.** A detector that returned the same score
on all 20 draws has a zero-width interval by construction.
:attr:`ReplicatedDetector.is_saturated` says so, so a renderer does not present it as an
unusually precise estimate.

**A level's maximum is not a detector's score.** ``access_levels.<L>.max_auroc`` is the
mean over draws of the *best* detector at that level, and taking a maximum over N noisy
estimates biases it upward: at L0 it lands at 0.548 while no individual L0 detector's
point estimate reaches 0.507. Reading it as "the best reward-only detector scores 0.548"
inverts the rung's meaning, which is a negative control -- reward-only detectors are
*supposed* to sit at chance on a proxy certified to be indistinguishable between the
variants, and the control is carried detector by detector, not by the max.
:meth:`ReplicatedLeaderboard.control_check` computes that per-detector statement from the
artifact so a renderer can state it instead of inferring it from the level aggregate.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rhob.detectors.redundancy import duplicate_source
from rhob.v3.leaderboard.access_summary import ACCESS_LEVELS

__all__ = [
    "CONTROL_LEVEL",
    "ChanceControl",
    "Interval",
    "LadderPartition",
    "LevelSeparation",
    "LevelStats",
    "ReplicatedDetector",
    "ReplicatedLeaderboard",
    "is_replicated_schema",
    "load_replicated_leaderboard",
]

#: The rung RHOB runs as a negative control. An L0 detector sees only the proxy reward,
#: and every family's proxy is certified by the admission gate to be indistinguishable
#: between the hacking and legitimate variants -- so an L0 detector that beat chance would
#: be evidence the *proxy matching* failed, not that reward-only detection works. The
#: level is a construction check on the benchmark, and is reported as one.
CONTROL_LEVEL = "L0"

#: What a renderer must print where the artifact has no measurement. Named rather than
#: inlined so every call site agrees, and so grepping for it finds them all: the whole
#: point is that an absent cell is visibly absent instead of silently becoming 0.500 or
#: an empty string that reads as zero.
NOT_MEASURED = "n/a"


def _num(value: Any) -> Optional[float]:
    """``value`` as a float, or ``None`` if it is not a real number (``bool`` excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


@dataclass(frozen=True)
class Interval:
    """A point estimate over replicates and its bootstrap interval.

    ``ci_lo``/``ci_hi`` are ``None`` when the artifact could not estimate an interval --
    the single-replicate case, which :func:`~scripts.aggregate_replication.bootstrap_ci`
    deliberately refuses to decorate. :meth:`text` renders that as the bare point plus a
    reason, never as a fabricated range.
    """

    mean: Optional[float]
    sd: Optional[float] = None
    ci_lo: Optional[float] = None
    ci_hi: Optional[float] = None
    n_replicates: int = 0
    min: Optional[float] = None
    max: Optional[float] = None
    note: Optional[str] = None

    @classmethod
    def from_json(cls, blob: Mapping[str, Any]) -> "Interval":
        n = blob.get("n_replicates")
        return cls(
            mean=_num(blob.get("mean")),
            sd=_num(blob.get("sd")),
            ci_lo=_num(blob.get("ci_lo")),
            ci_hi=_num(blob.get("ci_hi")),
            n_replicates=int(n) if isinstance(n, int) and not isinstance(n, bool) else 0,
            min=_num(blob.get("min")),
            max=_num(blob.get("max")),
            note=blob.get("note") if isinstance(blob.get("note"), str) else None,
        )

    @property
    def has_bounds(self) -> bool:
        return self.ci_lo is not None and self.ci_hi is not None

    @property
    def is_zero_width(self) -> bool:
        """Saturation: the same score on every draw. Not precision -- see the module docstring."""
        return self.sd == 0.0

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval is entirely on one side of zero (for differences)."""
        if not self.has_bounds:
            return False
        return self.ci_lo > 0 or self.ci_hi < 0

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies inside the interval. ``False`` when there is no interval.

        Used for the chance controls, where the reportable statement is "this interval
        does not exclude 0.500" -- not "this detector scored 0.500".
        """
        if not self.has_bounds:
            return False
        return self.ci_lo <= value <= self.ci_hi

    def text(self, digits: int = 3, signed: bool = False) -> str:
        """``"0.943 [0.938, 0.948]"``, or :data:`NOT_MEASURED` if there is no estimate."""
        if self.mean is None:
            return NOT_MEASURED
        fmt = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
        point = fmt.format(self.mean)
        if not self.has_bounds:
            return f"{point} (no interval; {self.n_replicates} draw)"
        return f"{point} [{fmt.format(self.ci_lo)}, {fmt.format(self.ci_hi)}]"


@dataclass(frozen=True)
class ReplicatedDetector:
    """One detector's board-wide interval plus its per-family intervals."""

    name: str
    access_level: str
    overall: Interval
    per_family: Mapping[str, Interval]
    #: Cells that produced a measurement, as recorded by the replicates. ``None`` if the
    #: artifact predates the field or the replicates disagreed.
    cells_measured: Optional[int]
    #: Scored by fitting on labels under cross-validation (:mod:`rhob.detectors.supervision`).
    label_fitted: bool = False
    #: The detector this one duplicates, if any. A duplicate is held out of every
    #: access-level aggregate (:mod:`rhob.detectors.redundancy`) and must not be read as
    #: an independent measurement at its level.
    duplicate_of: Optional[str] = None

    def family(self, name: str) -> Optional[Interval]:
        """This detector's interval for ``name``, or ``None`` if it never measured it.

        ``None`` means "no measurement", which is a different statement from "chance".
        Callers must render it as :data:`NOT_MEASURED`.
        """
        return self.per_family.get(name)

    @property
    def n_families_measured(self) -> int:
        return len(self.per_family)

    @property
    def is_saturated(self) -> bool:
        return self.overall.is_zero_width

    @property
    def counted_in_ladder(self) -> bool:
        return self.duplicate_of is None


@dataclass(frozen=True)
class LevelStats:
    """One access level's aggregate, under one partition of the detector suite."""

    level: str
    max_auroc: Interval
    mean_auroc: Interval
    #: How often each detector was the best at this level, over the draws. A level whose
    #: winner changes between draws does not have a stable best detector.
    best_detector_frequency: Mapping[str, int] = field(default_factory=dict)

    @property
    def best_detector(self) -> Optional[str]:
        return next(iter(self.best_detector_frequency), None)

    @property
    def best_detector_draws(self) -> int:
        return next(iter(self.best_detector_frequency.values()), 0)

    @property
    def total_draws(self) -> int:
        return sum(self.best_detector_frequency.values())

    @property
    def is_scored(self) -> bool:
        return self.max_auroc.mean is not None or self.mean_auroc.mean is not None


@dataclass(frozen=True)
class LevelSeparation:
    """A paired difference between two adjacent rungs, under one statistic."""

    lower: str
    higher: str
    statistic: str  # "max" | "mean"
    difference: Interval
    excludes_zero: bool
    replicates_with_hi_above_lo: int
    n_paired: int
    #: Pairing removes between-draw variation shared by the two rungs. It does not
    #: equalise denominators -- L1 aggregates 35 cells over 8 families where L2/L3
    #: aggregate 123 over 33 -- so a rung difference involving L1 is across populations
    #: as well as across access levels.
    denominators_equal: bool = False

    @property
    def key(self) -> str:
        return f"{self.higher}_minus_{self.lower}"

    @property
    def label(self) -> str:
        return f"{self.higher} - {self.lower}"

    @property
    def separates(self) -> bool:
        return self.excludes_zero


@dataclass(frozen=True)
class LadderPartition:
    """The access-level ladder over one partition of the detector suite."""

    #: Human-readable name for the partition, e.g. "all detectors".
    name: str
    levels: Mapping[str, LevelStats]
    separations: Mapping[str, LevelSeparation]

    def level(self, name: str) -> Optional[LevelStats]:
        return self.levels.get(name)

    def separation(self, lower: str, higher: str, statistic: str) -> Optional[LevelSeparation]:
        return self.separations.get(f"{higher}_minus_{lower}_{statistic}")

    def scored_levels(self) -> list[LevelStats]:
        return [s for s in (self.levels.get(lv) for lv in ACCESS_LEVELS) if s and s.is_scored]


@dataclass(frozen=True)
class ChanceControl:
    """Whether a level's detectors individually sit at chance, and what its max does.

    Two different statements live here and they routinely get confused, which is why the
    type carries both:

    - :attr:`holds` -- *every* detector at the level has an interval containing
      ``reference``. This is the control, and it is carried detector by detector.
    - :attr:`max_excludes_reference` -- the level's ``max_auroc`` interval sits entirely
      above it. That is the expected behaviour of a maximum over several noisy estimates,
      not a detection result, and :attr:`max_exceeds_every_detector` says when it has
      climbed past every individual detector's point estimate.
    """

    level: str
    reference: float
    detectors: tuple[ReplicatedDetector, ...]
    n_containing: int
    highest: Optional[ReplicatedDetector]
    level_max: Optional[Interval]
    level_mean: Optional[Interval]

    @property
    def n_detectors(self) -> int:
        return len(self.detectors)

    @property
    def holds(self) -> bool:
        """Every detector at the level has an interval containing ``reference``."""
        return self.n_detectors > 0 and self.n_containing == self.n_detectors

    @property
    def exceptions(self) -> list[ReplicatedDetector]:
        return [d for d in self.detectors if not d.overall.contains(self.reference)]

    @property
    def max_excludes_reference(self) -> bool:
        return self.level_max is not None and not self.level_max.contains(self.reference)

    @property
    def max_exceeds_every_detector(self) -> bool:
        """The level maximum sits above every individual detector's point estimate.

        The signature of selection bias: no detector achieves the number the level's
        "best detector" row reports, because that row is the mean over draws of whichever
        detector happened to win each draw.
        """
        if self.level_max is None or self.level_max.mean is None or self.highest is None:
            return False
        best_point = self.highest.overall.mean
        return best_point is not None and self.level_max.mean > best_point

    @property
    def mean_excludes_reference(self) -> bool:
        """Bare interval arithmetic on the level mean: does it exclude ``reference``?

        **This is not a verdict about which side of chance the suite mean sits on**, and a
        renderer must not present it as one. On the committed board it is ``True`` by a
        margin of about 8e-5 -- see :attr:`mean_margin_from_reference`, and read that
        property's docstring before quoting this one.
        """
        return self.level_mean is not None and not self.level_mean.contains(self.reference)

    @property
    def mean_margin_from_reference(self) -> Optional[float]:
        """How far the level-mean interval's nearer endpoint clears ``reference``.

        Positive means the interval lies clear of ``reference`` by that much; ``None``
        when there is no interval. The number exists so a caller can *check the margin
        against the estimator's own noise* rather than reading
        :attr:`mean_excludes_reference` as a finding.

        On the committed L0 board the margin is ~7.9e-5, while replaying the same
        bootstrap under 300 different generator seeds moves that endpoint over a range of
        ~2.4e-4 and puts it at or above 0.500 in 18 of the 300
        (``docs/figures/l0_control_robustness.md``). The exclusion is therefore smaller
        than the Monte-Carlo scatter of the endpoint that produces it: the interval places
        the suite mean on **neither** side of chance. Subtracting the endpoint from 0.500
        and reporting the remainder is the false-precision error itself, not a smaller
        version of it.
        """
        if self.level_mean is None or not self.level_mean.has_bounds:
            return None
        if self.level_mean.ci_hi < self.reference:
            return self.reference - self.level_mean.ci_hi
        if self.level_mean.ci_lo > self.reference:
            return self.level_mean.ci_lo - self.reference
        return 0.0


@dataclass(frozen=True)
class ReplicatedLeaderboard:
    """The whole replicated artifact, parsed."""

    path: Optional[Path]
    n_replicates: int
    method: Mapping[str, Any]
    draws: Sequence[Mapping[str, Any]]
    detectors: Mapping[str, ReplicatedDetector]
    all_detectors: LadderPartition
    unsupervised_only: LadderPartition
    label_fitted_detectors: tuple[str, ...]
    supervision_note: str
    zero_variance_detectors: tuple[str, ...]
    zero_variance_note: str
    #: Rungs whose separation verdict depends on whether you read the max or the mean.
    #: Reporting one statistic alone there presents a statistic-dependent result as
    #: settled, so a renderer showing a rung on this list must show both.
    statistic_disagreement: tuple[str, ...] = ()
    statistic_disagreement_note: str = ""

    # ------------------------------------------------------------------ accessors
    def standings(self) -> list[ReplicatedDetector]:
        """Every detector, best mean first. Detectors with no estimate sort last."""
        return sorted(
            self.detectors.values(),
            key=lambda d: (d.overall.mean if d.overall.mean is not None else float("-inf")),
            reverse=True,
        )

    def families(self) -> list[str]:
        """Every family any detector measured, sorted."""
        seen: set[str] = set()
        for d in self.detectors.values():
            seen.update(d.per_family)
        return sorted(seen)

    def access_levels_present(self) -> list[str]:
        present = {d.access_level for d in self.detectors.values()}
        ordered = [lv for lv in ACCESS_LEVELS if lv in present]
        return ordered + sorted(present - set(ordered))

    def duplicates(self) -> list[ReplicatedDetector]:
        """Detectors held out of every access-level aggregate as duplicates."""
        return [d for d in self.standings() if d.duplicate_of is not None]

    def detectors_at(self, level: str) -> list[ReplicatedDetector]:
        """Detectors at ``level`` that count toward its aggregate, best mean first."""
        return [
            d for d in self.standings() if d.access_level == level and d.counted_in_ladder
        ]

    def control_check(
        self, level: str = CONTROL_LEVEL, reference: float = 0.5
    ) -> ChanceControl:
        """Whether every detector at ``level`` has an interval containing ``reference``.

        Computed from the artifact rather than asserted, so the page states a verdict the
        committed data actually supports and a regenerated board that broke the control
        would change the page instead of being contradicted by it.
        """
        detectors = tuple(self.detectors_at(level))
        stats = self.all_detectors.level(level)
        return ChanceControl(
            level=level,
            reference=reference,
            detectors=detectors,
            n_containing=sum(1 for d in detectors if d.overall.contains(reference)),
            highest=next(
                (d for d in detectors if d.overall.mean is not None), None
            ),
            level_max=stats.max_auroc if stats else None,
            level_mean=stats.mean_auroc if stats else None,
        )

    @property
    def ci_percent(self) -> int:
        level = _num(self.method.get("ci_level")) or 0.95
        return int(round(level * 100))

    @property
    def bootstrap_resamples(self) -> Optional[int]:
        v = self.method.get("bootstrap_resamples")
        return int(v) if isinstance(v, int) and not isinstance(v, bool) else None

    @property
    def bootstrap_seed(self) -> Optional[int]:
        v = self.method.get("bootstrap_seed")
        return int(v) if isinstance(v, int) and not isinstance(v, bool) else None

    @property
    def seeds_per_variant(self) -> Optional[int]:
        """Rollout seeds per variant per cell, if every draw used the same count."""
        counts = {
            int(d["n_seeds"])
            for d in self.draws
            if isinstance(d.get("n_seeds"), int) and not isinstance(d.get("n_seeds"), bool)
        }
        return next(iter(counts)) if len(counts) == 1 else None


# --------------------------------------------------------------------------- parsing
def _intervals(blob: Any) -> dict[str, Interval]:
    if not isinstance(blob, Mapping):
        return {}
    return {
        str(k): Interval.from_json(v) for k, v in blob.items() if isinstance(v, Mapping)
    }


def _level_stats(blob: Any) -> dict[str, LevelStats]:
    levels: dict[str, LevelStats] = {}
    if not isinstance(blob, Mapping):
        return levels
    for level, payload in blob.items():
        if not isinstance(payload, Mapping):
            continue
        freq_raw = payload.get("best_detector_frequency")
        freq = (
            {str(k): int(v) for k, v in freq_raw.items() if isinstance(v, int)}
            if isinstance(freq_raw, Mapping)
            else {}
        )
        levels[str(level)] = LevelStats(
            level=str(level),
            max_auroc=Interval.from_json(payload.get("max_auroc") or {}),
            mean_auroc=Interval.from_json(payload.get("mean_auroc") or {}),
            best_detector_frequency=freq,
        )
    return levels


def _separations(blob: Any) -> dict[str, LevelSeparation]:
    seps: dict[str, LevelSeparation] = {}
    if not isinstance(blob, Mapping):
        return seps
    for key, payload in blob.items():
        if not isinstance(payload, Mapping):
            continue
        # Keys look like "L2_minus_L1_max": <higher>_minus_<lower>_<statistic>.
        parts = str(key).split("_")
        if len(parts) < 4 or parts[1] != "minus":
            continue
        higher, lower, statistic = parts[0], parts[2], parts[-1]
        seps[str(key)] = LevelSeparation(
            lower=lower,
            higher=higher,
            statistic=str(payload.get("statistic") or statistic),
            difference=Interval.from_json(payload),
            excludes_zero=bool(payload.get("excludes_zero")),
            replicates_with_hi_above_lo=int(payload.get("replicates_with_hi_above_lo") or 0),
            n_paired=int(payload.get("n_paired") or 0),
            denominators_equal=bool(payload.get("denominators_equal")),
        )
    return seps


def _str_tuple(values: Any) -> tuple[str, ...]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, Mapping)):
        return ()
    return tuple(str(v) for v in values)


def is_replicated_schema(data: Mapping[str, Any]) -> bool:
    """Whether ``data`` is the replicated artifact rather than one of the point schemas.

    Keyed on ``detectors`` being a *mapping*: ``leaderboard/v5_leaderboard.json`` also
    has a ``detectors`` key, but it holds an integer count, and dispatching on the key
    name alone would misread that file as replicated.
    """
    return isinstance(data.get("detectors"), Mapping)


def load_replicated_leaderboard(path: str | Path) -> ReplicatedLeaderboard:
    """Parse ``leaderboard/v5_replicated.json``.

    Raises:
        ValueError: if the file is not the replicated schema.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not is_replicated_schema(data):
        raise ValueError(
            f"{path}: not a replicated leaderboard (expected a 'detectors' object of "
            "per-detector intervals, as written by scripts/aggregate_replication.py)"
        )

    supervision = data.get("supervision") if isinstance(data.get("supervision"), Mapping) else {}
    label_fitted = _str_tuple(supervision.get("label_fitted_detectors"))
    fitted_set = set(label_fitted)

    detectors: dict[str, ReplicatedDetector] = {}
    for name, blob in data["detectors"].items():
        if not isinstance(blob, Mapping):
            continue
        cells = blob.get("cells_measured")
        detectors[str(name)] = ReplicatedDetector(
            name=str(name),
            access_level=str(blob.get("access_level") or "?"),
            overall=Interval.from_json(blob),
            per_family=_intervals(blob.get("per_family")),
            cells_measured=(
                int(cells) if isinstance(cells, int) and not isinstance(cells, bool) else None
            ),
            label_fitted=str(name) in fitted_set,
            duplicate_of=duplicate_source(str(name)),
        )

    degeneracy = data.get("degeneracy") if isinstance(data.get("degeneracy"), Mapping) else {}
    disagreement = (
        data.get("statistic_disagreement")
        if isinstance(data.get("statistic_disagreement"), Mapping)
        else {}
    )
    n = data.get("n_replicates")

    return ReplicatedLeaderboard(
        path=path,
        n_replicates=int(n) if isinstance(n, int) and not isinstance(n, bool) else 0,
        method=data.get("method") if isinstance(data.get("method"), Mapping) else {},
        draws=[d for d in (data.get("draws") or []) if isinstance(d, Mapping)],
        detectors=detectors,
        all_detectors=LadderPartition(
            name="all detectors",
            levels=_level_stats(data.get("access_levels")),
            separations=_separations(data.get("adjacent_level_separation")),
        ),
        unsupervised_only=LadderPartition(
            name="unsupervised only",
            levels=_level_stats(supervision.get("access_levels_unsupervised_only")),
            separations=_separations(
                supervision.get("adjacent_level_separation_unsupervised_only")
            ),
        ),
        label_fitted_detectors=label_fitted,
        supervision_note=str(supervision.get("why_it_matters") or ""),
        zero_variance_detectors=_str_tuple(degeneracy.get("detectors_with_zero_variance")),
        zero_variance_note=str(degeneracy.get("note") or ""),
        statistic_disagreement=_str_tuple(
            disagreement.get("rungs_where_max_and_mean_disagree_on_separation")
        ),
        statistic_disagreement_note=str(disagreement.get("note") or ""),
    )
