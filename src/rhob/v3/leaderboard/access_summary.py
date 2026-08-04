"""Per-access-level aggregates over a leaderboard, with duplicate detectors excluded.

This is the single place that answers "how well does level Lk do?". It reports both
the mean over the detectors at that level and the **max** (the best detector at that
level), because the two answer different questions and only the max speaks to the
claim RHOB actually cares about -- "is the information present at this access level?".
A level's mean is dragged down by however many weak detectors happen to have been
written for it, so a mean-only ladder measures the detector suite's composition as much
as the benchmark's structure. On the committed v5 board the two differ sharply at L2:
mean 0.7213 over 7 detectors, but max 0.9750 (Behavioral Threshold).

Detectors listed in :data:`~rhob.detectors.redundancy.DUPLICATE_DIAGNOSTICS` are
excluded from every level's statistics and reported separately, so a relabelled copy of
another detector cannot be counted twice. See that module for why ``Perfect Feature
Oracle`` is on the list.

The spread reported alongside the mean is the standard deviation **across the detectors
at a level**. It is a measure of how heterogeneous that level's detector suite is, not a
confidence interval on the level's mean and not a measurement uncertainty: each detector
contributes one number from a single shared evaluation draw, so these values are not
independent replicates and cannot be read as error bars.

Where the error bars actually are
---------------------------------
Sampling uncertainty on these figures is estimated separately, by re-running the whole
suite at independent ``(layout_seed, seed_base)`` draws and bootstrapping over the
replicates -- ``scripts/replicate_leaderboard.py`` and ``scripts/aggregate_replication.py``,
published to ``leaderboard/v5_replicated.json``. That aggregation calls **this** function
once per replicate rather than reimplementing the ladder, so the interval and the point
estimate it decorates are computed under identical rules (duplicates held out, degenerate
families held out of L0 and only L0). Quote a level's number from here; quote its
uncertainty from there. A single draw's ``std_auroc`` is not a substitute.

Degenerate families are held out of L0, and only L0
-----------------------------------------------------
RHOB's L0 rung is a *negative control*: "reward-only detectors sit at chance on a
carefully matched proxy". Six registered families do not have a matched proxy, they have
a **constant** one, and the admission ledger marks every one of their cells DEGENERATE
rather than certified (see ``admission/ADMISSION_LEDGER.md``). On a constant proxy every
L0 detector reads 0.5 for a reason that has nothing to do with matching, so averaging
those families into the L0 figure pads the control with cells that could not have come
out any other way. Pass the ledger's ``degenerate_families`` to
:func:`summarize_access_levels` and they are withheld from L0, with the withheld set and
the with-degenerate figure reported alongside so the exclusion is auditable rather than
silent.

The exclusion applies to **L0 only, deliberately**. Degeneracy is a property of the
proxy reward, which is the only channel L0 reads; the L1/L2/L3 detectors read state
visitation, behavior and true reward, and a constant-proxy family is a perfectly ordinary
benchmark item for them. Dropping it from those rungs would discard real measurements to
fix a problem they do not have.

One consequence has to be stated rather than hidden: excluding families means the L0 rung
is computed from the per-family AUROCs (a family-weighted mean over the families kept),
while the other rungs use each detector's ``overall_auroc`` (a cell-weighted mean over
all 123 cells), because the artifact records AUROCs per family and not per cell. On the
committed v5 board that re-weighting is worth 0.0003 AUROC at L0 -- family-weighted over
all 33 families is 0.4895 against the cell-weighted 0.4898 -- so it is not what moves the
number; the exclusion is. ``mean_auroc_including_degenerate`` is computed the same
family-weighted way for exactly this reason, so the difference between it and
``mean_auroc`` is attributable to the held-out families alone.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from rhob.detectors.redundancy import duplicate_source

ACCESS_LEVELS = ("L0", "L1", "L2", "L3")

#: The only level whose statistic is a claim about the proxy reward, and therefore the
#: only one a degenerate proxy invalidates. See the module docstring.
PROXY_LEVEL = "L0"


@dataclass
class AccessLevelSummary:
    """Aggregate statistics for one access level."""

    level: str
    #: (detector_name, overall_auroc) for every detector counted, best first.
    scored: list[tuple[str, float]] = field(default_factory=list)
    #: (detector_name, duplicated_detector_name) for every detector left out.
    excluded_duplicates: list[tuple[str, str]] = field(default_factory=list)
    #: Families held out of this level's statistics because the admission ledger marks
    #: their proxy degenerate. Empty on every level but :data:`PROXY_LEVEL`, and empty
    #: there too unless ``degenerate_families`` was passed.
    excluded_families: list[str] = field(default_factory=list)
    #: Families this level's numbers were actually computed over. ``None`` when no
    #: exclusion was applied and the level is reporting each detector's own
    #: ``overall_auroc``, which is a cell-weighted figure over the whole board.
    n_families_scored: int | None = None
    #: The same statistic with the degenerate families put back, so the cost of the
    #: exclusion is visible next to the exclusion. ``None`` when nothing was excluded.
    mean_auroc_including_degenerate: float | None = None

    @property
    def n_detectors(self) -> int:
        return len(self.scored)

    @property
    def mean_auroc(self) -> float | None:
        if not self.scored:
            return None
        return sum(a for _, a in self.scored) / len(self.scored)

    @property
    def std_auroc(self) -> float | None:
        """Population SD across the detectors at this level (see module docstring: not an error bar)."""
        mean = self.mean_auroc
        if mean is None:
            return None
        return (sum((a - mean) ** 2 for _, a in self.scored) / len(self.scored)) ** 0.5

    @property
    def max_auroc(self) -> float | None:
        return self.scored[0][1] if self.scored else None

    @property
    def best_detector(self) -> str | None:
        return self.scored[0][0] if self.scored else None


#: The committed ledger, relative to this file: ``src/rhob/v3/leaderboard/`` -> repo root.
_LEDGER_PATH = Path(__file__).resolve().parents[4] / "admission" / "admission_ledger.json"


def degenerate_families_from_ledger(path: Path | None = None) -> list[str]:
    """The families the admission ledger could not certify a matched proxy for.

    Read from the committed artifact rather than hardcoded, so the exclusion tracks the
    ledger: a family that gets an informative-but-matched proxy rejoins the L0 control by
    being re-certified, with no second list to remember to update.

    Returns an empty list when the ledger is absent (a source checkout that has never run
    ``scripts/admission_ledger.py``), which reproduces the pre-exclusion behaviour rather
    than failing the caller. A ledger that exists but does not answer the question is a
    different matter and raises.
    """
    ledger = _LEDGER_PATH if path is None else path
    if not ledger.exists():
        return []
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    families = payload.get("summary", {}).get("degenerate_families")
    if not isinstance(families, list):
        raise ValueError(
            f"{ledger} has no summary.degenerate_families block; it was written by an "
            "older version of scripts/admission_ledger.py and cannot say which families "
            "to hold out of the L0 negative control"
        )
    return sorted(str(f) for f in families)


def _family_mean(per_family: Mapping[str, object], skip: frozenset[str]) -> float | None:
    """Mean of a detector's per-family AUROCs, over the families not in ``skip``."""
    kept = [
        float(v)
        for k, v in per_family.items()
        if k not in skip and isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    return sum(kept) / len(kept) if kept else None


def summarize_access_levels(
    results: Mapping[str, Mapping[str, object]],
    *,
    exclude_duplicates: bool = True,
    degenerate_families: Iterable[str] = (),
) -> dict[str, AccessLevelSummary]:
    """Group a leaderboard's per-detector overall AUROCs by access level.

    Args:
        results: The ``"results"`` block of a leaderboard JSON: detector name ->
            dict with at least ``access_level`` and ``overall_auroc``. Entries whose
            detector errored during evaluation (no ``overall_auroc``, or ``null``)
            are skipped rather than treated as zero.
        exclude_duplicates: When True (the default, and the only correct setting for
            anything published), detectors registered as duplicates are held out of
            the statistics and listed in ``excluded_duplicates`` instead. Pass False
            only to reproduce the pre-fix numbers for comparison.
        degenerate_families: Families whose proxy the admission ledger could not certify
            as matched (its ``summary.degenerate_families``). They are withheld from
            :data:`PROXY_LEVEL` and from nowhere else -- see the module docstring for
            why the asymmetry is the point rather than an oversight. Every affected
            detector must carry a ``per_family`` block; a leaderboard that cannot say
            which families a number came from cannot have families taken out of it, and
            quietly leaving them in would be the defect this argument exists to remove.

    Returns:
        A dict keyed by every level in :data:`ACCESS_LEVELS`, always present even when
        a level has no detectors, so callers can render a complete ladder.

    Raises:
        ValueError: if ``degenerate_families`` is non-empty and a scored detector at
            :data:`PROXY_LEVEL` has no usable ``per_family`` block.
    """
    summaries = {lv: AccessLevelSummary(level=lv) for lv in ACCESS_LEVELS}
    skip = frozenset(degenerate_families)
    with_degenerate: list[float] = []

    for name, info in results.items():
        level = info.get("access_level")
        if level not in summaries:
            continue
        auroc = info.get("overall_auroc")
        if not isinstance(auroc, (int, float)) or isinstance(auroc, bool):
            continue
        source = duplicate_source(name) if exclude_duplicates else None
        if source is not None:
            summaries[level].excluded_duplicates.append((name, source))
            continue
        if skip and level == PROXY_LEVEL:
            per_family = info.get("per_family")
            if not isinstance(per_family, Mapping):
                raise ValueError(
                    f"cannot hold degenerate families out of {name!r}: the leaderboard "
                    "entry has no per_family block, so there is no way to tell which "
                    "families its overall AUROC was computed over"
                )
            kept = _family_mean(per_family, skip)
            everything = _family_mean(per_family, frozenset())
            if kept is None or everything is None:
                raise ValueError(
                    f"cannot hold degenerate families out of {name!r}: nothing is left "
                    f"of its per_family block after removing {sorted(skip)}"
                )
            summaries[level].scored.append((name, kept))
            with_degenerate.append(everything)
            summaries[level].n_families_scored = len(
                [k for k in per_family if k not in skip]
            )
            summaries[level].excluded_families = sorted(
                k for k in per_family if k in skip
            )
        else:
            summaries[level].scored.append((name, float(auroc)))

    if with_degenerate:
        summaries[PROXY_LEVEL].mean_auroc_including_degenerate = sum(
            with_degenerate
        ) / len(with_degenerate)

    for summary in summaries.values():
        summary.scored.sort(key=lambda pair: pair[1], reverse=True)
    return summaries


def render_access_summary_md(summaries: Mapping[str, AccessLevelSummary]) -> str:
    """Render the access-level ladder as a Markdown table, mean *and* max side by side."""
    lines = [
        "| Access level | Detectors | Mean AUROC | SD across detectors | Best AUROC | Best detector |",
        "|---|---|---|---|---|---|",
    ]
    for lv in ACCESS_LEVELS:
        s = summaries.get(lv)
        if s is None or s.n_detectors == 0:
            lines.append(f"| {lv} | 0 | - | - | - | - |")
            continue
        lines.append(
            f"| {lv} | {s.n_detectors} | {s.mean_auroc:.4f} | {s.std_auroc:.4f} | "
            f"{s.max_auroc:.4f} | {s.best_detector} |"
        )

    excluded = [
        (name, source, lv)
        for lv in ACCESS_LEVELS
        for name, source in (summaries[lv].excluded_duplicates if lv in summaries else [])
    ]
    lines.append("")
    lines.append(
        "SD is the spread across the detectors at a level, not a confidence interval: "
        "the detectors share one evaluation draw, so these are not independent replicates."
    )

    proxy = summaries.get(PROXY_LEVEL)
    if proxy is not None and proxy.excluded_families:
        including = proxy.mean_auroc_including_degenerate
        lines.append("")
        lines.append(
            f"{PROXY_LEVEL} is computed over {proxy.n_families_scored} families. "
            f"{len(proxy.excluded_families)} are held out because the admission ledger "
            f"marks their proxy DEGENERATE -- it is constant, so an L0 detector reads "
            f"0.5 on them by construction and not by matching: "
            + ", ".join(f"**{f}**" for f in proxy.excluded_families)
            + ". Putting them back would read "
            + (f"{including:.4f}" if including is not None else "-")
            + f" instead of {proxy.mean_auroc:.4f}. They are held out of {PROXY_LEVEL} "
            "only; the other levels do not read the proxy reward and score these "
            "families like any other."
        )
    if excluded:
        lines.append("")
        for name, source, lv in excluded:
            lines.append(
                f"Excluded from the {lv} aggregate: **{name}** is a relabelled duplicate of "
                f"**{source}** (identical scores on every family) and is reported only as a "
                f"cross-check, never as an independent measurement at {lv}."
            )
    return "\n".join(lines)
