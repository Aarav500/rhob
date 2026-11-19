"""Per-access-level aggregates over a leaderboard, with duplicate detectors excluded.

This is the single place that answers "how well does level Lk do?". It reports both
the mean over the detectors at that level and the **max** (the best detector at that
level), because the two answer different questions and only the max speaks to the
claim RHOB actually cares about -- "is the information present at this access level?".
A level's mean is dragged down by however many weak detectors happen to have been
written for it, so a mean-only ladder measures the detector suite's composition as much
as the benchmark's structure. On the committed v5 board (regenerated 2026-11-19) the two
differ sharply at L2: mean 0.5716 over 7 detectors, but max 0.9430 (Trajectory MLP); and
at L1: mean 0.5701 over 8, but max 0.9270 (State Divergence). The L1 figures are over the
35 cells (8 families) that emit ``state_counts`` and L2's over all 123, so the L1 and L2
maxima are not like for like.

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
carefully matched proxy". Six registered families have a proxy the admission ledger
cannot measure: it is constant, nearly constant, or identical in every run, so an L0
statistic can order almost none of their run pairs, and the ledger marks every one of
their cells DEGENERATE rather than certified (see ``admission/ADMISSION_LEDGER.md``). On
such a proxy an L0 detector's AUROC is the tie convention (0.5) or rounding error, for a reason
that has nothing to do with matching, so averaging those families into the L0 figure
pads the control with cells that could not have come out any other way. Pass the
ledger's ``degenerate_families`` to :func:`summarize_access_levels` and they are withheld
from L0, with the withheld set and the with-degenerate figure reported alongside so the
exclusion is auditable rather than silent.

DEGENERATE is the only status held out. Since 2026-11-19 the ledger covers all 33
families and 123 cells. It admits 42, certifying a matched proxy on each; it marks 30
DEGENERATE; and it measures 51 NOT ADMITTED (``proxy_matched`` failed on 24,
``proxy_distribution_matched`` on 40, ``behavioral_separated`` on 1; a cell can fail
more than one). 50 of the 51 fail a proxy criterion; the 51st,
``proxy_correlation_gaming`` @ 0.60, passes both proxy criteria and fails only
``behavioral_separated``. The NOT ADMITTED cells are still in the L0 mean. Whether the
negative control should also leave them out is an open question for the author
(``ROADMAP.md``).

The exclusion applies to **L0 only, deliberately**. Degeneracy is a property of the
proxy reward, which is the only channel L0 reads; the L1/L2/L3 detectors read state
visitation, behavior and true reward, and a constant-proxy family is a perfectly ordinary
benchmark item for them. Dropping it from those rungs would discard real measurements to
fix a problem they do not have.

One consequence has to be stated rather than hidden: excluding families means the L0 rung
is computed from the per-family AUROCs (a family-weighted mean over the families kept),
while the other rungs use each detector's ``overall_auroc`` (a cell-weighted mean over
the cells it has a value on, ``cells_measured``), because the artifact records AUROCs per
family and not per cell. On the committed v5 board that re-weighting is worth 0.0005
AUROC at L0 -- family-weighted over every family each detector has a value on is 0.4898
against the cell-weighted 0.4903 -- so it is not what moves the number; the exclusion
is. ``mean_auroc_including_degenerate`` is computed the same family-weighted way for
exactly this reason, so the difference between it and ``mean_auroc`` is attributable to
the held-out families alone.

A family that is ``null`` in a detector's ``per_family`` block is left out of that
detector's mean, never read as 0.5. A detector is ``null`` on a family that does not emit
the channel it reads (every L1 detector on 25 families), and on a family shorter than the
``min_episodes`` it declares. Reward Skewness is the only L0 detector that declares one
(100 episodes), so it is the only one computed over fewer families than the rest of the
rung; :attr:`AccessLevelSummary.n_families_scored_by_detector` records how many, and the
rendered table names it.

Eight other L0 detectors also need more episodes than some families run, and declare
none. Below the length their windows need, each returns a constant score (0.5, or 0.0
for Reward CUSUM), so on a family that short it ties on every pair and the board records
an AUROC of 0.5. That 0.5 is a fallback, not a measurement, and this module cannot tell
it apart from one: it is a number in ``per_family`` and it is averaged in.
:data:`UNDECLARED_HORIZON_FALLBACKS` lists the eight and the length each needs, and the
rendered table says so. On the board
regenerated 2026-11-19, Reward Peak, Reward Autocorrelation, Reward Trend and Variance
Window carry their fallback on all 19 families with horizon 40 or 60 (59 cells), and
Reward CUSUM, Max Plateau, Reward MLP and Gradient Reversal on all 7 with horizon 40
(22 cells). That is 104 of the per-family entries behind the L0 figures quoted from that
board (mean 0.4876, best 0.5351, Reward CUSUM). Recomputed with those entries left out,
the L0 mean would read 0.4834 and the best detector would still be Reward CUSUM, at
0.5474 over 20 families. Declaring each detector's horizon and re-scoring the boards is
an open item in ``ROADMAP.md``; it changes the L0 ladder.

The same kind of fallback is in the L1 and L2 figures: every L1 detector on
``continuous_camping``, whose 40 episodes are below each L1 window, and three L2
detectors (Feature Consistency, Feature Magnitude and Reward-Feature Correlation) on the
short-horizon families. :func:`undeclared_horizon_note` names only the L0 ones.
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

#: L0 detectors that declare no ``min_episodes`` and return a constant score from
#: ``classify`` on a run shorter than this many episodes (the length guard in each
#: ``rhob.detectors.l0_*`` module). On a family that short the constant ties on every
#: pair, so the board records an AUROC of 0.5 that is a fallback, not a measurement, and
#: :func:`summarize_access_levels` averages it in like any other number. Reward Skewness
#: is not listed: it declares ``min_episodes`` (100) and is ``null`` below it. Nor are
#: Reward Variance Ratio, Reward KDE and Spectral Reward, whose guards (21, 22 and 24
#: episodes) are below the shortest horizon a registered family runs (40).
#: ``tests/test_detectors/test_l0_window_fallbacks.py`` holds this table to the detector
#: code. An entry leaves it when its detector declares its horizon (``ROADMAP.md``).
UNDECLARED_HORIZON_FALLBACKS: dict[str, int] = {
    "Reward Autocorrelation": 100,
    "Reward Peak": 100,
    "Reward Trend": 100,
    "Variance Window": 100,
    "Gradient Reversal": 60,
    "Max Plateau": 50,
    "Reward CUSUM": 50,
    "Reward MLP": 50,
}


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
    #: Families this level's numbers were actually computed over: every family at least
    #: one detector has a value on, less the held-out ones. ``None`` when no exclusion was
    #: applied and the level is reporting each detector's own ``overall_auroc``, which is
    #: a cell-weighted figure over the cells that detector has a value on.
    n_families_scored: int | None = None
    #: Per detector, the families its figure was averaged over, when an exclusion was
    #: applied. A detector that is not applicable on some families (``null`` in its
    #: ``per_family`` block) has fewer than :attr:`n_families_scored`. A fallback 0.5 is
    #: not ``null`` and is counted (see :data:`UNDECLARED_HORIZON_FALLBACKS`).
    n_families_scored_by_detector: dict[str, int] = field(default_factory=dict)
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
    """The families whose proxy the admission ledger marks DEGENERATE (could not measure).

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


def _is_auroc(value: object) -> bool:
    """A measured per-family value; ``null`` (not applicable) is not one."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _family_mean(per_family: Mapping[str, object], skip: frozenset[str]) -> float | None:
    """Mean of a detector's per-family AUROCs, over the families not in ``skip``."""
    kept = [float(v) for k, v in per_family.items() if k not in skip and _is_auroc(v)]
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
            dict with at least ``access_level`` and ``overall_auroc``. Entries with no
            ``overall_auroc``, or ``null`` (the detector errored, or measured no cell)
            are skipped rather than treated as zero.
        exclude_duplicates: When True (the default, and the only correct setting for
            anything published), detectors registered as duplicates are held out of
            the statistics and listed in ``excluded_duplicates`` instead. Pass False
            only to reproduce the pre-fix numbers for comparison.
        degenerate_families: Families whose proxy the admission ledger could not measure
            and marks DEGENERATE (its ``summary.degenerate_families``). They are withheld from
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
    families_scored: set[str] = set()

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
            measured = {k for k, v in per_family.items() if k not in skip and _is_auroc(v)}
            families_scored |= measured
            summaries[level].n_families_scored_by_detector[name] = len(measured)
            summaries[level].excluded_families = sorted(
                k for k in per_family if k in skip
            )
        else:
            summaries[level].scored.append((name, float(auroc)))

    if with_degenerate:
        summaries[PROXY_LEVEL].mean_auroc_including_degenerate = sum(
            with_degenerate
        ) / len(with_degenerate)
        summaries[PROXY_LEVEL].n_families_scored = len(families_scored)

    for summary in summaries.values():
        summary.scored.sort(key=lambda pair: pair[1], reverse=True)
    return summaries


def _fewer_families_note(summary: AccessLevelSummary) -> str:
    """`` (Reward Skewness over 8 of them, ...)`` for detectors measured on fewer families."""
    total = summary.n_families_scored
    fewer = [
        (name, n)
        for name, n in sorted(summary.n_families_scored_by_detector.items())
        if total is not None and n < total
    ]
    if not fewer:
        return ""
    return (
        " ("
        + "; ".join(
            f"{name} over {n} of them: it is not applicable on the other {total - n}"
            for name, n in fewer
        )
        + ")"
    )


_NUMBER_WORDS = ("No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")


def undeclared_horizon_note(summary: AccessLevelSummary) -> str:
    """Name the detectors at this level whose short-run values are a fallback 0.5.

    Empty when no detector in :data:`UNDECLARED_HORIZON_FALLBACKS` is counted at this
    level. The Markdown table and ``scripts/plot_v5_results.py``'s figure both print it,
    so the two say the same thing.
    """
    counted = {name for name, _ in summary.scored}
    by_length: dict[int, list[str]] = {}
    for name, length in UNDECLARED_HORIZON_FALLBACKS.items():
        if name in counted:
            by_length.setdefault(length, []).append(name)
    if not by_length:
        return ""
    n = sum(len(names) for names in by_length.values())
    count = _NUMBER_WORDS[n] if n < len(_NUMBER_WORDS) else str(n)
    groups = "; ".join(
        f"{', '.join(sorted(names))} below {length} episodes"
        for length, names in sorted(by_length.items(), reverse=True)
    )
    subject, verbs = (
        (f"{count} {summary.level} detectors", ("declare", "return", "their"))
        if n != 1
        else (f"One {summary.level} detector", ("declares", "returns", "its"))
    )
    note = (
        f"{subject} {verbs[0]} no minimum horizon and {verbs[1]} a constant score on a "
        f"run shorter than {verbs[2]} windows ({groups}). On a family that short the "
        "constant ties on every pair and is recorded as 0.5. That 0.5 is a fallback, not "
        f"a measurement, and it is counted in the {summary.level} figures above."
    )
    if "Reward Skewness" in counted:
        note += " Only Reward Skewness declares its horizon; it is null below it instead."
    return note


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
            f"{PROXY_LEVEL} is computed over {proxy.n_families_scored} families"
            + _fewer_families_note(proxy)
            + f". {len(proxy.excluded_families)} are held out because the admission "
            "ledger marks their proxy DEGENERATE -- constant, nearly constant, or "
            "identical in every run, so an L0 detector's AUROC on them is the tie "
            "convention (0.5) or rounding error, not evidence of matching: "
            + ", ".join(f"**{f}**" for f in proxy.excluded_families)
            + ". Putting them back would read "
            + (f"{including:.4f}" if including is not None else "-")
            + f" instead of {proxy.mean_auroc:.4f}. They are held out of {PROXY_LEVEL} "
            "only; the other levels do not read the proxy reward and score these "
            "families like any other. Cells the ledger measures NOT ADMITTED are not "
            "held out and stay in the mean; whether they should is an open question "
            "(ROADMAP.md)."
        )
    fallback = undeclared_horizon_note(proxy) if proxy is not None else ""
    if fallback:
        lines.append("")
        lines.append(fallback)
    if excluded:
        lines.append("")
        for name, source, lv in excluded:
            lines.append(
                f"Excluded from the {lv} aggregate: **{name}** is a relabelled duplicate of "
                f"**{source}** (identical scores on every family) and is reported only as a "
                f"cross-check, never as an independent measurement at {lv}."
            )
    return "\n".join(lines)
