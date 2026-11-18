"""Verdicts on coded preconditions, plus the aggregation a published table may use.

Two layers, and which is which
------------------------------
The paper's three diagnostics -- D1 resolution, D2 power, D3 channel digest -- read a
benchmark's *artifacts*: score vectors, designs, wiring. D1 and D2 are computed by the
standalone ``cannotfail`` package in ``tools/cannotfail``. This module works one layer
up, on *coded rows*: a person's reading of what a precondition asserts and what backs
it. An earlier version of this docstring said the checks below were reconstructed
because the paper's three were unknown, and that they should be renamed if the paper's
turned out different. They did, but renaming would be wrong, because these are not
artifact diagnostics. They are the verdicts a row receives, and the first of them
consumes D1's measurement directly:

0. :func:`diagnose_constant_statistic` -- did D1, measured on the shipped design, find
   that the criterion's statistic took one value everywhere? Then its verdict is fixed
   by construction, whatever its claim shape. This exists because the checks below
   missed RHOB's ``onset_localizable``, a threshold criterion whose ``onset_std`` is
   exactly 0.0 in all 50 ledger cells: as a threshold claim it was never offered to
   the degeneracy check, and it came back merely ``undemonstrated``.

The other three correspond to defects found in this repository and documented in
``README.md``:

1. :func:`diagnose_direction` -- is an equivalence claim being tested with a
   difference test? This is the pre-audit ``proxy_matched``: ``abs(mean_auroc - 0.5)
   < 0.10``, which passed a genuinely leaking family at a true AUROC of 0.611
   **43.5%** of the time, and which noise made *easier* to pass.

2. :func:`diagnose_degeneracy` -- could the statistic have taken another value? This
   is the post-fix defect: an AUROC over a constant proxy is 0.5 by the tie
   convention, the bootstrap interval collapses to ``[0.5000, 0.5000]``, and the
   equivalence test then certifies against *any* margin. 15 of the first 35 cells
   passed this way.

3. :func:`diagnose_demonstrated_failure` -- did anything published show a negative
   verdict is reachable? This is the thesis criterion, and the weakest of the three:
   it is a statement about what was published, not about the check.

What this module will not compute
---------------------------------
There is no function here that returns "N of M preconditions are supported by an
equivalence test" over all preconditions. That denominator mixes criteria for which
a difference test is correct into a rate presented as a defect rate, and RHOB itself
would score 2 of 6 under it while being sound on both criteria where it matters.
:class:`SurveyResult` reports the inverted rate over equivalence-shaped criteria and
exposes the raw counts; anyone wanting the looser number can compute it and own it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from rhob.audit.criterion import (
    CodedCriterion,
    CriterionFinding,
    FailureEvidence,
    StatTest,
    Verdict,
)

__all__ = [
    "diagnose_constant_statistic",
    "diagnose_direction",
    "diagnose_degeneracy",
    "diagnose_demonstrated_failure",
    "diagnose",
    "run_survey",
    "SurveyResult",
    "InadmissibleRow",
]


def diagnose_constant_statistic(row: CodedCriterion) -> CriterionFinding | None:
    """Flag a criterion whose statistic was measured to take one value everywhere.

    Applies to every claim shape. A difference or threshold check is normally the safe
    direction, because noise pushes it toward failing -- but a constant has no noise,
    so a threshold on it passes in every cell or fails in every cell, and either way no
    observation the design can produce changes the verdict. A FAIL fixed this way is no
    more a measurement than a pass: RHOB's ``true_reward_diverges`` reported "measured
    and found not to diverge" on arms tied by arithmetic until ``584ba13``, after which
    the gate reports that case as DEGENERATE.
    """
    if row.observed_distinct_values != 1:
        return None

    return CriterionFinding(
        criterion=row,
        verdict=Verdict.DEGENERATE,
        reason=(
            "D1, measured: the criterion's statistic took a single value across every "
            "cell of the shipped design, so its verdict is fixed by construction and "
            "says nothing about any cell"
        ),
        remedy=(
            "report it as not measured rather than passed, and find what fixes the "
            "statistic -- usually a quantity the generator sets rather than samples"
        ),
    )


def diagnose_direction(row: CodedCriterion) -> CriterionFinding | None:
    """Flag an equivalence claim tested as a difference, or not tested at all.

    Returns ``None`` when the criterion is not equivalence-shaped -- a difference
    test on a difference claim is correct, and saying so is the point.
    """
    if not row.is_equivalence_shaped:
        return None

    if row.test_used is StatTest.EQUIVALENCE_TEST:
        return None

    if row.test_used is StatTest.NONE:
        return CriterionFinding(
            criterion=row,
            verdict=Verdict.INVERTED,
            reason=(
                "asserts an equivalence but nothing computes a verdict for it, so no "
                "observation can contradict it"
            ),
            remedy="state the margin and test it, or drop the criterion",
        )

    kind = (
        "a difference test"
        if row.test_used is StatTest.DIFFERENCE_TEST
        else "a point estimate against a constant"
    )
    return CriterionFinding(
        criterion=row,
        verdict=Verdict.INVERTED,
        reason=(
            f"asserts an equivalence but is tested with {kind}: failing to find a "
            "difference is reported as there being none, so noise makes the criterion "
            "easier to pass rather than harder"
        ),
        remedy=(
            "replace with an equivalence test (TOST) at the margin already implied by "
            "the criterion; the published claim does not change, the sample size does"
        ),
    )


def diagnose_degeneracy(row: CodedCriterion) -> CriterionFinding | None:
    """Flag an equivalence test with nothing guarding that its statistic can vary.

    Only equivalence tests carry this verdict. On a difference test ties push the
    statistic toward the null and therefore toward failing, so a tie cannot certify a
    false pass, which is the defect this check reads off the design. It can still
    publish "measured and found false" when nothing was measured: RHOB's
    ``true_reward_diverges`` did, on arms tied by arithmetic, and since ``584ba13`` the
    gate reports that case as DEGENERATE. Finding it is not this function's job: a
    coded row is flagged for it by :func:`diagnose_constant_statistic`, and only when
    the coder recorded ``observed_distinct_values``.
    """
    if not row.is_equivalence_shaped:
        return None
    if row.test_used is not StatTest.EQUIVALENCE_TEST:
        return None  # already reported by diagnose_direction
    if row.resolution_guarded:
        return None

    return CriterionFinding(
        criterion=row,
        verdict=Verdict.DEGENERACY_UNGUARDED,
        reason=(
            "equivalence test with no guard that the statistic could have taken "
            "another value: where every compared value ties, the interval collapses "
            "to a point and the criterion certifies against any margin"
        ),
        remedy=(
            "report a third outcome for 'not measurable' and hold those cells out of "
            "any control that assumes the criterion was tested"
        ),
    )


def diagnose_demonstrated_failure(row: CodedCriterion) -> CriterionFinding | None:
    """Flag a criterion with nothing published showing a negative is reachable.

    The mildest of the three, and the one most likely to be a documentation gap
    rather than a defect: a benchmark may well reject things routinely without ever
    publishing a rejection. That is why the verdict is ``UNDEMONSTRATED`` and the
    remedy is to publish, not to redesign.
    """
    if row.failure_demonstrated is not FailureEvidence.NONE:
        return None

    return CriterionFinding(
        criterion=row,
        verdict=Verdict.UNDEMONSTRATED,
        reason=(
            "nothing published shows this criterion returning a negative verdict: no "
            "power curve, no adversarial probe that it rejects, no observed rejection "
            "reported alongside the passes"
        ),
        remedy=(
            "publish the failures alongside the passes, or a power curve at one stated "
            "alternative -- usually a reporting change, not a design change"
        ),
    )


#: Applied in order; the first non-``None`` wins. A measured constant comes first: it is
#: an observation, not a reading of the design, and no change of test fixes it. Direction
#: precedes degeneracy because a criterion tested in the wrong direction has no
#: equivalence test for the degeneracy guard to be missing from, and both precede the
#: demonstrated-failure check because both are statements about the design rather than
#: the write-up.
_DIAGNOSTICS = (
    diagnose_constant_statistic,
    diagnose_direction,
    diagnose_degeneracy,
    diagnose_demonstrated_failure,
)


def diagnose(row: CodedCriterion) -> CriterionFinding:
    """Run the three diagnostics on one row and return its single verdict."""
    for check in _DIAGNOSTICS:
        finding = check(row)
        if finding is not None:
            return finding

    if not row.is_equivalence_shaped:
        # A tie can still fix a difference or threshold check at FAIL with nothing
        # measured. That case is reported only by diagnose_constant_statistic, from a
        # measured observed_distinct_values.
        return CriterionFinding(
            criterion=row,
            verdict=Verdict.NOT_APPLICABLE,
            reason=(
                f"{row.claim_shape} claim with a demonstrated failure: noise pushes it "
                "toward failing, not toward a false pass, so the equivalence diagnostics "
                "do not apply"
            ),
        )

    return CriterionFinding(
        criterion=row,
        verdict=Verdict.SOUND,
        reason=(
            "equivalence claim, equivalence test, degeneracy guarded, and a "
            "demonstrated failure"
        ),
    )


@dataclass(frozen=True)
class InadmissibleRow:
    """A row that was not scored, and why. Published rather than dropped."""

    criterion: CodedCriterion
    why: str


@dataclass(frozen=True)
class SurveyResult:
    """Scored rows, with the denominators a published table is allowed to quote."""

    findings: tuple[CriterionFinding, ...]
    skipped: tuple[InadmissibleRow, ...]

    @property
    def n_scored(self) -> int:
        return len(self.findings)

    @property
    def equivalence_shaped(self) -> tuple[CriterionFinding, ...]:
        """The only denominator over which an inverted rate means anything."""
        return tuple(f for f in self.findings if f.criterion.is_equivalence_shaped)

    @property
    def inverted(self) -> tuple[CriterionFinding, ...]:
        return tuple(f for f in self.findings if f.verdict is Verdict.INVERTED)

    @property
    def degeneracy_unguarded(self) -> tuple[CriterionFinding, ...]:
        return tuple(
            f for f in self.findings if f.verdict is Verdict.DEGENERACY_UNGUARDED
        )

    @property
    def degenerate(self) -> tuple[CriterionFinding, ...]:
        return tuple(f for f in self.findings if f.verdict is Verdict.DEGENERATE)

    @property
    def undemonstrated(self) -> tuple[CriterionFinding, ...]:
        return tuple(f for f in self.findings if f.verdict is Verdict.UNDEMONSTRATED)

    def inverted_rate(self) -> tuple[int, int]:
        """``(inverted, equivalence-shaped)`` -- the headline the survey may report.

        Returns ``(0, 0)`` when no equivalence-shaped criterion was coded, which is a
        real possible outcome and must be reported as "no equivalence-shaped criteria
        found", never as a rate.
        """
        denom = len(self.equivalence_shaped)
        return len(self.inverted), denom

    def by_benchmark(self) -> dict[str, Counter[str]]:
        """Per-benchmark verdict counts, for the table's rows."""
        out: dict[str, Counter[str]] = {}
        for finding in self.findings:
            out.setdefault(finding.criterion.benchmark, Counter())[
                finding.verdict.value
            ] += 1
        return out

    def benchmarks_with_findings(self) -> tuple[str, ...]:
        """Benchmarks carrying at least one adverse verdict, sorted.

        This is the outreach list, and it is deliberately narrower than "benchmarks
        surveyed". A benchmark whose every criterion came back ``NOT_APPLICABLE`` or
        ``SOUND`` does not belong in a table of findings and does not get an email.
        """
        return tuple(
            sorted({f.criterion.benchmark for f in self.findings if f.is_finding})
        )


def run_survey(rows: Iterable[CodedCriterion]) -> SurveyResult:
    """Score coded rows, skipping any that are not checkable.

    Inadmissible rows are returned rather than dropped, so the count of what was not
    scored is published alongside what was.
    """
    findings: list[CriterionFinding] = []
    skipped: list[InadmissibleRow] = []

    for row in rows:
        ok, why = row.is_admissible()
        if not ok:
            skipped.append(InadmissibleRow(criterion=row, why=why))
            continue
        findings.append(diagnose(row))

    return SurveyResult(findings=tuple(findings), skipped=tuple(skipped))


def format_table(result: SurveyResult, criteria: Sequence[CodedCriterion] = ()) -> str:
    """Render the findings as a markdown table.

    Every row carries its source, so a reader -- in particular an author of the
    benchmark named -- can check the coding against the thing coded.
    """
    del criteria  # sources travel on the findings themselves
    lines = [
        "| Benchmark | Criterion | Verdict | Why | Source |",
        "|---|---|---|---|---|",
    ]
    for f in sorted(
        result.findings, key=lambda x: (x.criterion.benchmark, x.criterion.criterion)
    ):
        lines.append(
            f"| `{f.criterion.benchmark}` | `{f.criterion.criterion}` | "
            f"{f.verdict.value} | {f.reason} | {f.criterion.source} |"
        )

    n_inv, n_eq = result.inverted_rate()
    lines.append("")
    if n_eq == 0:
        lines.append(
            "**No equivalence-shaped criteria were coded.** The inverted rate is "
            "undefined, not zero."
        )
    else:
        n_other = result.n_scored - n_eq
        verb = "is" if n_other == 1 else "are"
        lines.append(
            f"**Inverted: {n_inv} of {n_eq} equivalence-shaped criteria.** "
            f"({result.n_scored} criteria scored in total; the other {n_other} "
            f"{verb} difference- or threshold-shaped, where a difference test is "
            "correct.)"
        )
    if result.degenerate:
        n_deg = len(result.degenerate)
        noun, their = ("criterion", "Its verdict is") if n_deg == 1 else ("criteria", "Their verdicts are")
        lines.append(
            f"**Degenerate: {n_deg} {noun}** whose statistic was measured to take one "
            f"value across the whole design. {their} fixed by construction, whatever the "
            "claim shape, and counted separately from the inverted rate."
        )
    if result.skipped:
        lines.append(
            f"**Not scored: {len(result.skipped)} rows** were inadmissible "
            "(no source, no coder, or the degeneracy question unasked)."
        )
    return "\n".join(lines)
