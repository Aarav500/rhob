"""How one benchmark precondition gets coded, and what verdicts it can receive.

Why this module exists
----------------------
The paper's survey of published benchmarks currently lives only as prose: there is
no list of benchmarks in this repository, no coding sheet, and no runnable
diagnostic. That is a problem for a project whose own thesis is::

    A criterion is not a check until it is accompanied by a demonstration that it
    can fail -- a power curve, or an adversarial probe that passes it -- and a
    negative control is decorative unless the failures are published alongside the
    passes.

A survey that names other people's benchmarks and cannot itself be re-run is that
same defect pointed outward. This module is the machine-readable half of the fix:
the unit of the survey is a :class:`CodedCriterion`, one row per precondition, with
the coder's judgements recorded as enum fields rather than left in a paragraph.
:mod:`rhob.audit.diagnostics` turns rows into verdicts; ``docs/SURVEY_PROTOCOL.md``
states the decision rules a coder applies to produce a row.

The denominator problem
-----------------------
The statistic this module exists to stop being quoted loosely is *"N of M
preconditions are supported by an equivalence test"*, with ``M`` the count of all
preconditions.

That denominator is wrong, and RHOB is the counterexample. RHOB ships six
admission criteria. Two are equivalence tests. The other four are difference tests
**and should be** -- as ``README.md`` puts it, "ties push the statistic *toward* the
null and therefore toward failing -- the safe direction". Scored against all
preconditions, post-audit RHOB reads 2 of 6 and looks mostly broken. Scored against
the criteria that actually assert an equivalence, it reads 2 of 2 and is sound.

So a difference test is a defect only where the claim is an equivalence -- where
"I failed to find a difference" is being reported as "there is no difference". That
is :attr:`Verdict.INVERTED`, and the rate that means anything is inverted over
*equivalence-shaped* criteria. :class:`~rhob.audit.diagnostics.SurveyResult` reports
that denominator and declines to report the other one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ClaimShape(str, Enum):
    """What the criterion asserts, independent of how it is tested.

    This is the coder's judgement call and the one the whole survey turns on, so
    ``docs/SURVEY_PROTOCOL.md`` fixes the decision rule and requires two independent
    coders with disagreements recorded rather than resolved silently.
    """

    #: "These two things are the same." Passing requires evidence *for* sameness.
    #: Absence of a detected difference is not that evidence.
    EQUIVALENCE = "equivalence"

    #: "These two things differ" / "this quantity exceeds that one." Failing to show
    #: it is the safe direction: noise pushes the verdict toward rejection.
    DIFFERENCE = "difference"

    #: "This quantity lies on this side of a fixed constant", with no comparison
    #: between groups. Scored like a difference claim unless the constant is itself
    #: an equivalence margin in disguise.
    THRESHOLD = "threshold"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class StatTest(str, Enum):
    """The statistical procedure the benchmark actually applies to the criterion."""

    #: TOST or any procedure requiring an interval to sit inside a stated margin.
    EQUIVALENCE_TEST = "equivalence_test"

    #: A null-hypothesis test, CI-excludes-zero, or any procedure whose pass
    #: condition is "no significant difference was found".
    DIFFERENCE_TEST = "difference_test"

    #: A point estimate compared to a constant, with no interval at all.
    POINT_THRESHOLD = "point_threshold"

    #: The criterion is stated but nothing computes a verdict for it.
    NONE = "none"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class FailureEvidence(str, Enum):
    """What the benchmark published to show this criterion *can* return a negative.

    Ordered loosely by strength. ``NONE`` is the common case and is not by itself an
    accusation -- it is the thing the survey counts.
    """

    #: A power curve, or any table of pass rates against stated alternatives. The
    #: strongest form: it says how often the check catches a violation of known size.
    POWER_CURVE = "power_curve"

    #: A deliberately broken input that the criterion rejects, published as such.
    ADVERSARIAL_PROBE = "adversarial_probe"

    #: The criterion rejected something real, and the rejection is in the artifact
    #: alongside the passes.
    OBSERVED_FAILURE = "observed_failure"

    #: Nothing published establishes that a negative verdict is reachable.
    NONE = "none"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class Verdict(str, Enum):
    """The diagnostic outcome for one coded criterion.

    ``UNDEMONSTRATED`` is deliberately the mildest finding and is expected to be the
    majority verdict in any real survey. It says only that the benchmark did not
    publish evidence the check can fail. It does **not** say the check cannot fail,
    and a table that reports the two as the same thing is making the error this
    project is about.
    """

    #: Test matches the claim, and a failure is demonstrated.
    SOUND = "sound"

    #: Test matches the claim, but nothing shows a negative verdict is reachable.
    UNDEMONSTRATED = "undemonstrated"

    #: An equivalence claim tested with a difference test: absence of evidence
    #: reported as evidence of absence. This is RHOB's own pre-audit defect.
    INVERTED = "inverted"

    #: An equivalence test with nothing checking that its statistic could have taken
    #: another value. RHOB's post-fix defect: a tied statistic certifies against any
    #: margin, so the check cannot fail.
    DEGENERACY_UNGUARDED = "degeneracy_unguarded"

    #: A difference- or threshold-shaped claim, where failing is the safe direction.
    #: Carries no finding; excluded from the equivalence denominator.
    NOT_APPLICABLE = "not_applicable"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class CodedCriterion:
    """One precondition of one benchmark, as coded by a human following the protocol.

    Every field except ``notes`` is a judgement the protocol defines a rule for. The
    ``source`` field is what makes a row checkable by a third party -- and by the
    benchmark's own authors, who are the people most able to say the coding is wrong.

    Attributes:
        benchmark: Short identifier, e.g. ``"rhob"``.
        criterion: The criterion's own name in its benchmark, e.g. ``"proxy_matched"``.
        source: Where it is defined, precisely enough to check: a file and line, or a
            paper section. A row without this is not admissible -- see
            :meth:`is_admissible`.
        claim_shape: What the criterion asserts.
        test_used: What the benchmark computes.
        failure_demonstrated: What was published showing a negative is reachable.
        resolution_guarded: For an equivalence test, whether anything checks that the
            statistic could have taken a different value. ``None`` where not
            applicable.
        coder: Identifier of the person who coded the row.
        notes: Free text. Required when the coding is not obvious from the source.
    """

    benchmark: str
    criterion: str
    source: str
    claim_shape: ClaimShape
    test_used: StatTest
    failure_demonstrated: FailureEvidence = FailureEvidence.NONE
    resolution_guarded: bool | None = None
    coder: str = ""
    notes: str = ""

    def is_admissible(self) -> tuple[bool, str]:
        """Whether this row may enter a published table, and why not if it may not.

        The bar is deliberately low and entirely about checkability: a named source,
        a named coder. A row that fails this is not evidence about anyone's
        benchmark, and :func:`~rhob.audit.diagnostics.run_survey` refuses to score it
        rather than quietly counting it.
        """
        if not self.source.strip():
            return False, "no source: the row cannot be checked against the benchmark"
        if not self.coder.strip():
            return False, "no coder: the judgement has no author"
        # The degeneracy question is only answerable where an equivalence test was
        # actually run: a tied statistic can only over-certify a test that reads an
        # interval. Requiring it of every equivalence-shaped row would skip precisely
        # the rows that carry the INVERTED finding -- an equivalence claim with no
        # equivalence test has no interval for a guard to be missing from -- and the
        # survey would silently drop its own results.
        if (
            self.claim_shape is ClaimShape.EQUIVALENCE
            and self.test_used is StatTest.EQUIVALENCE_TEST
            and self.resolution_guarded is None
        ):
            return False, (
                "equivalence test with resolution_guarded unset: the degeneracy "
                "question applies here and was not asked"
            )
        return True, ""

    @property
    def is_equivalence_shaped(self) -> bool:
        """Whether this criterion belongs in the equivalence denominator."""
        return self.claim_shape is ClaimShape.EQUIVALENCE


@dataclass(frozen=True)
class CriterionFinding:
    """A verdict on one row, with the reason stated in the row's own terms."""

    criterion: CodedCriterion
    verdict: Verdict
    reason: str
    #: Set when the finding is one a benchmark's authors could act on. Used to build
    #: the outreach list: a finding nobody can fix is not worth an email.
    remedy: str = field(default="")

    @property
    def is_finding(self) -> bool:
        """Whether this says something adverse about the criterion."""
        return self.verdict in (
            Verdict.INVERTED,
            Verdict.DEGENERACY_UNGUARDED,
            Verdict.UNDEMONSTRATED,
        )
