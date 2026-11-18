"""The survey frame as the paper publishes it, and a check that a port reproduces it.

Why this exists
---------------
The paper's Appendix F prints, per benchmark, how many preconditions were enumerated,
and beneath it how the 77 split by the kind of support behind them (0 equivalence test,
15 difference test, 40 descriptive, 22 no support). It does not print the rows
themselves -- which criterion, from which source, under which test. Those live in the
coding sheet, which is not in this repository.

So the port of that sheet into ``audit/*.json`` can be checked against something it did
not write. :func:`reconcile` is that check: a port that drops a row, adds one, or moves a
precondition between kinds of support comes back with a named discrepancy rather than a
plausible-looking table. Without it, a transcription error would surface, if at all,
only as a slightly different headline -- the kind of defect this package exists to make
visible.

What it does not check
----------------------
The descriptive / no-support split (40 / 22) is reported but not enforced against
:class:`~rhob.audit.criterion.StatTest`, because the paper's "descriptive" has no exact
counterpart there: a reported statistic compared to a constant is ``POINT_THRESHOLD``, one
compared to nothing is ``NONE``, and which a given row is belongs to the coder. Their sum
(62) is enforced. ``claim_shape`` is not checked at all -- the paper never coded it, which
is the reason the two-coder pass exists.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rhob.audit.criterion import CodedCriterion, StatTest

__all__ = ["FrameMember", "SurveyFrame", "load_frame", "reconcile"]


@dataclass(frozen=True)
class FrameMember:
    """One benchmark in the pre-registered frame.

    ``axis2_claims`` is ``None`` for a member that was never audited, which is different
    from ``0`` (audited, no preconditions found) and must stay different.
    """

    benchmark: str
    venue: str
    axis2_claims: int | None
    status: str

    @property
    def audited(self) -> bool:
        return self.axis2_claims is not None


@dataclass(frozen=True)
class SurveyFrame:
    members: tuple[FrameMember, ...]
    support_totals: dict[str, int]
    benchmarks_with_difference_test_claims: int

    @property
    def total_claims(self) -> int:
        return sum(m.axis2_claims or 0 for m in self.members)

    def member(self, name: str) -> FrameMember | None:
        key = name.casefold()
        return next((m for m in self.members if m.benchmark.casefold() == key), None)


def load_frame(path: str | Path) -> SurveyFrame:
    """Read the frame, refusing one whose totals disagree with each other."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    members = tuple(
        FrameMember(
            benchmark=m["benchmark"],
            venue=m["venue"],
            axis2_claims=m["axis2_claims"],
            status=m["status"],
        )
        for m in data["frame"]
    )
    frame = SurveyFrame(
        members=members,
        support_totals=dict(data["support_totals"]),
        benchmarks_with_difference_test_claims=data["benchmarks_with_difference_test_claims"],
    )
    by_support = sum(frame.support_totals.values())
    if by_support != frame.total_claims:
        raise ValueError(
            f"{path}: per-benchmark claims sum to {frame.total_claims} but the support "
            f"totals sum to {by_support}; the transcription is wrong somewhere"
        )
    return frame


def reconcile(rows: Iterable[CodedCriterion], frame: SurveyFrame) -> tuple[str, ...]:
    """Every way ``rows`` fails to reproduce the published frame. Empty means it does.

    Checked: each audited member carries exactly its published number of rows; no row
    names a benchmark outside the frame or one the paper did not audit; equivalence-test
    and difference-test totals match; the remainder matches descriptive + no support; and
    the difference-test rows span the published number of benchmarks.
    """
    rows = tuple(rows)
    problems: list[str] = []

    per_benchmark: Counter[str] = Counter()
    for row in rows:
        member = frame.member(row.benchmark)
        if member is None:
            problems.append(f"{row.benchmark}/{row.criterion}: benchmark is not in the frame")
            continue
        if not member.audited:
            problems.append(
                f"{row.benchmark}/{row.criterion}: the paper did not audit {member.benchmark}, "
                "so a row for it is new coding, not a port"
            )
            continue
        per_benchmark[member.benchmark] += 1

    for member in frame.members:
        if not member.audited:
            continue
        have = per_benchmark.get(member.benchmark, 0)
        if have != member.axis2_claims:
            problems.append(
                f"{member.benchmark}: {have} row(s), the paper enumerates {member.axis2_claims}"
            )

    tests = Counter(r.test_used for r in rows if frame.member(r.benchmark) is not None)
    want_eq = frame.support_totals["equivalence_test"]
    want_diff = frame.support_totals["difference_test"]
    want_rest = frame.support_totals["descriptive"] + frame.support_totals["no_support"]

    if tests[StatTest.EQUIVALENCE_TEST] != want_eq:
        problems.append(
            f"equivalence_test rows: {tests[StatTest.EQUIVALENCE_TEST]}, the paper reports {want_eq}"
        )
    if tests[StatTest.DIFFERENCE_TEST] != want_diff:
        problems.append(
            f"difference_test rows: {tests[StatTest.DIFFERENCE_TEST]}, the paper reports {want_diff}"
        )
    rest = tests[StatTest.POINT_THRESHOLD] + tests[StatTest.NONE]
    if rest != want_rest:
        problems.append(
            f"point_threshold + none rows: {rest}, the paper reports {want_rest} "
            "(descriptive + no support)"
        )

    diff_benchmarks = {
        frame.member(r.benchmark).benchmark  # type: ignore[union-attr]
        for r in rows
        if r.test_used is StatTest.DIFFERENCE_TEST and frame.member(r.benchmark) is not None
    }
    if len(diff_benchmarks) != frame.benchmarks_with_difference_test_claims:
        problems.append(
            f"difference-test rows span {len(diff_benchmarks)} benchmark(s), the paper reports "
            f"{frame.benchmarks_with_difference_test_claims}"
        )

    return tuple(problems)
