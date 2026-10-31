"""A port of the paper's survey sheet must reproduce the paper's own Appendix F.

The rows built here are synthetic. They are shaped to match the published per-benchmark
totals so the reconciliation can be exercised; they are not the paper's rows and must
never be read as them. For the same reason this file names no benchmark it builds rows
for: which members carry the synthetic rows is chosen mechanically from the frame, so no
synthetic count sits next to a real benchmark's name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rhob.audit import ClaimShape, CodedCriterion, StatTest
from rhob.audit.frame import load_frame, reconcile

FRAME = Path(__file__).resolve().parents[2] / "audit" / "survey_s1_frame.json"


def _difference_carriers(frame) -> dict[str, int]:
    """Which members carry the synthetic difference-test rows, and how many each.

    The published total (15) is spread as evenly as it goes over the published number of
    benchmarks (4), onto the first audited members, in frame order, with room for the
    largest share. Which members the paper's real rows name is exactly what this file does
    not know, so the choice is an arbitrary rule, not a reading of the paper.
    """
    total = frame.support_totals["difference_test"]
    k = frame.benchmarks_with_difference_test_claims
    shares = [total // k + (1 if i < total % k else 0) for i in range(k)]
    chosen = [m.benchmark for m in frame.members if m.audited and m.axis2_claims >= max(shares)]
    assert len(chosen) >= k, "the frame has too few members with room for the shares"
    return dict(zip(chosen[:k], shares))


def _rows(frame):
    carriers = _difference_carriers(frame)
    rows = []
    for member in frame.members:
        if not member.audited:
            continue
        n_diff = carriers.get(member.benchmark, 0)
        for i in range(member.axis2_claims):
            test = StatTest.DIFFERENCE_TEST if i < n_diff else StatTest.NONE
            rows.append(
                CodedCriterion(
                    benchmark=member.benchmark,
                    criterion=f"c{i}",
                    source="synthetic",
                    claim_shape=ClaimShape.DIFFERENCE,
                    test_used=test,
                    coder="synthetic",
                )
            )
    return rows


@pytest.fixture(scope="module")
def frame():
    return load_frame(FRAME)


def test_frame_transcription_is_internally_consistent(frame):
    assert len(frame.members) == 19, "the pre-registered frame is 19 benchmarks"
    assert frame.total_claims == 77
    assert sum(frame.support_totals.values()) == 77
    assert frame.support_totals["equivalence_test"] == 0
    assert frame.support_totals["difference_test"] == 15
    with_claims = [m for m in frame.members if m.audited and m.axis2_claims > 0]
    assert len(with_claims) == 15, "77 preconditions across 15 benchmarks"
    assert sum(1 for m in frame.members if not m.audited) == 3


def test_unaudited_is_not_zero(frame):
    """FLIP was audited and had no claims; NINCO was never audited. Not the same thing."""
    assert frame.member("FLIP").axis2_claims == 0
    assert frame.member("FLIP").audited
    assert frame.member("NINCO").axis2_claims is None
    assert not frame.member("NINCO").audited


def test_a_faithful_port_reconciles(frame):
    assert reconcile(_rows(frame), frame) == ()


def test_a_dropped_row_is_named(frame):
    victim = next(m for m in reversed(frame.members) if m.audited and m.axis2_claims)
    last = f"c{victim.axis2_claims - 1}"
    rows = [r for r in _rows(frame) if not (r.benchmark == victim.benchmark and r.criterion == last)]
    problems = reconcile(rows, frame)
    want = f"{victim.benchmark}: {victim.axis2_claims - 1} row(s)"
    assert any(p.startswith(want) for p in problems), problems


def test_a_row_moved_between_kinds_of_support_is_caught(frame):
    rows = _rows(frame)
    i = next(k for k, r in enumerate(rows) if r.test_used is StatTest.NONE)
    r = rows[i]
    rows[i] = CodedCriterion(
        benchmark=r.benchmark, criterion=r.criterion, source=r.source,
        claim_shape=r.claim_shape, test_used=StatTest.EQUIVALENCE_TEST,
        resolution_guarded=True, coder=r.coder,
    )
    problems = reconcile(rows, frame)
    assert any("equivalence_test rows: 1" in p for p in problems), problems


def test_difference_tests_spread_over_the_wrong_number_of_benchmarks(frame):
    rows = _rows(frame)
    carriers = _difference_carriers(frame)
    donor = next(iter(carriers))
    outsider = next(
        m.benchmark
        for m in frame.members
        if m.audited and m.axis2_claims and m.benchmark not in carriers
    )
    # Move one of the donor's difference-test labels onto a row of a benchmark that
    # carries none, so five benchmarks carry them while every total still matches.
    i = next(k for k, r in enumerate(rows) if r.benchmark == outsider)
    r = rows[i]
    rows[i] = CodedCriterion(
        benchmark=outsider, criterion=r.criterion, source=r.source,
        claim_shape=r.claim_shape, test_used=StatTest.DIFFERENCE_TEST, coder=r.coder,
    )
    j = next(
        k for k, r in enumerate(rows)
        if r.benchmark == donor and r.test_used is StatTest.DIFFERENCE_TEST
    )
    s = rows[j]
    rows[j] = CodedCriterion(
        benchmark=donor, criterion=s.criterion, source=s.source,
        claim_shape=s.claim_shape, test_used=StatTest.NONE, coder=s.coder,
    )
    problems = reconcile(rows, frame)
    assert any("span 5 benchmark(s)" in p for p in problems), problems
    assert len(problems) == 1, problems


def test_a_row_for_an_unaudited_benchmark_is_new_coding(frame):
    unaudited = next(m.benchmark for m in frame.members if not m.audited)
    rows = _rows(frame) + [
        CodedCriterion(
            benchmark=unaudited, criterion="x", source="synthetic",
            claim_shape=ClaimShape.DIFFERENCE, test_used=StatTest.NONE, coder="synthetic",
        )
    ]
    problems = reconcile(rows, frame)
    assert any(f"did not audit {unaudited}" in p for p in problems), problems
