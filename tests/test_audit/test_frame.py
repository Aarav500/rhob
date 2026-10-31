"""A port of the paper's survey sheet must reproduce the paper's own Appendix F.

The rows built here are synthetic. They are shaped to match the published per-benchmark
totals so the reconciliation can be exercised; they are not the paper's rows and must
never be read as them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rhob.audit import ClaimShape, CodedCriterion, StatTest
from rhob.audit.frame import load_frame, reconcile

FRAME = Path(__file__).resolve().parents[2] / "audit" / "survey_s1_frame.json"

# Which synthetic benchmarks carry the 15 difference-test rows. Four, as published;
# which four the paper's real rows name is exactly what this file does not know.
_DIFF = {"RAID": 5, "TSB-AD": 4, "ADBench": 3, "WILDS": 3}


def _rows(frame):
    rows = []
    for member in frame.members:
        if not member.audited:
            continue
        n_diff = _DIFF.get(member.benchmark, 0)
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
    rows = [r for r in _rows(frame) if not (r.benchmark == "WILDS" and r.criterion == "c14")]
    problems = reconcile(rows, frame)
    assert any(p.startswith("WILDS: 14 row(s)") for p in problems), problems


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
    # Move one RAID difference-test row's label onto OGB, so five benchmarks carry them.
    i = next(k for k, r in enumerate(rows) if r.benchmark == "OGB")
    r = rows[i]
    rows[i] = CodedCriterion(
        benchmark="OGB", criterion=r.criterion, source=r.source,
        claim_shape=r.claim_shape, test_used=StatTest.DIFFERENCE_TEST, coder=r.coder,
    )
    j = next(k for k, r in enumerate(rows) if r.benchmark == "RAID" and r.test_used is StatTest.DIFFERENCE_TEST)
    s = rows[j]
    rows[j] = CodedCriterion(
        benchmark="RAID", criterion=s.criterion, source=s.source,
        claim_shape=s.claim_shape, test_used=StatTest.NONE, coder=s.coder,
    )
    problems = reconcile(rows, frame)
    assert any("span 5 benchmark(s)" in p for p in problems), problems


def test_a_row_for_an_unaudited_benchmark_is_new_coding(frame):
    rows = _rows(frame) + [
        CodedCriterion(
            benchmark="MMTEB", criterion="x", source="synthetic",
            claim_shape=ClaimShape.DIFFERENCE, test_used=StatTest.NONE, coder="synthetic",
        )
    ]
    problems = reconcile(rows, frame)
    assert any("did not audit MMTEB" in p for p in problems), problems
