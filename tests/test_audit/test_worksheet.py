"""The worksheet pass must not let an uncoded judgement through as a benign default.

The failure mode these guard against is quiet: a blank ``claim_shape`` defaulting to
``difference`` turns every uncoded precondition into a non-finding, which would make
the survey under-report in exactly the direction that flatters its author.
"""

from __future__ import annotations

import pytest

from rhob.audit import ClaimShape, CodedCriterion, FailureEvidence, StatTest
from rhob.audit.worksheet import (
    WORKSHEET_COLUMNS,
    agreement_report,
    read_worksheet,
    shape_census,
    write_worksheet,
)


def _row(criterion: str, shape: ClaimShape = ClaimShape.EQUIVALENCE, **kw) -> CodedCriterion:
    base = dict(
        benchmark="bench_x",
        criterion=criterion,
        source="paper §3.1",
        claim_shape=shape,
        test_used=StatTest.DIFFERENCE_TEST,
        failure_demonstrated=FailureEvidence.NONE,
        resolution_guarded=None,
        coder="",
        notes="",
    )
    base.update(kw)
    return CodedCriterion(**base)


def test_emitted_worksheet_blanks_the_shape_call(tmp_path):
    path = write_worksheet([_row("decontamination")], tmp_path / "a.csv", coder="ash")
    text = path.read_text(encoding="utf-8")

    assert "claim_shape" in text
    assert "equivalence" not in text.split("\n")[-2], "the shape must be blanked"
    assert "bench_x,decontamination" in text
    assert "ash" in text


def test_worksheet_carries_the_decision_rule_for_the_coder(tmp_path):
    path = write_worksheet([_row("c")], tmp_path / "a.csv", coder="ash")
    header = path.read_text(encoding="utf-8").split("\n")[0]
    assert header.startswith("#")
    assert "EASIER" in header and "HARDER" in header


def test_reading_back_an_uncoded_shape_raises(tmp_path):
    path = write_worksheet([_row("c")], tmp_path / "a.csv", coder="ash")
    with pytest.raises(ValueError, match="uncoded"):
        read_worksheet(path)


def test_round_trip_once_the_coder_fills_it_in(tmp_path):
    path = tmp_path / "a.csv"
    write_worksheet([_row("decontamination")], path, coder="ash")

    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "bench_x,decontamination,paper §3.1,,", "bench_x,decontamination,paper §3.1,equivalence,"
    )
    path.write_text(text, encoding="utf-8")

    rows = read_worksheet(path)
    assert len(rows) == 1
    assert rows[0].claim_shape is ClaimShape.EQUIVALENCE
    assert rows[0].coder == "ash"


def test_worksheet_columns_put_shape_before_test_used():
    """The coder should call the shape from the source, not from what was done about it."""
    cols = list(WORKSHEET_COLUMNS)
    assert cols.index("claim_shape") < cols.index("test_used")
    assert cols.index("source") < cols.index("claim_shape")


# --- agreement -----------------------------------------------------------------


def test_agreement_counts_matches_and_reports_disagreements():
    a = [_row("c1", ClaimShape.EQUIVALENCE), _row("c2", ClaimShape.DIFFERENCE)]
    b = [_row("c1", ClaimShape.EQUIVALENCE), _row("c2", ClaimShape.EQUIVALENCE)]

    report = agreement_report(a, b)
    assert (report.n_agreed, report.n_compared) == (1, 2)
    assert report.rate == 0.5
    assert report.disagreements == (("bench_x", "c2", "difference", "equivalence"),)


def test_rows_only_one_coder_saw_are_reported_not_dropped():
    a = [_row("c1"), _row("only_a")]
    b = [_row("c1")]

    report = agreement_report(a, b)
    assert report.n_compared == 1
    assert report.only_in_a == ("bench_x/only_a",)
    assert report.only_in_b == ()


def test_empty_comparison_is_vacuous_not_perfect():
    report = agreement_report([], [])
    assert report.n_compared == 0
    assert report.rate == 1.0, "documented as vacuous; summary must state the count"
    assert "0/0" in report.summary()


# --- the census that sets the denominator ---------------------------------------


def test_shape_census_gives_the_published_denominator():
    rows = [
        _row("c1", ClaimShape.EQUIVALENCE),
        _row("c2", ClaimShape.EQUIVALENCE),
        _row("c3", ClaimShape.DIFFERENCE),
        _row("c4", ClaimShape.THRESHOLD),
    ]
    census = shape_census(rows)
    assert census["equivalence"] == 2
    assert sum(census.values()) == 4
