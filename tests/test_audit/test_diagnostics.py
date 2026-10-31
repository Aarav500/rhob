"""The diagnostics must fail RHOB where RHOB was wrong.

A survey instrument that cannot produce a finding against its own author is the
thing this package exists to detect, so the load-bearing tests here are the ones
asserting that the pre-audit RHOB coding comes back INVERTED on both of its
equivalence-shaped criteria.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rhob.audit import (
    ClaimShape,
    CodedCriterion,
    FailureEvidence,
    StatTest,
    Verdict,
    diagnose,
    load_snapshot,
    run_survey,
)

SELF_CODING = Path(__file__).resolve().parents[2] / "audit" / "rhob_self_coding.json"


def _row(**kw) -> CodedCriterion:
    base = dict(
        benchmark="b",
        criterion="c",
        source="file.py:1",
        claim_shape=ClaimShape.DIFFERENCE,
        test_used=StatTest.DIFFERENCE_TEST,
        failure_demonstrated=FailureEvidence.OBSERVED_FAILURE,
        coder="tester",
    )
    base.update(kw)
    return CodedCriterion(**base)


# --- The direction diagnostic ------------------------------------------------


def test_equivalence_claim_tested_as_difference_is_inverted():
    row = _row(
        claim_shape=ClaimShape.EQUIVALENCE,
        test_used=StatTest.DIFFERENCE_TEST,
        resolution_guarded=True,
    )
    assert diagnose(row).verdict is Verdict.INVERTED


def test_equivalence_claim_with_no_test_at_all_is_inverted():
    row = _row(
        claim_shape=ClaimShape.EQUIVALENCE,
        test_used=StatTest.NONE,
        resolution_guarded=True,
    )
    assert diagnose(row).verdict is Verdict.INVERTED


def test_difference_claim_tested_as_difference_is_not_a_finding():
    """The safe direction. Flagging this would make the survey wrong about RHOB."""
    finding = diagnose(_row(claim_shape=ClaimShape.DIFFERENCE))
    assert finding.verdict is Verdict.NOT_APPLICABLE
    assert not finding.is_finding


def test_threshold_claim_is_not_a_finding():
    finding = diagnose(
        _row(claim_shape=ClaimShape.THRESHOLD, test_used=StatTest.POINT_THRESHOLD)
    )
    assert finding.verdict is Verdict.NOT_APPLICABLE


# --- The degeneracy diagnostic -----------------------------------------------


def test_unguarded_equivalence_test_is_flagged():
    row = _row(
        claim_shape=ClaimShape.EQUIVALENCE,
        test_used=StatTest.EQUIVALENCE_TEST,
        resolution_guarded=False,
    )
    assert diagnose(row).verdict is Verdict.DEGENERACY_UNGUARDED


def test_guarded_equivalence_test_with_evidence_is_sound():
    row = _row(
        claim_shape=ClaimShape.EQUIVALENCE,
        test_used=StatTest.EQUIVALENCE_TEST,
        resolution_guarded=True,
        failure_demonstrated=FailureEvidence.POWER_CURVE,
    )
    assert diagnose(row).verdict is Verdict.SOUND


# --- The demonstrated-failure diagnostic -------------------------------------


def test_no_published_failure_is_undemonstrated():
    row = _row(failure_demonstrated=FailureEvidence.NONE)
    assert diagnose(row).verdict is Verdict.UNDEMONSTRATED


def test_direction_outranks_demonstrated_failure():
    """A design defect is reported ahead of a reporting gap."""
    row = _row(
        claim_shape=ClaimShape.EQUIVALENCE,
        test_used=StatTest.DIFFERENCE_TEST,
        failure_demonstrated=FailureEvidence.NONE,
        resolution_guarded=False,
    )
    assert diagnose(row).verdict is Verdict.INVERTED


# --- Admissibility ------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        {"source": ""},
        {"coder": ""},
        {
            "claim_shape": ClaimShape.EQUIVALENCE,
            "test_used": StatTest.EQUIVALENCE_TEST,
            "resolution_guarded": None,
        },
    ],
)
def test_inadmissible_rows_are_skipped_and_reported(kw):
    result = run_survey([_row(**kw)])
    assert result.n_scored == 0
    assert len(result.skipped) == 1
    assert result.skipped[0].why


# --- The denominator ----------------------------------------------------------


def test_inverted_rate_denominator_is_equivalence_shaped_only():
    rows = [
        _row(
            criterion="eq_bad",
            claim_shape=ClaimShape.EQUIVALENCE,
            test_used=StatTest.DIFFERENCE_TEST,
            resolution_guarded=False,
        ),
        _row(criterion="diff_1"),
        _row(criterion="diff_2"),
        _row(criterion="diff_3"),
    ]
    result = run_survey(rows)
    assert result.inverted_rate() == (1, 1), "difference claims must not dilute the rate"


def test_no_equivalence_criteria_reports_zero_denominator_not_a_rate():
    result = run_survey([_row(), _row(criterion="d2")])
    assert result.inverted_rate() == (0, 0)


# --- The self-coding, which is the instrument's calibration -------------------


def test_pre_audit_rhob_is_inverted_on_both_equivalence_criteria():
    rows = load_snapshot(SELF_CODING, "rhob_pre_audit")
    result = run_survey(rows)

    assert result.n_scored == 6
    assert not result.skipped

    inverted = {f.criterion.criterion for f in result.inverted}
    assert inverted == {"proxy_matched", "proxy_distribution_matched"}
    assert result.inverted_rate() == (2, 2)


def test_post_audit_rhob_clears_both_equivalence_criteria():
    rows = load_snapshot(SELF_CODING, "rhob_post_audit")
    result = run_survey(rows)

    assert result.inverted_rate() == (0, 2)
    sound = {f.criterion.criterion for f in result.findings if f.verdict is Verdict.SOUND}
    assert sound == {"proxy_matched", "proxy_distribution_matched"}


def test_post_audit_rhob_has_one_finding_left():
    """Every criterion has a probe family that fails it, so none is merely undemonstrated.

    An earlier coding said three were, which was wrong: the probes in
    tests/test_v3/test_criteria_discriminate.py were committed before that coding. What the
    audit did not fix is onset_localizable, which can fail on a probe and still cannot fail
    on any cell the benchmark ships.
    """
    result = run_survey(load_snapshot(SELF_CODING, "rhob_post_audit"))
    assert not result.undemonstrated
    assert {f.criterion.criterion for f in result.findings if f.is_finding} == {"onset_localizable"}
    verdicts = {f.criterion.criterion: f.verdict for f in result.findings}
    assert verdicts["behavioral_separated"] is Verdict.NOT_APPLICABLE
    assert verdicts["camping_quality"] is Verdict.NOT_APPLICABLE


def test_self_coding_sources_still_point_at_their_criteria():
    """A row whose source is a README line must name its criterion on that line.

    The README gained 87 lines above its criteria table on 2026-10-31 and every source in
    this file silently went stale. A calibration whose citations do not resolve cannot be
    checked by anyone, which is the property the coding exists to have.
    """
    import json
    import re

    readme = (SELF_CODING.parents[1] / "README.md").read_text(encoding="utf-8").splitlines()
    data = json.loads(SELF_CODING.read_text(encoding="utf-8"))
    checked = 0
    for snapshot in data["snapshots"].values():
        for row in snapshot["rows"]:
            m = re.match(r"README\.md:(\d+)(?:-(\d+))?", row["source"])
            if not m:
                continue
            lo = int(m.group(1)); hi = int(m.group(2) or lo)
            span = "\n".join(readme[lo - 1:hi])
            assert row["criterion"] in span, (row["criterion"], row["source"], readme[lo - 1][:80])
            checked += 1
    assert checked == 12


def test_post_audit_onset_localizable_is_degenerate():
    """The regression that motivated diagnose_constant_statistic.

    onset_std is exactly 0.0 in all 50 ledger cells, so the threshold passes every cell
    by construction. Before the measured-constant check this row came back merely
    undemonstrated, because a threshold claim was never offered to the degeneracy check.
    """
    result = run_survey(load_snapshot(SELF_CODING, "rhob_post_audit"))
    assert {f.criterion.criterion for f in result.degenerate} == {"onset_localizable"}
    assert result.inverted_rate() == (0, 2), "a degenerate row must not enter the inverted rate"


# --- The measured-constant diagnostic ------------------------------------------


def test_threshold_claim_on_a_constant_statistic_is_degenerate():
    row = _row(
        claim_shape=ClaimShape.THRESHOLD,
        test_used=StatTest.POINT_THRESHOLD,
        observed_distinct_values=1,
    )
    finding = diagnose(row)
    assert finding.verdict is Verdict.DEGENERATE
    assert finding.is_finding


def test_a_statistic_that_varies_is_not_degenerate():
    row = _row(
        claim_shape=ClaimShape.THRESHOLD,
        test_used=StatTest.POINT_THRESHOLD,
        observed_distinct_values=2,
    )
    assert diagnose(row).verdict is Verdict.NOT_APPLICABLE


def test_unmeasured_is_not_degenerate():
    """None means nobody ran D1. That is the usual case and must not read as a finding."""
    row = _row(claim_shape=ClaimShape.THRESHOLD, test_used=StatTest.POINT_THRESHOLD)
    assert row.observed_distinct_values is None
    assert diagnose(row).verdict is Verdict.NOT_APPLICABLE


def test_measured_constant_outranks_direction():
    """No change of test fixes a constant, so it is reported ahead of INVERTED."""
    row = _row(
        claim_shape=ClaimShape.EQUIVALENCE,
        test_used=StatTest.DIFFERENCE_TEST,
        observed_distinct_values=1,
    )
    assert diagnose(row).verdict is Verdict.DEGENERATE


def test_zero_distinct_values_is_inadmissible():
    result = run_survey([_row(observed_distinct_values=0)])
    assert result.n_scored == 0
    assert "below 1" in result.skipped[0].why


def test_the_loose_denominator_would_misreport_post_audit_rhob():
    """Why SurveyResult does not expose 'supported by an equivalence test / all'.

    Post-audit RHOB is sound on both criteria where the distinction bites, and the
    loose statistic still reads 2 of 6. This test pins the gap that motivates the
    denominator choice, so nobody reintroduces the loose one as a convenience.
    """
    result = run_survey(load_snapshot(SELF_CODING, "rhob_post_audit"))

    loose = sum(
        1
        for f in result.findings
        if f.criterion.test_used is StatTest.EQUIVALENCE_TEST
    )
    assert (loose, result.n_scored) == (2, 6)
    assert result.inverted_rate() == (0, 2)


def test_survey_result_has_no_loose_rate_helper():
    """Guards the design decision rather than the arithmetic."""
    result = run_survey(load_snapshot(SELF_CODING, "rhob_post_audit"))
    for name in dir(result):
        assert "equivalence_support" not in name


def test_equivalence_claim_without_an_equivalence_test_stays_admissible():
    """The regression that matters: these rows ARE the headline finding.

    An equivalence claim tested with a difference test has no interval for a
    degeneracy guard to be missing from, so demanding resolution_guarded of it would
    skip every INVERTED row and the survey would drop its own results.
    """
    row = _row(
        claim_shape=ClaimShape.EQUIVALENCE,
        test_used=StatTest.DIFFERENCE_TEST,
        resolution_guarded=None,
    )
    ok, why = row.is_admissible()
    assert ok, why

    result = run_survey([row])
    assert result.n_scored == 1
    assert not result.skipped
    assert result.inverted_rate() == (1, 1)
