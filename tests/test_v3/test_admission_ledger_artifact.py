"""The committed admission ledger must keep FAIL and DEGENERATE apart, in every field.

The tri-state exists so that "we measured this and the family is wrong" and "we could
not measure this at all" stay distinguishable. It is easy to build the distinction into
the certificate and then lose it in the summary -- which is exactly what happened: the
ledger published ``n_failing_by_criterion: {proxy_matched: 15}`` for a criterion with
**zero** measured failures and 15 unmeasurable cells, re-merging the two at the last
step, in the one block a reader is most likely to quote.

So the counts are pinned here against the artifact itself, by recomputing them from the
per-cell outcomes. A summary that disagrees with its own results block is worse than no
summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rhob.v3.admission_gate import CRITERIA

_LEDGER = Path(__file__).resolve().parents[2] / "admission" / "admission_ledger.json"


@pytest.fixture(scope="module")
def ledger() -> dict:
    assert _LEDGER.exists(), (
        f"{_LEDGER} is missing; regenerate it with `python scripts/admission_ledger.py`"
    )
    return json.loads(_LEDGER.read_text(encoding="utf-8"))


def test_the_three_statuses_partition_the_grid(ledger):
    s, rows = ledger["summary"], ledger["results"]
    assert s["n_cells"] == len(rows)
    assert s["n_admitted"] + s["n_degenerate"] + s["n_not_admitted"] == s["n_cells"]
    # A cell that failed something real is NOT ADMITTED even if it is also degenerate:
    # the measured failure is the actionable finding and outranks the unmeasurable one.
    for row in rows:
        if any(v == "fail" for v in row["outcomes"].values()):
            assert row["status"] == "NOT ADMITTED", row["family"]


def test_every_per_criterion_count_is_recomputable_from_the_cells(ledger):
    s, rows = ledger["summary"], ledger["results"]
    for criterion in CRITERIA:
        failed = sum(1 for r in rows if r["outcomes"][criterion] == "fail")
        degenerate = sum(1 for r in rows if r["outcomes"][criterion] == "degenerate")
        assert s["n_failed_by_criterion"][criterion] == failed, criterion
        assert s["n_degenerate_by_criterion"][criterion] == degenerate, criterion
        # The old, ambiguous count kept its meaning under a name that admits to it.
        assert s["n_not_established_by_criterion"][criterion] == failed + degenerate


def test_a_degenerate_cell_is_never_counted_as_a_measured_failure(ledger):
    """The C7 defect, stated as an assertion about the shipped numbers.

    On the current grid ``proxy_matched`` has no measured failures at all; every cell it
    did not establish, it could not measure. Publishing that as "15 failing" told a
    reader to go hunt for a proxy leak in families whose data cannot show one.
    """
    s = ledger["summary"]
    for criterion in CRITERIA:
        failed = s["n_failed_by_criterion"][criterion]
        degenerate = s["n_degenerate_by_criterion"][criterion]
        assert failed >= 0 and degenerate >= 0
        # No cell can be both at once -- the outcome is a single tri-state value.
        assert failed + degenerate <= s["n_cells"]
    assert s["n_failed_by_criterion"]["proxy_matched"] == 0
    assert s["n_degenerate_by_criterion"]["proxy_matched"] == s["n_degenerate"]


def test_the_degenerate_family_list_matches_the_cells(ledger):
    s, rows = ledger["summary"], ledger["results"]
    from_cells = sorted({r["family"] for r in rows if r["degenerate_criteria"]})
    assert s["degenerate_families"] == from_cells
    # Every one of them is degenerate at *every* tier it is scored at. If that ever
    # stops being true, the L0 exclusion in `rhob.v3.leaderboard.access_summary` becomes
    # a family-level approximation to a cell-level fact and has to be revisited.
    for family in from_cells:
        cells = [r for r in rows if r["family"] == family]
        assert all(r["degenerate_criteria"] for r in cells), family


#: The exact keys ``scripts/admission_ledger.py`` writes under ``summary``, in order.
#: Pinned as a set because ``docs/API_SPECIFICATION.md`` documents this block key by key
#: and drifted from it: it went on listing the removed ``n_failing_by_criterion`` (and
#: never picked up ``n_not_admitted`` or ``n_not_established_by_criterion``) through the
#: whole change that split that field in two, because nothing compared the two artifacts.
#: A reader writing a consumer against the documented schema would have had a KeyError.
_SUMMARY_KEYS = {
    "n_cells",
    "n_admitted",
    "n_degenerate",
    "n_not_admitted",
    "n_failed_by_criterion",
    "n_degenerate_by_criterion",
    "n_not_established_by_criterion",
    "degenerate_families",
}

#: Same, for one cell of the ``results`` block.
_RESULT_KEYS = {
    "family",
    "difficulty",
    "passed",
    "status",
    "criteria",
    "outcomes",
    "degenerate_criteria",
    "details",
    "metrics",
    "design",
    "error",
}


def test_the_published_schema_is_exactly_these_keys(ledger):
    """Anything documenting the ledger has this to check itself against.

    Both directions matter: a key the artifact stopped writing breaks a consumer, and a
    key it started writing is one the docs never told anyone about. Asserting equality
    rather than containment is what makes the second half fail too.
    """
    assert set(ledger) == {"schema", "provenance", "design", "summary", "results",
                           "timing_seconds"}
    assert set(ledger["summary"]) == _SUMMARY_KEYS
    for row in ledger["results"]:
        assert set(row) == _RESULT_KEYS, row["family"]


def test_the_guard_thresholds_are_recorded_with_the_numbers(ledger):
    """A verdict is uninterpretable without the cut that produced it."""
    gate = ledger["provenance"]["gate"]
    assert gate["equivalence_margin"] == 0.10
    assert gate["informativeness_floor"] == 1e-4
    for row in ledger["results"]:
        metrics = row["metrics"].get("proxy_matched")
        if not metrics:
            continue
        assert "statistic_resolution" in metrics
        assert "proxy_informativeness" in metrics
        assert metrics["informativeness_floor"] == gate["informativeness_floor"]
