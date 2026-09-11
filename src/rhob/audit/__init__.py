"""Falsifiability diagnostics for benchmark preconditions, and the survey that uses them.

This package exists so that a claim about somebody else's benchmark can be checked
by somebody else -- including by the authors of the benchmark named, who are the
people best placed to say the coding is wrong.

Start with ``docs/SURVEY_PROTOCOL.md`` for the decision rules a coder applies, and
``audit/rhob_self_coding.json`` for the worked example: RHOB's own six admission
criteria, coded before and after the 2026-08 audit.
"""

from __future__ import annotations

from rhob.audit.criterion import (
    ClaimShape,
    CodedCriterion,
    CriterionFinding,
    FailureEvidence,
    StatTest,
    Verdict,
)
from rhob.audit.diagnostics import (
    InadmissibleRow,
    SurveyResult,
    diagnose,
    diagnose_degeneracy,
    diagnose_demonstrated_failure,
    diagnose_direction,
    format_table,
    run_survey,
)
from rhob.audit.loader import load_coding, load_snapshot

__all__ = [
    "ClaimShape",
    "CodedCriterion",
    "CriterionFinding",
    "FailureEvidence",
    "InadmissibleRow",
    "SurveyResult",
    "StatTest",
    "Verdict",
    "diagnose",
    "diagnose_degeneracy",
    "diagnose_demonstrated_failure",
    "diagnose_direction",
    "format_table",
    "load_coding",
    "load_snapshot",
    "run_survey",
]
