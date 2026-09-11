"""Read coding sheets off disk into :class:`~rhob.audit.criterion.CodedCriterion` rows.

A coding sheet is JSON so that it can be diffed, reviewed, and corrected by someone
who does not run Python -- in particular by an author of a benchmark it names, who
should be able to send back a patch to a row rather than an argument about a table.

The loader is strict: an unknown enum value or a missing required field raises
rather than defaulting, because a row that silently defaults to the benign value is
a row nobody coded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rhob.audit.criterion import ClaimShape, CodedCriterion, FailureEvidence, StatTest

__all__ = ["load_coding", "load_snapshot", "parse_row"]

_REQUIRED = ("benchmark", "criterion", "source", "claim_shape", "test_used")


def parse_row(raw: dict[str, Any]) -> CodedCriterion:
    """Build one row, raising on anything the coder left unstated."""
    missing = [k for k in _REQUIRED if not raw.get(k)]
    if missing:
        raise ValueError(
            f"coded row is missing required field(s) {missing}: {raw!r}. A row that "
            "cannot be checked is not evidence about a benchmark."
        )

    try:
        shape = ClaimShape(raw["claim_shape"])
        test = StatTest(raw["test_used"])
        failure = FailureEvidence(raw.get("failure_demonstrated", "none"))
    except ValueError as exc:
        raise ValueError(f"unrecognised coding value in row {raw!r}: {exc}") from exc

    guarded = raw.get("resolution_guarded")
    if guarded is not None and not isinstance(guarded, bool):
        raise ValueError(
            f"resolution_guarded must be true, false or null, got {guarded!r}"
        )

    return CodedCriterion(
        benchmark=raw["benchmark"],
        criterion=raw["criterion"],
        source=raw["source"],
        claim_shape=shape,
        test_used=test,
        failure_demonstrated=failure,
        resolution_guarded=guarded,
        coder=raw.get("coder", ""),
        notes=raw.get("notes", ""),
    )


def load_coding(path: str | Path) -> dict[str, tuple[CodedCriterion, ...]]:
    """Load every snapshot in a coding sheet, keyed by snapshot name."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    snapshots = data.get("snapshots")
    if not isinstance(snapshots, dict):
        raise ValueError(f"{path}: no 'snapshots' object")

    out: dict[str, tuple[CodedCriterion, ...]] = {}
    for name, snapshot in snapshots.items():
        rows = snapshot.get("rows", [])
        out[name] = tuple(parse_row(r) for r in rows)
    return out


def load_snapshot(path: str | Path, name: str) -> tuple[CodedCriterion, ...]:
    """Load one named snapshot from a coding sheet."""
    all_snapshots = load_coding(path)
    if name not in all_snapshots:
        raise KeyError(f"{path}: no snapshot named {name!r}; have {sorted(all_snapshots)}")
    return all_snapshots[name]
