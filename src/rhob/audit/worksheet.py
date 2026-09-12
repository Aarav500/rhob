"""CSV worksheets for the two-coder pass, and the agreement report that follows it.

Why a CSV step exists at all
----------------------------
The paper's §5 sheet records, per precondition, **whether an equivalence test
supports it**. That is one of the two fields a finding needs. The other is
:class:`~rhob.audit.criterion.ClaimShape` -- whether the precondition asserts an
equivalence in the first place -- and it is the field the chosen denominator is
computed over.

Those are not the same judgement and neither implies the other. A precondition with
no equivalence test is a finding only if it is equivalence-shaped; a difference claim
tested with a difference test is correct, and four of RHOB's own six criteria are
exactly that. So porting §5 into ``audit/*.json`` does **not** reproduce the headline
under the new denominator: it yields rows with ``test_used`` known and
``claim_shape`` unknown, which :meth:`CodedCriterion.is_admissible` rejects.

The missing field has to be coded by a person, twice, per ``docs/SURVEY_PROTOCOL.md``
§2 and §5. This module is the paperwork for that: emit a worksheet per coder with
everything already known pre-filled and the judgement columns blank, read them back,
and report how often the two coders disagreed -- a number the protocol requires be
published with the result.

Expected direction of the effect
--------------------------------
Narrowing the denominator is likely to *strengthen* the finding rather than weaken
it. "0 of 77 preconditions have an equivalence test" invites the reply that most of
the 77 never needed one. "N of N equivalence-shaped preconditions are inverted", with
the shape call published per row and contestable by the benchmark's own authors, does
not have that reply available to it.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from rhob.audit.criterion import ClaimShape, CodedCriterion, FailureEvidence, StatTest
from rhob.audit.loader import parse_row

__all__ = [
    "WORKSHEET_COLUMNS",
    "AgreementReport",
    "agreement_report",
    "read_worksheet",
    "write_worksheet",
]

#: Column order. ``claim_shape`` sits immediately after ``source`` because the coder
#: should make the call from the source text, before seeing what the benchmark did
#: about it -- knowing there is no equivalence test invites coding the shape to match.
WORKSHEET_COLUMNS = (
    "benchmark",
    "criterion",
    "source",
    "claim_shape",
    "test_used",
    "failure_demonstrated",
    "resolution_guarded",
    "coder",
    "notes",
)

_SHAPE_RULE = (
    "claim_shape: equivalence | difference | threshold. Rule (protocol §2): if a "
    "noisier or weaker measurement makes the criterion EASIER to pass, it is "
    "equivalence. If noise makes it HARDER to pass, it is difference or threshold."
)


def write_worksheet(
    rows: Iterable[CodedCriterion],
    path: str | Path,
    *,
    coder: str,
    blank_fields: Sequence[str] = ("claim_shape",),
) -> Path:
    """Emit a coding worksheet with ``blank_fields`` cleared for a human to fill.

    Args:
        rows: Partially-known rows -- typically benchmark, criterion, source and
            whatever the source sheet already recorded.
        path: Destination CSV.
        coder: Stamped into every row, so a returned worksheet carries its author.
        blank_fields: Columns to clear. Defaults to the shape call, which is the one
            the paper's sheet does not contain.

    Returns:
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as fh:
        fh.write(f"# {_SHAPE_RULE}\n")
        fh.write("# Leave a cell blank only if you cannot make the call; say why in notes.\n")
        writer = csv.DictWriter(fh, fieldnames=list(WORKSHEET_COLUMNS))
        writer.writeheader()
        for row in rows:
            record = {
                "benchmark": row.benchmark,
                "criterion": row.criterion,
                "source": row.source,
                "claim_shape": row.claim_shape.value,
                "test_used": row.test_used.value,
                "failure_demonstrated": row.failure_demonstrated.value,
                "resolution_guarded": (
                    "" if row.resolution_guarded is None else str(row.resolution_guarded).lower()
                ),
                "coder": coder,
                "notes": row.notes,
            }
            for field_name in blank_fields:
                record[field_name] = ""
            writer.writerow(record)
    return out


def _parse_guard(value: str) -> bool | None:
    text = value.strip().lower()
    if text in ("", "null", "none", "n/a"):
        return None
    if text in ("true", "yes", "y", "1"):
        return True
    if text in ("false", "no", "n", "0"):
        return False
    raise ValueError(f"resolution_guarded: cannot read {value!r} as true/false/blank")


def read_worksheet(path: str | Path) -> tuple[CodedCriterion, ...]:
    """Read a filled worksheet back into rows, raising on an uncoded judgement.

    A blank ``claim_shape`` raises rather than defaulting. The whole point of the
    pass is that somebody made the call; a row that silently defaults to the benign
    value is a row nobody coded.
    """
    src = Path(path)
    rows: list[CodedCriterion] = []

    with src.open(newline="", encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]

    for lineno, raw in enumerate(csv.DictReader(lines), start=2):
        if not (raw.get("benchmark") or "").strip():
            continue
        blank = [f for f in ("claim_shape", "test_used") if not (raw.get(f) or "").strip()]
        if blank:
            raise ValueError(
                f"{src}:{lineno}: {raw.get('benchmark')}/{raw.get('criterion')} has "
                f"uncoded {blank}. Code it or drop the row; do not leave it to default."
            )
        raw = dict(raw)
        raw["resolution_guarded"] = _parse_guard(raw.get("resolution_guarded") or "")
        rows.append(parse_row(raw))

    return tuple(rows)


@dataclass(frozen=True)
class AgreementReport:
    """How often two independent coders made the same call, and where they did not.

    ``docs/SURVEY_PROTOCOL.md`` §5 requires this be published with the survey. A high
    disagreement rate on ``claim_shape`` does not invalidate the result, but it does
    bound how much weight the per-row verdicts carry, and hiding it would be the same
    species of omission the survey is about.
    """

    field: str
    n_compared: int
    n_agreed: int
    disagreements: tuple[tuple[str, str, str, str], ...]
    only_in_a: tuple[str, ...] = ()
    only_in_b: tuple[str, ...] = ()

    @property
    def rate(self) -> float:
        """Fraction agreed. Returns 1.0 for an empty comparison, which is vacuous."""
        return 1.0 if self.n_compared == 0 else self.n_agreed / self.n_compared

    def summary(self) -> str:
        lines = [
            f"{self.field}: {self.n_agreed}/{self.n_compared} agreed "
            f"({self.rate:.1%})"
        ]
        for benchmark, criterion, a, b in self.disagreements:
            lines.append(f"  disagree  {benchmark}/{criterion}: A={a}  B={b}")
        for key in self.only_in_a:
            lines.append(f"  only in A: {key}")
        for key in self.only_in_b:
            lines.append(f"  only in B: {key}")
        return "\n".join(lines)


def agreement_report(
    coder_a: Sequence[CodedCriterion],
    coder_b: Sequence[CodedCriterion],
    field: str = "claim_shape",
) -> AgreementReport:
    """Compare two coders' worksheets on one field.

    Rows are matched on ``(benchmark, criterion)``. Rows present for only one coder
    are reported rather than dropped -- a coder who skipped a precondition made a
    judgement about scope, and that is part of the result too.
    """
    index_a = {(r.benchmark, r.criterion): r for r in coder_a}
    index_b = {(r.benchmark, r.criterion): r for r in coder_b}

    shared = sorted(set(index_a) & set(index_b))
    disagreements: list[tuple[str, str, str, str]] = []
    agreed = 0

    for key in shared:
        val_a = getattr(index_a[key], field)
        val_b = getattr(index_b[key], field)
        if val_a == val_b:
            agreed += 1
        else:
            disagreements.append(
                (key[0], key[1], str(getattr(val_a, "value", val_a)), str(getattr(val_b, "value", val_b)))
            )

    fmt = lambda k: f"{k[0]}/{k[1]}"  # noqa: E731
    return AgreementReport(
        field=field,
        n_compared=len(shared),
        n_agreed=agreed,
        disagreements=tuple(disagreements),
        only_in_a=tuple(fmt(k) for k in sorted(set(index_a) - set(index_b))),
        only_in_b=tuple(fmt(k) for k in sorted(set(index_b) - set(index_a))),
    )


def shape_census(rows: Iterable[CodedCriterion]) -> Counter[str]:
    """How the coded preconditions split across claim shapes.

    This is the number that decides whether the survey has a finding at all: the
    equivalence-shaped count is the denominator, and if it is small the headline is
    about a small number of criteria and must say so.
    """
    return Counter(r.claim_shape.value for r in rows)


#: Re-exported so a caller building skeleton rows does not need two imports.
__all__ += ["ClaimShape", "FailureEvidence", "StatTest", "shape_census"]
