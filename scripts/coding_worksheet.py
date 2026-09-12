#!/usr/bin/env python3
"""Emit coding worksheets for the two-coder pass, and merge them back.

The paper's §5 sheet records whether each precondition has an equivalence test. It
does not record whether the precondition is equivalence-*shaped*, and that is the
field the published denominator is computed over (docs/SURVEY_PROTOCOL.md §2, §4).
So the port is not a format conversion -- it is a re-coding pass, and this is its
paperwork.

    # 1. Put the paper's §5 rows in a skeleton CSV (benchmark, criterion, source,
    #    test_used, and anything else already known). Leave claim_shape blank.
    # 2. Emit one worksheet per coder:
    python scripts/coding_worksheet.py emit audit/skeleton.csv --coders ash,second

    # 3. Each coder fills claim_shape independently, without conferring.
    # 4. Merge, see the disagreement rate, write the survey sheet:
    python scripts/coding_worksheet.py merge \\
        audit/worksheets/ash.csv audit/worksheets/second.csv \\
        --snapshot published_survey --out audit/survey.json

Merging refuses to write while the two coders still disagree, because the protocol
requires disagreements be recorded and resolved deliberately rather than by whichever
file was passed second. Use --accept a|b per the resolution you actually reached.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rhob.audit.worksheet import (  # noqa: E402
    agreement_report,
    read_worksheet,
    shape_census,
    write_worksheet,
)


def _display(path: Path) -> str:
    """Repo-relative where possible, absolute otherwise.

    Worksheets are routinely written outside the checkout -- a scratch directory, or
    a shared folder the second coder can reach -- so this must not raise.
    """
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def cmd_emit(args: argparse.Namespace) -> int:
    rows = read_worksheet(args.skeleton) if args.coded else _read_skeleton(args.skeleton)
    out_dir = Path(args.out_dir)
    for coder in args.coders.split(","):
        coder = coder.strip()
        if not coder:
            continue
        path = write_worksheet(rows, out_dir / f"{coder}.csv", coder=coder)
        print(f"wrote {_display(path)}  ({len(rows)} rows, claim_shape blank)")
    print("\nCoders fill claim_shape independently. Do not confer before merging.")
    return 0


def _read_skeleton(path: str) -> tuple:
    """Read a skeleton whose claim_shape is deliberately blank.

    ``read_worksheet`` refuses blanks by design, so a skeleton is filled with a
    placeholder shape purely to construct the row objects; ``write_worksheet`` then
    blanks the column again for the coder.
    """
    import csv

    from rhob.audit.criterion import ClaimShape, CodedCriterion, FailureEvidence, StatTest

    rows = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    for raw in csv.DictReader(lines):
        if not (raw.get("benchmark") or "").strip():
            continue
        rows.append(
            CodedCriterion(
                benchmark=raw["benchmark"].strip(),
                criterion=raw["criterion"].strip(),
                source=(raw.get("source") or "").strip(),
                claim_shape=ClaimShape(raw.get("claim_shape") or "difference"),
                test_used=StatTest(raw.get("test_used") or "none"),
                failure_demonstrated=FailureEvidence(
                    raw.get("failure_demonstrated") or "none"
                ),
                resolution_guarded=None,
                coder="",
                notes=(raw.get("notes") or "").strip(),
            )
        )
    return tuple(rows)


def cmd_merge(args: argparse.Namespace) -> int:
    a = read_worksheet(args.coder_a)
    b = read_worksheet(args.coder_b)

    report = agreement_report(a, b, field="claim_shape")
    print(report.summary())

    census = shape_census(a if args.accept == "a" else b)
    print("\nshape census (accepted coding):")
    for shape, n in sorted(census.items()):
        print(f"  {shape}: {n}")
    n_eq = census.get("equivalence", 0)
    print(
        f"\nDenominator for the published headline: {n_eq} equivalence-shaped "
        f"precondition(s) out of {sum(census.values())} coded."
    )
    if n_eq == 0:
        print("  -> report as 'no equivalence-shaped preconditions found', not as a rate.")

    if report.disagreements and not args.accept:
        print(
            "\nRefusing to write: coders disagree on "
            f"{len(report.disagreements)} row(s). Resolve them deliberately, then "
            "re-run with --accept a|b, and record the resolution in each row's notes.",
            file=sys.stderr,
        )
        return 1

    if not args.out:
        return 0

    accepted = a if args.accept == "a" else b
    payload = {
        "_about": [
            "Merged two-coder survey sheet. Produced by scripts/coding_worksheet.py.",
            f"claim_shape agreement: {report.n_agreed}/{report.n_compared} "
            f"({report.rate:.1%}). Publish this rate with the result.",
        ],
        "protocol": "docs/SURVEY_PROTOCOL.md",
        "agreement": {
            "field": report.field,
            "n_compared": report.n_compared,
            "n_agreed": report.n_agreed,
            "rate": round(report.rate, 4),
            "disagreements": [
                {"benchmark": d[0], "criterion": d[1], "coder_a": d[2], "coder_b": d[3]}
                for d in report.disagreements
            ],
        },
        "snapshots": {
            args.snapshot: {
                "_about": "Coded from the paper's §5 sheet, re-coded for claim shape.",
                "rows": [
                    {
                        "benchmark": r.benchmark,
                        "criterion": r.criterion,
                        "source": r.source,
                        "claim_shape": r.claim_shape.value,
                        "test_used": r.test_used.value,
                        "failure_demonstrated": r.failure_demonstrated.value,
                        "resolution_guarded": r.resolution_guarded,
                        "coder": r.coder,
                        "notes": r.notes,
                    }
                    for r in accepted
                ],
            }
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {_display(out)}")
    print(f"next: python scripts/audit_table.py {_display(out)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    emit = sub.add_parser("emit", help="write one blank-shape worksheet per coder")
    emit.add_argument("skeleton")
    emit.add_argument("--coders", default="a,b")
    emit.add_argument("--out-dir", default=str(REPO / "audit" / "worksheets"))
    emit.add_argument(
        "--coded",
        action="store_true",
        help="the skeleton is already a fully coded worksheet, not a skeleton",
    )
    emit.set_defaults(func=cmd_emit)

    merge = sub.add_parser("merge", help="compare two coders and write the survey sheet")
    merge.add_argument("coder_a")
    merge.add_argument("coder_b")
    merge.add_argument("--out", default=None)
    merge.add_argument("--snapshot", default="published_survey")
    merge.add_argument(
        "--accept",
        choices=("a", "b"),
        default=None,
        help="whose coding to write after disagreements were resolved deliberately",
    )
    merge.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
