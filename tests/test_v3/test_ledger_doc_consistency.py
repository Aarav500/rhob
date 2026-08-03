"""The repo's prose about admission must agree with ``admission/admission_ledger.json``.

Why this file exists
--------------------
The degeneracy guard landed and the ledger went from "30 of 35 cells admitted" to a
three-way split in the same change -- and every doc that quoted the old numbers kept
quoting them. ``README.md`` said 30 admitted next to a shipped artifact that said 15;
``REPRODUCIBILITY.md`` and ``CHANGELOG.md`` listed ``distributional_shift``,
``monitored_sandbagging`` and ``orbit_chirality`` as "Admitted at every tier" when every
one of their cells is DEGENERATE. None of that was caught by anything, because nothing
in the repo read the artifact and the prose together.

Nothing stops that recurring except a test, so: this one parses the numbers back out of
the Markdown and fails with both values side by side. It is deliberately about the
*headline* claims -- the counts a reader takes away and the admitted/degenerate labels on
named families -- rather than every number in the docs, because those are the claims that
misinform if they go stale.

The three checks
----------------
1. **The headline triple.** Every doc that summarizes the ledger must state admitted,
   degenerate and total, and those numbers must be the ledger's.
2. **No degenerate family is called admitted.** A family with a degenerate cell cannot be
   described as admitted at every tier, in any doc, in any phrasing this recognizes.
3. **The degenerate list is complete.** A doc that summarizes the ledger must mention
   every degenerate family somewhere -- a doc that names three of six reads as naming the
   set, which is precisely how the stale trio survived.
4. **Per-criterion cell counts.** "``proxy_matched`` was measured on 20 of the 35 cells"
   has to still be arithmetic about the current grid.

What is deliberately *not* checked is whether any particular sentence enumerates the set
correctly, because the docs legitimately discuss one family at a time and no regex can
tell that from an enumeration that stopped early. Check 3 is the tractable half of that
question; the rest is a reviewer's job.

Historical statements are not policed and must not be: the docs legitimately describe
what the *first* ledger measured, and rewriting history to match the current artifact
would be worse than the drift. Check 1 anchors on lines that state the current triple,
so a sentence about an earlier ledger is simply not one of them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from rhob.v3.admission_gate import CRITERIA

_REPO = Path(__file__).resolve().parents[2]
_LEDGER_JSON = _REPO / "admission" / "admission_ledger.json"

#: Docs that quote the ledger's headline. The ledger's own Markdown is included because
#: it is generated from the same JSON and a mismatch there means the renderer drifted.
_HEADLINE_DOCS = (
    _REPO / "README.md",
    _REPO / "REPRODUCIBILITY.md",
    _REPO / "CHANGELOG.md",
    _REPO / "admission" / "ADMISSION_LEDGER.md",
)

#: Every Markdown file whose family-level claims are checked. Wider than the headline
#: set: a stale "admitted at every tier" is just as wrong in a tutorial as in the README.
_ALL_DOCS = tuple(
    sorted(
        set(_HEADLINE_DOCS)
        | set((_REPO / "docs").rglob("*.md"))
        | {_REPO / "ROADMAP.md", _REPO / "CONTRIBUTING.md"}
    )
)

#: Ways the docs have actually claimed a family is admitted everywhere. Matched against
#: the same line as the family name.
_ADMITTED_EVERYWHERE = re.compile(
    r"admitted\s+(?:at|in)\s+(?:every|all|each)\b|"
    r"(?:every|all)\s+tiers?\s+admitted|"
    r"admitted\s+\d+\s*/\s*\d+",
    re.IGNORECASE,
)

_INT = re.compile(r"\d+")


@pytest.fixture(scope="module")
def ledger() -> dict:
    assert _LEDGER_JSON.exists(), (
        f"{_LEDGER_JSON} is missing. It is a committed artifact -- regenerate it with "
        "`python scripts/admission_ledger.py` before the docs can be checked against it."
    )
    return json.loads(_LEDGER_JSON.read_text(encoding="utf-8"))


def _headline_lines(text: str) -> list[str]:
    """Lines that state a ledger headline: admitted *and* degenerate *and* >= 3 numbers.

    Requiring all three keeps prose that mentions only one of the counts (or only the
    word) out of the check, while catching every line that actually summarizes the grid.
    """
    out = []
    for line in text.splitlines():
        low = line.lower()
        if "admitted" in low and "degenerate" in low and len(_INT.findall(line)) >= 3:
            out.append(line.strip())
    return out


def test_headline_counts_in_the_docs_match_the_ledger(ledger):
    """Admitted / degenerate / total, as the docs state them, against the artifact."""
    summary = ledger["summary"]
    expected = {
        "admitted": summary["n_admitted"],
        "degenerate": summary["n_degenerate"],
        "cells": summary["n_cells"],
    }
    assert expected["admitted"] + expected["degenerate"] + summary["n_not_admitted"] == (
        expected["cells"]
    ), f"the ledger's own summary does not add up: {summary}"

    for doc in _HEADLINE_DOCS:
        assert doc.exists(), f"{doc} is missing"
        lines = _headline_lines(doc.read_text(encoding="utf-8"))
        assert lines, (
            f"{doc.relative_to(_REPO)} never states the ledger headline. It must contain "
            f"a line giving all three of admitted={expected['admitted']}, "
            f"degenerate={expected['degenerate']}, total cells={expected['cells']} -- "
            "otherwise a reader has no way to tell which ledger the prose describes."
        )
        for line in lines:
            found = {int(n) for n in _INT.findall(line)}
            missing = {k: v for k, v in expected.items() if v not in found}
            assert not missing, (
                f"{doc.relative_to(_REPO)} states a ledger headline that disagrees with "
                f"{_LEDGER_JSON.relative_to(_REPO)}.\n"
                f"  doc says : {line}\n"
                f"  ledger   : {expected['admitted']} admitted, "
                f"{expected['degenerate']} degenerate, "
                f"{summary['n_not_admitted']} not admitted, "
                f"{expected['cells']} cells\n"
                f"  missing from the line: {missing}"
            )


def test_no_degenerate_family_is_described_as_admitted(ledger):
    """A family with an unmeasurable proxy is not admitted, anywhere, in any phrasing."""
    degenerate = set(ledger["summary"]["degenerate_families"])
    assert degenerate, "the ledger reports no degenerate families; this check is vacuous"

    offences = []
    for doc in _ALL_DOCS:
        if not doc.exists():
            continue
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            if not _ADMITTED_EVERYWHERE.search(line):
                continue
            for family in sorted(degenerate):
                if family in line:
                    offences.append(f"{doc.relative_to(_REPO)}:{lineno}: {line.strip()}")
    assert not offences, (
        "these lines describe a family the ledger marks DEGENERATE as admitted:\n  "
        + "\n  ".join(offences)
        + f"\nThe ledger's degenerate families are: {sorted(degenerate)}"
    )


def test_every_degenerate_family_is_named_in_the_headline_docs(ledger):
    """Omission is the other half of the drift, and it is the half that reads as fine.

    A doc that names three of the six degenerate families reads as naming the set, and
    that is exactly how the stale trio survived a ledger that had grown to six. Whether
    a *particular* sentence enumerates correctly is not something a regex can judge --
    the docs legitimately discuss one family at a time -- but a document that describes
    the degenerate set at all and never mentions a member of it is unambiguously
    incomplete.
    """
    degenerate = sorted(ledger["summary"]["degenerate_families"])
    for doc in _HEADLINE_DOCS:
        text = doc.read_text(encoding="utf-8")
        missing = [f for f in degenerate if f not in text]
        assert not missing, (
            f"{doc.relative_to(_REPO)} never mentions {missing}, which the ledger marks "
            f"DEGENERATE at every scored tier. The full set is {degenerate}."
        )


#: Phrases that mark a sentence as describing an *earlier* ledger. The docs deliberately
#: record what the first equivalence ledger measured -- "15 of its 35 cells" is a true
#: statement about a superseded artifact -- and rewriting history to match the current
#: one would be a worse failure than the drift this file exists to catch.
_HISTORICAL = re.compile(
    r"first[\w \-]{0,30}ledger|pre-audit|earlier version|used to|previously|originally|"
    r"before the (?:guard|fix|audit)|at the time",
    re.IGNORECASE,
)

_CELL_CLAIM = re.compile(r"(\d+)\s*(?:/|of)\s*(?:the\s+|its\s+)?(\d+)\s+(?:ledger\s+)?cells")


def test_per_criterion_cell_counts_match_the_current_ledger(ledger):
    """"``proxy_matched`` was measured on 20 of the 35 cells" has to still be true.

    The headline check above only inspects lines that give the whole triple. This one
    catches the other shape the drift takes: a per-criterion count quoted in passing,
    which stays quietly wrong for a whole release because nobody re-derives it.

    Deliberately narrow. It fires only on a line that names one of the six criteria
    *and* makes an "N of M cells" claim, because the docs also count cells against the
    whole 123-cell benchmark grid and against per-family tiers, and a scan that could
    not tell those apart would cry wolf until someone deleted it.
    """
    s = ledger["summary"]
    allowed = {s["n_admitted"], s["n_degenerate"], s["n_not_admitted"], s["n_cells"]}
    for block in ("n_failed_by_criterion", "n_degenerate_by_criterion",
                  "n_not_established_by_criterion"):
        allowed |= set(s[block].values())
    allowed |= {s["n_cells"] - v for v in set(allowed)}

    offences = []
    for doc in _HEADLINE_DOCS:
        # Paragraph granularity, not line: Markdown wraps, so the marker that makes a
        # sentence historical ("the first ledger ... 15 of its 35 cells") routinely
        # lands on a different line from the number it excuses.
        for para in re.split(r"\n\s*\n", doc.read_text(encoding="utf-8")):
            if _HISTORICAL.search(para) or not any(c in para for c in CRITERIA):
                continue
            for numerator, denominator in _CELL_CLAIM.findall(para):
                if int(denominator) != s["n_cells"] or int(numerator) not in allowed:
                    offences.append(
                        f"{doc.relative_to(_REPO)}: '{numerator} of {denominator} "
                        f"cells' -- {' '.join(para.split())[:160]}"
                    )
    assert not offences, (
        f"the ledger has {s['n_cells']} cells: {s['n_admitted']} admitted, "
        f"{s['n_degenerate']} degenerate, {s['n_not_admitted']} not admitted; "
        f"per criterion, failed={s['n_failed_by_criterion']} "
        f"degenerate={s['n_degenerate_by_criterion']}. These claims do not match it and "
        "are not marked as historical:\n  " + "\n  ".join(offences)
    )


def test_the_ledger_markdown_and_json_report_the_same_grid(ledger):
    """The published Markdown is generated from this JSON, and must still say so."""
    md = (_REPO / "admission" / "ADMISSION_LEDGER.md").read_text(encoding="utf-8")
    summary = ledger["summary"]
    assert f"{summary['n_admitted']} / {summary['n_cells']} cells admitted" in md
    assert f"{summary['n_degenerate']} degenerate" in md
    assert f"{summary['n_not_admitted']} not admitted" in md
    # Every degenerate cell is listed by name, not just counted.
    for row in ledger["results"]:
        if row["degenerate_criteria"]:
            assert f"| {row['family']} | {row['difficulty']:.2f} |" in md
