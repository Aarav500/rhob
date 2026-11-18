#!/usr/bin/env python3
"""Axis A: D2 power for the paper's sub-class A claims, by the pre-registered rules.

Sub-class A is an equivalence asserted from a non-significant difference test, 15 of the
77 preconditions in the paper's survey. For each such claim this asks how often its
difference test would have rejected, at the design it shipped, if the claim were violated
by a true AUROC of 0.7 (0.6 and 0.8 as secondaries), and applies the decision rules fixed
in docs/SURVEY_AXIS_A_PREREG.md before any row was filled. The arithmetic is
``cannotfail``'s ``min_two_sided_p`` and ``difference_power``; nothing here re-derives it.

    # every input lives outside the repository or under audit/private/, which
    # .gitignore excludes; the template's one row is synthetic
    python scripts/survey_axis_a_power.py audit/private/axis_a_sheet.csv \\
        --sealed-record audit/private/axis_a_sealed.txt \\
        --coder-sheets audit/private/axis_a_coder_1.csv audit/private/axis_a_coder_2.csv
    python scripts/survey_axis_a_power.py audit/private/axis_a_sheet.csv --aggregate-only ...
    python scripts/survey_axis_a_power.py audit/private/axis_a_sheet.csv --benchmark NAME
    # checks that compute nothing: the sealed record alone, before its digest is recorded;
    # a coder's own sheet; the reconciled sheet against the record and both own sheets
    python scripts/survey_axis_a_power.py --check --sealed-record audit/private/axis_a_sealed.txt
    python scripts/survey_axis_a_power.py audit/private/axis_a_coder_1.csv --check
    python scripts/survey_axis_a_power.py audit/private/axis_a_sheet.csv --check \\
        --sealed-record audit/private/axis_a_sealed.txt \\
        --coder-sheets audit/private/axis_a_coder_1.csv audit/private/axis_a_coder_2.csv

The default output is a per-row table, which names benchmarks, followed by the aggregate
counts. ``--aggregate-only`` prints the counts alone: no benchmark, criterion, source,
coder or claim id. ``--benchmark NAME`` prints one benchmark's rows, for that group's
email, and no counts. ``--check`` validates its inputs and prints only counts, never a
value or a verdict. Until every
group's outreach window has closed (docs/AUTHOR_OUTREACH.md), the default and
``--aggregate-only`` output stay with the two coders, the counts included, and
``--benchmark`` output goes only to that group: the paper puts the 15 claims in 4
benchmarks, so some counts could single one out. Saved output goes under audit/private/ or
outside the repository.

The script computes all six analyses of prereg section 7.3. Its last line gives an outcome
only when every input is the registered one: the pinned power.py and frame, the sealed
record whose digest section 13 records, with the same identification columns as the sheet,
and the two coders' own sheets, three different files. ``disagreed_fields`` must name
exactly the coded columns the own sheets hold differently, and every other coded column
must hold the value both own sheets hold (not checked with ``--after-responses``).
Otherwise the last line says none, and why.

Exit status: 0 when every input is valid, whatever its verdicts; 1 when an input is
invalid, is missing, is not UTF-8 text, or sits anywhere in this checkout outside
audit/private/, the template included. An invalid input is reported by line and column,
or as a whole file, without echoing any cell, and nothing is computed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "cannotfail" / "src"))

import cannotfail  # noqa: E402
from cannotfail import power as cf_power  # noqa: E402
from cannotfail.power import difference_power, min_two_sided_p  # noqa: E402

PREREG = "docs/SURVEY_AXIS_A_PREREG.md"
FRAME = REPO / "audit" / "survey_s1_frame.json"
TEMPLATE = REPO / "audit" / "axis_a_sheet_TEMPLATE.csv"
PRIVATE_DIR = REPO / "audit" / "private"

PRIMARY_AUROC = 0.7
GRID = (0.6, 0.7, 0.8)
TARGET_POWER = 0.8
ALPHA_RANGE = (1e-10, 0.5)
# SHA-256, with LF line endings, of tools/cannotfail/src/cannotfail/power.py as last
# changed in 88a3f10, and of audit/survey_s1_frame.json. The pre-registration names both
# (section 5); a run on any other file is not this analysis, and reports no outcome.
REGISTERED_POWER_SHA256 = "63a94676b51294cc9528b3c830d66e610b3f459893327edfa4675122a5429cd2"
REGISTERED_FRAME_SHA256 = "4952ac6e2cfab407f68517c091ad839bd055c379070d298725fcdd80b4543936"

COLUMNS = (
    "claim_id",
    "benchmark",
    "criterion",
    "source_location",
    "sub_class_a",
    "test_used",
    "n_arm_a",
    "n_arm_b",
    "unit_of_replication",
    "alpha",
    "sidedness",
    "defaults_applied",
    "exact_test",
    "scores",
    "clustered",
    "other_tests",
    "coder_1",
    "coder_2",
    "disagreed_fields",
    "notes",
)
IDENTIFICATION = ("claim_id", "benchmark", "criterion", "source_location")
# What the two coders fill independently from the source; the identification fields
# before them come from the sealed record.
CODED_FIELDS = (
    "sub_class_a",
    "test_used",
    "n_arm_a",
    "n_arm_b",
    "unit_of_replication",
    "alpha",
    "sidedness",
    "defaults_applied",
    "exact_test",
    "scores",
    "clustered",
    "other_tests",
)

RANK_TESTS = frozenset({"mann_whitney"})
MEAN_TESTS = frozenset({"t_test", "bootstrap_mean_difference"})
UNCOMPUTED_TESTS = frozenset(
    {"paired_t", "wilcoxon_signed_rank", "chi_square", "fisher_exact", "ks", "other"}
)
TESTS = RANK_TESTS | MEAN_TESTS | UNCOMPUTED_TESTS
DEFAULTS = ("alpha", "sidedness", "test", "conflict")
EXACT = ("yes", "no", "unknown")
SCORES = ("continuous", "ordinal", "binary", "unknown")
CLUSTERED = ("yes", "no", "unknown")
SIDES = ("two_sided", "one_sided")

NOT_MEMBER = "not sub-class A on re-coding"
NOT_COMPUTED = "not computed"
CANNOT_REJECT = "cannot reject at any data"
UNDERPOWERED = "underpowered"
ADEQUATE = "adequately powered"
VERDICTS = (NOT_MEMBER, NOT_COMPUTED, CANNOT_REJECT, UNDERPOWERED, ADEQUATE)
COMPUTED = (CANNOT_REJECT, UNDERPOWERED, ADEQUATE)
WEAK = (CANNOT_REJECT, UNDERPOWERED)

NOT_COMPUTED_REASONS = {
    "test": "test neither rank-based nor a two-sample mean difference",
    "n": "n per arm not stated in the source",
    "floor": "mean-difference test whose exact rank floor is above alpha",
}
# Why a rank row whose exact floor is above alpha is not "cannot reject" (prereg rule 3).
RULE3_BLOCKERS = {
    "exact": "the source does not show an exact test",
    "default": "alpha or sidedness rests on a default",
    "other_tests": "another test of the claim could reject",
}

INCONCLUSIVE = "inconclusive"
MOSTLY_WEAK = "most could not reliably fail"
SPLIT = "split"
MOSTLY_ADEQUATE = "most adequately powered"
NOT_ROBUST = "not robust"

SENSITIVITIES = (
    ("b", "clustered or unknown rows dropped"),
    ("c", "t-test family at the efficiency-adjusted n"),
    ("d", "claims resting on several tests at their combined chance to fail"),
    ("e", "binary or unknown scores at the two-proportion power"),
    ("f", "a defaulted sidedness taken as one-sided"),
)

RECORD_TITLE = "axis-A sealed record"
RECORD_KEYS = ("salt", "paper_underpowered_bar", "expected")
RECORD_COLUMNS = (*IDENTIFICATION, "paper_20")
PAPER_20 = ("underpowered", "not_underpowered", "no", "unknown")
# The line of prereg section 13 that records the sealed record's digest once it is made.
RECORD_DIGEST_LINE = re.compile(r"\*\*Sealed record: SHA-256 `([0-9a-f]{64})`")

_NORMAL = NormalDist()


class SheetError(ValueError):
    """An input cannot be used; ``problems`` says where and why, one per entry."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class DiffTest:
    """One difference test: the coded one, or an entry of ``other_tests``."""

    test_used: str
    n_a: int | None
    n_b: int | None
    alpha: float
    one_sided: bool


@dataclass(frozen=True)
class Claim:
    line: int
    claim_id: str
    benchmark: str
    criterion: str
    source_location: str
    sub_class_a: bool
    test_used: str
    n_a: int | None
    n_b: int | None
    unit: str
    alpha: float
    one_sided: bool
    clustered: str
    coder_1: str
    coder_2: str
    disagreed: tuple[str, ...]
    notes: str
    defaults: frozenset[str] = frozenset()
    exact_test: str = "unknown"
    scores: str = "unknown"
    other_tests: tuple[DiffTest, ...] = ()

    @property
    def coded(self) -> DiffTest:
        return DiffTest(self.test_used, self.n_a, self.n_b, self.alpha, self.one_sided)

    @property
    def identification(self) -> tuple[str, str, str, str]:
        return (self.claim_id, self.benchmark.casefold(), self.criterion, self.source_location)


@dataclass(frozen=True)
class Assessment:
    claim: Claim
    verdict: str
    reason: str = ""
    min_p: float | None = None
    power: dict[float, float] | None = None
    adjusted_verdict: str | None = None  # (c)
    blocked: str = ""  # why rule 3 did not apply to a rank row below its floor
    combined: float | None = None  # (d)
    binary_power: float | None = None  # (e)
    one_sided_power: float | None = None  # (f)


@dataclass(frozen=True)
class Frame:
    expected_rows: int
    expected_benchmarks: int
    audited: dict[str, int]
    unaudited: frozenset[str]
    sha256: str = ""


@dataclass(frozen=True)
class SealedRecord:
    sha256: str
    salt: str
    expected_w: int
    expected_j: int
    # claim_id -> (its identification columns, its paper_20 value)
    claims: dict[str, tuple[tuple[str, str, str, str], str]]


@dataclass(frozen=True)
class CoderCheck:
    first_salted: str | None
    second_salted: str | None
    rows_not_shared: int
    rows_mismatched: int
    second: tuple[Assessment, ...] = field(default=())


@dataclass(frozen=True)
class Context:
    frame: Frame
    record: SealedRecord | None = None
    registered_record: str | None = None
    coders: CoderCheck | None = None
    sheet_salted: str | None = None
    after_responses: bool = False


# --- reading ----------------------------------------------------------------------


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def salted_sha256(salt: str, path: Path) -> str:
    """The SHA-256 of the salt, a newline and the file's bytes (prereg section 13)."""
    return hashlib.sha256(salt.encode("ascii") + b"\n" + Path(path).read_bytes()).hexdigest()


def load_frame(path: Path) -> Frame:
    """The totals the sheet must reproduce: the paper's difference-test claims."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    audited, unaudited = {}, set()
    for member in data["frame"]:
        key = member["benchmark"].casefold()
        if member["axis2_claims"] is None:
            unaudited.add(key)
        else:
            audited[key] = int(member["axis2_claims"])
    return Frame(
        expected_rows=int(data["support_totals"]["difference_test"]),
        expected_benchmarks=int(data["benchmarks_with_difference_test_claims"]),
        audited=audited,
        unaudited=frozenset(unaudited),
        sha256=lf_sha256(path),
    )


def _whole(text: str) -> int | None:
    text = text.strip()
    return int(text) if text.isascii() and text.isdigit() else None


def _level(text: str) -> float | None:
    """A level in the registered range, or None."""
    text = text.strip()
    if not text.isascii():
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if ALPHA_RANGE[0] <= value <= ALPHA_RANGE[1] else None


def _names(text: str) -> tuple[str, ...]:
    return tuple(f.strip().lower() for f in text.split(";") if f.strip())


def _diff_test(entry: str) -> DiffTest | None:
    """One ``other_tests`` entry, ``test_used:n_arm_a:n_arm_b:alpha:sidedness``, or None."""
    parts = [p.strip() for p in entry.split(":")]
    if len(parts) != 5:
        return None
    test, raw_a, raw_b, raw_alpha, sided = parts
    test, sided = test.lower(), sided.lower()
    n_a, n_b = _whole(raw_a), _whole(raw_b)
    alpha = _level(raw_alpha)
    if test not in TESTS or sided not in SIDES or alpha is None or bool(raw_a) != bool(raw_b):
        return None
    if raw_a and (n_a is None or n_b is None or n_a < 1 or n_b < 1):
        return None
    return DiffTest(test, n_a, n_b, alpha, sided == "one_sided")


def _parse_row(raw: dict, line: int, frame: Frame, problems: list[str]) -> Claim | None:
    if None in raw:
        problems.append(f"line {line}: more cells than the header has columns")
        return None
    if any(raw.get(k) is None for k in COLUMNS):
        problems.append(f"line {line}: fewer cells than the header has columns")
        return None
    # A line break inside a quoted cell is compared as LF, as the sealed record's is.
    cell = {k: raw[k].replace("\r\n", "\n").strip() for k in COLUMNS}
    before = len(problems)

    def bad(column: str, reason: str) -> None:
        problems.append(f"line {line}: {column}: {reason}")

    def one_of(column: str, allowed) -> str:
        value = cell[column].lower()
        if value not in allowed:
            bad(column, f"must be one of {', '.join(sorted(allowed))}")
        return value

    for column in ("claim_id", "benchmark", "criterion", "source_location", "unit_of_replication"):
        if not cell[column]:
            bad(column, "required, and blank")
    if cell["benchmark"].casefold() in frame.unaudited:
        bad("benchmark", "a member the paper did not audit; out of scope (prereg section 11)")

    sub = one_of("sub_class_a", ("yes", "no"))
    test = one_of("test_used", TESTS)

    raw_a, raw_b = cell["n_arm_a"], cell["n_arm_b"]
    n_a, n_b = _whole(raw_a), _whole(raw_b)
    if bool(raw_a) != bool(raw_b):
        bad("n_arm_a/n_arm_b", "give both, or leave both blank when the source does not state n")
    for column, text, n in (("n_arm_a", raw_a, n_a), ("n_arm_b", raw_b, n_b)):
        if text and (n is None or n < 1):
            bad(column, "must be a whole number of at least 1, in ASCII digits")

    alpha = _level(cell["alpha"])
    if alpha is None:
        bad("alpha", f"must be a number from {ALPHA_RANGE[0]:g} to {ALPHA_RANGE[1]:g}")
    sided = one_of("sidedness", SIDES)

    defaults = _names(cell["defaults_applied"])
    unknown_defaults = sum(1 for d in defaults if d not in DEFAULTS)
    if unknown_defaults:
        bad("defaults_applied", f"{unknown_defaults} entry(ies) not among {', '.join(DEFAULTS)}")
    if len(set(defaults)) != len(defaults):
        bad("defaults_applied", "names a default twice")
    # defaults_applied describes the claim as a whole, any of its tests (prereg section 3),
    # so it is not checked against the coded test's own values.

    exact = one_of("exact_test", EXACT)
    scores = one_of("scores", SCORES)
    clustered = one_of("clustered", CLUSTERED)

    others: list[DiffTest] = []
    for i, entry in enumerate((e for e in cell["other_tests"].split(";") if e.strip()), 1):
        parsed = _diff_test(entry)
        if parsed is None:
            bad(
                "other_tests",
                f"entry {i} must be test_used:n_arm_a:n_arm_b:alpha:sidedness, each value as "
                "its own column allows, both n blank when not stated (prereg section 3)",
            )
        else:
            others.append(parsed)

    c1, c2 = cell["coder_1"], cell["coder_2"]
    if not c1 or not c2:
        bad("coder_1/coder_2", "two coders are required (prereg section 9)")
    elif c1.casefold() == c2.casefold():
        bad("coder_1/coder_2", "must name two different coders (prereg section 9)")

    disagreed = _names(cell["disagreed_fields"])
    unknown = sum(1 for f in disagreed if f not in CODED_FIELDS)
    if unknown:
        bad("disagreed_fields", f"{unknown} entry(ies) not among the coded fields "
            f"{', '.join(CODED_FIELDS)}")
    if len(set(disagreed)) != len(disagreed):
        bad("disagreed_fields", "names a field twice")

    if len(problems) > before:
        return None
    claim = Claim(
        line=line,
        claim_id=cell["claim_id"],
        benchmark=cell["benchmark"],
        criterion=cell["criterion"],
        source_location=cell["source_location"],
        sub_class_a=sub == "yes",
        test_used=test,
        n_a=n_a,
        n_b=n_b,
        unit=cell["unit_of_replication"],
        alpha=alpha,
        one_sided=sided == "one_sided",
        clustered=clustered,
        coder_1=c1,
        coder_2=c2,
        disagreed=disagreed,
        notes=cell["notes"],
        defaults=frozenset(defaults),
        exact_test=exact,
        scores=scores,
        other_tests=tuple(others),
    )
    coded = rank_key(claim.coded)
    for i, other in enumerate(claim.other_tests, 1):
        if rank_key(other) > coded:
            bad(
                "other_tests",
                f"entry {i} is more powerful at AUROC {PRIMARY_AUROC} than the coded test; "
                "the row codes the most powerful (prereg section 2)",
            )
    return None if len(problems) > before else claim


def read_sheet(path: Path, frame: Frame) -> tuple[Claim, ...]:
    """Every row of the sheet, or :class:`SheetError` naming every invalid cell. A file
    that is not UTF-8 text, or not a readable CSV, is refused as a whole, echoing nothing."""
    try:
        return _read_sheet(path, frame)
    except UnicodeDecodeError:
        raise SheetError(["the file is not UTF-8 text; save it as CSV UTF-8"]) from None
    except csv.Error:
        raise SheetError(["the file is not a readable CSV"]) from None


def _read_sheet(path: Path, frame: Frame) -> tuple[Claim, ...]:
    problems: list[str] = []
    claims: list[Claim] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        header = [h.strip() for h in (reader.fieldnames or [])]
        if not header:
            raise SheetError(["the sheet is empty: no header row"])
        known = [h for h in header if h in COLUMNS]
        if not known:
            # Most likely a data row where the header should be. Print none of it.
            raise SheetError([
                "line 1 is not the header: it holds none of the column names. "
                "It may be a data row; is the header line missing?"
            ])
        missing = [c for c in COLUMNS if c not in header]
        unexpected = len(header) - len(known)
        repeated = sorted({h for h in known if header.count(h) > 1})
        if missing:
            problems.append(f"header: missing column(s) {', '.join(missing)}")
            if len(missing) > len(COLUMNS) // 2:
                problems.append("header: line 1 may not be the header")
        if unexpected:
            problems.append(f"header: {unexpected} unexpected column name(s)")
        if repeated:
            problems.append(f"header: repeated column(s) {', '.join(repeated)}")
        if problems:
            raise SheetError(problems)
        reader.fieldnames = header
        first_line: dict[str, int] = {}
        for raw in reader:
            if None not in raw and not any((v or "").strip() for v in raw.values()):
                continue
            line = reader.line_num
            claim = _parse_row(raw, line, frame, problems)
            if claim is None:
                continue
            if claim.claim_id in first_line:
                first = first_line[claim.claim_id]
                problems.append(f"line {line}: claim_id: duplicates the claim_id on line {first}")
                continue
            first_line[claim.claim_id] = line
            claims.append(claim)
    if not claims and not problems:
        problems.append("the sheet has a header and no claim rows")
    if problems:
        raise SheetError(problems)
    return tuple(claims)


def read_record(path: Path, frame: Frame) -> SealedRecord:
    """The sealed record of prereg section 13, or :class:`SheetError`. Echoes nothing."""
    data = Path(path).read_bytes()
    problems: list[str] = []
    text = data.decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
    head, sep, rest = text.partition("\nclaims:\n")
    lines = head.split("\n")
    if lines[0].strip() != RECORD_TITLE:
        problems.append(f"sealed record: the first line must be {RECORD_TITLE!r}")
    if not sep:
        problems.append("sealed record: no line 'claims:' before the claims table")
    keys: dict[str, str] = {}
    for number, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        key, colon, value = line.partition(":")
        key = key.strip()
        if not colon or key not in RECORD_KEYS:
            problems.append(
                f"sealed record: line {number}: not one of the keys {', '.join(RECORD_KEYS)}"
            )
        elif key in keys:
            problems.append(f"sealed record: line {number}: repeats a key")
        else:
            keys[key] = value.strip()
    missing = [k for k in RECORD_KEYS if k not in keys]
    if missing:
        problems.append(f"sealed record: missing {', '.join(missing)}")
    salt = keys.get("salt", "")
    if "salt" in keys and not re.fullmatch(r"[0-9a-f]{64}", salt):
        problems.append("sealed record: salt must be 64 lowercase hexadecimal digits")
    if "paper_underpowered_bar" in keys and not keys["paper_underpowered_bar"]:
        problems.append("sealed record: paper_underpowered_bar is blank")
    expected = re.fullmatch(r"W=(\d+) j=(\d+)", keys.get("expected", ""))
    if "expected" in keys and not (
        expected and int(expected[2]) <= int(expected[1]) <= frame.expected_rows
    ):
        problems.append(f"sealed record: expected must be W=<w> j=<j>, 0 <= j <= W <= "
                        f"{frame.expected_rows}")
    claims: dict[str, tuple[tuple[str, str, str, str], str]] = {}
    if sep:
        reader = csv.DictReader(io.StringIO(rest))
        if tuple(h.strip() for h in (reader.fieldnames or [])) != RECORD_COLUMNS:
            problems.append(f"sealed record: the claims header must be {','.join(RECORD_COLUMNS)}")
        else:
            reader.fieldnames = list(RECORD_COLUMNS)
            for raw in reader:
                number = reader.line_num
                if None in raw or any(raw.get(k) is None for k in RECORD_COLUMNS):
                    problems.append(f"sealed record: claims line {number}: not the header's "
                                    "number of cells")
                    continue
                cell = {k: raw[k].strip() for k in RECORD_COLUMNS}
                if not all(cell[k] for k in IDENTIFICATION):
                    problems.append(f"sealed record: claims line {number}: a blank "
                                    "identification cell")
                elif cell["paper_20"].lower() not in PAPER_20:
                    problems.append(f"sealed record: claims line {number}: paper_20 must be "
                                    f"one of {', '.join(PAPER_20)}")
                elif cell["claim_id"] in claims:
                    problems.append(f"sealed record: claims line {number}: repeats a claim_id")
                else:
                    ident = (cell["claim_id"], cell["benchmark"].casefold(), cell["criterion"],
                             cell["source_location"])
                    claims[cell["claim_id"]] = (ident, cell["paper_20"].lower())
            if len(claims) != frame.expected_rows and not problems:
                problems.append(f"sealed record: {len(claims)} claim(s); the paper counts "
                                f"{frame.expected_rows}")
            elif not problems:
                # Its benchmarks must be the frame's, as the sheet's must be (prereg 13).
                names = [ident[1] for ident, _ in claims.values()]
                problems += [f"sealed record: {p}" for p in reconcile_names(names, frame)]
    if problems:
        raise SheetError(problems)
    return SealedRecord(
        sha256=hashlib.sha256(data).hexdigest(),
        salt=salt,
        expected_w=int(expected[1]),
        expected_j=int(expected[2]),
        claims=claims,
    )


def record_digest_in(text: str) -> str | None:
    """The sealed record's digest as prereg section 13 records it, or None."""
    found = RECORD_DIGEST_LINE.findall(text)
    return found[0] if len(found) == 1 else None


def registered_record_digest() -> str | None:
    return record_digest_in((REPO / PREREG).read_text(encoding="utf-8"))


def tracked_path_refusal(path: Path) -> str | None:
    """Why an input at ``path`` must not be read, or None when it may.

    A filled sheet names other groups' benchmarks, and the sealed record and the coders'
    own sheets hold the same rows, so inside this checkout they may sit only under
    audit/private/, which .gitignore excludes. The template is refused too: it is a
    tracked file, and filled in place it would be one commit from publication.
    """
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(REPO) or resolved.is_relative_to(PRIVATE_DIR.resolve()):
        return None
    return (
        "refusing to read an input from a path inside this checkout that git would track. "
        "Every filled row names a third-party benchmark, so the sheet, the sealed record and "
        "the coders' own sheets live outside the repository or under "
        f"{PRIVATE_DIR.relative_to(REPO).as_posix()}/, which .gitignore excludes; copy the "
        f"template there before filling it ({PREREG}, sections 10 and 14)."
    )


# --- the pre-registered computation --------------------------------------------------


def floor_p(n_a: int, n_b: int, one_sided: bool) -> float:
    """The smallest p an exact Mann-Whitney test can return at this size and sidedness."""
    p = min_two_sided_p(n_a, n_b)
    return p / 2.0 if one_sided else p


def power_grid(n_a: int, n_b: int, alpha: float, one_sided: bool) -> dict[float, float]:
    """Power at each AUROC in the grid (prereg section 5).

    A one-sided test at alpha shares its rejection threshold with a two-sided test at
    2 * alpha; the opposite tail that adds is small and errs toward 'adequately powered'.
    ``difference_power`` is not symmetric in its arms, so each figure is the larger of the
    two orders, which errs the same way and makes the sheet's order of the arms irrelevant.
    """
    level = 2.0 * alpha if one_sided else alpha
    return {
        theta: max(difference_power(n_a, n_b, theta, alpha=level),
                   difference_power(n_b, n_a, theta, alpha=level))
        for theta in GRID
    }


def binary_power(n_a: int, n_b: int, alpha: float, one_sided: bool) -> float:
    """Analysis (e): a two-sided two-proportion z-test at 0.3 against 0.7, the binary pair
    with AUROC 0.7 and, for equal arms, the lowest power (prereg section 5, step 6)."""
    level = 2.0 * alpha if one_sided else alpha
    p_lo, p_hi = 0.3, 0.7
    pooled = (p_lo * n_a + p_hi * n_b) / (n_a + n_b)
    s0 = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b))
    s1 = math.sqrt(p_lo * (1.0 - p_lo) / n_a + p_hi * (1.0 - p_hi) / n_b)
    z = _NORMAL.inv_cdf(1.0 - level / 2.0)
    gap = p_hi - p_lo
    return _NORMAL.cdf((gap - z * s0) / s1) + _NORMAL.cdf((-gap - z * s0) / s1)


def decide(min_p: float | None, alpha: float, power_primary: float) -> str:
    """The per-claim rule. ``min_p`` is None where rule 3 does not apply."""
    if min_p is not None and min_p > alpha:
        return CANNOT_REJECT
    if power_primary < TARGET_POWER:
        return UNDERPOWERED
    return ADEQUATE


def efficiency_adjusted(n: int) -> int:
    """The n at which the Mann-Whitney calculation matches a t-test at ``n`` on normal
    data: the Mann-Whitney test's efficiency relative to the t-test there is 3/pi."""
    return math.ceil(n * math.pi / 3.0)


def test_power(test: DiffTest) -> float | None:
    """Power at the primary AUROC as section 5 computes it, or None where rule 2 leaves the
    test not computed."""
    if test.test_used in UNCOMPUTED_TESTS or test.n_a is None or test.n_b is None:
        return None
    if test.test_used in MEAN_TESTS and floor_p(test.n_a, test.n_b, test.one_sided) > test.alpha:
        return None
    return power_grid(test.n_a, test.n_b, test.alpha, test.one_sided)[PRIMARY_AUROC]


def rank_key(test: DiffTest) -> tuple[bool, float, float]:
    """The order prereg section 2 codes tests by: a computed power first, then the higher
    power, then the smaller minimum p. Tests with no computed power are not ordered among
    themselves, and the source's order breaks what is left."""
    power = test_power(test)
    if power is None:
        return (False, 0.0, 0.0)
    return (True, power, -floor_p(test.n_a, test.n_b, test.one_sided))


def _cannot_reject(test: DiffTest) -> bool:
    return (
        test.test_used in RANK_TESTS
        and test.n_a is not None
        and floor_p(test.n_a, test.n_b, test.one_sided) > test.alpha
    )


def _rule3_ready(claim: Claim) -> bool:
    """Rule 3's conditions on the row: an exact test shown, and no level or sidedness that
    rests on a default."""
    return claim.exact_test == "yes" and not claim.defaults & {"alpha", "sidedness"}


def rule3_blocker(claim: Claim) -> str:
    """Why rule 3 does not apply to a rank row whose floor is above alpha; '' when it does."""
    if claim.exact_test != "yes":
        return "exact"
    if claim.defaults & {"alpha", "sidedness"}:
        return "default"
    if not all(_cannot_reject(t) for t in claim.other_tests):
        return "other_tests"
    return ""


def chance_term(test: DiffTest, ready: bool) -> float:
    """p_i in analysis (d): the test's power, or its alpha where that is larger or where no
    power is computed; 0 only for a rank test that cannot reject under rule 3's conditions."""
    if _cannot_reject(test):
        return 0.0 if ready else test.alpha
    power = test_power(test)
    return test.alpha if power is None else max(power, test.alpha)


def assess(claim: Claim) -> Assessment:
    if not claim.sub_class_a:
        return Assessment(claim, NOT_MEMBER)
    if claim.test_used in UNCOMPUTED_TESTS:
        return Assessment(claim, NOT_COMPUTED, reason="test")
    if claim.n_a is None or claim.n_b is None:
        return Assessment(claim, NOT_COMPUTED, reason="n")
    floor = floor_p(claim.n_a, claim.n_b, claim.one_sided)
    if claim.test_used in MEAN_TESTS and floor > claim.alpha:
        # The exact rank floor does not bind a mean-difference test, so where it is above
        # alpha the Mann-Whitney stand-in breaks down (prereg section 4.2, item 3).
        return Assessment(claim, NOT_COMPUTED, reason="floor")
    power = power_grid(claim.n_a, claim.n_b, claim.alpha, claim.one_sided)
    blocked, adjusted, min_p = "", None, None
    if claim.test_used in RANK_TESTS:
        min_p = floor
        blocked = rule3_blocker(claim) if floor > claim.alpha else ""
        verdict = decide(None if blocked else floor, claim.alpha, power[PRIMARY_AUROC])
    else:
        verdict = decide(None, claim.alpha, power[PRIMARY_AUROC])
        adj = power_grid(efficiency_adjusted(claim.n_a), efficiency_adjusted(claim.n_b),
                         claim.alpha, claim.one_sided)[PRIMARY_AUROC]
        adjusted = decide(None, claim.alpha, adj)
    combined = None
    if claim.other_tests:
        ready = _rule3_ready(claim)
        combined = 1.0 - math.prod(1.0 - chance_term(t, ready)
                                   for t in (claim.coded, *claim.other_tests))
    binary = None
    if claim.scores in ("binary", "unknown") and verdict != CANNOT_REJECT:
        binary = binary_power(claim.n_a, claim.n_b, claim.alpha, claim.one_sided)
    one_sided = None
    if "sidedness" in claim.defaults:
        one_sided = power_grid(claim.n_a, claim.n_b, claim.alpha, True)[PRIMARY_AUROC]
    return Assessment(
        claim, verdict, min_p=min_p, power=power, adjusted_verdict=adjusted, blocked=blocked,
        combined=combined, binary_power=binary, one_sided_power=one_sided,
    )


def outcome(assessments) -> str:
    """The overall outcome over the confirmed rows (prereg section 7.2).

    The outcome that weakens the paper is checked first, so that it stands whatever the
    coverage; the coverage rule can only make a strengthening or split result inconclusive.
    """
    confirmed = [a for a in assessments if a.verdict != NOT_MEMBER]
    computed = [a for a in confirmed if a.verdict in COMPUTED]
    weak = sum(a.verdict in WEAK for a in computed)
    adequate = len(computed) - weak
    if adequate > weak:
        return MOSTLY_ADEQUATE
    if 2 * len(computed) <= len(confirmed):
        return INCONCLUSIVE
    if weak > adequate:
        return MOSTLY_WEAK
    return SPLIT


def _toward_adequate(a: Assessment, power: float | None) -> Assessment:
    """A row moves only toward adequately powered, and only where it was computed."""
    if a.verdict in COMPUTED and power is not None and power >= TARGET_POWER:
        return Assessment(a.claim, ADEQUATE, power=a.power)
    return a


def sensitivity_outcomes(assessments) -> dict[str, str]:
    """Analyses (b) to (f) of prereg section 7.3, by name."""
    adjusted = [
        Assessment(a.claim, ADEQUATE) if a.adjusted_verdict == ADEQUATE else a
        for a in assessments
    ]
    return {
        "clustered or unknown rows dropped": outcome(
            [a for a in assessments if a.claim.clustered == "no"]
        ),
        "t-test family at the efficiency-adjusted n": outcome(adjusted),
        "claims resting on several tests at their combined chance to fail": outcome(
            [_toward_adequate(a, a.combined) for a in assessments]
        ),
        "binary or unknown scores at the two-proportion power": outcome(
            [_toward_adequate(a, a.binary_power) for a in assessments]
        ),
        "a defaulted sidedness taken as one-sided": outcome(
            [_toward_adequate(a, a.one_sided_power) for a in assessments]
        ),
    }


def reported_outcome(primary: str, sensitivities: dict[str, str]) -> str:
    """A sensitivity analysis can withdraw an outcome that strengthens the paper, never
    one that weakens it."""
    if primary == MOSTLY_WEAK and any(s != MOSTLY_WEAK for s in sensitivities.values()):
        return NOT_ROBUST
    return primary


def _over(confirmed: int, benchmarks: int, frame: Frame) -> str:
    return (f"{confirmed} of {frame.expected_rows} confirmed, over {benchmarks} of "
            f"{frame.expected_benchmarks} benchmarks")


def membership_correction(confirmed: int, benchmarks: int, frame: Frame) -> str | None:
    """The prereg 7.1 line, when fewer rows are confirmed than the paper counts."""
    if confirmed >= frame.expected_rows:
        return None
    return (
        f"membership correction: {_over(confirmed, benchmarks, frame)}; "
        "the sub-class A count is wrong (prereg 7.1; weakens the paper)"
    )


def reportable(blockers: list[str], confirmed: int, benchmarks: int, frame: Frame,
               primary: str, sensitivities: dict[str, str]) -> str:
    """What the last line may say (prereg section 7). A run on inputs that are not the
    registered ones has no reportable outcome; a membership correction comes first; and a
    strengthening outcome stands only if every sensitivity analysis returns it."""
    if blockers:
        return "none: " + "; ".join(blockers)
    power = reported_outcome(primary, sensitivities)
    if membership_correction(confirmed, benchmarks, frame):
        return (
            f"membership correction, {_over(confirmed, benchmarks, frame)} "
            f"(prereg 7.1; weakens the paper); then, over the {confirmed} confirmed: {power}"
        )
    return power


def reconcile(claims, frame: Frame) -> list[str]:
    """Every way the sheet fails to be the paper's sub-class A. Names no benchmark."""
    return reconcile_names([c.benchmark for c in claims], frame)


def reconcile_names(names, frame: Frame) -> list[str]:
    """The same, for a list of benchmark names, one per row: the sheet's or the sealed
    record's (prereg sections 7.1 and 13)."""
    problems = []
    per_benchmark = Counter(name.casefold() for name in names)
    if len(names) != frame.expected_rows:
        problems.append(f"{len(names)} row(s); Appendix F counts {frame.expected_rows} claims")
    if len(per_benchmark) != frame.expected_benchmarks:
        problems.append(
            f"{len(per_benchmark)} benchmark(s); the paper's section 5 counts "
            f"{frame.expected_benchmarks}"
        )
    outside = sum(1 for b in per_benchmark if b not in frame.audited)
    if outside:
        problems.append(f"{outside} benchmark name(s) not in the frame")
    over = sum(1 for b, k in per_benchmark.items() if b in frame.audited and k > frame.audited[b])
    if over:
        problems.append(f"{over} benchmark(s) with more rows than Appendix F enumerates for them")
    return problems


def power_source_digest() -> str:
    return lf_sha256(Path(cf_power.__file__))


def _coded_values(claim: Claim) -> tuple:
    """The coded columns as compared (prereg section 9), as the script reads them:
    ``defaults_applied`` and ``other_tests`` as unordered lists, numbers as numbers, text
    ignoring case and surrounding spaces, and the two arms in the order written."""
    return (
        claim.sub_class_a, claim.test_used, claim.n_a, claim.n_b, claim.unit.casefold(),
        claim.alpha, claim.one_sided, claim.defaults, claim.exact_test, claim.scores,
        claim.clustered, Counter(claim.other_tests),
    )


def check_coders(claims, first, second, salt: str | None, paths,
                 after_responses: bool = False) -> CoderCheck:
    """Compare the coders' own sheets with each other and with the reconciled sheet
    (prereg section 9). A row matches when ``disagreed_fields`` names exactly the coded
    columns the own sheets hold differently, and every other coded column holds the value
    both own sheets hold. The second part is skipped after author responses, whose copy
    changes values by design (prereg section 7.4)."""
    not_shared, mismatched = coder_counts(claims, first, second, after_responses)
    return CoderCheck(
        first_salted=salted_sha256(salt, paths[0]) if salt else None,
        second_salted=salted_sha256(salt, paths[1]) if salt else None,
        rows_not_shared=not_shared,
        rows_mismatched=mismatched,
        second=tuple(assess(c) for c in second),
    )


def coder_counts(claims, first, second, after_responses: bool = False) -> tuple[int, int]:
    """Rows not shared between the sheet and the own sheets, and rows that do not match
    them (prereg section 9). Counts only; it assesses nothing, so ``--check`` uses it."""
    sheet = {c.claim_id: c for c in claims}
    a = {c.claim_id: c for c in first}
    b = {c.claim_id: c for c in second}
    idents = {c.identification for c in claims}
    not_shared = sum(
        1 for own in (first, second) for c in own if c.identification not in idents
    ) + sum(1 for c in claims if c.claim_id not in a or c.claim_id not in b)
    mismatched = 0
    for claim_id, claim in sheet.items():
        if claim_id not in a or claim_id not in b:
            continue
        va, vb = _coded_values(a[claim_id]), _coded_values(b[claim_id])
        differ = {name for name, x, y in zip(CODED_FIELDS, va, vb) if x != y}
        changed = {
            name for name, x, r in zip(CODED_FIELDS, va, _coded_values(claim))
            if name not in differ and x != r
        }
        if differ != set(claim.disagreed) or (changed and not after_responses):
            mismatched += 1
    return not_shared, mismatched


# --- output --------------------------------------------------------------------------


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "n/a"
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def _table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(r, widths)).rstrip() for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _row_note(a: Assessment) -> str:
    notes = []
    if a.reason:
        notes.append(NOT_COMPUTED_REASONS[a.reason])
    if a.blocked:
        notes.append(f"exact floor above alpha, but {RULE3_BLOCKERS[a.blocked]}")
    if a.adjusted_verdict:
        notes.append(f"t-test stand-in; at adjusted n: {a.adjusted_verdict}")
    if a.combined is not None:
        notes.append(f"{1 + len(a.claim.other_tests)} tests; combined {a.combined:.3f}")
    if a.binary_power is not None:
        notes.append(f"two-proportion {a.binary_power:.3f}")
    if a.one_sided_power is not None:
        notes.append(f"one-sided {a.one_sided_power:.3f}")
    return "; ".join(notes)


def per_row_table(assessments) -> str:
    rows = [
        ["claim", "benchmark", "test", "n a/b", "sided", "alpha", "exact", "scores",
         "clustered", "min p", "pw 0.6", "pw 0.7", "pw 0.8", "verdict", "note"]
    ]
    for a in assessments:
        c = a.claim
        n = f"{c.n_a}/{c.n_b}" if c.n_a is not None else "-"
        pw = [f"{a.power[t]:.3f}" for t in GRID] if a.power else ["-"] * len(GRID)
        min_p = _fmt_p(a.min_p) if a.verdict in COMPUTED else "-"
        rows.append([
            c.claim_id, c.benchmark, c.test_used, n, "one" if c.one_sided else "two",
            f"{c.alpha:g}", c.exact_test, c.scores, c.clustered, min_p, *pw, a.verdict,
            _row_note(a),
        ])
    return _table(rows)


def _split(assessments) -> str:
    v = Counter(a.verdict for a in assessments)
    return (f"cannot reject {v[CANNOT_REJECT]}, underpowered {v[UNDERPOWERED]}, adequately "
            f"powered {v[ADEQUATE]}, not computed {v[NOT_COMPUTED]}")


def _count(assessments, rank: bool) -> int:
    kinds = RANK_TESTS if rank else MEAN_TESTS
    return sum(1 for a in assessments if a.claim.test_used in kinds)


def _differs_from_paper(a: Assessment, paper: str) -> bool | None:
    if a.verdict not in COMPUTED or paper not in ("underpowered", "not_underpowered"):
        return None
    return (paper == "underpowered") != (a.verdict in WEAK)


def aggregate_lines(assessments, ctx: Context) -> list[str]:
    """Counts only. Nothing here may name a benchmark, a criterion, a source or a claim."""
    frame = ctx.frame
    claims = [a.claim for a in assessments]
    by_verdict = Counter(a.verdict for a in assessments)
    confirmed = [a for a in assessments if a.verdict != NOT_MEMBER]
    computed = [a for a in confirmed if a.verdict in COMPUTED]
    weak = [a for a in computed if a.verdict in WEAK]
    adequate = [a for a in computed if a.verdict == ADEQUATE]
    benchmarks = len({a.claim.benchmark.casefold() for a in confirmed})
    problems = reconcile(claims, frame)
    digest = power_source_digest()
    digest_ok = digest == REGISTERED_POWER_SHA256
    frame_ok = frame.sha256 == REGISTERED_FRAME_SHA256

    blockers = []
    if not digest_ok:
        blockers.append("power.py is not the registered source (prereg 5)")
    if not frame_ok:
        blockers.append("the frame is not the registered transcription (prereg 5)")
    if problems:
        blockers.append("the sheet is not the paper's sub-class A (prereg 7.1)")

    out = [
        "Axis A power re-run: aggregate counts. These lines name no benchmark, criterion, "
        "source, coder or claim.",
        "Private until every group's outreach window has closed: until then it stays with the "
        f"two coders ({PREREG}, section 10).",
        f"rules: {PREREG}",
        f"cannotfail {cannotfail.__version__}; power.py sha256 {digest} "
        f"({'matches' if digest_ok else 'DIFFERS FROM'} the registered source)",
        f"frame sha256 {frame.sha256} "
        f"({'matches' if frame_ok else 'DIFFERS FROM'} the registered transcription)",
    ]
    record = ctx.record
    if record is None:
        blockers.append("no sealed record was given (prereg 13)")
        out.append("sealed record: not given, so the rows are not checked against it")
    else:
        registered = ctx.registered_record
        if registered is None:
            blockers.append("section 13 records no sealed-record digest yet (prereg 13)")
            state = "section 13 records no digest yet"
        elif registered != record.sha256:
            blockers.append("the sealed record is not the one section 13 records (prereg 13)")
            state = "DIFFERS FROM the digest section 13 records"
        else:
            state = "matches the digest section 13 records"
        out.append(f"sealed record sha256 {record.sha256} ({state})")
        sealed = {ident for ident, _ in record.claims.values()}
        outside = sum(1 for c in claims if c.identification not in sealed)
        if outside:
            blockers.append("the sheet's rows are not the sealed record's (prereg 13)")
        out.append(
            "rows whose identification columns are not in the sealed record: "
            f"{outside} of {len(claims)}"
        )
        if ctx.sheet_salted:
            out.append(f"this sheet, salted sha256 {ctx.sheet_salted}")
    coders = ctx.coders
    if coders is None:
        blockers.append("the coders' own sheets were not given (prereg 9)")
        out.append("coders' own sheets: not given")
    else:
        if coders.rows_not_shared:
            blockers.append("the coders' own sheets do not hold this sheet's rows (prereg 9)")
        if coders.rows_mismatched:
            blockers.append("this sheet does not match the coders' own sheets (prereg 9)")
        out.append(
            "coders' own sheets: rows not shared with this sheet "
            f"{coders.rows_not_shared}; rows that do not match them (disagreed_fields, or a "
            f"value both own sheets hold) {coders.rows_mismatched}"
        )
        if coders.first_salted:
            out.append(f"first coder's sheet, salted sha256 {coders.first_salted}")
            out.append(f"second coder's sheet, salted sha256 {coders.second_salted}")
    sheet_benchmarks = len({c.benchmark.casefold() for c in claims})
    out.append(f"rows: {len(claims)} over {sheet_benchmarks} benchmarks")
    if problems:
        out.append(
            "reconciles with the paper's counts: NO. This is not the paper's sub-class A, and "
            "nothing below may be reported as the power re-run."
        )
        out += [f"  - {p}" for p in problems]
    else:
        out.append(
            f"reconciles with the paper's counts: yes ({frame.expected_rows} claims, Appendix F; "
            f"over {frame.expected_benchmarks} benchmarks, section 5)"
        )
        # Reported before any power figure (prereg section 7.1).
        membership = membership_correction(len(confirmed), benchmarks, frame)
        if membership:
            out.append(membership)

    disagreed_rows = sum(1 for c in claims if c.disagreed)
    per_field = Counter(f for c in claims for f in c.disagreed)
    field_text = ", ".join(f"{f} {k}" for f, k in sorted(per_field.items())) or "none"
    defaults = Counter(d for c in claims for d in c.defaults)
    out += [
        f"rows with a coder disagreement: {disagreed_rows} of {len(claims)} "
        f"(per field: {field_text})",
        "rows resting on a rule for a silent or conflicting source: "
        + ", ".join(f"{d} {defaults[d]}" for d in DEFAULTS),
        f"{NOT_MEMBER}: {by_verdict[NOT_MEMBER]}",
        f"confirmed sub-class A: {len(confirmed)}",
        f"  {CANNOT_REJECT}: {by_verdict[CANNOT_REJECT]}",
        f"  {UNDERPOWERED} (power below {TARGET_POWER} at AUROC {PRIMARY_AUROC}): "
        f"{by_verdict[UNDERPOWERED]}",
        f"  {ADEQUATE}: {by_verdict[ADEQUATE]}",
        f"  {NOT_COMPUTED}: {by_verdict[NOT_COMPUTED]}",
    ]
    reasons = Counter(a.reason for a in confirmed if a.verdict == NOT_COMPUTED)
    out += [
        f"    {NOT_COMPUTED_REASONS[r]}: {reasons[r]}" for r in NOT_COMPUTED_REASONS if reasons[r]
    ]
    blocked = Counter(a.blocked for a in computed if a.blocked)
    out.append(
        "underpowered rows whose exact floor is above alpha, where rule 3 did not apply: "
        f"{sum(blocked.values())} ("
        + ", ".join(f"{RULE3_BLOCKERS[b]} {blocked[b]}" for b in RULE3_BLOCKERS) + ")"
    )
    below = {t: sum(1 for a in computed if a.power[t] < TARGET_POWER) for t in GRID}
    scores = Counter(a.claim.scores for a in computed)
    out += [
        "computed rows with power below "
        f"{TARGET_POWER}: " + ", ".join(f"{below[t]} at AUROC {t}" for t in GRID)
        + f", of {len(computed)}",
        f"computed rows by test: rank-based {_count(computed, rank=True)}, "
        f"t-test stand-in {_count(computed, rank=False)}",
        "computed rows by scores: " + ", ".join(f"{s} {scores[s]}" for s in SCORES),
        f"computed rows whose units are clustered or unknown: "
        f"{sum(a.claim.clustered != 'no' for a in computed)} of {len(computed)}",
        f"adequately powered rows: clustered or unknown "
        f"{sum(a.claim.clustered != 'no' for a in adequate)}, "
        f"t-test stand-in {_count(adequate, rank=False)}, of {len(adequate)}",
        "t-test stand-in rows whose verdict changes at the efficiency-adjusted n: "
        f"{sum(1 for a in computed if a.adjusted_verdict and a.adjusted_verdict != a.verdict)}",
        "computed rows resting on several tests: "
        f"{sum(1 for a in computed if a.claim.other_tests)}",
        f"benchmarks with at least one claim that could not reliably fail: "
        f"{len({a.claim.benchmark.casefold() for a in weak})} of {benchmarks}",
    ]
    if record is not None:
        paper = {cid: p for cid, (_, p) in record.claims.items()}
        among = [a for a in confirmed if paper.get(a.claim.claim_id) in
                 ("underpowered", "not_underpowered")]
        unsure = [a for a in confirmed if paper.get(a.claim.claim_id) == "unknown"]
        rest = [a for a in confirmed if paper.get(a.claim.claim_id) == "no"]
        differs = [d for d in (_differs_from_paper(a, paper[a.claim.claim_id]) for a in among)
                   if d is not None]
        out += [
            f"confirmed claims among the paper's 20 designs (sealed record): {len(among)}; "
            f"of those, the paper found underpowered: "
            f"{sum(1 for a in among if paper[a.claim.claim_id] == 'underpowered')}",
            f"  among the paper's 20: {_split(among)}",
            f"  possibly among them (paper_20 unknown): {_split(unsure)}",
            f"  not among them (paper_20 no): {_split(rest)}",
            "  cannot reject at any data among the paper's 20, each against its '0 met the "
            f"CANNOT_FAIL bar' (prereg 7.2): {sum(1 for a in among if a.verdict == CANNOT_REJECT)}",
            f"  verdicts among the paper's 20 that differ from the paper's: {sum(differs)} of "
            f"{len(differs)} compared",
            f"the author's expected counts (sealed record): W {record.expected_w}, "
            f"j {record.expected_j}; found: W {len(weak)}, j {by_verdict[CANNOT_REJECT]}",
        ]
    if coders is not None:
        second_confirmed = [a for a in coders.second if a.verdict != NOT_MEMBER]
        out.append(
            f"on the second coder's own sheet: confirmed {len(second_confirmed)}; "
            f"outcome, all rows: {outcome(coders.second)}"
        )
    primary = outcome(assessments)
    sens = sensitivity_outcomes(assessments)
    out.append(f"outcome, (a) all rows: {primary}")
    out += [f"outcome, ({letter}) {name}: {sens[name]}" for letter, name in SENSITIVITIES]
    final = reportable(blockers, len(confirmed), benchmarks, frame, primary, sens)
    if ctx.after_responses:
        out.append(
            "outcome after author responses, reported beside the registered outcome; it "
            f"leads only if it weakens the paper (prereg 7.4): {final}"
        )
    else:
        out.append(f"outcome as it may be reported: {final}")
    return out


def _read_or_exit(reader, label: str):
    try:
        return reader()
    except SheetError as err:
        print(f"invalid {label}, nothing computed:", file=sys.stderr)
        for problem in err.problems:
            print(f"  {problem}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "sheet",
        nargs="?",
        help="a CSV in the format of audit/axis_a_sheet_TEMPLATE.csv (optional only for "
        "--check --sealed-record, which then checks the record alone)",
    )
    parser.add_argument(
        "--sealed-record",
        help="the private sealed record of prereg section 13, whose digest section 13 records",
    )
    parser.add_argument(
        "--coder-sheets",
        nargs=2,
        metavar=("FIRST", "SECOND"),
        help="the two coders' own sheets, coder_1's then coder_2's (prereg section 9)",
    )
    parser.add_argument(
        "--after-responses",
        action="store_true",
        help="label the run as the analysis after author responses (prereg section 7.4)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "print the aggregate counts only. Like the rows, they stay with the two coders "
            "until every outreach window has closed, and are made public after that"
        ),
    )
    mode.add_argument(
        "--benchmark",
        metavar="NAME",
        help="print only this benchmark's rows, for that group's email, and no counts",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help=(
            "validate the inputs and print only 'valid' or the errors, and counts: with "
            "--sealed-record, the rows not in it and its claims the sheet lacks; with "
            "--coder-sheets, the rows not shared with them and the rows that do not match "
            "them. Computes nothing"
        ),
    )
    args = parser.parse_args(argv)
    if args.check and args.after_responses:
        parser.error("--check does not take --after-responses")
    if args.sheet is None and not (args.check and args.sealed_record and not args.coder_sheets):
        parser.error("a sheet is required, except for --check --sealed-record alone")

    labels = [("sheet", args.sheet)] if args.sheet else []
    if args.sealed_record:
        labels.append(("sealed record", args.sealed_record))
    if args.coder_sheets:
        labels += [("first coder's sheet", args.coder_sheets[0]),
                   ("second coder's sheet", args.coder_sheets[1])]
    for label, path in labels:
        refusal = tracked_path_refusal(Path(path))
        if refusal:
            print(refusal, file=sys.stderr)
            return 1
        if not Path(path).is_file():
            print(f"no such file: the {label}", file=sys.stderr)
            return 1
    sheets = [(label, Path(path)) for label, path in labels if label != "sealed record"]
    for i, (label, path) in enumerate(sheets):
        for other, other_path in sheets[i + 1:]:
            if path.samefile(other_path):
                print(f"the {label} and the {other} are the same file; the reconciled sheet "
                      "and each coder's own sheet are three files (prereg section 9)",
                      file=sys.stderr)
                return 1

    frame = load_frame(FRAME)
    if args.sheet is None:
        # --check --sealed-record alone: the record, before its digest is recorded.
        record = _read_or_exit(lambda: read_record(Path(args.sealed_record), frame),
                               "sealed record")
        if record is None:
            return 1
        print(f"sealed record: valid, {len(record.claims)} claims; nothing computed")
        return 0
    sheet = Path(args.sheet)
    claims = _read_or_exit(lambda: read_sheet(sheet, frame), "sheet")
    if claims is None:
        return 1

    if args.check:
        print(f"valid: {len(claims)} rows; nothing computed")
        failed = 0
        if args.sealed_record:
            record = _read_or_exit(lambda: read_record(Path(args.sealed_record), frame),
                                   "sealed record")
            if record is None:
                return 1
            sealed = {ident for ident, _ in record.claims.values()}
            held = {c.identification for c in claims}
            outside = sum(1 for c in claims if c.identification not in sealed)
            absent = sum(1 for ident in sealed if ident not in held)
            print("rows whose identification columns are not in the sealed record: "
                  f"{outside} of {len(claims)}")
            print(f"sealed-record claims this sheet does not hold: {absent} of {len(sealed)}")
            failed += outside + absent
        if args.coder_sheets:
            own = []
            for label, path in labels[-2:]:
                parsed = _read_or_exit(lambda p=path: read_sheet(Path(p), frame), label)
                if parsed is None:
                    return 1
                own.append(parsed)
            not_shared, mismatched = coder_counts(claims, own[0], own[1])
            print(f"coders' own sheets: rows not shared with this sheet {not_shared}; rows that "
                  f"do not match them {mismatched}")
            failed += not_shared + mismatched
        return 1 if failed else 0

    if args.benchmark is not None:
        wanted = args.benchmark.strip().casefold()
        rows = [assess(c) for c in claims if c.benchmark.casefold() == wanted]
        if not rows:
            print("no row of the sheet has that benchmark", file=sys.stderr)
            return 1
        print(
            "PRIVATE: one benchmark's rows, for that group's email only. Send it to no one "
            f"else ({PREREG}, section 10).\n"
        )
        print(per_row_table(rows))
        print("\nEND OF PRIVATE ROWS")
        return 0

    record = None
    if args.sealed_record:
        record = _read_or_exit(lambda: read_record(Path(args.sealed_record), frame),
                               "sealed record")
        if record is None:
            return 1
    coders = None
    if args.coder_sheets:
        own = []
        for label, path in labels[-2:]:
            parsed = _read_or_exit(lambda p=path: read_sheet(Path(p), frame), label)
            if parsed is None:
                return 1
            own.append(parsed)
        coders = check_coders(claims, own[0], own[1], record.salt if record else None,
                              [Path(p) for p in args.coder_sheets], args.after_responses)

    ctx = Context(
        frame=frame,
        record=record,
        registered_record=registered_record_digest() if record else None,
        coders=coders,
        sheet_salted=salted_sha256(record.salt, sheet) if record else None,
        after_responses=args.after_responses,
    )
    assessments = [assess(c) for c in claims]
    if not args.aggregate_only:
        print(
            "PRIVATE: this table names benchmarks. It stays with the two coders until every "
            f"group's window has closed ({PREREG}, section 10).\n"
        )
        print(per_row_table(assessments))
        print("\nEND OF PRIVATE TABLE\n")
    print("\n".join(aggregate_lines(assessments, ctx)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
