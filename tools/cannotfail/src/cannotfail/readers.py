"""Read a maintainer's score file into the columns ``cannotfail resolution`` needs.

Every reader returns the same thing: for each requested column, one string per scored
item, the text a CSV cell would hold, with a missing value as ``""``. Scores and labels are
parsed once, afterwards, so a score means the same thing whichever format it came from.

How a column is named depends on the format:

- **CSV and TSV**: the header's literal name. A dot in it is just a character.
- **JSON, JSONL and Parquet**: a dotted path. ``scores.detector.value`` is the field
  ``value`` inside ``detector`` inside ``scores``, and an integer segment indexes a list,
  so ``filtered_resps.0.0`` is the first item of the first item. A key that itself contains
  a dot is matched whole before the path is split at that dot: in
  ``{"a.b": 1, "a": {"b": 2}}``, ``a.b`` is 1. A path that resolves in some records and
  not others is blank in the others.
- **Inspect logs**: a column of ``inspect_ai.analysis.samples_df``, such as
  ``score_includes`` or ``metadata_family``, read only with ``--format inspect``.

Parquet needs pyarrow (``cannotfail[parquet]``) and Inspect logs need inspect_ai, pandas
and pyarrow (``cannotfail[inspect]``). Both are imported only when such a file is read, so
the base install stays NumPy only.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import warnings
from collections.abc import Sequence

import numpy as np

__all__ = ["FORMATS", "InputError", "detect_format", "read_columns"]

#: The values ``--format`` accepts.
FORMATS = ("csv", "tsv", "json", "jsonl", "parquet", "inspect")

_INDEX = re.compile(r"[0-9]+\Z")
_SHOWN_KEYS = 12
_PARQUET_HINT = 'reading Parquet needs pyarrow: pip install "cannotfail[parquet]"'
_INSPECT_HINT = 'reading Inspect logs needs inspect_ai, pandas and pyarrow: pip install "cannotfail[inspect]"'


class InputError(ValueError):
    """Bad input: reported on stderr with exit status 2."""


def detect_format(path: str) -> str:
    """The format a file's extension implies. Inspect logs are never guessed."""
    low = path.lower()
    if low.endswith(".json"):
        return "json"
    if low.endswith((".jsonl", ".ndjson")):
        return "jsonl"
    if low.endswith((".parquet", ".pq")):
        return "parquet"
    if low.endswith((".tsv", ".tab")):
        return "tsv"
    if low.endswith(".eval"):
        raise InputError(
            f"{path}: an .eval file is an Inspect log; pass --format inspect, which needs "
            'pip install "cannotfail[inspect]"'
        )
    return "csv"


def read_columns(
    path: str,
    wanted: Sequence[str],
    *,
    fmt: str | None = None,
    delimiter: str | None = None,
    score: str | None = None,
) -> dict[str, list[str]]:
    """Read the named columns of ``path``, one string per scored item.

    ``score`` names the score column, which only the Inspect reader treats specially: it
    converts Inspect's letter grades there (see :func:`_read_inspect`).
    """
    wanted = list(dict.fromkeys(wanted))
    fmt = fmt or detect_format(path)
    if fmt in ("csv", "tsv"):
        if delimiter is None:
            delimiter = "\t" if fmt == "tsv" else ","
        return _read_delimited(path, delimiter, wanted)
    if fmt in ("json", "jsonl"):
        return _columns_from_records(path, _read_json_records(path, fmt == "jsonl"), wanted)
    if fmt == "parquet":
        return _read_parquet(path, wanted)
    if fmt == "inspect":
        return _read_inspect(path, wanted, score)
    raise InputError(f"unknown format {fmt!r}; expected one of {', '.join(FORMATS)}")


# --- one value as a cell ---------------------------------------------------------------

def _cell(value) -> str:
    """One value as the text a CSV cell would hold. ``None`` becomes blank."""
    if isinstance(value, np.generic):  # repr(np.float64(0.5)) is not a number
        value = value.item()
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)  # round-trips exactly, unlike str() on some platforms
    return str(value)


# --- CSV and TSV -----------------------------------------------------------------------

def _read_delimited(path: str, delimiter: str, wanted: Sequence[str]) -> dict[str, list[str]]:
    """Read the named columns. Lines starting with ``#`` are skipped only above the header.

    A ``#`` inside the data is data: a cell or item id may legitimately start with one,
    and dropping such a row silently would hide a cell from ``--fail-on``.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:  # -sig: Excel writes a BOM
        text = fh.read()
    lines = text.splitlines(keepends=True)
    start = 0
    while start < len(lines) and (lines[start].lstrip().startswith("#") or not lines[start].strip()):
        start += 1
    reader = csv.DictReader(io.StringIO("".join(lines[start:])), delimiter=delimiter)
    header = reader.fieldnames or []
    if not header:
        raise InputError(f"{path}: no header row")
    missing = [c for c in wanted if c not in header]
    if missing:
        hint = ""
        if "." in missing[0]:
            hint = " (CSV and TSV columns are matched by their literal header, not as a dotted path)"
        raise InputError(f"{path}: no column {missing[0]!r}; columns are {header}{hint}")
    cols: dict[str, list[str]] = {c: [] for c in wanted}
    for row in reader:
        for c in wanted:
            cols[c].append((row.get(c) or "").strip())
    if not cols[wanted[0]]:
        raise InputError(f"{path}: the header is there but there are no data rows")
    return cols


# --- JSON and JSONL --------------------------------------------------------------------

def _read_json_records(path: str, jsonl: bool) -> list[dict]:
    """A ``.json`` list of objects, or ``.jsonl`` with one object per line.

    A list of ``{"id": ..., "score": ...}`` objects is a common leaderboard predictions
    format, and JSONL is what evaluation harnesses log per sample, so a maintainer can run
    the check on the file they already have.
    """
    with open(path, encoding="utf-8-sig") as fh:
        if jsonl:
            records = []
            for i, line in enumerate(fh, start=1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise InputError(f"{path}: line {i} is not JSON: {exc.msg}") from None
        else:
            try:
                records = json.load(fh)
            except json.JSONDecodeError as exc:
                raise InputError(f"{path}: not valid JSON: {exc.msg} at line {exc.lineno}") from None
    if isinstance(records, dict):
        lists = [k for k, v in records.items() if isinstance(v, list) and v and all(isinstance(r, dict) for r in v)]
        hint = ""
        if "eval" in records and "samples" in records:
            hint = "; it looks like an Inspect log, which --format inspect reads (pip install \"cannotfail[inspect]\")"
        elif lists:
            hint = f"; its key {lists[0]!r} holds a list of objects, which you could extract first, e.g. with jq '.[{json.dumps(lists[0])}]'"
        raise InputError(
            f"{path}: expected a list of objects, one per scored item; this file holds one object, "
            f"with keys {_keys(records)}{hint}"
        )
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise InputError(f"{path}: expected a list of objects, one per scored item")
    if not records:
        raise InputError(f"{path}: {'no records' if jsonl else 'the list is empty'}")
    return records


def _resolve(obj, parts: list[str], depth: int = 0):
    """Follow ``parts`` into ``obj``: ``(True, value, None)`` or ``(False, None, failure)``.

    ``failure`` is ``(depth, container, remaining parts)`` at the deepest point any reading
    of the path reached. At each object the longest run of parts that is itself a key is
    tried first, so a key containing a dot is matched whole before it is split.
    """
    if not parts:
        return True, obj, None
    deepest = (depth, obj, parts)
    if isinstance(obj, dict):
        for k in range(len(parts), 0, -1):
            key = ".".join(parts[:k])
            if key in obj:
                ok, value, fail = _resolve(obj[key], parts[k:], depth + k)
                if ok:
                    return True, value, None
                if fail[0] > deepest[0]:
                    deepest = fail
    elif isinstance(obj, (list, tuple)):
        if _INDEX.match(parts[0]) and int(parts[0]) < len(obj):
            ok, value, fail = _resolve(obj[int(parts[0])], parts[1:], depth + 1)
            if ok:
                return True, value, None
            if fail[0] > deepest[0]:
                deepest = fail
    return False, None, deepest


def _keys(obj: dict) -> str:
    keys = sorted(str(k) for k in obj)
    more = len(keys) - _SHOWN_KEYS
    return f"{keys[:_SHOWN_KEYS]}" + (f" and {more} more" if more > 0 else "")


def _not_found(path: str, name: str, parts: list[str], failure, n_records: int) -> str:
    depth, at, rest = failure
    where = f"in any of the {n_records} records" if n_records > 1 else "in the one record"
    if depth == 0:
        return f"{path}: no field {name!r} {where}; the keys at the top level are {_keys(at)}"
    reached = ".".join(parts[:depth])
    if isinstance(at, dict):
        what = f"an object with keys {_keys(at)}"
    elif isinstance(at, (list, tuple)):
        what = f"a list of {len(at)}, so the next part must be an index from 0 to {len(at) - 1}, not {rest[0]!r}"
        if not at:
            what = "an empty list"
    else:
        what = f"the value {at!r}, which has no fields"
    return f"{path}: no field {name!r} {where}. The deepest it reaches is {reached!r}, {what}."


def _holds(name: str, value) -> str:
    if isinstance(value, dict):
        eg = f"; name one of its fields, e.g. {name + '.' + str(next(iter(value)))!r}" if value else ""
        return f"{name!r} holds an object with keys {_keys(value)}, not a value{eg}"
    eg = f"; index it, e.g. {name + '.0'!r}" if value else ""
    return f"{name!r} holds a list of {len(value)}, not a value{eg}"


def _columns_from_records(path: str, records: list[dict], wanted: Sequence[str]) -> dict[str, list[str]]:
    cols: dict[str, list[str]] = {}
    for name in wanted:
        parts = name.split(".")
        cells: list[str] = []
        deepest = None
        found = False
        for i, record in enumerate(records, start=1):
            ok, value, failure = _resolve(record, parts)
            if ok:
                if isinstance(value, (dict, list, tuple)):
                    raise InputError(f"{path}: record {i}: {_holds(name, value)}")
                cells.append(_cell(value))
                found = True
            else:
                cells.append("")
                if deepest is None or failure[0] > deepest[0]:
                    deepest = failure
        if not found:
            raise InputError(_not_found(path, name, parts, deepest, len(records)))
        cols[name] = cells
    return cols


# --- Parquet ---------------------------------------------------------------------------

def _read_parquet(path: str, wanted: Sequence[str]) -> dict[str, list[str]]:
    """Read only the top-level columns the paths start in, then follow them as in JSON.

    A struct column comes back as an object and a list column as a list, so a nested
    field is named by the same dotted path it would have in JSON.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise InputError(f"{path}: {_PARQUET_HINT}") from None
    except Exception as exc:  # installed, but built for another NumPy, say
        raise InputError(f"{path}: pyarrow is installed but failed to import ({type(exc).__name__}: {exc}); "
                         f"{_PARQUET_HINT}") from None
    try:
        names = pq.read_schema(path).names
    except (OSError, ValueError) as exc:  # pyarrow's ArrowInvalid is a ValueError
        raise InputError(f"{path}: not a readable Parquet file: {exc}") from None
    need: set[str] = set()
    for name in wanted:
        tops = [n for n in names if name == n or name.startswith(n + ".")]
        if not tops:
            raise InputError(f"{path}: no column {name!r}; the top-level columns are {names}")
        need.update(tops)
    indices = [i for i, n in enumerate(names) if n in need]
    try:
        if any("." in names[i] for i in indices):
            # Some pyarrow versions read a dotted name in `columns` as a path into a
            # struct. Selecting by position cannot be misread.
            table = pq.read_table(path).select(indices)
        else:
            table = pq.read_table(path, columns=[names[i] for i in indices])
    except (OSError, ValueError) as exc:
        raise InputError(f"{path}: not a readable Parquet file: {exc}") from None
    if table.num_rows == 0:
        raise InputError(f"{path}: the file has columns but no rows")
    return _columns_from_records(path, table.to_pylist(), wanted)


# --- Inspect logs ----------------------------------------------------------------------

def _read_inspect(path: str, wanted: Sequence[str], score: str | None) -> dict[str, list[str]]:
    """An Inspect log, or a directory of them, through ``inspect_ai.analysis.samples_df``.

    One row per sample and epoch, with the task and model columns added so that a
    directory of logs can be grouped by them. In the score column, Inspect's letter grades
    (``C``, ``I``, ``P``, ``N``) and booleans are converted by Inspect's own
    ``value_to_float``, to the 1, 0, 0.5, 0 and 1/0 its built-in metrics average. Nothing
    else is converted: an unrecognised string is reported, not turned into 0 as
    ``value_to_float`` would.
    """
    try:
        import pandas as pd
        import pyarrow  # noqa: F401  (samples_df needs it, and says so less plainly)
        from inspect_ai.analysis import EvalModel, EvalTask, SampleSummary, samples_df
        from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, PARTIAL, value_to_float
    except ImportError as exc:
        raise InputError(f"{path}: {_INSPECT_HINT} ({exc})") from None
    except Exception as exc:  # installed, but built for another NumPy, say
        raise InputError(f"{path}: inspect_ai, pandas or pyarrow is installed but failed to import "
                         f"({type(exc).__name__}: {exc}); {_INSPECT_HINT}") from None
    if "://" not in path and not os.path.exists(path):
        raise InputError(f"{path}: no such file or directory")
    try:
        with warnings.catch_warnings():
            # Deprecations inside inspect_ai's own use of pandas are for its developers,
            # not for someone reading a log, and would land in the middle of the report.
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            df = samples_df(path, columns=EvalTask + EvalModel + SampleSummary, quiet=True)
    except Exception as exc:  # whatever inspect_ai raises on a file it cannot read
        raise InputError(f"{path}: inspect_ai could not read it as an Inspect log: {type(exc).__name__}: {exc}") from None
    if len(df) == 0:
        raise InputError(f"{path}: no samples in the log")
    columns = [str(c) for c in df.columns]
    missing = [c for c in wanted if c not in columns]
    if missing:
        raise InputError(f"{path}: no column {missing[0]!r} in samples_df; its columns are {columns}")

    grades = {CORRECT, INCORRECT, PARTIAL, NOANSWER}
    to_float = value_to_float()

    def plain(v):
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):  # a list or array: not a missing value
            pass
        return v.item() if isinstance(v, np.generic) else v

    cols: dict[str, list[str]] = {}
    for name in wanted:
        values = [plain(v) for v in df[name].tolist()]
        if name == score:
            values = [to_float(v) if isinstance(v, bool) or (isinstance(v, str) and v in grades) else v
                      for v in values]
        cols[name] = [_cell(v) for v in values]
    return cols
