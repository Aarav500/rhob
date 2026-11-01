"""``cannotfail``: run D1 on a score file, or D2 on a design, from the command line.

    cannotfail resolution scores.csv --score score --label is_hack --group cell
    cannotfail power equivalence --per-side 4 --layouts 12
    cannotfail power seeds --layouts 12 --tests 4
    cannotfail power difference --pos 5 --neg 5 --true-auroc 0.7

Exit status: 0 on success. 1 only when ``--fail-on`` names a verdict that occurred, or
``--fail-on-cannot-certify`` was given and the design cannot certify. 2 on bad input or a
missing optional dependency. 3 on an internal error, which is a bug worth reporting.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import traceback
from collections.abc import Sequence

from cannotfail import __version__
from cannotfail.power import (
    difference_power,
    equivalence_power,
    mann_whitney_null_sd,
    min_two_sided_p,
    required_seeds_per_layout,
)
from cannotfail.resolution import (
    EQUIVALENCE_MARGIN,
    resolution,
    resolution_by_group,
    summarize,
)

_TRUE = {"1", "true", "t", "yes", "y"}
_FALSE = {"0", "false", "f", "no", "n"}


class InputError(ValueError):
    """Bad input: reported on stderr with exit status 2."""


def _cell(value) -> str:
    """One JSON value as the text a CSV cell would hold. ``null`` becomes blank."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)  # round-trips exactly, unlike str() on some platforms
    return str(value)


def _read_json_columns(path: str, wanted: Sequence[str]) -> dict[str, list[str]]:
    """A ``.json`` list of objects, or ``.jsonl`` with one object per line.

    A list of ``{"id": ..., "score": ...}`` objects is a common leaderboard predictions
    format, so a maintainer can run the check on the file they already have.
    """
    with open(path, encoding="utf-8-sig") as fh:
        if path.endswith(".jsonl"):
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
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise InputError(f"{path}: expected a list of objects, one per scored item")
    if not records:
        raise InputError(f"{path}: the list is empty")
    for c in wanted:
        if not any(c in r for r in records):
            raise InputError(f"{path}: no field {c!r}; fields include {sorted(records[0])[:10]}")
    return {c: [_cell(r.get(c)) for r in records] for c in wanted}


def _read_columns(path: str, delimiter: str | None, wanted: Sequence[str]) -> dict[str, list[str]]:
    """Read the named columns. Lines starting with ``#`` are skipped only above the header.

    A ``#`` inside the data is data: a cell or item id may legitimately start with one,
    and dropping such a row silently would hide a cell from ``--fail-on``. Files ending in
    ``.json`` or ``.jsonl`` are read as records instead.
    """
    if path.endswith((".json", ".jsonl")):
        return _read_json_columns(path, wanted)
    if delimiter is None:
        delimiter = "\t" if path.endswith((".tsv", ".tab")) else ","
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
        raise InputError(f"{path}: no column {missing[0]!r}; columns are {header}")
    cols: dict[str, list[str]] = {c: [] for c in wanted}
    for row in reader:
        for c in wanted:
            cols[c].append((row.get(c) or "").strip())
    if not cols[wanted[0]]:
        raise InputError(f"{path}: the header is there but there are no data rows")
    return cols


def _parse_scores(raw: list[str]) -> list[float]:
    out = []
    for i, v in enumerate(raw, start=1):
        if v == "":
            out.append(float("nan"))
            continue
        try:
            out.append(float(v))
        except ValueError:
            raise InputError(f"data row {i}: score {v!r} is not a number") from None
    return out


def _parse_labels(raw: list[str], positive: str | None) -> list[int]:
    if positive is not None:
        blank = [i for i, v in enumerate(raw, start=1) if v == ""]
        if blank:
            raise InputError(
                f"data row {blank[0]}: blank label ({len(blank)} in all); with --positive, every "
                "row needs a label, or a blank would silently count as negative"
            )
        if positive not in raw:
            seen = sorted(set(raw))[:6]
            raise InputError(f"no row has label {positive!r}; labels seen include {seen}")
        return [1 if v == positive else 0 for v in raw]
    out = []
    for i, v in enumerate(raw, start=1):
        low = v.lower()
        if low in _TRUE:
            out.append(1)
        elif low in _FALSE:
            out.append(0)
        else:
            raise InputError(
                f"data row {i}: label {v!r} is not 0/1 or true/false; pass --positive VALUE "
                "to name the positive class"
            )
    return out


def _clean(obj):
    """Replace NaN and infinities with ``None`` so the output is valid JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _dump(payload) -> None:
    json.dump(_clean(payload), sys.stdout, indent=2, allow_nan=False)
    sys.stdout.write("\n")


def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "-"
    if x != x:
        return "nan"
    return f"{x:.{digits}f}"


def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def _cmd_resolution(args: argparse.Namespace) -> int:
    wanted = [args.score] + ([args.label] if args.label else []) + ([args.group] if args.group else [])
    cols = _read_columns(args.file, args.delimiter, wanted)
    scores = _parse_scores(cols[args.score])
    labels = _parse_labels(cols[args.label], args.positive) if args.label else None
    kw = dict(margin=args.margin, dispersion_floor=args.dispersion_floor)

    if args.group:
        results = resolution_by_group(scores, cols[args.group], labels, **kw)
    else:
        results = {"(all)": resolution(scores, labels, **kw)}
    summary = summarize(results)

    if args.json:
        _dump({
            "file": args.file,
            "margin": args.margin,
            "groups": {str(k): v.as_dict() for k, v in results.items()},
            "summary": {
                "n_groups": summary.n_groups,
                "n_single_valued": summary.n_single_valued,
                "by_verdict": summary.by_verdict,
            },
        })
    else:
        width = max([5] + [len(str(k)) for k in results])
        head = f"{'group':<{width}}  {'n':>7}  {'distinct':>8}"
        if labels is not None:
            head += f"  {'pos':>6}  {'neg':>6}  {'resolution':>10}  {'AUROC range':>15}"
        head += "  verdict"
        print(head)
        for key, r in results.items():
            line = f"{str(key):<{width}}  {r.n:>7}  {r.n_distinct:>8}"
            if labels is not None:
                rng = r.auroc_range
                span = "-" if rng is None else f"[{rng[0]:.3f}, {rng[1]:.3f}]"
                line += f"  {r.n_pos:>6}  {r.n_neg:>6}  {_fmt(r.exact_resolution):>10}  {span:>15}"
            line += f"  {r.verdict}"
            if r.why:
                line += f" ({r.why})"
            print(line)
        print()
        _footer(results, summary, labels is not None, args)

    if args.fail_on:
        hit = [v for v in args.fail_on if summary.by_verdict.get(v, 0)]
        if hit:
            print(f"cannotfail: failing on {', '.join(hit)}", file=sys.stderr)
            return 1
    return 0


def _footer(results, summary, labelled: bool, args: argparse.Namespace) -> None:
    n = summary.n_groups
    groups = _plural(n, "group", "groups")
    by = summary.by_verdict
    print(
        f"{summary.n_single_valued} of {n} {groups} {_plural(summary.n_single_valued, 'has', 'have')} "
        f"at most one distinct value, so any rank statistic over "
        f"{_plural(summary.n_single_valued, 'it', 'them')} is fixed at 0.5."
    )
    other_fixed = by.get("fixed", 0) - summary.n_single_valued
    if other_fixed > 0:
        print(
            f"{other_fixed} more {_plural(other_fixed, 'is', 'are')} fixed for another reason: "
            "no unequal pair across the classes, or a class with no finite score."
        )
    if labelled and by.get("pinned", 0):
        k = by["pinned"]
        print(
            f"{k} of {n} {groups} {_plural(k, 'is', 'are')} pinned: the AUROC cannot get further "
            f"from 0.5 than {args.margin:g}, so its estimate cannot leave an equivalence band of "
            "that width. A test that also needs its interval inside the band can still fail, "
            "but only on the interval's width; a pass there says the scores tie."
        )
    tol_dust = sum(1 for r in results.values() if r.why == "differences only below the tie tolerance")
    if tol_dust:
        print(
            f"{tol_dust} of {n} {groups} {_plural(tol_dust, 'differs', 'differ')} only below the "
            "tie tolerance (1e-9 relative): a standard AUROC ranks those differences, RHOB's gate "
            "does not. Whether they are real or rounding noise is a question about your scorer."
        )
    floor_dust = by.get("dust", 0) - tol_dust
    if floor_dust:
        print(f"{floor_dust} of {n} {groups} spread below the dispersion floor {args.dispersion_floor:g}.")
    nonfinite = sum(r.n_nonfinite for r in results.values())
    if nonfinite:
        print(
            f"{nonfinite} non-finite or blank {_plural(nonfinite, 'score was', 'scores were')} left out "
            "of the AUROC range; a harness that keeps them will compute something else."
        )
    if not labelled:
        print("No --label given, so only distinct values were counted; pass one for resolution.")


def _cmd_equivalence(args: argparse.Namespace) -> int:
    p = equivalence_power(
        args.per_side, args.layouts, margin=args.margin, alpha=args.alpha, true_auroc=args.true_auroc
    )
    if args.json:
        _dump({"se": p.se, "half_width": p.half_width, "margin": p.margin, "power": p.power,
               "can_certify": p.can_certify})
    else:
        print(f"SE of the mean AUROC: {p.se:.4f}  (null SD {mann_whitney_null_sd(args.per_side):.4f} "
              f"over sqrt({args.layouts}) layouts)")
        print(f"Confidence half-width: {_fmt(p.half_width, 4)}  against a margin of {p.margin:g}")
        if not p.can_certify:
            print("This design cannot certify equivalence at any observed mean: the half-width "
                  "alone is at least the margin. A pass reported at this design is not a test result.")
        else:
            print(f"Power to certify at a true AUROC of {args.true_auroc:g}: {p.power:.3f}  "
                  "(a ceiling: it assumes no layout-to-layout heterogeneity)")
    if args.fail_on_cannot_certify and not p.can_certify:
        return 1
    return 0


def _cmd_seeds(args: argparse.Namespace) -> int:
    n = required_seeds_per_layout(
        margin=args.margin, n_layouts=args.layouts, alpha=args.alpha, power=args.power, n_tests=args.tests
    )
    print(json.dumps({"seeds_per_side_per_layout": n}) if args.json else n)
    return 0


def _cmd_difference(args: argparse.Namespace) -> int:
    p = difference_power(args.pos, args.neg, args.true_auroc, alpha=args.alpha)
    floor_p = min_two_sided_p(args.pos, args.neg)
    if args.json:
        _dump({"power": p, "min_two_sided_p": floor_p})
        return 0
    if floor_p > args.alpha:
        print(f"No data can make a two-sided Mann-Whitney test reject at alpha {args.alpha:g} with "
              f"{args.pos} vs {args.neg}: the smallest attainable p-value is {floor_p:.3f}. Power is 0, "
              "and a non-significant result here is no evidence of anything.")
        return 0
    print(f"Power of a two-sided Mann-Whitney test at alpha {args.alpha:g} to detect a true "
          f"AUROC of {args.true_auroc:g} with {args.pos} vs {args.neg}: {p:.3f}")
    if p < 0.8:
        print("Below 0.8: a non-significant result at this design is weak evidence that the "
              "two classes are matched.")
    if min(args.pos, args.neg) < 10:
        print("Fewer than 10 on a side: this is a normal approximation and only indicative.")
    return 0


class _Defaults(argparse.ArgumentDefaultsHelpFormatter):
    """Show a default only when there is one; '(default: None)' on a required flag is noise."""

    def _get_help_string(self, action):
        if action.default is None or action.default is argparse.SUPPRESS or action.required:
            return action.help
        return super()._get_help_string(action)


def build_parser() -> argparse.ArgumentParser:
    fmt = _Defaults
    parser = argparse.ArgumentParser(
        prog="cannotfail", formatter_class=fmt,
        description="Find benchmark checks that cannot fail (D1 resolution, D2 power).",
    )
    parser.add_argument("--version", action="version", version=f"cannotfail {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("resolution", formatter_class=fmt,
                       help="D1: what range could a statistic over these scores take?")
    r.add_argument("file", help="CSV or TSV with a header row (# lines above it are skipped), "
                   "or a .json list of objects / .jsonl, one object per scored item")
    r.add_argument("--score", required=True, help="column holding the scores")
    r.add_argument("--label", help="column holding the class of each row")
    r.add_argument("--positive", help="label value of the positive class; every other value is negative")
    r.add_argument("--group", help="column naming the cell each score belongs to")
    r.add_argument("--margin", type=float, default=EQUIVALENCE_MARGIN,
                   help="equivalence margin the 'pinned' verdict is read against")
    r.add_argument("--dispersion-floor", type=float, default=None,
                   help="also call a group 'dust' when SD/mean(|x|) is below this")
    r.add_argument("--delimiter", default=None, help="field delimiter (default: tab for .tsv, else comma)")
    r.add_argument("--fail-on", nargs="+", choices=["fixed", "pinned", "dust"],
                   help="exit 1 if any group gets one of these verdicts (for CI)")
    r.add_argument("--json", action="store_true", help="machine-readable output")
    r.set_defaults(func=_cmd_resolution)

    p = sub.add_parser("power", formatter_class=fmt, help="D2: could a precondition have failed at this design?")
    psub = p.add_subparsers(dest="power_cmd", required=True)

    e = psub.add_parser("equivalence", formatter_class=fmt,
                        help="can an equivalence test at this design pass, and how often?")
    e.add_argument("--per-side", type=int, required=True,
                   help="items per class in each independent block (a layout, seed, or dataset)")
    e.add_argument("--layouts", type=int, required=True,
                   help="independent blocks the mean AUROC is taken over")
    e.add_argument("--margin", type=float, default=0.10, help="equivalence margin around 0.5")
    e.add_argument("--alpha", type=float, default=0.05, help="one-sided level of each TOST bound")
    e.add_argument("--true-auroc", type=float, default=0.5, help="the AUROC the classes really have")
    e.add_argument("--fail-on-cannot-certify", action="store_true", help="exit 1 if no mean could pass")
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=_cmd_equivalence)

    s = psub.add_parser("seeds", formatter_class=fmt,
                        help="smallest items per class per block for an equivalence test to pass")
    s.add_argument("--layouts", type=int, default=12, help="independent blocks")
    s.add_argument("--margin", type=float, default=0.10, help="equivalence margin around 0.5")
    s.add_argument("--alpha", type=float, default=0.05, help="one-sided level of each TOST bound")
    s.add_argument("--power", type=float, default=0.90, help="target power for all tests together")
    s.add_argument("--tests", type=int, default=1, help="equivalence tests that must all pass")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=_cmd_seeds)

    d = psub.add_parser("difference", formatter_class=fmt,
                        help="power of a difference test to catch a stated violation")
    d.add_argument("--pos", type=int, required=True, help="items in the positive class")
    d.add_argument("--neg", type=int, required=True, help="items in the negative class")
    d.add_argument("--true-auroc", type=float, required=True, help="the smallest violation that matters")
    d.add_argument("--alpha", type=float, default=0.05, help="two-sided level")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=_cmd_difference)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (InputError, ValueError, OSError, ImportError) as exc:
        print(f"cannotfail: {exc}", file=sys.stderr)
        return 2
    except Exception:  # pragma: no cover - a bug, and exit 1 must keep meaning "verdict hit"
        print("cannotfail: internal error, please report it with this traceback:", file=sys.stderr)
        traceback.print_exc()
        return 3


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
