"""``cannotfail``: run D1 on a score file, or D2 on a design, from the command line.

    cannotfail resolution scores.csv --score score --label is_hack --group cell
    cannotfail power equivalence --per-side 4 --layouts 12
    cannotfail power seeds --layouts 12 --tests 4
    cannotfail power difference --pos 5 --neg 5 --true-auroc 0.7

Exit status: 0 on success; 1 when ``--fail-on`` names a verdict that occurred, or when
an equivalence design cannot certify and ``--fail-on-cannot-certify`` was given; 2 on
bad input.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence

from cannotfail import __version__
from cannotfail.power import (
    difference_power,
    equivalence_power,
    mann_whitney_null_sd,
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


def _read_columns(path: str, delimiter: str | None, wanted: Sequence[str]) -> dict[str, list[str]]:
    if delimiter is None:
        delimiter = "\t" if path.endswith((".tsv", ".tab")) else ","
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader((line for line in fh if not line.lstrip().startswith("#")), delimiter=delimiter)
        header = reader.fieldnames or []
        missing = [c for c in wanted if c not in header]
        if missing:
            raise InputError(f"{path}: no column {missing[0]!r}; columns are {header}")
        cols: dict[str, list[str]] = {c: [] for c in wanted}
        for row in reader:
            for c in wanted:
                cols[c].append((row.get(c) or "").strip())
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
            raise InputError(f"row {i}: score {v!r} is not a number") from None
    return out


def _parse_labels(raw: list[str], positive: str | None) -> list[int]:
    if positive is not None:
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
                f"row {i}: label {v!r} is not 0/1 or true/false; pass --positive VALUE "
                "to name the positive class"
            )
    return out


def _fmt(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "-"
    if x != x:
        return "nan"
    return f"{x:.{digits}f}"


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
        payload = {
            "file": args.file,
            "margin": args.margin,
            "groups": {str(k): v.as_dict() for k, v in results.items()},
            "summary": {
                "n_groups": summary.n_groups,
                "n_single_valued": summary.n_single_valued,
                "by_verdict": summary.by_verdict,
            },
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
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
                line += f"  {r.n_pos:>6}  {r.n_neg:>6}  {_fmt(r.resolution):>10}  {span:>15}"
            line += f"  {r.verdict}"
            print(line)
        print()
        n = summary.n_groups
        noun = "group" if n == 1 else "groups"
        print(
            f"{summary.n_single_valued} of {n} {noun} have a score vector with a single "
            "distinct value: any rank statistic over them is fixed."
        )
        pinned = summary.by_verdict.get("pinned", 0)
        if labels is not None and pinned:
            print(
                f"{pinned} of {n} {noun} are pinned: their AUROC cannot leave "
                f"0.5 +/- {args.margin:g}, so an equivalence test at that margin cannot fail there."
            )
        if summary.by_verdict.get("dust", 0):
            print(
                f"{summary.by_verdict['dust']} of {n} {noun} differ only below the dispersion "
                f"floor {args.dispersion_floor:g}."
            )
        if labels is None:
            print("No --label given, so only distinct values were counted; pass one for resolution.")

    if args.fail_on:
        hit = [v for v in args.fail_on if summary.by_verdict.get(v, 0)]
        if hit:
            print(f"cannotfail: failing on {', '.join(hit)}", file=sys.stderr)
            return 1
    return 0


def _cmd_equivalence(args: argparse.Namespace) -> int:
    p = equivalence_power(
        args.per_side, args.layouts, margin=args.margin, alpha=args.alpha, true_auroc=args.true_auroc
    )
    if args.json:
        json.dump(
            {"se": p.se, "half_width": p.half_width, "margin": p.margin, "power": p.power,
             "can_certify": p.can_certify},
            sys.stdout, indent=2,
        )
        sys.stdout.write("\n")
    else:
        print(f"SE of the mean AUROC: {p.se:.4f}  (null SD {mann_whitney_null_sd(args.per_side):.4f} over sqrt({args.layouts}))")
        print(f"Confidence half-width: {p.half_width:.4f}  against a margin of {p.margin:g}")
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
    if args.json:
        print(json.dumps({"power": p}))
    else:
        print(f"Power of a two-sided Mann-Whitney test at alpha {args.alpha:g} to detect a true "
              f"AUROC of {args.true_auroc:g} with {args.pos} vs {args.neg}: {p:.3f}")
        if p < 0.8:
            print("Below 0.8: a non-significant result at this design is weak evidence that the "
                  "two classes are matched.")
        if min(args.pos, args.neg) < 10:
            print("Fewer than 10 on a side: this is a normal approximation and only indicative.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cannotfail",
        description="Find benchmark checks that cannot fail (D1 resolution, D2 power).",
    )
    parser.add_argument("--version", action="version", version=f"cannotfail {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("resolution", help="D1: what range could a statistic over these scores take?")
    r.add_argument("file", help="CSV or TSV with a header row; lines starting with # are skipped")
    r.add_argument("--score", required=True, help="column holding the scores")
    r.add_argument("--label", help="column holding 0/1 or true/false class labels")
    r.add_argument("--positive", help="label value to treat as the positive class (others are negative)")
    r.add_argument("--group", help="column naming the cell each score belongs to")
    r.add_argument("--margin", type=float, default=EQUIVALENCE_MARGIN,
                   help="equivalence margin the 'pinned' verdict is read against (default 0.10)")
    r.add_argument("--dispersion-floor", type=float, default=None,
                   help="flag groups whose SD/mean(|x|) is below this as 'dust' (off by default)")
    r.add_argument("--delimiter", default=None, help="field delimiter (default: tab for .tsv, else comma)")
    r.add_argument("--fail-on", nargs="+", choices=["fixed", "pinned", "dust"],
                   help="exit 1 if any group gets one of these verdicts (for CI)")
    r.add_argument("--json", action="store_true", help="machine-readable output")
    r.set_defaults(func=_cmd_resolution)

    p = sub.add_parser("power", help="D2: could a precondition have failed at this design?")
    psub = p.add_subparsers(dest="power_cmd", required=True)

    e = psub.add_parser("equivalence", help="can an equivalence test at this design pass, and how often?")
    e.add_argument("--per-side", type=int, required=True, help="runs per class per layout")
    e.add_argument("--layouts", type=int, required=True, help="independent layouts (clusters)")
    e.add_argument("--margin", type=float, default=0.10)
    e.add_argument("--alpha", type=float, default=0.05)
    e.add_argument("--true-auroc", type=float, default=0.5)
    e.add_argument("--fail-on-cannot-certify", action="store_true", help="exit 1 if no mean could pass")
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=_cmd_equivalence)

    s = psub.add_parser("seeds", help="smallest runs per side per layout for an equivalence test to pass")
    s.add_argument("--layouts", type=int, default=12)
    s.add_argument("--margin", type=float, default=0.10)
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--power", type=float, default=0.90)
    s.add_argument("--tests", type=int, default=1, help="equivalence tests that must all pass")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=_cmd_seeds)

    d = psub.add_parser("difference", help="power of a difference test to catch a stated violation")
    d.add_argument("--pos", type=int, required=True)
    d.add_argument("--neg", type=int, required=True)
    d.add_argument("--true-auroc", type=float, required=True, help="the smallest violation that matters")
    d.add_argument("--alpha", type=float, default=0.05)
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=_cmd_difference)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (InputError, ValueError, OSError) as exc:
        print(f"cannotfail: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
