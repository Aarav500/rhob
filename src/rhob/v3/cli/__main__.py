"""``python -m rhob <command>`` -- the RHOB v3 CLI. Stdlib argparse only.

Commands:
    families               List registered environment families.
    evaluate               Evaluate a detector and write a submission JSON.
    validate SUBMISSION    Validate a submission JSON's schema.
    submit SUBMISSION      Add a validated submission to the leaderboard.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from rhob.detectors.posthoc import PosthocDetector
from rhob.v3.benchmark import Benchmark
from rhob.v3.leaderboard.board import Leaderboard, LeaderboardEntry

_REQUIRED_SUBMISSION_KEYS = {"detector_name", "access_level", "overall_auroc", "n_cells"}
_DEFAULT_LEADERBOARD_DIR = Path("leaderboard")


def _load_detector(path: str) -> PosthocDetector:
    """Import a user's Python file and instantiate its PosthocDetector subclass."""
    module_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    candidates = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type) and issubclass(obj, PosthocDetector) and obj is not PosthocDetector
    ]
    if not candidates:
        raise ValueError(f"no PosthocDetector subclass found in {path!r}")
    return candidates[0]()


def cmd_families(_args: argparse.Namespace) -> int:
    rows = Benchmark.list_families()
    if not rows:
        print("(no families registered)")
        return 0
    header = f"{'family':<24}{'mechanism':<14}{'complexity':<18}{'difficulty range'}"
    print(header)
    print("-" * len(header))
    for r in rows:
        lo, hi = r["difficulty_range"]
        print(f"{r['family']:<24}{r['mechanism']:<14}{r['complexity']:<18}[{lo:.2f}, {hi:.2f}]")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    detector = _load_detector(args.detector)
    results = Benchmark.evaluate(
        detector,
        families=args.families,
        difficulties=args.difficulties,
        n_seeds=args.n_seeds,
        verbose=not args.quiet,
    )
    results.summary()

    entry = LeaderboardEntry.from_results(results, author=args.author or "anonymous")
    payload = {
        "detector_name": entry.detector_name,
        "access_level": entry.access_level,
        "author": entry.author,
        "timestamp": entry.timestamp,
        "overall_auroc": entry.overall_auroc,
        "n_cells": entry.n_cells,
        "family_auroc": entry.family_auroc,
        "mechanism_auroc": entry.mechanism_auroc,
        "tier_auroc": entry.tier_auroc,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSubmission written to {out_path}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.submission)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"INVALID: could not parse {path}: {exc}")
        return 1
    missing = _REQUIRED_SUBMISSION_KEYS - set(data)
    if missing:
        print(f"INVALID: missing keys {sorted(missing)}")
        return 1
    if data["access_level"] not in ("L0", "L1", "L2", "L3"):
        print(f"INVALID: access_level must be one of L0/L1/L2/L3, got {data['access_level']!r}")
        return 1
    print(f"VALID: {data['detector_name']} ({data['access_level']}), overall AUROC={data['overall_auroc']}")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    if cmd_validate(args) != 0:
        return 1
    data = json.loads(Path(args.submission).read_text())
    if args.name:
        data["detector_name"] = args.name
    if args.author:
        data["author"] = args.author

    board_path = _DEFAULT_LEADERBOARD_DIR / "leaderboard.json"
    board = Leaderboard.load(board_path)
    board.add(LeaderboardEntry(**{k: data[k] for k in LeaderboardEntry.__dataclass_fields__ if k in data}))
    board.save(board_path)

    (_DEFAULT_LEADERBOARD_DIR / "README.md").write_text(
        "# RHOB v3 Leaderboard\n\n" + board.render_standings_md() + "\n"
    )
    (_DEFAULT_LEADERBOARD_DIR / "by_mechanism.md").write_text(board.render_by_mechanism_md() + "\n")
    (_DEFAULT_LEADERBOARD_DIR / "by_difficulty.md").write_text(board.render_by_difficulty_md() + "\n")
    print(f"Added {data['detector_name']} to the leaderboard at {board_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rhob", description="RHOB v3 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("families", help="List registered environment families").set_defaults(func=cmd_families)

    p_eval = sub.add_parser("evaluate", help="Evaluate a detector across the benchmark suite")
    p_eval.add_argument("--detector", required=True, help="Path to a .py file defining a PosthocDetector subclass")
    p_eval.add_argument("--families", default="all")
    p_eval.add_argument("--difficulties", default="all")
    p_eval.add_argument("--n-seeds", type=int, default=20, dest="n_seeds")
    p_eval.add_argument("--output", default="submission.json")
    p_eval.add_argument("--author", default=None)
    p_eval.add_argument("--quiet", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    p_val = sub.add_parser("validate", help="Validate a submission JSON")
    p_val.add_argument("submission")
    p_val.set_defaults(func=cmd_validate)

    p_sub = sub.add_parser("submit", help="Add a submission to the leaderboard")
    p_sub.add_argument("submission")
    p_sub.add_argument("--name", default=None)
    p_sub.add_argument("--author", default=None)
    p_sub.set_defaults(func=cmd_submit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
