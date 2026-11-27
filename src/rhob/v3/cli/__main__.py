"""``python -m rhob <command>`` -- the RHOB v3 CLI. Stdlib argparse only.

Commands:
    families               List registered environment families.
    evaluate               Evaluate a detector and write a submission JSON.
    validate SUBMISSION    Validate a submission JSON's schema.
    submit SUBMISSION      Add a validated submission to the community board
                           (``leaderboard/community/`` by default; ``--board-dir``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rhob.detectors.posthoc import PosthocDetector
from rhob.v3.benchmark import Benchmark
from rhob.v3.leaderboard.board import Leaderboard, LeaderboardEntry

_REQUIRED_SUBMISSION_KEYS = {"detector_name", "access_level", "overall_auroc", "n_cells"}

#: Where ``submit`` records community submissions, relative to the working directory. Not
#: ``leaderboard/`` itself: that directory holds the benchmark's own boards
#: (``v5_replicated.json``, ``v5_leaderboard.json``, the transfer artifacts) and a
#: hand-written index ``README.md``, none of which ``submit`` may write. Not under
#: ``submissions/`` either: ``.github/workflows/leaderboard_validate.yml`` validates every
#: ``submissions/*.json`` as a submission, and ``board.json`` is not one.
_DEFAULT_LEADERBOARD_DIR = Path("leaderboard") / "community"

#: The first line of every Markdown file ``submit`` renders. ``submit`` only overwrites a
#: file in the board directory that starts with it, so it cannot replace a README it did
#: not write (``--board-dir leaderboard`` would otherwise overwrite the index).
_COMMUNITY_HEADER = "# RHOB community submissions (self-reported, single draw)"

_RENDERED_MD = ("README.md", "by_mechanism.md", "by_difficulty.md")


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


def _render_community_md(board: Leaderboard) -> dict[str, str]:
    """The three Markdown files ``submit`` writes next to ``board.json``, by file name."""
    note = (
        "Self-reported results added with `python -m rhob submit`, each from one draw of\n"
        "the suite run by its submitter. They are not the benchmark's board: that is\n"
        "`leaderboard/v5_replicated.json` (intervals over independent draws). `Cells` is the\n"
        "submission's `n_cells`; `rhob evaluate` writes the number of measured cells the\n"
        "overall AUROC averages, with not-applicable cells left out.\n"
    )
    return {
        "README.md": f"{_COMMUNITY_HEADER}\n\n{note}\n{board.render_standings_md()}\n",
        "by_mechanism.md": (
            f"{_COMMUNITY_HEADER}\n\n## By mechanism\n\n{board.render_by_mechanism_md()}\n"
        ),
        "by_difficulty.md": (
            f"{_COMMUNITY_HEADER}\n\n## By difficulty tier\n\n{board.render_by_difficulty_md()}\n"
        ),
    }


def _foreign_markdown(board_dir: Path) -> list[Path]:
    """Files ``submit`` would overwrite in ``board_dir`` that it did not write itself."""
    foreign = []
    for name in _RENDERED_MD:
        path = board_dir / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="replace") as fh:
            first = fh.readline().rstrip("\r\n")
        if first != _COMMUNITY_HEADER:
            foreign.append(path)
    return foreign


def cmd_submit(args: argparse.Namespace) -> int:
    """Validate a submission and add it to the entries board in ``--board-dir``.

    Every check runs before anything is written, so a refusal (exit 1) leaves the board
    directory exactly as it was. Refused: an invalid submission; a ``board.json`` that is
    not an entries-schema board (one of the benchmark's own boards, say); and a
    Markdown file in the directory that ``submit`` did not render.
    """
    if cmd_validate(args) != 0:
        return 1
    data = json.loads(Path(args.submission).read_text(encoding="utf-8"))
    if args.name:
        data["detector_name"] = args.name
    if args.author:
        data["author"] = args.author
    # The documented minimum is four keys (submissions/README.md); LeaderboardEntry also
    # needs these two, so supply them rather than fail on a valid submission.
    data["author"] = data.get("author") or "anonymous"
    data["timestamp"] = data.get("timestamp") or datetime.now(timezone.utc).isoformat()

    board_dir = Path(args.board_dir)
    board_path = board_dir / "board.json"
    try:
        board = Leaderboard.load(board_path)
    except ValueError as exc:
        print(f"REFUSED: {exc}. Nothing was written; pass --board-dir to use another directory.")
        return 1
    foreign = _foreign_markdown(board_dir)
    if foreign:
        names = ", ".join(str(p) for p in foreign)
        print(
            f"REFUSED: {names} not written by `rhob submit` (first line is not "
            f"{_COMMUNITY_HEADER!r}). Nothing was written; pass --board-dir to use another "
            "directory."
        )
        return 1

    fields = LeaderboardEntry.__dataclass_fields__
    board.add(LeaderboardEntry(**{k: data[k] for k in fields if k in data}))
    board.save(board_path)
    for name, text in _render_community_md(board).items():
        (board_dir / name).write_text(text, encoding="utf-8")
    print(f"Added {data['detector_name']} to the community board at {board_path}")
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

    p_sub = sub.add_parser(
        "submit", help="Add a validated submission to the community board (leaderboard/community/)"
    )
    p_sub.add_argument("submission")
    p_sub.add_argument("--name", default=None)
    p_sub.add_argument("--author", default=None)
    p_sub.add_argument(
        "--board-dir",
        type=Path,
        default=_DEFAULT_LEADERBOARD_DIR,
        dest="board_dir",
        help=(
            "Directory holding the community board.json and its rendered tables "
            f"(default: {_DEFAULT_LEADERBOARD_DIR.as_posix()}, relative to the working directory)"
        ),
    )
    p_sub.set_defaults(func=cmd_submit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
