"""Derive every family's proxy calibration once, serially, into the disk cache.

Run before a parallel admission run. ``generate_pair`` does two things: it derives the
family's calibration (expensive -- 74s of HalfCheetah rollouts for one ``mujoco_camping``
tier) and it builds the rollout closures (free). So calling it at every scored difficulty
costs exactly the calibrations and nothing else, and populates
``src/rhob/v3/calibration_cache.py``'s on-disk layer for the workers that follow.

Without this, ``pytest -n N`` on the per-tier smoke screen has every worker miss the
cache at the same moment and derive the same amplitude concurrently -- parallelism that
costs more CPU than it saves. With it, the parallel phase is pure rollouts, which is the
part that actually wants the cores.

Prints what it derived and what it found already warm, so a run that unexpectedly
recomputes everything is visible rather than just slow.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Run against *this* checkout's sources, not whatever `rhob` happens to be installed.
# `pip install -e .` records one source tree, so inside a git worktree a bare
# `import rhob` resolves to the other one -- silently. This script writes a calibration
# cache keyed on a hash of the family modules' source, so importing the wrong tree
# would key entries against source that is not the source being tested.
# tests/test_scripts_run_their_own_checkout.py enforces this for every script here.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import rhob.v3.families  # noqa: F401,E402 -- import for @FamilyRegistry.register side effects
from rhob.v3.calibration_cache import cache_dir, enabled  # noqa: E402
from rhob.v3.registry import FamilyRegistry  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        nargs="*",
        default=None,
        help="Only these families (default: every registered family).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first family that raises instead of reporting and continuing.",
    )
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=None,
        help=(
            "Stop starting new families once this much wall clock has elapsed, and name "
            "the ones left cold. Warming is an optimisation and must never be the reason "
            "a run produces no results: mujoco_joint_limit_gaming alone costs 38 minutes "
            "to derive, so an unbounded cold warm can consume a CI job's entire budget "
            "before a single test executes. Anything left cold is derived by the tests "
            "themselves, just less efficiently."
        ),
    )
    args = parser.parse_args(argv)

    if not enabled():
        print("RHOB_CALIB_CACHE is off; nothing to warm.", file=sys.stderr)
        return 0

    names = sorted(args.families) if args.families else FamilyRegistry.list_families()
    print(f"cache dir: {cache_dir()}", flush=True)
    print(f"warming {len(names)} families\n", flush=True)

    failures: list[tuple[str, float, str]] = []
    skipped: list[str] = []
    total = 0.0
    started_all = time.perf_counter()
    # Largest single-family cost seen so far this run. The budget is checked BETWEEN
    # families and a calibration is not interruptible, so a guard that only asks "is the
    # budget spent?" can start a family that runs far past it: measured in CI, the warm
    # step blew a 45-minute budget and was killed by the 50-minute step timeout, losing
    # every unflushed progress line with it. Refusing to start a family when less than
    # the worst observed cost remains bounds the overrun to one family instead of one
    # family plus the whole remaining list. It cannot bound the FIRST expensive family,
    # which has no prior observation -- so the step timeout still has to exceed the
    # budget by the worst known single-family cost (mujoco_joint_limit_gaming, ~38 min).
    worst_seen = 0.0
    for index, name in enumerate(names):
        remaining = (
            None
            if args.budget_seconds is None
            else args.budget_seconds - (time.perf_counter() - started_all)
        )
        if remaining is not None and remaining < worst_seen and index > 0:
            skipped = list(names[index:])
            print(
                f"\n  {remaining:.0f}s left, less than the worst family seen "
                f"({worst_seen:.0f}s); leaving {len(skipped)} families cold rather than "
                f"starting one that would overrun",
                flush=True,
            )
            break
        if (
            args.budget_seconds is not None
            and time.perf_counter() - started_all >= args.budget_seconds
        ):
            skipped = list(names[index:])
            print(
                f"\n  budget of {args.budget_seconds:.0f}s reached; "
                f"leaving {len(skipped)} families cold",
                flush=True,
            )
            break
        try:
            family = FamilyRegistry.get(name)
            difficulties = family.default_difficulties()
        except Exception as exc:  # noqa: BLE001 -- one broken family must not stop the rest
            print(f"  {name}: SKIPPED ({type(exc).__name__}: {exc})", flush=True)
            failures.append((name, 0.0, f"{type(exc).__name__}: {exc}"))
            if args.fail_fast:
                return 1
            continue

        started = time.perf_counter()
        try:
            for difficulty in difficulties:
                # Calibration happens here; the rollout closures it returns are not called.
                family.generate_pair(difficulty, seed=0)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - started
            print(f"  {name}: FAILED after {elapsed:.1f}s ({type(exc).__name__}: {exc})", flush=True)
            failures.append((name, elapsed, f"{type(exc).__name__}: {exc}"))
            if args.fail_fast:
                return 1
            continue

        elapsed = time.perf_counter() - started
        total += elapsed
        worst_seen = max(worst_seen, elapsed)
        # A warm family costs milliseconds; a cold one costs its calibrations. The
        # distinction is the useful signal, so say which happened.
        state = "already warm" if elapsed < 1.0 else "derived"
        print(f"  {name}: {state} ({len(difficulties)} tiers, {elapsed:.1f}s)", flush=True)

    warmed = len(names) - len(failures) - len(skipped)
    print(f"\ntotal {total:.1f}s across {warmed} families", flush=True)
    if skipped:
        # Named, not merely counted: comparing two runs, a reader needs to know which
        # families the test phase had to derive for itself.
        print(f"left cold by the budget ({len(skipped)}): {', '.join(skipped)}", flush=True)
    if failures:
        print(f"{len(failures)} family/families could not be warmed:", flush=True)
        for name, _, why in failures:
            print(f"  {name}: {why}", flush=True)
        # Not fatal by default: a family that cannot calibrate will fail its own test
        # with a better message than this script can give.
    return 0


if __name__ == "__main__":
    sys.exit(main())
