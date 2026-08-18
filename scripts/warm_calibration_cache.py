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
    args = parser.parse_args(argv)

    if not enabled():
        print("RHOB_CALIB_CACHE is off; nothing to warm.", file=sys.stderr)
        return 0

    names = sorted(args.families) if args.families else FamilyRegistry.list_families()
    print(f"cache dir: {cache_dir()}")
    print(f"warming {len(names)} families\n")

    failures: list[tuple[str, float, str]] = []
    total = 0.0
    for name in names:
        try:
            family = FamilyRegistry.get(name)
            difficulties = family.default_difficulties()
        except Exception as exc:  # noqa: BLE001 -- one broken family must not stop the rest
            print(f"  {name}: SKIPPED ({type(exc).__name__}: {exc})")
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
            print(f"  {name}: FAILED after {elapsed:.1f}s ({type(exc).__name__}: {exc})")
            failures.append((name, elapsed, f"{type(exc).__name__}: {exc}"))
            if args.fail_fast:
                return 1
            continue

        elapsed = time.perf_counter() - started
        total += elapsed
        # A warm family costs milliseconds; a cold one costs its calibrations. The
        # distinction is the useful signal, so say which happened.
        state = "already warm" if elapsed < 1.0 else "derived"
        print(f"  {name}: {state} ({len(difficulties)} tiers, {elapsed:.1f}s)")

    print(f"\ntotal {total:.1f}s across {len(names) - len(failures)} families")
    if failures:
        print(f"{len(failures)} family/families could not be warmed:")
        for name, _, why in failures:
            print(f"  {name}: {why}")
        # Not fatal by default: a family that cannot calibrate will fail its own test
        # with a better message than this script can give.
    return 0


if __name__ == "__main__":
    sys.exit(main())
