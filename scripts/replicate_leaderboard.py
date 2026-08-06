"""Run ONE replicate of the full leaderboard at an independent draw of the benchmark.

The published leaderboard is a single draw: one environment layout per (family,
difficulty) and one seed sequence, five runs per variant. At ``n_seeds=5`` the
Mann-Whitney null standard error is ``sqrt((n+m+1)/(12nm))`` = 0.19 -- wider than most
of the differences that leaderboard reports. Ranking two detectors 0.62 vs 0.55 on one
draw is not evidence that the first is better.

This script produces one replicate. ``scripts/aggregate_replication.py`` combines them
into per-detector and per-family intervals. Run replicates as independent processes:
they share nothing, so R of them on an R-core box costs the wall-clock of one.

    for i in $(seq 0 19); do python scripts/replicate_leaderboard.py --replicate-id $i & done

A replicate is identified by ``(layout_seed, seed_base)``, derived from ``--replicate-id``
so the mapping is reproducible and recorded in every artifact. ``seed_base`` is spaced by
1e6 because ``MatchedPair.rollout`` draws the hacking variant at ``seed_base + s`` and the
legitimate variant at ``seed_base + 1000 + s``; a spacing below 1000 would make one
replicate's legitimate runs collide with the next replicate's hacking runs, quietly
correlating draws that the aggregate then treats as independent.

Both coordinates matter and neither is redundant:
  * ``layout_seed`` resamples the *environment* -- and, under sign randomization, each
    family's behavioral orientation.
  * ``seed_base`` resamples the *runs* within a fixed environment.
Holding either fixed would understate the variance of the published number.

``tests/test_v3/test_replication_draws_differ.py`` fails if these seeds stop reaching the
simulation, which would make every replicate a copy of replicate 0 and every resulting
interval zero-width.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import rhob from THIS checkout, not from whatever tree `pip install -e .` recorded.
# See tests/test_scripts_run_their_own_checkout.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# The detector list and per-detector summariser are imported from the leaderboard script
# rather than restated here, so a replicate cannot drift onto a different detector set or
# a different aggregation rule than the artifact it is supposed to be replicating.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v5_leaderboard_and_transfer import DETECTORS, _summarize  # noqa: E402

from rhob.v3.benchmark import Benchmark  # noqa: E402
from rhob.v3.provenance import provenance_block  # noqa: E402
from rhob.v3.registry import FamilyRegistry  # noqa: E402

# Spacing between replicates' seed bases. Must exceed MatchedPair.rollout's internal
# LEGIT_SEED_OFFSET (1000) by a wide margin -- see the module docstring.
SEED_BASE_STRIDE = 1_000_000


def draw_for(replicate_id: int) -> tuple[int, int]:
    """The ``(layout_seed, seed_base)`` this replicate id denotes.

    Replicate 0 is ``(0, 0)``: it reproduces the historical single-draw leaderboard
    exactly, which makes it the control the other replicates are read against.
    """
    return replicate_id, replicate_id * SEED_BASE_STRIDE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate-id", type=int, required=True, help="Replicate index (>=0).")
    parser.add_argument("--n-seeds", type=int, default=5, help="Rollout seeds per variant per cell.")
    parser.add_argument(
        "--families", nargs="+", default=None, help="Subset of families (default: all)."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/replication"),
        help="Directory for replicate_NNN.json.",
    )
    args = parser.parse_args()

    if args.replicate_id < 0:
        parser.error("--replicate-id must be >= 0")

    layout_seed, seed_base = draw_for(args.replicate_id)
    families = args.families if args.families else "all"
    family_names = FamilyRegistry.list_families() if families == "all" else list(families)

    print(f"replicate {args.replicate_id}: layout_seed={layout_seed} seed_base={seed_base}")
    print(f"  {len(DETECTORS)} detectors x {len(family_names)} families, n_seeds={args.n_seeds}")

    results_by_detector: dict[str, dict] = {}
    for i, detector in enumerate(DETECTORS, 1):
        name = detector.name
        try:
            results = Benchmark.evaluate(
                detector,
                families=families,
                difficulties="all",
                n_seeds=args.n_seeds,
                verbose=False,
                layout_seed=layout_seed,
                seed_base=seed_base,
            )
            record = _summarize(results, detector.access_level)
            # Per-cell AUROCs are kept, not just the per-family means: the aggregate needs
            # to be able to put an interval on a single cell, which is the resolution at
            # which the difficulty and access-level claims are actually made.
            record["cells_detail"] = [
                {
                    "family": c.family,
                    "difficulty": round(float(c.difficulty), 4),
                    "auroc": (
                        None
                        if (not c.applicable or c.discrimination_auroc != c.discrimination_auroc)
                        else round(float(c.discrimination_auroc), 6)
                    ),
                }
                for c in results.cells
            ]
            results_by_detector[name] = record
            overall = record["overall_auroc"]
            shown = "N/A" if overall is None else f"{overall:.3f}"
            print(f"  [{i}/{len(DETECTORS)}] {name:<38} {shown}")
        except Exception as e:  # noqa: BLE001 -- one bad detector must not lose the replicate
            print(f"  [{i}/{len(DETECTORS)}] {name:<38} ERROR: {e}")
            results_by_detector[name] = {"access_level": detector.access_level, "error": str(e)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"replicate_{args.replicate_id:03d}.json"
    with open(out, "w") as f:
        json.dump(
            {
                "replicate_id": args.replicate_id,
                "layout_seed": layout_seed,
                "seed_base": seed_base,
                "n_seeds": args.n_seeds,
                "families_evaluated": family_names,
                "provenance": provenance_block(script="scripts/replicate_leaderboard.py"),
                "results": results_by_detector,
            },
            f,
            indent=2,
        )
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
