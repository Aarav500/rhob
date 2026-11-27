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
  * ``layout_seed`` resamples the *environment* in the families that build it from a
    seed (``FamilyRegistry.generate_suite`` passes it to ``generate_pair_at``), and,
    under sign randomization, every family's behavioral orientation.
  * ``seed_base`` resamples the *runs* within a fixed environment.
Holding either fixed would understate the variance of the published number.

That held only from 2026-11-27. Before then ``generate_suite`` dropped ``layout_seed``,
so every replicate was built and oriented at layout 0 whatever it recorded, and resampled
its rollout seeds alone. A replicate therefore says which code drew it: ``draw_schema``
(:data:`DRAW_SCHEMA`) is 2 from the fix on, and absent from every file written before
it. ``scripts/merge_rescored_detector.py`` refuses to merge into a draw without it,
``scripts/check_redraw.py`` requires it of a re-draw, and
``scripts/aggregate_replication.py`` reports it in its ``replicates`` block.

Each record's ``cells_detail`` is written by ``_cells_detail`` from
``scripts/v5_leaderboard_and_transfer.py``, the function that writes the board's
``--cells-out`` sidecar, so replicate 0 (the board's draw) can be compared with the board
cell for cell.

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

from v5_leaderboard_and_transfer import DETECTORS, _cells_detail, _summarize  # noqa: E402

from rhob.v3.benchmark import Benchmark  # noqa: E402
from rhob.v3.provenance import provenance_block  # noqa: E402
from rhob.v3.registry import FamilyRegistry  # noqa: E402

# Spacing between replicates' seed bases. Must exceed MatchedPair.rollout's internal
# LEGIT_SEED_OFFSET (1000) by a wide margin -- see the module docstring.
SEED_BASE_STRIDE = 1_000_000

#: What a replicate's ``(layout_seed, seed_base)`` reached, written as the top-level
#: ``"draw_schema"`` of every replicate file.
#:
#: - absent (read as 1): written before 2026-11-27, when ``FamilyRegistry.generate_suite``
#:   dropped ``layout_seed``. Every cell was built, and oriented, at layout 0; only the
#:   rollout seeds (``seed_base``) were resampled, whatever ``layout_seed`` says.
#: - 2: ``layout_seed`` is forwarded to the families (``generate_pair_at(d,
#:   seed=layout_seed)``), so the layout -- where a family builds one from its seed --
#:   and the orientation are resampled with the rollout seeds.
#:
#: Raise it whenever a change alters what a draw coordinate reaches, so that a merge or
#: an aggregate cannot pool draws made under two meanings of the same seeds.
DRAW_SCHEMA = 2


def draw_for(replicate_id: int) -> tuple[int, int]:
    """The ``(layout_seed, seed_base)`` this replicate id denotes.

    Replicate 0 is ``(0, 0)``, the draw ``leaderboard/v5_leaderboard.json`` is scored
    at, which makes it the control the other replicates are read against. A re-draw made
    from one commit, beside the board, must reproduce the board exactly: replicate 0's
    ``cells_detail`` and the board's ``--cells-out`` sidecar come from the same function,
    and ``scripts/check_redraw.py`` refuses the set if any cell differs (unless told how
    many may).

    The set committed before that re-draw was not such a copy. Its ``replicate_000.json``
    (drawn 2026-08-04, ``draw_schema`` absent, Reward Skewness re-scored 2026-09-03 and
    19 more detectors 2026-11-26) and the board re-scored on 2026-11-26 agree on 974 of
    990 per-family entries -- 610 of the 626 non-null values, plus 364 entries that are
    ``null`` (not applicable) in both -- and 14 of the 16 that differ are
    ``sequence_keyword_stuffing``. Before 2026-11-26, 145 more of the agreeing entries
    were a fallback 0.5 on families shorter than a detector's windows, which is 0.5 in
    any draw -- 104 of eight L0 detectors, 33 of three L2 detectors and 8 of the L1
    detectors on ``continuous_camping``. Those detectors have declared ``min_episodes``
    since 2026-11-26, and the 145 cells are now among the entries ``null`` on both
    boards (219 + 145 = 364).
    """
    return replicate_id, replicate_id * SEED_BASE_STRIDE


def main(argv: list[str] | None = None) -> None:
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
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=None,
        help=(
            "Subset of detector names (default: all). For re-scoring one row after a "
            "detector fix; the merged replicate must record which rows were re-scored."
        ),
    )
    args = parser.parse_args(argv)

    if args.replicate_id < 0:
        parser.error("--replicate-id must be >= 0")

    layout_seed, seed_base = draw_for(args.replicate_id)
    families = args.families if args.families else "all"
    family_names = FamilyRegistry.list_families() if families == "all" else list(families)

    print(f"replicate {args.replicate_id}: layout_seed={layout_seed} seed_base={seed_base}")
    detectors = [d for d in DETECTORS if args.detectors is None or d.name in args.detectors]
    if args.detectors and len(detectors) != len(args.detectors):
        known = {d.name for d in DETECTORS}
        parser.error(f"unknown detector(s): {sorted(set(args.detectors) - known)}")
    print(f"  {len(detectors)} detectors x {len(family_names)} families, n_seeds={args.n_seeds}")

    results_by_detector: dict[str, dict] = {}
    for i, detector in enumerate(detectors, 1):
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
            record["cells_detail"] = _cells_detail(results)
            results_by_detector[name] = record
            overall = record["overall_auroc"]
            shown = "N/A" if overall is None else f"{overall:.3f}"
            print(f"  [{i}/{len(detectors)}] {name:<38} {shown}")
        except Exception as e:  # noqa: BLE001 -- one bad detector must not lose the replicate
            print(f"  [{i}/{len(detectors)}] {name:<38} ERROR: {e}")
            results_by_detector[name] = {"access_level": detector.access_level, "error": str(e)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"replicate_{args.replicate_id:03d}.json"
    with open(out, "w") as f:
        json.dump(
            {
                "replicate_id": args.replicate_id,
                "draw_schema": DRAW_SCHEMA,
                "layout_seed": layout_seed,
                "seed_base": seed_base,
                "n_seeds": args.n_seeds,
                "families_evaluated": family_names,
                "detectors_evaluated": [d.name for d in detectors],
                "provenance": provenance_block(script="scripts/replicate_leaderboard.py"),
                "results": results_by_detector,
            },
            f,
            indent=2,
        )
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
