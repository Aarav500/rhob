#!/usr/bin/env python
"""v5 Leaderboard + Cross-Family Generalization Experiment.

Evaluates the full detector suite across every registered family (families="all"
resolves dynamically via FamilyRegistry, so this automatically covers new families as
they're added) with cross-family transfer analysis.

Two properties of the emitted JSON are load-bearing and easy to miss:

1. **Not-applicable cells are ``null``, never 0.5.** A detector is scored only on
   families that emit the channel its access level is defined over. A family that
   ships no state-visitation histogram gives every L1 detector nothing to read, and
   the old artifact recorded the detectors' 0.5 no-signal constant there as though it
   were a measurement -- on 25 of the 33 families and 88 of the 123 cells.
   ``per_family`` now carries ``null`` for such families, they are listed under
   ``not_applicable_families``, and they are excluded from ``overall_auroc``. A ``0.5``
   in this file therefore means "measured, and at chance" -- which is a real and
   interesting result -- and is now distinguishable from "never measured".
2. **The whole file is one draw.** Every detector is scored on the identical rolled-out
   runs from a single layout seed and a single set of rollout seeds; there is no
   replication and there are no confidence intervals; the null SE near AUROC 0.5 is
   0.19 at n=m=5. The ``sampling`` block records exactly that, so a reader does not
   have to infer it from this script's source. For intervals, run
   ``scripts/replicate_leaderboard.py`` over several draws and aggregate with
   ``scripts/aggregate_replication.py``.

Run: python scripts/v5_leaderboard_and_transfer.py [--families ...] [--detectors ...]
     [--n-seeds N] [--out PATH]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# Run the checkout this file lives in, not whatever the editable install points at.
# In a git worktree those differ, and the failure is silent: the script imports a
# same-named module from the OTHER tree and publishes results attributed to this one.
# Enforced for every script by tests/test_scripts_run_their_own_checkout.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rhob.detectors import (  # noqa: E402
    AngularMomentumDetector,
    BehavioralThresholdDetector,
    BimodalOccupancyDetector,
    CentroidDriftDetector,
    CentroidTrackerDetector,
    FeatureConsistencyDetector,
    FeatureMagnitudeDetector,
    GradientReversalDetector,
    MaxPlateauDetector,
    OccupancyPolarizationDetector,
    PerfectFeatureOracleDetector,
    RewardAutocorrelationDetector,
    RewardCUSUMDetector,
    RewardKDEDetector,
    RewardMLPDetector,
    RewardPeakDetector,
    RewardSkewnessDetector,
    RewardThresholdDetector,
    RewardTrendDetector,
    RewardVarianceRatioDetector,
    RewardFeatureCorrelationDetector,
    SpectralRewardDetector,
    StateCoverageRateDetector,
    StateDivergenceDetector,
    StateFrequencyAnomalyDetector,
    TrajectoryMLPDetector,
    TransitionEntropyDetector,
    TrueRewardOracleDetector,
    VarianceWindowDetector,
    VisitationEntropyTrendDetector,
)
from rhob.v3.benchmark import Benchmark  # noqa: E402
from rhob.v3.leaderboard.access_summary import (  # noqa: E402
    degenerate_families_from_ledger,
    render_access_summary_md,
    summarize_access_levels,
)
from rhob.v3.provenance import provenance_block, sampling_block  # noqa: E402
from rhob.v3.registry import FamilyRegistry  # noqa: E402

# All 30 detectors. 29 of them are independent measurements; PerfectFeatureOracleDetector
# is a relabelled duplicate of BehavioralThresholdDetector (it inherits classify/detect_onset
# unchanged) and is kept in the run only so the committed artifact keeps its row as a
# cross-check. It is held out of the access-level aggregates -- see
# rhob.detectors.redundancy and rhob.v3.leaderboard.access_summary.
DETECTORS = [
    RewardThresholdDetector(),
    RewardCUSUMDetector(),
    RewardVarianceRatioDetector(),
    SpectralRewardDetector(),
    RewardPeakDetector(),
    RewardAutocorrelationDetector(),
    RewardSkewnessDetector(),
    RewardTrendDetector(),
    VarianceWindowDetector(),
    MaxPlateauDetector(),
    GradientReversalDetector(),
    RewardMLPDetector(),
    RewardKDEDetector(),
    StateFrequencyAnomalyDetector(),
    CentroidDriftDetector(),
    OccupancyPolarizationDetector(),
    BimodalOccupancyDetector(),
    TransitionEntropyDetector(),
    StateCoverageRateDetector(),
    VisitationEntropyTrendDetector(),
    StateDivergenceDetector(),
    BehavioralThresholdDetector(),
    AngularMomentumDetector(),
    CentroidTrackerDetector(),
    FeatureMagnitudeDetector(),
    FeatureConsistencyDetector(),
    RewardFeatureCorrelationDetector(),
    TrajectoryMLPDetector(),
    TrueRewardOracleDetector(),
    PerfectFeatureOracleDetector(),
]


# Rollout seed layout, mirroring MatchedPair.rollout (src/rhob/v3/base_pair.py): the
# hacking variant is rolled out at seeds ``seed_base + s`` and the legitimate variant at
# ``seed_base + 1000 + s`` for ``s in range(n_seeds)``. The leaderboard uses the default
# ``seed_base=0``.
ROLLOUT_SEED_BASE = 0
LEGIT_SEED_OFFSET = 1000

# FamilyRegistry.generate_suite() calls ``fam.generate_pair(d)`` without a seed, i.e.
# with BaseFamily.generate_pair's default ``seed=0``. The leaderboard therefore samples
# exactly one environment layout per (family, difficulty) -- one draw, not a
# distribution -- which is the fact the ``sampling`` block below makes explicit.
LAYOUT_SEEDS = [0]


def _jsonable(value: float | None) -> float | None:
    """Round for the artifact, mapping NaN to ``null`` (JSON has no NaN literal).

    ``json.dump`` would happily emit a bare ``NaN`` token, which is invalid JSON and
    fails in every strict parser downstream (including the Gradio space).
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), 3)


def _summarize(results, level: str) -> dict:
    """Turn one detector's :class:`BenchmarkResults` into its leaderboard record.

    Families whose cells were all not-applicable get ``per_family[family] = null`` and
    an entry in ``not_applicable_families``; they contribute nothing to
    ``overall_auroc``. This is the distinction between "measured, scored 0.5" and
    "this detector's input channel does not exist in this family".
    """
    scored_by_family: dict[str, list[float]] = {}
    na_by_family: dict[str, int] = {}
    na_reasons: dict[str, int] = {}

    for cell in results.cells:
        if cell.applicable and not math.isnan(cell.discrimination_auroc):
            scored_by_family.setdefault(cell.family, []).append(float(cell.discrimination_auroc))
        else:
            na_by_family[cell.family] = na_by_family.get(cell.family, 0) + 1
            reason = cell.na_reason or "cell produced a NaN AUROC (degenerate labels)"
            na_reasons[reason] = na_reasons.get(reason, 0) + 1

    per_family: dict[str, float | None] = {}
    for family in sorted(set(scored_by_family) | set(na_by_family)):
        values = scored_by_family.get(family)
        per_family[family] = round(float(np.mean(values)), 3) if values else None

    return {
        "access_level": level,
        "overall_auroc": _jsonable(results.overall_auroc),
        "cells": len(results.cells),
        "cells_measured": len(results.scored_cells),
        "cells_not_applicable": sum(na_by_family.values()),
        "not_applicable_families": sorted(f for f, v in per_family.items() if v is None),
        "not_applicable_reasons": na_reasons,
        "per_family": per_family,
    }


def build_board(
    results_by_detector: dict, family_names: list[str], n_seeds: int, prov: dict
) -> dict:
    """The board artifact, from each detector's record (see :func:`_summarize`)."""
    return {
        # Kept at the top level for the existing readers that look for it
        # (rhob.v3.leaderboard.adapters, space/app.py); the authoritative copy
        # is provenance.generated_utc.
        "timestamp": prov["generated_utc"],
        "version": "v5",
        "detectors": len(results_by_detector),
        "families": len(family_names),
        "families_evaluated": family_names,
        "provenance": prov,
        "sampling": sampling_block(
            n_seeds=n_seeds,
            n_layouts=len(LAYOUT_SEEDS),
            layout_seeds=LAYOUT_SEEDS,
            rollout_seeds_hacking=[ROLLOUT_SEED_BASE + s for s in range(n_seeds)],
            rollout_seeds_legit=[
                ROLLOUT_SEED_BASE + LEGIT_SEED_OFFSET + s for s in range(n_seeds)
            ],
            n_replicates=1,
        ),
        "cell_semantics": {
            "per_family_null": (
                "not applicable -- the family does not emit the RunData channel "
                "this detector's access level is defined over, so no measurement "
                "exists. Excluded from overall_auroc; never imputed as 0.5."
            ),
            "per_family_number": "measured mean AUROC over that family's difficulty cells",
        },
        "results": results_by_detector,
    }


class ShardError(ValueError):
    """A set of per-detector boards that does not add up to one full board."""


def merge_shards(
    shard_dir: Path,
    detectors: list[str] | None = None,
    families: list[str] | None = None,
) -> dict:
    """Assemble the full board from one-detector runs (``--detectors NAME``) under ``shard_dir``.

    Each shard is the board JSON of one run over every family for one detector, so the
    30 detectors can be scored on separate machines at once
    (``.github/workflows/regenerate.yml``). That gives the board a single run would have
    written, because each detector's record depends only on the rollouts, and those are
    deterministic in the seeds: every shard rolls out the identical runs (the sampling
    block's ``shared_draw_across_detectors`` holds whether the runs are shared in memory
    or regenerated).

    Refused unless the shards are exactly the 30 detectors in :data:`DETECTORS`, each
    once; none recorded an error; all evaluated every registered family under the same
    sampling block; and all came from one clean commit under one Python and one set of
    package versions. ``provenance.sharding`` records what a single run would not:
    the shard count, the first and last shard timestamps, and every platform string.

    ``detectors`` and ``families`` narrow what is expected (tests use them to merge a
    cheap subset); the CLI always expects all 30 detectors on every registered family.
    """
    import copy
    import os

    paths = sorted(shard_dir.rglob("*.json"))
    if not paths:
        raise ShardError(f"no shard JSON under {shard_dir}")
    expected = [d.name for d in DETECTORS if detectors is None or d.name in detectors]
    by_name: dict[str, tuple[Path, dict]] = {}
    for path in paths:
        board = json.loads(path.read_text(encoding="utf-8"))
        if board.get("version") != "v5" or "provenance" not in board or "sampling" not in board:
            raise ShardError(f"{path}: not a v5 board with provenance and sampling blocks")
        if len(board.get("results", {})) != 1:
            raise ShardError(
                f"{path}: {len(board.get('results', {}))} detectors; a shard is exactly one"
            )
        (name, record), = board["results"].items()
        if "error" in record:
            raise ShardError(f"{path}: {name} raised: {record['error']}")
        if name in by_name:
            raise ShardError(f"{path}: {name} already supplied by {by_name[name][0]}")
        by_name[name] = (path, board)
    missing = [n for n in expected if n not in by_name]
    extra = sorted(set(by_name) - set(expected))
    if missing or extra:
        raise ShardError(f"shards are not the {len(expected)} detectors: missing {missing}, "
                         f"unknown {extra}")

    ordered = [by_name[n] for n in expected]
    ref_path, ref = ordered[0]
    wanted_families = FamilyRegistry.list_families() if families is None else list(families)
    if ref["families_evaluated"] != wanted_families:
        raise ShardError(f"{ref_path}: evaluated {len(ref['families_evaluated'])} families, "
                         f"not the {len(wanted_families)} expected")

    def environment(prov: dict) -> dict:
        return {
            "commit": (prov.get("git") or {}).get("commit"),
            "python": prov.get("python"),
            "python_implementation": prov.get("python_implementation"),
            "packages": prov.get("packages"),
        }

    ref_env = environment(ref["provenance"])
    if not ref_env["commit"]:
        raise ShardError(f"{ref_path}: no git commit recorded")
    for path, board in ordered:
        if (board["provenance"].get("git") or {}).get("dirty") is not False:
            raise ShardError(f"{path}: produced from a dirty or unknown tree")
        for field, value in environment(board["provenance"]).items():
            if value != ref_env[field]:
                raise ShardError(f"{path}: {field} differs from {ref_path}")
        if board["families_evaluated"] != ref["families_evaluated"]:
            raise ShardError(f"{path}: families_evaluated differs from {ref_path}")
        if board["sampling"] != ref["sampling"]:
            raise ShardError(f"{path}: sampling block differs from {ref_path}")

    stamps = sorted(b["provenance"]["generated_utc"] for _, b in ordered)
    prov = copy.deepcopy(ref["provenance"])
    prov["generated_utc"] = stamps[-1]
    prov["argv"] = list(sys.argv)
    ci_run = None
    if os.environ.get("GITHUB_RUN_ID"):
        ci_run = (
            f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
            f"{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/"
            f"{os.environ['GITHUB_RUN_ID']}"
        )
    prov["sharding"] = {
        "unit": "one detector over every family per run",
        "n_shards": len(ordered),
        "generated_utc_first": stamps[0],
        "generated_utc_last": stamps[-1],
        "platforms": sorted({b["provenance"].get("platform") for _, b in ordered}),
        "ci_run": ci_run,
    }
    results = {name: board["results"][name] for name, (_, board) in zip(expected, ordered)}
    merged = build_board(
        results, ref["families_evaluated"], ref["sampling"]["n_seeds_per_variant"], prov
    )
    if merged["sampling"] != ref["sampling"]:
        raise ShardError("the shards' sampling block is not the one this script writes")
    return merged


def print_summary(results_by_detector: dict) -> None:
    """The run's console report: successes, N/A totals, top 10, and the access ladder."""
    scored = [
        (name, data)
        for name, data in results_by_detector.items()
        if data.get("overall_auroc") is not None
    ]
    print(f"     Successful: {len(scored)}/{len(results_by_detector)}")
    total_na = sum(
        data.get("cells_not_applicable", 0) for data in results_by_detector.values()
    )
    total_cells = sum(data.get("cells", 0) for data in results_by_detector.values())
    print(f"     Not-applicable cells (excluded, not imputed): {total_na}/{total_cells}")

    # Print top detectors
    print()
    print("Top 10 Detectors:")
    scored.sort(key=lambda x: x[1]["overall_auroc"], reverse=True)
    for i, (name, data) in enumerate(scored[:10], 1):
        print(
            f"  {i:2}. {name:<40} {data['access_level']:<3} {data['overall_auroc']:.3f}"
            f"  ({data['cells_measured']}/{data['cells']} cells measured)"
        )

def main():
    """Run v5 leaderboard and transfer analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families", nargs="+", default=None,
        help="Family names to evaluate (default: every registered family).",
    )
    parser.add_argument(
        "--detectors", nargs="+", default=None,
        help="Detector names to evaluate, matched against detector.name (default: all 30).",
    )
    parser.add_argument("--n-seeds", type=int, default=5, help="Rollout seeds per variant per cell.")
    parser.add_argument(
        "--out", type=Path, default=Path("leaderboard/v5_leaderboard.json"),
        help="Output JSON path. Point this elsewhere for smoke runs so the published "
             "artifact is not overwritten by a subset evaluation.",
    )
    parser.add_argument(
        "--list-detectors", action="store_true",
        help="Print the detector names as a JSON list and exit (the shard list for "
             ".github/workflows/regenerate.yml).",
    )
    parser.add_argument(
        "--merge-shards", type=Path, default=None, metavar="DIR",
        help="Assemble the full board from one-detector runs under DIR instead of "
             "evaluating anything; see merge_shards.",
    )
    args = parser.parse_args()

    if args.list_detectors:
        print(json.dumps([d.name for d in DETECTORS]))
        return 0
    if args.merge_shards is not None:
        if args.families or args.detectors:
            parser.error("--merge-shards takes no --families or --detectors")
        try:
            board = merge_shards(args.merge_shards)
        except ShardError as exc:
            print(f"refusing to merge: {exc}", file=sys.stderr)
            return 1
        board_path = Path(args.out)
        board_path.parent.mkdir(parents=True, exist_ok=True)
        with open(board_path, "w") as f:
            json.dump(board, f, indent=2)
        print(f"[OK] Merged {board['provenance']['sharding']['n_shards']} shards into {board_path}")
        results_by_detector = board["results"]
    else:
        results_by_detector = evaluate(parser, args)

    print_summary(results_by_detector)

    # Access-level ladder. Printed with the max alongside the mean, with duplicate
    # detectors held out, and with the ledger's degenerate-proxy families held out of L0
    # (and only L0 -- they are ordinary items for every level that does not read the
    # proxy reward), so a regeneration surfaces the corrected ladder immediately instead
    # of leaving it to be recomputed downstream. The exclusion set is read from
    # admission/admission_ledger.json rather than hardcoded, so it tracks the ledger; see
    # rhob.v3.leaderboard.access_summary for both arguments.
    print()
    print("By access level:")
    print(
        render_access_summary_md(
            summarize_access_levels(
                results_by_detector,
                degenerate_families=degenerate_families_from_ledger(),
            )
        )
    )
    return 0


def evaluate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> dict:
    """Score the selected detectors on the selected families and write the board."""
    families = args.families if args.families else "all"
    family_names = FamilyRegistry.list_families() if families == "all" else list(families)
    if args.detectors:
        wanted = {d.lower() for d in args.detectors}
        detectors = [d for d in DETECTORS if d.name.lower() in wanted]
        missing = wanted - {d.name.lower() for d in detectors}
        if missing:
            parser.error(f"unknown detector name(s): {sorted(missing)}")
    else:
        detectors = list(DETECTORS)

    print("v5 Benchmark: Full Leaderboard")
    print("=" * 70)
    print(f"Evaluating {len(detectors)} detectors across {len(family_names)} families...")
    print()

    results_by_detector = {}
    for i, detector in enumerate(detectors, 1):
        name = detector.name
        level = detector.access_level
        print(f"[{i}/{len(detectors)}] {name:<35} ({level})", end="", flush=True)

        try:
            results = Benchmark.evaluate(
                detector,
                families=families,
                difficulties="all",
                n_seeds=args.n_seeds,  # Reduced for speed
                verbose=False,
            )
            record = _summarize(results, level)
            results_by_detector[name] = record

            overall = record["overall_auroc"]
            n_na = record["cells_not_applicable"]
            shown = "N/A (no measurable cell)" if overall is None else f"{overall:.3f}"
            na_note = f"  [{n_na}/{record['cells']} cells N/A]" if n_na else ""
            print(f" -> {shown} [OK]{na_note}")

        except Exception as e:
            print(f" -> ERROR: {e}")
            results_by_detector[name] = {"access_level": level, "error": str(e)}

    # Save full leaderboard
    board_path = Path(args.out)
    board_path.parent.mkdir(parents=True, exist_ok=True)
    prov = provenance_block(script="scripts/v5_leaderboard_and_transfer.py")
    with open(board_path, "w") as f:
        json.dump(build_board(results_by_detector, family_names, args.n_seeds, prov), f, indent=2)

    print()
    print("=" * 70)
    print(f"[OK] Leaderboard saved to {board_path}")
    return results_by_detector


if __name__ == "__main__":
    raise SystemExit(main())
