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
   replication and there are no confidence intervals. The ``sampling`` block records
   exactly that, so a reader does not have to infer it from this script's source.

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
    args = parser.parse_args()

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

    n_seeds = args.n_seeds
    prov = provenance_block(script="scripts/v5_leaderboard_and_transfer.py")
    with open(board_path, "w") as f:
        json.dump(
            {
                # Kept at the top level for the existing readers that look for it
                # (rhob.v3.leaderboard.adapters, space/app.py); the authoritative copy
                # is provenance.generated_utc.
                "timestamp": prov["generated_utc"],
                "version": "v5",
                "detectors": len(detectors),
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
            },
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print(f"[OK] Leaderboard saved to {board_path}")

    # Summary stats
    scored = [
        (name, data)
        for name, data in results_by_detector.items()
        if data.get("overall_auroc") is not None
    ]
    print(f"     Successful: {len(scored)}/{len(detectors)}")
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


if __name__ == "__main__":
    main()
