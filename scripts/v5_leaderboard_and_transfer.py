#!/usr/bin/env python
"""v5 Leaderboard + Cross-Family Generalization Experiment.

Evaluates the full detector suite across every registered family (families="all"
resolves dynamically via FamilyRegistry, so this automatically covers new families as
they're added) with cross-family transfer analysis.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from rhob.detectors import (
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
from rhob.v3.benchmark import Benchmark
from rhob.v3.registry import FamilyRegistry

# All 30 detectors
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


def main():
    """Run v5 leaderboard and transfer analysis."""
    n_families = len(FamilyRegistry.list_families())
    print("v5 Benchmark: Full Leaderboard")
    print("=" * 70)
    print(f"Evaluating {len(DETECTORS)} detectors across {n_families} families...")
    print()

    results_by_detector = {}
    for i, detector in enumerate(DETECTORS, 1):
        name = detector.name
        level = detector.access_level
        print(f"[{i}/{len(DETECTORS)}] {name:<35} ({level})", end="", flush=True)

        try:
            results = Benchmark.evaluate(
                detector,
                families="all",
                difficulties="all",
                n_seeds=5,  # Reduced for speed
                verbose=False,
            )

            overall = results.overall_auroc
            results_by_detector[name] = {
                "access_level": level,
                "overall_auroc": round(float(overall), 3),
                "cells": len(results.cells),
                "per_family": {},
            }

            # Per-family AUROC
            for cell in results.cells:
                family_key = cell.family
                if family_key not in results_by_detector[name]["per_family"]:
                    results_by_detector[name]["per_family"][family_key] = []
                results_by_detector[name]["per_family"][family_key].append(
                    round(float(cell.discrimination_auroc), 3)
                )

            # Average per family
            for family_key in results_by_detector[name]["per_family"]:
                values = results_by_detector[name]["per_family"][family_key]
                results_by_detector[name]["per_family"][family_key] = round(
                    np.mean(values), 3
                )

            print(f" -> {overall:.3f} [OK]")

        except Exception as e:
            print(f" -> ERROR: {e}")
            results_by_detector[name] = {"access_level": level, "error": str(e)}

    # Save full leaderboard
    board_path = Path("leaderboard/v5_leaderboard.json")
    board_path.parent.mkdir(parents=True, exist_ok=True)

    with open(board_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "version": "v5",
                "detectors": len(DETECTORS),
                "families": n_families,
                "results": results_by_detector,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print(f"[OK] Leaderboard saved to {board_path}")

    # Summary stats
    successful = sum(
        1 for r in results_by_detector.values() if r.get("overall_auroc")
    )
    print(f"     Successful: {successful}/{len(DETECTORS)}")

    # Print top detectors
    print()
    print("Top 10 Detectors:")
    results = [
        (name, data)
        for name, data in results_by_detector.items()
        if data.get("overall_auroc")
    ]
    results.sort(key=lambda x: x[1]["overall_auroc"], reverse=True)
    for i, (name, data) in enumerate(results[:10], 1):
        print(
            f"  {i:2}. {name:<40} {data['access_level']:<3} {data['overall_auroc']:.3f}"
        )


if __name__ == "__main__":
    main()
