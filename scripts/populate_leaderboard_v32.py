#!/usr/bin/env python
"""Populate leaderboard with v3.2 detector suite (29 total)."""

import json
from datetime import datetime
from pathlib import Path

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

# All 29 detectors (excluding Ensemble which needs special init)
DETECTORS = [
    # L0 (13)
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
    # L1 (8)
    StateFrequencyAnomalyDetector(),
    CentroidDriftDetector(),
    OccupancyPolarizationDetector(),
    BimodalOccupancyDetector(),
    TransitionEntropyDetector(),
    StateCoverageRateDetector(),
    VisitationEntropyTrendDetector(),
    StateDivergenceDetector(),
    # L2 (7 - skip EnsembleDetector)
    BehavioralThresholdDetector(),
    AngularMomentumDetector(),
    CentroidTrackerDetector(),
    FeatureMagnitudeDetector(),
    FeatureConsistencyDetector(),
    RewardFeatureCorrelationDetector(),
    TrajectoryMLPDetector(),
    # L3 (2)
    TrueRewardOracleDetector(),
    PerfectFeatureOracleDetector(),
]


def main():
    """Run leaderboard evaluation."""
    results_by_detector = {}
    board_path = Path("leaderboard/leaderboard.json")
    board_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {len(DETECTORS)} detectors...")
    print("=" * 60)

    for i, detector in enumerate(DETECTORS, 1):
        name = detector.name
        level = detector.access_level
        print(f"[{i}/{len(DETECTORS)}] {name} (access {level})", end="", flush=True)

        try:
            results = Benchmark.evaluate(
                detector,
                families="all",
                difficulties="all",
                n_seeds=10,
                verbose=False,
            )

            overall = results.overall_auroc
            results_by_detector[name] = {
                "access_level": level,
                "overall_auroc": round(float(overall), 3),
                "n_cells": len(results.cells),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            print(f" => AUROC={overall:.3f} [OK]")

        except Exception as e:
            print(f" => ERROR: {e}")
            results_by_detector[name] = {
                "access_level": level,
                "overall_auroc": None,
                "error": str(e),
            }

    # Save results
    with open(board_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "total_detectors": len(DETECTORS),
                "results": results_by_detector,
            },
            f,
            indent=2,
        )

    print("=" * 60)
    print(f"[OK] Leaderboard saved to {board_path}")
    print(f"  Total detectors: {len(DETECTORS)}")
    print(f"  Successful: {sum(1 for r in results_by_detector.values() if r.get('overall_auroc') is not None)}")


if __name__ == "__main__":
    main()
