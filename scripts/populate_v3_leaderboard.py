"""Populate the v3 leaderboard with the full baseline-detector suite.

Rolls out each registered (family, difficulty) pair *once* and reuses those runs
across every detector (re-rolling out per detector would be ~14x more expensive for
no benefit, since the underlying runs don't depend on which detector reads them).
Writes ``leaderboard/{README,by_mechanism,by_difficulty}.md`` plus the
machine-readable ``leaderboard.json``.
"""

from __future__ import annotations

from pathlib import Path

from rhob.detectors import (
    BehavioralThresholdDetector,
    EnsembleDetector,
    PerfectFeatureOracleDetector,
    RewardCUSUMDetector,
    RewardKDEDetector,
    RewardMLPDetector,
    RewardThresholdDetector,
    RewardVarianceRatioDetector,
    SpectralRewardDetector,
    StateCoverageRateDetector,
    StateDivergenceDetector,
    TrajectoryMLPDetector,
    TrueRewardOracleDetector,
    VisitationEntropyTrendDetector,
)
from rhob.v3.benchmark import BenchmarkResults, CellResult, _evaluate_cell
from rhob.v3.leaderboard.board import Leaderboard
from rhob.v3.registry import FamilyRegistry

import os

OUT_DIR = Path("leaderboard")
# Reduced from the full 20-seed x all-5-tiers sweep to keep this tractable in a single
# session: 2 continuous difficulty tiers (easy + hard) instead of 4, 10 seeds/variant
# instead of 20. Still exercises every family, every detector, and the full difficulty
# range end to end; `Benchmark.evaluate(..., n_seeds=20, difficulties="all")` is the
# complete sweep for a production leaderboard run.
N_SEEDS = 10
CONTINUOUS_DIFFICULTIES = [0.90, 0.70]
# Set RHOB_SKIP_CONTINUOUS=1 to populate gridworld-only (fast, no DQN training) --
# useful when continuous-tier DQN training is impractically slow in the current
# environment; re-run without the flag later to add the continuous-tier rows.
SKIP_CONTINUOUS = os.environ.get("RHOB_SKIP_CONTINUOUS") == "1"


def main() -> None:
    detectors = [
        RewardThresholdDetector(),
        RewardCUSUMDetector(),
        RewardVarianceRatioDetector(),
        RewardKDEDetector(),
        SpectralRewardDetector(),
        RewardMLPDetector(window_size=50),
        StateDivergenceDetector(baseline_episodes=50, steady_window=100),
        VisitationEntropyTrendDetector(baseline_episodes=50, late_window=100),
        StateCoverageRateDetector(baseline_episodes=50, late_window=100),
        BehavioralThresholdDetector(),
        TrajectoryMLPDetector(window=100),
        EnsembleDetector(
            [BehavioralThresholdDetector(steady_window=100), BehavioralThresholdDetector(steady_window=50)],
            name="Ensemble(BehavioralThreshold x2)",
        ),
        TrueRewardOracleDetector(),
        PerfectFeatureOracleDetector(),
    ]

    print("Generating the pair suite (rolled out once, shared across all detectors) ...")
    gridworld_pairs = FamilyRegistry.generate_suite("gridworld_camping", "all")
    continuous_pairs = [] if SKIP_CONTINUOUS else FamilyRegistry.generate_suite(
        "continuous_camping", CONTINUOUS_DIFFICULTIES
    )
    if SKIP_CONTINUOUS:
        print("  RHOB_SKIP_CONTINUOUS=1: skipping continuous_camping (DQN training deferred)")
    pairs = gridworld_pairs + continuous_pairs
    rolled_out = []
    for pair in pairs:
        print(f"  rolling out {pair.family} @ target-L2={pair.difficulty:.2f} ...")
        runs_a, runs_b, onsets_a = pair.rollout(N_SEEDS)
        rolled_out.append((pair, runs_a, runs_b, onsets_a))
    print(f"Done: {len(rolled_out)} pairs rolled out.\n")

    board = Leaderboard()
    for detector in detectors:
        print(f"=== {detector.name} ({detector.access_level}) ===")
        results = BenchmarkResults(detector_name=detector.name, access_level=detector.access_level)
        for pair, runs_a, runs_b, onsets_a in rolled_out:
            auroc, mae = _evaluate_cell(detector, runs_a, runs_b, onsets_a, detector.access_level, pair.n_episodes)
            results.cells.append(
                CellResult(pair.family, pair.mechanism.value, pair.difficulty, auroc, mae, N_SEEDS)
            )
            print(f"  {pair.family} @ L2*={pair.difficulty:.2f}: AUROC={auroc:.3f}  onsetMAE={mae:.3f}")
        results.summary()
        board.submit(results, author="rhob-baselines")
        print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    board.save(OUT_DIR / "leaderboard.json")
    (OUT_DIR / "README.md").write_text("# RHOB v3 Leaderboard\n\n" + board.render_standings_md() + "\n")
    (OUT_DIR / "by_mechanism.md").write_text(board.render_by_mechanism_md() + "\n")
    (OUT_DIR / "by_difficulty.md").write_text(board.render_by_difficulty_md() + "\n")
    print(f"Leaderboard written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
