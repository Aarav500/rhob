#!/usr/bin/env python
"""Cross-family generalization experiment: train on Families 1-6, test on 7-9.

This is the real experiment behind the paper's Cross-Family Transfer Analysis
(Section 6.3, Table ``tab:transfer``). Earlier drafts of that table used
illustrative placeholder numbers; this script produces the real ones from the
actual RHOB v3/v5 environments, families, and detectors.

Methodology:
  1. "Train" AUROC: in-distribution performance of each detector on Families
     1-6, measured with the same 5-fold stratified cross-validation used
     throughout the paper (via ``Benchmark.evaluate``), so it is directly
     comparable to every other in-distribution number reported.
  2. Each detector is then fit ONCE on the full pool of Families 1-6 runs
     (all default difficulties, all seeds) to produce a single frozen,
     deployable model -- standard practice: cross-validation is for honest
     scoring, the shipped model uses all available training data.
  3. That frozen model is evaluated -- with no further fitting -- on
     held-out Families 7-9 (goal misgeneralization, physics exploitation,
     distributional shift), which it has never seen in any form.
  4. A Top-5 L2 ensemble (4 stateless behavioral detectors + the frozen
     Trajectory MLP) is evaluated the same way, to measure whether combining
     detectors recovers some of the transfer gap.

Run: python scripts/cross_family_transfer.py [--n-seeds-train N] [--n-seeds-test N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import rhob.v3.families  # noqa: F401  -- importing registers all 9 families
from rhob.v3.access import restrict
from rhob.v3.benchmark import Benchmark
from rhob.v3.registry import FamilyRegistry

from rhob.detectors.l0_reward_mlp import RewardMLPDetector
from rhob.detectors.l1_state_divergence import StateDivergenceDetector
from rhob.detectors.l2_trajectory_mlp import TrajectoryMLPDetector
from rhob.detectors.l2_behavioral_threshold import BehavioralThresholdDetector
from rhob.detectors.l2_angular_momentum import AngularMomentumDetector
from rhob.detectors.l2_centroid_tracker import CentroidTrackerDetector
from rhob.detectors.l2_feature_magnitude import FeatureMagnitudeDetector
from rhob.detectors.l2_ensemble import EnsembleDetector

TRAIN_FAMILIES = [
    "gridworld_camping",
    "continuous_camping",
    "proxy_correlation_gaming",
    "shortcut_exploitation",
    "novelty_farming",
    "orbit_chirality",
]
TEST_FAMILIES = [
    "goal_misgeneralization",
    "physics_exploitation",
    "distributional_shift",
]

TEST_SEED_BASE = 50_000  # disjoint from any training rollout seeds


def pooled_runs(families: list[str], level: str, n_seeds: int):
    """Roll out every default-difficulty pair for ``families``, restricted to ``level``."""
    runs_a_all, runs_b_all = [], []
    for fam_name in families:
        fam = FamilyRegistry.get(fam_name)
        for d in fam.default_difficulties():
            pair = fam.generate_pair(d)
            runs_a, runs_b, _ = pair.rollout(n_seeds)
            runs_a_all.extend(restrict(r, level) for r in runs_a)
            runs_b_all.extend(restrict(r, level) for r in runs_b)
    return runs_a_all, runs_b_all


def transfer_eval(detector, families: list[str], level: str, n_seeds: int):
    """Score a frozen, already-fitted detector on held-out families.

    Returns (per_family_auroc dict, average_auroc).

    Some L1/L2 features (e.g. raw state-visitation histograms) are dimensioned by
    a family's own state-space discretization and are not even representationally
    compatible with a different family's discretization -- a stronger failure mode
    than "same representation, different distribution." A ``classify`` call that
    raises on such a structural mismatch is treated as carrying zero transferable
    signal and scored at chance (0.5), rather than crashing the experiment.
    """
    per_family = {}
    n_incompatible = 0
    for fam_name in families:
        fam = FamilyRegistry.get(fam_name)
        cell_aurocs = []
        for d in fam.default_difficulties():
            pair = fam.generate_pair(d)
            runs_a, runs_b, _ = pair.rollout(n_seeds, seed_base=TEST_SEED_BASE)
            restricted_a = [restrict(r, level) for r in runs_a]
            restricted_b = [restrict(r, level) for r in runs_b]
            labels = np.array([1] * len(restricted_a) + [0] * len(restricted_b))
            scores = []
            for r in restricted_a + restricted_b:
                try:
                    scores.append(detector.classify(r))
                except (ValueError, RuntimeError):
                    n_incompatible += 1
                    scores.append(0.5)
            if len(set(labels)) > 1:
                cell_aurocs.append(float(roc_auc_score(labels, scores)))
        per_family[fam_name] = float(np.mean(cell_aurocs)) if cell_aurocs else float("nan")
    if n_incompatible:
        print(
            f"  [note] {n_incompatible} classify() calls hit a representational "
            f"mismatch (e.g. incompatible state-space dimensionality) and were "
            f"scored at chance (0.5) rather than crashing.",
            flush=True,
        )
    avg = float(np.mean([v for v in per_family.values() if v == v]))
    return per_family, avg


def run_one(name: str, detector, level: str, n_seeds_train: int, n_seeds_test: int) -> dict:
    print(f"=== {name} ({level}) ===", flush=True)
    t0 = time.time()

    train_bench = Benchmark.evaluate(
        detector, families=TRAIN_FAMILIES, difficulties="all", n_seeds=n_seeds_train, verbose=False
    )
    train_auroc = train_bench.overall_auroc
    print(f"  in-distribution (5-fold CV) train AUROC: {train_auroc:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    train_a, train_b = pooled_runs(TRAIN_FAMILIES, level, n_seeds_train)
    if hasattr(detector, "fit"):
        try:
            detector.fit(train_a, train_b)
        except (ValueError, RuntimeError) as e:
            print(f"  [warn] pooled fit on Families 1-6 failed ({e}); "
                  f"detector will score at chance for the transfer step.", flush=True)

    per_family, avg_transfer = transfer_eval(detector, TEST_FAMILIES, level, n_seeds_test)
    gap = (train_auroc - avg_transfer) / train_auroc * 100 if train_auroc else float("nan")

    print(f"  transfer per-family: { {k: round(v, 3) for k, v in per_family.items()} }", flush=True)
    print(f"  avg transfer AUROC: {avg_transfer:.3f}   gap: {gap:.1f}%   [{time.time()-t0:.0f}s total]", flush=True)

    return {
        "access_level": level,
        "train_auroc": round(train_auroc, 3),
        "per_family_transfer": {k: round(v, 3) for k, v in per_family.items()},
        "avg_transfer_auroc": round(avg_transfer, 3),
        "generalization_gap_pct": round(gap, 1),
    }, detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds-train", type=int, default=10)
    parser.add_argument("--n-seeds-test", type=int, default=20)
    args = parser.parse_args()

    results: dict[str, dict] = {}

    reward_mlp_result, _ = run_one(
        "Reward MLP", RewardMLPDetector(), "L0", args.n_seeds_train, args.n_seeds_test
    )
    results["Reward MLP"] = reward_mlp_result

    state_div_result, _ = run_one(
        "State Divergence", StateDivergenceDetector(), "L1", args.n_seeds_train, args.n_seeds_test
    )
    results["State Divergence"] = state_div_result

    traj_mlp_result, fitted_traj_mlp = run_one(
        "Trajectory MLP", TrajectoryMLPDetector(), "L2", args.n_seeds_train, args.n_seeds_test
    )
    results["Trajectory MLP"] = traj_mlp_result

    print("=== Ensemble (Top 5, L2) ===", flush=True)
    t0 = time.time()
    members = [
        BehavioralThresholdDetector(),
        AngularMomentumDetector(),
        CentroidTrackerDetector(),
        FeatureMagnitudeDetector(),
        fitted_traj_mlp,  # already fit on pooled Families 1-6 above
    ]
    ensemble = EnsembleDetector(members, name="Ensemble (Top 5)")

    train_bench = Benchmark.evaluate(
        ensemble, families=TRAIN_FAMILIES, difficulties="all", n_seeds=args.n_seeds_train, verbose=False
    )
    train_auroc = train_bench.overall_auroc
    per_family, avg_transfer = transfer_eval(ensemble, TEST_FAMILIES, "L2", args.n_seeds_test)
    gap = (train_auroc - avg_transfer) / train_auroc * 100 if train_auroc else float("nan")
    print(f"  train AUROC: {train_auroc:.3f}  transfer: {avg_transfer:.3f}  gap: {gap:.1f}%  [{time.time()-t0:.0f}s]", flush=True)

    results["Ensemble (Top 5)"] = {
        "access_level": "L2",
        "train_auroc": round(train_auroc, 3),
        "per_family_transfer": {k: round(v, 3) for k, v in per_family.items()},
        "avg_transfer_auroc": round(avg_transfer, 3),
        "generalization_gap_pct": round(gap, 1),
    }

    out_path = Path("leaderboard/cross_family_transfer.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "train_families": TRAIN_FAMILIES,
                "test_families": TEST_FAMILIES,
                "n_seeds_train": args.n_seeds_train,
                "n_seeds_test": args.n_seeds_test,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\n[OK] Saved to {out_path}")


if __name__ == "__main__":
    main()
