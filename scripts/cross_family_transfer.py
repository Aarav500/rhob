#!/usr/bin/env python
"""Cross-family generalization experiment: train on Families 1-6, test on 7-14.

This is the real experiment behind the paper's Cross-Family Transfer Analysis
(Section 6.3, Table ``tab:transfer``). Earlier drafts of that table used
illustrative placeholder numbers; this script produces the real ones from the
actual RHOB v3/v5 environments, families, and detectors.

TEST_FAMILIES was expanded from 3 to 8 families (v1.4): the 5 families added in
v1.4 (reward_channel_tampering, sensor_calibration_tampering, monitored_sandbagging,
eval_probe_sandbagging, rlhf_reward_model_overopt) introduce 3 mechanism types
(REWARD_TAMPERING, DECEPTIVE_ALIGNMENT, RM_OVEROPTIMIZATION) that appear in neither
TRAIN_FAMILIES nor the original 3 test families -- testing transfer to them is a
strictly more meaningful "mechanistically unseen" evaluation than the original split,
which only exercised 2 unseen mechanism types on top of what training already covers.
TRAIN_FAMILIES is unchanged, so "train AUROC" numbers stay comparable to prior results.

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
     held-out Families 7-14 (goal misgeneralization, physics exploitation,
     distributional shift, plus the 5 v1.4 families spanning 3 mechanism
     types unseen during training), which it has never seen in any form.
  4. A Top-5 L2 ensemble (4 stateless behavioral detectors + the frozen
     Trajectory MLP) is evaluated the same way, to measure whether combining
     detectors recovers some of the transfer gap.

Families that do not emit the channel a detector reads are reported as ``null``
(not applicable) and left out of both the pooled fit and the transfer average, rather
than being scored at the detector's no-signal constant. This bites here: two of the
eight test families (monitored_sandbagging, eval_probe_sandbagging) and four of the six
train families (proxy_correlation_gaming, shortcut_exploitation, novelty_farming,
orbit_chirality) ship ``state_counts=None`` -- 25 of the 33 registered families do --
so the L1 State Divergence row used to be part real measurement and part hardcoded 0.5,
and its "train AUROC" was computed over 20 fabricated cells and 5 measured ones.

Run: python scripts/cross_family_transfer.py [--n-seeds-train N] [--n-seeds-test N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import rhob.v3.families  # noqa: F401  -- importing registers every family
from rhob.v3.access import restrict
from rhob.v3.benchmark import Benchmark, missing_channels, offer_population, required_channels
from rhob.v3.provenance import provenance_block, sampling_block
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
    "reward_channel_tampering",
    "sensor_calibration_tampering",
    "monitored_sandbagging",
    "eval_probe_sandbagging",
    "rlhf_reward_model_overopt",
]

TEST_SEED_BASE = 50_000  # disjoint from any training rollout seeds

# Mirrors MatchedPair.rollout (src/rhob/v3/base_pair.py): the legitimate variant is
# rolled out at ``seed_base + 1000 + s``, the hacking variant at ``seed_base + s``.
LEGIT_SEED_OFFSET = 1000

# Both pooled_runs() and transfer_eval() call ``fam.generate_pair(d)`` with
# BaseFamily.generate_pair's default ``seed=0``, so the whole experiment rests on a
# single environment layout per (family, difficulty). Recorded in the sampling block
# rather than left implicit in the source.
LAYOUT_SEEDS = [0]


def pooled_runs(families: list[str], level: str, n_seeds: int, required: tuple[str, ...] = ()):
    """Roll out every default-difficulty pair for ``families``, restricted to ``level``.

    Families that do not emit every channel in ``required`` are skipped: a run with no
    state-visitation histogram contributes nothing to an L1 detector's fit except a
    silently-dropped row (``StateDivergenceDetector.fit`` filters them out anyway), and
    counting it as training data overstates how much data the frozen model saw.
    Returns ``(runs_a, runs_b, skipped_families)``.
    """
    runs_a_all, runs_b_all = [], []
    skipped: list[str] = []
    for fam_name in families:
        fam = FamilyRegistry.get(fam_name)
        fam_a, fam_b = [], []
        for d in fam.default_difficulties():
            pair = fam.generate_pair_at(d)
            runs_a, runs_b, _ = pair.rollout(n_seeds)
            fam_a.extend(restrict(r, level) for r in runs_a)
            fam_b.extend(restrict(r, level) for r in runs_b)
        if required and missing_channels(fam_a + fam_b, required):
            skipped.append(fam_name)
            continue
        runs_a_all.extend(fam_a)
        runs_b_all.extend(fam_b)
    return runs_a_all, runs_b_all, skipped


def transfer_eval(detector, families: list[str], level: str, n_seeds: int):
    """Score a frozen, already-fitted detector on held-out families.

    Returns ``(per_family_auroc dict, average_auroc, na_families)``. A family's entry
    is ``None`` when it does not emit the channel this detector's access level is
    defined over; such families are excluded from the average rather than scored at
    the detector's no-signal constant, which would be a property of the fallback
    branch and not a transfer measurement.

    Some L1/L2 features (e.g. raw state-visitation histograms) are dimensioned by
    a family's own state-space discretization and are not even representationally
    compatible with a different family's discretization -- a stronger failure mode
    than "same representation, different distribution." That is a genuine transfer
    failure (the channel exists, the model just cannot consume this family's version
    of it), so a ``classify`` call that raises on such a structural mismatch is
    treated as carrying zero transferable signal and scored at chance (0.5), rather
    than crashing the experiment. That is a different situation from the channel not
    existing at all, which is what ``None`` above marks.
    """
    required = required_channels(detector)
    per_family: dict[str, float | None] = {}
    na_families: list[str] = []
    n_incompatible = 0
    for fam_name in families:
        fam = FamilyRegistry.get(fam_name)
        cell_aurocs = []
        family_na = False
        for d in fam.default_difficulties():
            pair = fam.generate_pair_at(d)
            runs_a, runs_b, _ = pair.rollout(n_seeds, seed_base=TEST_SEED_BASE)
            restricted_a = [restrict(r, level) for r in runs_a]
            restricted_b = [restrict(r, level) for r in runs_b]
            if missing_channels(restricted_a + restricted_b, required):
                family_na = True
                continue
            # Same unlabeled-population hook Benchmark._evaluate_cell offers, so a
            # detector that orients itself from the cell can do so on a held-out family
            # too. Without it the transfer number would penalize an L2 detector for a
            # coin flip (the family's drawn behavioral sign) rather than for failing to
            # generalize -- and the frozen model's own training families give it no way
            # to know this one's orientation.
            offer_population(detector, restricted_a + restricted_b)
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
        if cell_aurocs:
            per_family[fam_name] = float(np.mean(cell_aurocs))
        elif family_na:
            per_family[fam_name] = None
            na_families.append(fam_name)
        else:
            per_family[fam_name] = float("nan")
    if n_incompatible:
        print(
            f"  [note] {n_incompatible} classify() calls hit a representational "
            f"mismatch (e.g. incompatible state-space dimensionality) and were "
            f"scored at chance (0.5) rather than crashing.",
            flush=True,
        )
    if na_families:
        print(
            f"  [note] not applicable ({'/'.join(required)} absent) and excluded "
            f"from the transfer average: {', '.join(na_families)}",
            flush=True,
        )
    scored = [v for v in per_family.values() if v is not None and v == v]
    avg = float(np.mean(scored)) if scored else float("nan")
    return per_family, avg, na_families


def run_one_multitrial(
    name: str,
    build_detector,
    level: str,
    n_seeds_train: int,
    n_seeds_test: int,
    n_trials: int,
) -> dict:
    """Repeat ``run_one`` across ``n_trials`` independently-seeded model instances
    and report mean/std, not a single draw.

    Neural-net detectors here (Reward MLP, Trajectory MLP) do not fix their
    torch random seed, so weight initialization differs -- often drastically
    so -- from run to run. An earlier version of this experiment reported a
    single unseeded run's transfer AUROC as *the* result; repeating the exact
    same fit procedure 10 times on the Trajectory MLP produced held-out
    AUROCs on distributional_shift ranging from 0.00 to 1.00 (genuinely
    bimodal, not noise around a stable mean). A single-run number for a
    detector with that much seed-sensitivity is not a reproducible
    measurement and should not be reported as one.
    """
    train_aurocs, transfer_aurocs, gaps = [], [], []
    per_family_trials: dict[str, list[float]] = {}
    na_families: list[str] = []
    last_result: dict = {}

    for trial in range(n_trials):
        detector = build_detector(trial)
        result, _ = run_one(name, detector, level, n_seeds_train, n_seeds_test)
        last_result = result
        train_aurocs.append(result["train_auroc"])
        transfer_aurocs.append(result["avg_transfer_auroc"])
        gaps.append(result["generalization_gap_pct"])
        na_families = result["test_families_not_applicable"]
        for fam, v in result["per_family_transfer"].items():
            # ``None`` = the family emits no channel this detector can read. Channel
            # presence is a property of the family, not of the trial's random seed, so
            # such a family has an empty trial list and stays null in the aggregate.
            if v is not None:
                per_family_trials.setdefault(fam, []).append(v)
            else:
                per_family_trials.setdefault(fam, [])

    def _agg(values: list[float | None], fn, digits: int = 3) -> float | None:
        """Aggregate the trials that produced a number; ``None`` if none did."""
        present = [v for v in values if v is not None]
        return round(float(fn(present)), digits) if present else None

    def _show(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.3f}"

    per_family_mean = {k: _agg(v, np.mean) for k, v in per_family_trials.items()}
    per_family_std = {k: _agg(v, np.std) for k, v in per_family_trials.items()}

    print(
        f"  [{name}] across {n_trials} seeded trials: "
        f"train={_show(_agg(train_aurocs, np.mean))}+/-{_show(_agg(train_aurocs, np.std))}  "
        f"transfer={_show(_agg(transfer_aurocs, np.mean))}"
        f"+/-{_show(_agg(transfer_aurocs, np.std))}",
        flush=True,
    )

    return {
        "access_level": level,
        "n_trials": n_trials,
        "train_auroc_mean": _agg(train_aurocs, np.mean),
        "train_auroc_std": _agg(train_aurocs, np.std),
        "train_cells_measured": last_result.get("train_cells_measured"),
        "train_cells_not_applicable": last_result.get("train_cells_not_applicable"),
        "train_families_not_applicable": last_result.get("train_families_not_applicable", []),
        "train_families_excluded_from_pooled_fit": last_result.get(
            "train_families_excluded_from_pooled_fit", []
        ),
        "per_family_transfer_mean": per_family_mean,
        "per_family_transfer_std": per_family_std,
        "test_families_not_applicable": na_families,
        "avg_transfer_auroc_mean": _agg(transfer_aurocs, np.mean),
        "avg_transfer_auroc_std": _agg(transfer_aurocs, np.std),
        "generalization_gap_pct_mean": _agg(gaps, np.mean, digits=1),
        "all_trial_transfer_aurocs": [_round_or_none(v) for v in transfer_aurocs],
    }


def _round_or_none(value: float | None, digits: int = 3) -> float | None:
    """Round for the artifact; ``None`` (not applicable) and NaN both become ``null``."""
    if value is None or value != value:
        return None
    return round(float(value), digits)


def run_one(name: str, detector, level: str, n_seeds_train: int, n_seeds_test: int) -> dict:
    print(f"=== {name} ({level}) ===", flush=True)
    t0 = time.time()

    train_bench = Benchmark.evaluate(
        detector, families=TRAIN_FAMILIES, difficulties="all", n_seeds=n_seeds_train, verbose=False
    )
    train_auroc = train_bench.overall_auroc
    na_train = train_bench.na_families
    note = f"  ({len(na_train)} train families N/A: {', '.join(na_train)})" if na_train else ""
    print(
        f"  in-distribution (5-fold CV) train AUROC: {train_auroc:.3f} "
        f"over {len(train_bench.scored_cells)} measured cells{note}  [{time.time()-t0:.0f}s]",
        flush=True,
    )

    required = required_channels(detector)
    train_a, train_b, skipped_train = pooled_runs(TRAIN_FAMILIES, level, n_seeds_train, required)
    if skipped_train:
        print(
            f"  [note] excluded from the pooled fit ({'/'.join(required)} absent): "
            f"{', '.join(skipped_train)}",
            flush=True,
        )
    if hasattr(detector, "fit"):
        try:
            detector.fit(train_a, train_b)
        except (ValueError, RuntimeError) as e:
            print(f"  [warn] pooled fit on Families 1-6 failed ({e}); "
                  f"detector will score at chance for the transfer step.", flush=True)

    per_family, avg_transfer, na_test = transfer_eval(detector, TEST_FAMILIES, level, n_seeds_test)
    gap = (train_auroc - avg_transfer) / train_auroc * 100 if train_auroc else float("nan")

    shown = {k: ("N/A" if v is None else round(v, 3)) for k, v in per_family.items()}
    print(f"  transfer per-family: {shown}", flush=True)
    print(f"  avg transfer AUROC: {avg_transfer:.3f}   gap: {gap:.1f}%   [{time.time()-t0:.0f}s total]", flush=True)

    return {
        "access_level": level,
        "train_auroc": _round_or_none(train_auroc),
        "train_cells_measured": len(train_bench.scored_cells),
        "train_cells_not_applicable": len(train_bench.na_cells),
        "train_families_not_applicable": na_train,
        "train_families_excluded_from_pooled_fit": skipped_train,
        "per_family_transfer": {k: _round_or_none(v) for k, v in per_family.items()},
        "test_families_not_applicable": na_test,
        "avg_transfer_auroc": _round_or_none(avg_transfer),
        "generalization_gap_pct": _round_or_none(gap, 1),
    }, detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds-train", type=int, default=10)
    parser.add_argument("--n-seeds-test", type=int, default=20)
    parser.add_argument(
        "--n-trials", type=int, default=5,
        help="Independent model-init trials for neural-net detectors (Reward MLP, "
             "Trajectory MLP, Ensemble). Their torch weight init is not seeded to a "
             "fixed value across trials, so a single trial is not a reproducible "
             "measurement -- see run_one_multitrial.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("leaderboard/cross_family_transfer.json"),
        help="Output JSON path. Point this elsewhere for smoke runs so the published "
             "artifact is not overwritten by a reduced-seed evaluation.",
    )
    parser.add_argument(
        "--only", nargs="+", default=None,
        choices=["Reward MLP", "State Divergence", "Trajectory MLP", "Ensemble (Top 5)"],
        help="Run only these detector rows (smoke testing). Default: all four.",
    )
    args = parser.parse_args()
    selected = set(args.only) if args.only else None

    def wanted(row: str) -> bool:
        return selected is None or row in selected

    results: dict[str, dict] = {}

    if wanted("Reward MLP"):
        results["Reward MLP"] = run_one_multitrial(
            "Reward MLP",
            lambda trial: RewardMLPDetector(seed=trial),
            "L0", args.n_seeds_train, args.n_seeds_test, args.n_trials,
        )

    if wanted("State Divergence"):
        state_div_result, _ = run_one(
            "State Divergence", StateDivergenceDetector(), "L1", args.n_seeds_train, args.n_seeds_test
        )
        results["State Divergence"] = state_div_result

    if wanted("Trajectory MLP"):
        results["Trajectory MLP"] = run_one_multitrial(
            "Trajectory MLP",
            lambda trial: TrajectoryMLPDetector(seed=trial),
            "L2", args.n_seeds_train, args.n_seeds_test, args.n_trials,
        )

    def build_ensemble(trial: int):
        traj_mlp = TrajectoryMLPDetector(seed=trial)
        train_a, train_b, _ = pooled_runs(
            TRAIN_FAMILIES, "L2", args.n_seeds_train, required_channels(traj_mlp)
        )
        traj_mlp.fit(train_a, train_b)
        members = [
            BehavioralThresholdDetector(),
            AngularMomentumDetector(),
            CentroidTrackerDetector(),
            FeatureMagnitudeDetector(),
            traj_mlp,
        ]
        return EnsembleDetector(members, name="Ensemble (Top 5)")

    if wanted("Ensemble (Top 5)"):
        results["Ensemble (Top 5)"] = run_one_multitrial(
            "Ensemble (Top 5)", build_ensemble, "L2", args.n_seeds_train, args.n_seeds_test,
            args.n_trials,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "train_families": TRAIN_FAMILIES,
                "test_families": TEST_FAMILIES,
                "n_seeds_train": args.n_seeds_train,
                "n_seeds_test": args.n_seeds_test,
                "n_trials": args.n_trials,
                "provenance": provenance_block(script="scripts/cross_family_transfer.py"),
                # Train and test draw from disjoint seed bases (TEST_SEED_BASE), so the
                # held-out evaluation never reuses a rollout the model was fitted on.
                # ``n_replicates=1``: the *environment* draw is a single one, shared by
                # every trial and every detector. ``model_init_trials`` is the separate
                # axis this experiment does replicate -- independently-seeded neural-net
                # weight initializations (see run_one_multitrial). Replicating model init
                # says nothing about sampling error in the environment draw.
                "sampling": sampling_block(
                    n_seeds=args.n_seeds_train,
                    n_layouts=len(LAYOUT_SEEDS),
                    layout_seeds=LAYOUT_SEEDS,
                    rollout_seeds_hacking=list(range(args.n_seeds_train)),
                    rollout_seeds_legit=[
                        LEGIT_SEED_OFFSET + s for s in range(args.n_seeds_train)
                    ],
                    n_replicates=1,
                    model_init_trials=args.n_trials,
                    test_seed_base=TEST_SEED_BASE,
                    n_seeds_test=args.n_seeds_test,
                    rollout_seeds_hacking_test=[
                        TEST_SEED_BASE + s for s in range(args.n_seeds_test)
                    ],
                    rollout_seeds_legit_test=[
                        TEST_SEED_BASE + LEGIT_SEED_OFFSET + s for s in range(args.n_seeds_test)
                    ],
                ),
                "cell_semantics": {
                    "per_family_null": (
                        "not applicable -- the family does not emit the RunData channel "
                        "this detector's access level is defined over. Excluded from the "
                        "transfer average and from the pooled fit; never imputed as 0.5."
                    ),
                },
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\n[OK] Saved to {out_path}")


if __name__ == "__main__":
    main()
