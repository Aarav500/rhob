"""Evaluate all 6 baseline detectors on all 4 admitted pairs.

Produces the paper's main results: a 6x4 discrimination-AUROC table and a 6x4
onset-timing-MAE table, plus validation checks confirming the information
barrier (L0 near chance) and behavioral sufficiency (L2 matches admission).
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rhob.detectors import (  # noqa: E402
    BehavioralThresholdDetector,
    RewardCUSUMDetector,
    RewardMLPDetector,
    RewardThresholdDetector,
    StateDivergenceDetector,
    TrajectoryMLPDetector,
)
from rhob.detectors.posthoc import PosthocDetector, RunData  # noqa: E402
from generate_eval_data import PairEvalData  # noqa: E402

L0_BARRIER_MAX = 0.65
# This evaluation draws its own 20-per-class sample (seeds 10_000+/20_000+), independent
# of the admission certificate's sample (seeds 0-39/1000-1039) -- same population, different
# draw. At n=20/class, AUROC's sampling SE is ~sqrt(p(1-p)/n) ~ 0.08-0.09 for p in [0.7, 0.9],
# so the tolerance must be a consistency check against sampling noise, not exact agreement.
L2_CONSISTENCY_TOL = 0.12
CV_FOLDS = 5


@dataclass
class DetectorResult:
    detector_name: str
    access_level: str
    pair_id: str
    discrimination_auroc: float
    onset_mae: float
    n_runs: int
    n_onset_correct: int


def _stratified_folds(n_a: int, n_b: int, k: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return k (train_idx, test_idx) splits over indices [0, n_a+n_b), stratified by class.

    Indices [0, n_a) are class A, [n_a, n_a+n_b) are class B.
    """
    rng = np.random.default_rng(seed)
    idx_a = rng.permutation(n_a)
    idx_b = n_a + rng.permutation(n_b)
    folds_a = np.array_split(idx_a, k)
    folds_b = np.array_split(idx_b, k)

    splits = []
    all_idx = np.arange(n_a + n_b)
    for i in range(k):
        test_idx = np.concatenate([folds_a[i], folds_b[i]])
        train_idx = np.setdiff1d(all_idx, test_idx)
        splits.append((train_idx, test_idx))
    return splits


def _simple_discrimination(detector: PosthocDetector, runs_a: list[RunData], runs_b: list[RunData]) -> float:
    scores = [detector.classify(r) for r in runs_a] + [detector.classify(r) for r in runs_b]
    labels = [1] * len(runs_a) + [0] * len(runs_b)
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _simple_onset(detector: PosthocDetector, runs_a: list[RunData], onsets_true: list[int], n_episodes: int) -> tuple[float, int]:
    errors = []
    n_correct = 0
    for run, true_onset in zip(runs_a, onsets_true):
        pred = detector.detect_onset(run)
        if pred < 0 or true_onset < 0:
            errors.append(1.0)
        else:
            err = abs(pred - true_onset) / n_episodes
            errors.append(err)
            if abs(pred - true_onset) <= max(3, n_episodes // 100):
                n_correct += 1
    return float(np.mean(errors)) if errors else float("nan"), n_correct


def _cv_reward_mlp(
    runs_a: list[RunData], runs_b: list[RunData], onsets_a: list[int], n_episodes: int, window_size: int
) -> tuple[float, float, int]:
    """5-fold CV for RewardMLPDetector: returns (auroc, onset_mae, n_onset_correct)."""
    n_a, n_b = len(runs_a), len(runs_b)
    all_runs = runs_a + runs_b
    labels = np.array([1] * n_a + [0] * n_b)
    oof_scores = np.full(n_a + n_b, np.nan)
    onset_preds: dict[int, int] = {}

    for train_idx, test_idx in _stratified_folds(n_a, n_b, CV_FOLDS):
        train_a = [all_runs[i] for i in train_idx if labels[i] == 1]
        train_b = [all_runs[i] for i in train_idx if labels[i] == 0]
        detector = RewardMLPDetector(window_size=window_size)
        detector.fit(train_a, train_b, epochs=80)
        for i in test_idx:
            oof_scores[i] = detector.classify(all_runs[i])
            if i < n_a:  # onset only meaningful for Variant A
                onset_preds[i] = detector.detect_onset(all_runs[i])

    auroc = float(roc_auc_score(labels, oof_scores)) if len(set(labels)) > 1 else float("nan")

    errors, n_correct = [], 0
    for i in range(n_a):
        pred = onset_preds.get(i, -1)
        true_onset = onsets_a[i]
        if pred < 0 or true_onset < 0:
            errors.append(1.0)
        else:
            err = abs(pred - true_onset) / n_episodes
            errors.append(err)
            if abs(pred - true_onset) <= max(3, n_episodes // 100):
                n_correct += 1
    onset_mae = float(np.mean(errors)) if errors else float("nan")
    return auroc, onset_mae, n_correct


def _cv_state_divergence(
    runs_a: list[RunData], runs_b: list[RunData], onsets_a: list[int], n_episodes: int, baseline_episodes: int, steady_window: int
) -> tuple[float, float, int]:
    """5-fold CV for StateDivergenceDetector: returns (auroc, onset_mae, n_onset_correct)."""
    n_a, n_b = len(runs_a), len(runs_b)
    all_runs = runs_a + runs_b
    labels = np.array([1] * n_a + [0] * n_b)
    oof_scores = np.full(n_a + n_b, np.nan)
    onset_preds: dict[int, int] = {}

    for train_idx, test_idx in _stratified_folds(n_a, n_b, CV_FOLDS):
        train_a = [all_runs[i] for i in train_idx if labels[i] == 1]
        train_b = [all_runs[i] for i in train_idx if labels[i] == 0]
        detector = StateDivergenceDetector(baseline_episodes=baseline_episodes, steady_window=steady_window)
        detector.fit(train_a, train_b)
        for i in test_idx:
            oof_scores[i] = detector.classify(all_runs[i])
            if i < n_a:
                onset_preds[i] = detector.detect_onset(all_runs[i])  # oracle-free, doesn't need fit()

    auroc = float(roc_auc_score(labels, oof_scores)) if len(set(labels)) > 1 else float("nan")

    errors, n_correct = [], 0
    for i in range(n_a):
        pred = onset_preds.get(i, -1)
        true_onset = onsets_a[i]
        if pred < 0 or true_onset < 0:
            errors.append(1.0)
        else:
            err = abs(pred - true_onset) / n_episodes
            errors.append(err)
            if abs(pred - true_onset) <= max(3, n_episodes // 100):
                n_correct += 1
    onset_mae = float(np.mean(errors)) if errors else float("nan")
    return auroc, onset_mae, n_correct


def _cv_trajectory_mlp(
    runs_a: list[RunData], runs_b: list[RunData], onsets_a: list[int], n_episodes: int, window: int
) -> tuple[float, float, int]:
    """5-fold CV for TrajectoryMLPDetector: returns (auroc, onset_mae, n_onset_correct)."""
    n_a, n_b = len(runs_a), len(runs_b)
    all_runs = runs_a + runs_b
    labels = np.array([1] * n_a + [0] * n_b)
    oof_scores = np.full(n_a + n_b, np.nan)
    onset_preds: dict[int, int] = {}

    for train_idx, test_idx in _stratified_folds(n_a, n_b, CV_FOLDS):
        train_a = [all_runs[i] for i in train_idx if labels[i] == 1]
        train_b = [all_runs[i] for i in train_idx if labels[i] == 0]
        detector = TrajectoryMLPDetector(window=window)
        detector.fit(train_a, train_b, epochs=80)
        for i in test_idx:
            oof_scores[i] = detector.classify(all_runs[i])
            if i < n_a:
                onset_preds[i] = detector.detect_onset(all_runs[i])

    auroc = float(roc_auc_score(labels, oof_scores)) if len(set(labels)) > 1 else float("nan")

    errors, n_correct = [], 0
    for i in range(n_a):
        pred = onset_preds.get(i, -1)
        true_onset = onsets_a[i]
        if pred < 0 or true_onset < 0:
            errors.append(1.0)
        else:
            err = abs(pred - true_onset) / n_episodes
            errors.append(err)
            if abs(pred - true_onset) <= max(3, n_episodes // 100):
                n_correct += 1
    onset_mae = float(np.mean(errors)) if errors else float("nan")
    return auroc, onset_mae, n_correct


def evaluate_pair(pair_data: PairEvalData) -> list[DetectorResult]:
    """Run all 6 detectors on one pair's evaluation data."""
    n_ep = pair_data.n_episodes
    is_gridworld = "pair_01" in pair_data.pair_id
    # Window sizes scaled to each pair's episode budget: gridworld has 500
    # episodes (activation at 250); continuous pairs have 40 (activation at 10).
    # `steady` for continuous matches ContinuousConfig.steady_window (25) exactly --
    # the same trailing window the admission gate itself uses to certify L2, so the
    # L2-consistency check compares like with like rather than adding avoidable
    # window-size noise on top of the admission measurement.
    window = 50 if is_gridworld else 8
    baseline = 50 if is_gridworld else 10
    steady = 100 if is_gridworld else 25

    results = []

    # --- L0 ---
    rt = RewardThresholdDetector(baseline_episodes=baseline, late_window=steady)
    rc = RewardCUSUMDetector(baseline_window=baseline)
    for detector in (rt, rc):
        auroc = _simple_discrimination(detector, pair_data.runs_a, pair_data.runs_b)
        mae, n_correct = _simple_onset(detector, pair_data.runs_a, pair_data.onsets_a, n_ep)
        results.append(
            DetectorResult(detector.name, detector.access_level, pair_data.pair_id, auroc, mae, len(pair_data.runs_a), n_correct)
        )

    mlp_auroc, mlp_mae, mlp_correct = _cv_reward_mlp(
        pair_data.runs_a, pair_data.runs_b, pair_data.onsets_a, n_ep, window_size=window
    )
    results.append(
        DetectorResult("Reward MLP", "L0", pair_data.pair_id, mlp_auroc, mlp_mae, len(pair_data.runs_a), mlp_correct)
    )

    # --- L1 ---
    sd_auroc, sd_mae, sd_correct = _cv_state_divergence(
        pair_data.runs_a, pair_data.runs_b, pair_data.onsets_a, n_ep,
        baseline_episodes=baseline, steady_window=steady,
    )
    results.append(
        DetectorResult("State Divergence", "L1", pair_data.pair_id, sd_auroc, sd_mae, len(pair_data.runs_a), sd_correct)
    )

    # --- L2 ---
    bt = BehavioralThresholdDetector(steady_window=steady, baseline_episodes=baseline)
    auroc = _simple_discrimination(bt, pair_data.runs_a, pair_data.runs_b)
    mae, n_correct = _simple_onset(bt, pair_data.runs_a, pair_data.onsets_a, n_ep)
    results.append(DetectorResult(bt.name, bt.access_level, pair_data.pair_id, auroc, mae, len(pair_data.runs_a), n_correct))

    traj_auroc, traj_mae, traj_correct = _cv_trajectory_mlp(
        pair_data.runs_a, pair_data.runs_b, pair_data.onsets_a, n_ep, window=window
    )
    results.append(
        DetectorResult("Trajectory MLP", "L2", pair_data.pair_id, traj_auroc, traj_mae, len(pair_data.runs_a), traj_correct)
    )

    return results


def run_validation_checks(all_results: list[DetectorResult], admission_l2: dict[str, float]) -> list[str]:
    """Check the L0 barrier, L2 consistency, and monotonicity. Returns a list of findings."""
    findings = []

    l0_results = [r for r in all_results if r.access_level == "L0"]
    l0_leaks = [r for r in l0_results if not np.isnan(r.discrimination_auroc) and r.discrimination_auroc > L0_BARRIER_MAX]
    if l0_leaks:
        for r in l0_leaks:
            findings.append(f"L0 BARRIER FAILED: {r.detector_name} on {r.pair_id} scored AUROC={r.discrimination_auroc:.3f} (>{L0_BARRIER_MAX})")
    else:
        findings.append(f"L0 barrier holds: all L0 detectors scored AUROC <= {L0_BARRIER_MAX} on all pairs.")

    bt_results = [r for r in all_results if r.detector_name == "Behavioral Threshold"]
    for r in bt_results:
        ref = admission_l2.get(r.pair_id)
        if ref is None:
            continue
        diff = abs(r.discrimination_auroc - ref)
        if diff > L2_CONSISTENCY_TOL:
            findings.append(f"L2 CONSISTENCY FAILED: {r.pair_id} Behavioral Threshold AUROC={r.discrimination_auroc:.3f} vs admission {ref:.3f} (diff={diff:.3f})")
        else:
            findings.append(f"L2 consistent on {r.pair_id}: Behavioral Threshold={r.discrimination_auroc:.3f} vs admission={ref:.3f} (diff={diff:.3f})")

    order = ["tier1/pair_01_gridworld", "tier2/pair_02_easy", "tier2/pair_03_medium", "tier2/pair_04_hard"]
    for detector_name in ("Behavioral Threshold", "Trajectory MLP"):
        vals = []
        for pid in order:
            match = [r.discrimination_auroc for r in all_results if r.detector_name == detector_name and r.pair_id == pid]
            if match:
                vals.append((pid, match[0]))
        is_monotone = all(vals[i][1] >= vals[i + 1][1] - 0.02 for i in range(len(vals) - 1))  # small tolerance
        findings.append(
            f"{'Monotone' if is_monotone else 'NOT MONOTONE'} degradation for {detector_name}: "
            + " >= ".join(f"{pid.split('/')[-1]}={v:.3f}" for pid, v in vals)
        )

    return findings


def write_results_csv(all_results: list[DetectorResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["detector", "access_level", "pair_id", "discrimination_auroc", "onset_mae", "n_runs", "n_onset_correct"])
        for r in all_results:
            writer.writerow([r.detector_name, r.access_level, r.pair_id, f"{r.discrimination_auroc:.4f}", f"{r.onset_mae:.4f}", r.n_runs, r.n_onset_correct])


def format_table(all_results: list[DetectorResult], value_attr: str) -> str:
    pairs = ["tier1/pair_01_gridworld", "tier2/pair_02_easy", "tier2/pair_03_medium", "tier2/pair_04_hard"]
    detector_order = ["Reward Threshold", "Reward CUSUM", "Reward MLP", "State Divergence", "Behavioral Threshold", "Trajectory MLP"]

    header = f"{'Detector':<22}" + "".join(f"{p.split('/')[-1]:>18}" for p in pairs)
    lines = [header, "-" * len(header)]
    for det in detector_order:
        row = f"{det:<22}"
        for pid in pairs:
            match = [getattr(r, value_attr) for r in all_results if r.detector_name == det and r.pair_id == pid]
            row += f"{match[0]:>18.3f}" if match else f"{'N/A':>18}"
        lines.append(row)
    return "\n".join(lines)


def main() -> None:
    from generate_eval_data import ADMISSION_L2, generate_all_pairs

    print("Generating evaluation data for all 4 pairs (this trains a fresh tabular agent")
    print("per gridworld run and one shared DQN camper for the continuous tier) ...\n")
    data = generate_all_pairs(n_runs=20)

    all_results: list[DetectorResult] = []
    for pair_id, pair_data in data.items():
        print(f"\nEvaluating detectors on {pair_id} ...")
        results = evaluate_pair(pair_data)
        all_results.extend(results)
        for r in results:
            print(f"  {r.detector_name:<22} ({r.access_level})  AUROC={r.discrimination_auroc:.3f}  onset_MAE={r.onset_mae:.3f}")

    out_dir = Path("results/detector_evaluation")
    write_results_csv(all_results, out_dir / "results_table.csv")

    print("\n" + "=" * 90)
    print("DISCRIMINATION AUROC")
    print("=" * 90)
    print(format_table(all_results, "discrimination_auroc"))

    print("\n" + "=" * 90)
    print("ONSET TIMING MAE (normalized by episode budget)")
    print("=" * 90)
    print(format_table(all_results, "onset_mae"))

    findings = run_validation_checks(all_results, ADMISSION_L2)
    print("\n" + "=" * 90)
    print("VALIDATION CHECKS")
    print("=" * 90)
    for f in findings:
        print(f"  {f}")

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("DISCRIMINATION AUROC\n" + format_table(all_results, "discrimination_auroc") + "\n\n")
        f.write("ONSET TIMING MAE\n" + format_table(all_results, "onset_mae") + "\n\n")
        f.write("VALIDATION CHECKS\n" + "\n".join(findings) + "\n")
    print(f"\nResults written to {out_dir}/")


if __name__ == "__main__":
    main()
