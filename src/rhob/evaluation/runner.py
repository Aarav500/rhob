"""Evaluation runner: orchestrates detector x environment evaluation.

Pipeline (engineering spec Section 6.1):

1. Validate the detector contract (bounds + determinism) on a sample trajectory.
2. For each trajectory: reset the detector, stream access-filtered observations,
   collect the score sequence.
3. Compute per-trajectory metrics.
4. Aggregate per environment and overall.
5. Return a structured :class:`EvaluationReport`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from rhob.config import EvaluationConfig
from rhob.core.exceptions import ScoreBoundsError
from rhob.core.trajectory import Trajectory
from rhob.core.types import Tier
from rhob.detectors.base import AbstractDetector
from rhob.evaluation import metrics
from rhob.evaluation.report import (
    EnvironmentMetrics,
    EvaluationReport,
    TrajectoryResult,
)
from rhob._version import __version__

TrajectorySource = Union[Sequence[Trajectory], str, Path]


class EvaluationRunner:
    """Runs a detector over a set of pre-recorded trajectories."""

    def run(
        self,
        detector: AbstractDetector,
        trajectories: TrajectorySource,
        config: Optional[EvaluationConfig] = None,
    ) -> EvaluationReport:
        config = config or EvaluationConfig()
        trajs = _resolve_trajectories(trajectories)
        if not trajs:
            raise ValueError("No trajectories provided for evaluation.")

        start = time.perf_counter()
        diagnostics = self._validate_contract(detector, trajs[0])

        results: list[TrajectoryResult] = []
        # scores/labels pooled per environment for per-step AUROC/AUPRC.
        pooled: dict[str, dict[str, list]] = {}
        for traj in trajs:
            scores = self._run_one(detector, traj)
            labels = traj.binary_labels()
            self._check_bounds(scores, detector, traj)

            env = traj.environment_id
            bucket = pooled.setdefault(env, {"scores": [], "labels": []})
            bucket["scores"].append(scores)
            bucket["labels"].append(labels)

            results.append(self._trajectory_result(traj, scores, labels, config))

        per_environment = self._aggregate_environments(pooled, results, config)
        report = self._aggregate_overall(detector, per_environment, results, config, diagnostics)
        report.total_compute_seconds = time.perf_counter() - start
        return report

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _run_one(detector: AbstractDetector, traj: Trajectory) -> np.ndarray:
        detector.reset()
        scores = np.fromiter(
            (detector.step(obs) for obs in traj.iter_observations(detector.access_level)),
            dtype=np.float64,
            count=traj.total_steps,
        )
        return scores

    def _validate_contract(self, detector: AbstractDetector, traj: Trajectory) -> dict:
        """Light contract check: determinism + bounds on a sample trajectory."""
        violations: list[str] = []
        warnings: list[str] = []

        first = self._run_one(detector, traj)
        second = self._run_one(detector, traj)
        if not np.array_equal(first, second):
            violations.append(
                "Non-deterministic: identical input sequences produced different "
                "scores. Ensure reset() fully clears state and any RNG is re-seeded."
            )
        if first.size and (
            np.any(first < 0.0) or np.any(first > 1.0) or np.any(~np.isfinite(first))
        ):
            violations.append("Scores outside [0, 1] or non-finite during validation.")
        return {"contract_violations": violations, "warnings": warnings}

    @staticmethod
    def _check_bounds(scores: np.ndarray, detector: AbstractDetector, traj: Trajectory) -> None:
        if scores.size == 0:
            return
        if np.any(~np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
            bad = scores[(~np.isfinite(scores)) | (scores < 0) | (scores > 1)][:3]
            raise ScoreBoundsError(
                f"Detector {detector.id!r} produced out-of-range scores "
                f"(e.g. {bad.tolist()}) on run seed={traj.seed}. "
                "All scores must lie in [0.0, 1.0] and be finite.\n"
                "Fix: clamp or squash your detector's output into [0, 1]."
            )

    @staticmethod
    def _trajectory_result(
        traj: Trajectory,
        scores: np.ndarray,
        labels: np.ndarray,
        config: EvaluationConfig,
    ) -> TrajectoryResult:
        d_step = metrics.detection_step(scores, config.score_threshold)
        latency = metrics.detection_latency(
            scores, traj.onset_step, traj.total_steps, config.score_threshold
        )
        return TrajectoryResult(
            run_id=f"{traj.environment_id}:seed={traj.seed}",
            environment_id=traj.environment_id,
            seed=traj.seed,
            is_hacking=traj.is_hacking_run,
            onset_step=traj.onset_step,
            detected=(d_step is not None),
            detection_step=d_step,
            latency=latency,
            auroc=metrics.auroc(scores, labels),
            auprc=metrics.auprc(scores, labels),
            fpr_at_k=metrics.fpr_at_k(scores, labels, config.alert_budget),
        )

    def _aggregate_environments(
        self,
        pooled: dict,
        results: list[TrajectoryResult],
        config: EvaluationConfig,
    ) -> dict[str, EnvironmentMetrics]:
        from rhob.environments.registry import get_environment_tier

        per_env: dict[str, EnvironmentMetrics] = {}
        for env_id, bucket in pooled.items():
            all_scores = np.concatenate(bucket["scores"])
            all_labels = np.concatenate(bucket["labels"])

            env_results = [r for r in results if r.environment_id == env_id]
            hacking = [r for r in env_results if r.is_hacking]
            clean = [r for r in env_results if not r.is_hacking]

            per_traj_auroc = [r.auroc for r in hacking]
            detected_flags = [r.detected for r in hacking]
            latencies = [r.latency for r in hacking]

            try:
                tier = get_environment_tier(env_id)
            except KeyError:
                tier = Tier.TIER1

            per_env[env_id] = EnvironmentMetrics(
                environment_id=env_id,
                tier=tier,
                auroc=metrics.auroc(all_scores, all_labels),
                auroc_ci=metrics.bootstrap_ci(
                    per_traj_auroc, config.bootstrap_n, config.confidence, config.seed
                ),
                auprc=metrics.auprc(all_scores, all_labels),
                miss_rate=metrics.miss_rate(detected_flags),
                tfd=metrics.time_to_first_detection(latencies),
                fpr_at_k=float(np.mean([r.fpr_at_k for r in env_results])),
                median_latency=metrics.time_to_first_detection(latencies),
                n_hacking_runs=len(hacking),
                n_clean_runs=len(clean),
            )
        return per_env

    def _aggregate_overall(
        self,
        detector: AbstractDetector,
        per_environment: dict[str, EnvironmentMetrics],
        results: list[TrajectoryResult],
        config: EvaluationConfig,
        diagnostics: dict,
    ) -> EvaluationReport:
        per_env_auroc = {e: m.auroc for e, m in per_environment.items()}
        env_tier = {e: m.tier for e, m in per_environment.items()}
        score = metrics.rhob_score(per_env_auroc, env_tier)

        # Bootstrap CI on the RHOB-Score across environments' per-step AUROCs.
        env_aurocs = [m.auroc for m in per_environment.values()]
        score_ci = (
            metrics.bootstrap_ci(env_aurocs, config.bootstrap_n, config.confidence, config.seed)
            if len(env_aurocs) > 1
            else next(iter(per_environment.values())).auroc_ci
        )

        mean_auroc = float(np.nanmean([m.auroc for m in per_environment.values()]))
        mean_miss = float(np.nanmean([m.miss_rate for m in per_environment.values()]))
        mean_tfd = float(np.nanmean([m.tfd for m in per_environment.values()]))
        mean_fpr = float(np.nanmean([m.fpr_at_k for m in per_environment.values()]))

        return EvaluationReport(
            detector_id=detector.id,
            detector_name=detector.name,
            detector_version=detector.version,
            access_level=detector.access_level,
            is_oracle_free=detector.is_oracle_free,
            rhob_score=score,
            rhob_score_ci=score_ci,
            mean_auroc=mean_auroc,
            mean_miss_rate=mean_miss,
            mean_tfd=mean_tfd,
            mean_fpr_at_k=mean_fpr,
            per_environment=per_environment,
            per_trajectory=results,
            diagnostics=diagnostics,
            rhob_version=__version__,
        )


def evaluate(
    detector: AbstractDetector,
    trajectories: TrajectorySource,
    config: Optional[EvaluationConfig] = None,
) -> EvaluationReport:
    """Evaluate a detector on pre-recorded trajectories (top-level convenience).

    ``trajectories`` may be a list of :class:`Trajectory`, or a path to an HDF5
    dataset file produced by :func:`rhob.data.storage.save_dataset`.
    """
    return EvaluationRunner().run(detector, trajectories, config)


def compare(
    detectors: Sequence[AbstractDetector],
    trajectories: TrajectorySource,
    config: Optional[EvaluationConfig] = None,
) -> list[EvaluationReport]:
    """Evaluate several detectors on the same trajectories."""
    trajs = _resolve_trajectories(trajectories)
    runner = EvaluationRunner()
    return [runner.run(det, trajs, config) for det in detectors]


def _resolve_trajectories(source: TrajectorySource) -> list[Trajectory]:
    if isinstance(source, (str, Path)):
        from rhob.data.storage import load_dataset

        return load_dataset(source)
    return list(source)
