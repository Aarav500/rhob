"""Structured evaluation reports and comparison tables."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from rhob.core.types import AccessLevel, Tier


def _fmt(value: float, nd: int = 3) -> str:
    """Format a float, rendering NaN/inf compactly."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, float) and np.isinf(value):
        return "∞"
    return f"{value:.{nd}f}"


@dataclass
class TrajectoryResult:
    """Per-trajectory evaluation outcome."""

    run_id: str
    environment_id: str
    seed: int
    is_hacking: bool
    onset_step: Optional[int]
    detected: bool
    detection_step: Optional[int]
    latency: float
    auroc: float
    auprc: float
    fpr_at_k: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "environment_id": self.environment_id,
            "seed": self.seed,
            "is_hacking": self.is_hacking,
            "onset_step": self.onset_step,
            "detected": self.detected,
            "detection_step": self.detection_step,
            "latency": _nan_to_none(self.latency),
            "auroc": _nan_to_none(self.auroc),
            "auprc": _nan_to_none(self.auprc),
            "fpr_at_k": _nan_to_none(self.fpr_at_k),
        }


@dataclass
class EnvironmentMetrics:
    """Aggregate metrics for one environment."""

    environment_id: str
    tier: Tier
    auroc: float  # pooled per-step AUROC (hacking + clean)
    auroc_ci: tuple[float, float]  # bootstrap CI over per-trajectory AUROCs
    auprc: float
    miss_rate: float
    tfd: float  # median normalized latency among detected hacking runs
    fpr_at_k: float
    median_latency: float
    n_hacking_runs: int
    n_clean_runs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "tier": self.tier.value,
            "auroc": _nan_to_none(self.auroc),
            "auroc_ci": [_nan_to_none(self.auroc_ci[0]), _nan_to_none(self.auroc_ci[1])],
            "auprc": _nan_to_none(self.auprc),
            "miss_rate": _nan_to_none(self.miss_rate),
            "tfd": _nan_to_none(self.tfd),
            "fpr_at_k": _nan_to_none(self.fpr_at_k),
            "median_latency": _nan_to_none(self.median_latency),
            "n_hacking_runs": self.n_hacking_runs,
            "n_clean_runs": self.n_clean_runs,
        }


@dataclass
class EvaluationReport:
    """Complete result of evaluating one detector on the benchmark."""

    detector_id: str
    detector_name: str
    detector_version: str
    access_level: AccessLevel
    is_oracle_free: bool
    rhob_score: float
    rhob_score_ci: tuple[float, float]
    mean_auroc: float
    mean_miss_rate: float
    mean_tfd: float
    mean_fpr_at_k: float
    per_environment: dict[str, EnvironmentMetrics]
    per_trajectory: list[TrajectoryResult] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    rhob_version: str = ""
    total_compute_seconds: float = 0.0

    # ---------------------------------------------------------------- serialize
    def to_dict(self, include_trajectories: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "detector_id": self.detector_id,
            "detector_name": self.detector_name,
            "detector_version": self.detector_version,
            "access_level": str(self.access_level),
            "is_oracle_free": self.is_oracle_free,
            "aggregate": {
                "rhob_score": _nan_to_none(self.rhob_score),
                "rhob_score_ci": [
                    _nan_to_none(self.rhob_score_ci[0]),
                    _nan_to_none(self.rhob_score_ci[1]),
                ],
                "mean_auroc": _nan_to_none(self.mean_auroc),
                "mean_miss_rate": _nan_to_none(self.mean_miss_rate),
                "mean_tfd": _nan_to_none(self.mean_tfd),
                "mean_fpr_at_k": _nan_to_none(self.mean_fpr_at_k),
            },
            "per_environment": {k: v.to_dict() for k, v in self.per_environment.items()},
            "diagnostics": self.diagnostics,
            "rhob_version": self.rhob_version,
            "total_compute_seconds": self.total_compute_seconds,
        }
        if include_trajectories:
            data["per_trajectory"] = [r.to_dict() for r in self.per_trajectory]
        return data

    def to_json(self, include_trajectories: bool = True, indent: int = 2) -> str:
        return json.dumps(self.to_dict(include_trajectories), indent=indent)

    def to_markdown(self) -> str:
        lines = [
            f"# RHOB Evaluation: {self.detector_name} ({self.detector_id})",
            "",
            f"- **Access level:** {self.access_level}",
            f"- **Oracle-free:** {self.is_oracle_free}",
            f"- **RHOB-Score:** {_fmt(self.rhob_score)} "
            f"(99% CI [{_fmt(self.rhob_score_ci[0])}, {_fmt(self.rhob_score_ci[1])}])",
            f"- **Mean AUROC:** {_fmt(self.mean_auroc)} | "
            f"**Miss rate:** {_fmt(self.mean_miss_rate)} | "
            f"**TFD:** {_fmt(self.mean_tfd)} | "
            f"**FPR@k:** {_fmt(self.mean_fpr_at_k)}",
            "",
            "## Per-environment",
            "",
            "| Environment | Tier | AUROC | 99% CI | AUPRC | Miss | TFD | FPR@k | n_hack | n_clean |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for env_id, m in self.per_environment.items():
            ci = f"[{_fmt(m.auroc_ci[0])}, {_fmt(m.auroc_ci[1])}]"
            lines.append(
                f"| {env_id} | {m.tier.value} | {_fmt(m.auroc)} | {ci} | {_fmt(m.auprc)} | "
                f"{_fmt(m.miss_rate)} | {_fmt(m.tfd)} | {_fmt(m.fpr_at_k)} | "
                f"{m.n_hacking_runs} | {m.n_clean_runs} |"
            )
        return "\n".join(lines)


def results_table(reports: list[EvaluationReport]) -> str:
    """Build a Markdown comparison table across detectors (leaderboard-style)."""
    lines = [
        "| Method | Access | RHOB-Score | 99% CI | Mean AUROC | Miss | TFD | FPR@k |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(reports, key=lambda x: (-_safe(x.rhob_score), x.mean_tfd)):
        ci = f"[{_fmt(r.rhob_score_ci[0])}, {_fmt(r.rhob_score_ci[1])}]"
        lines.append(
            f"| {r.detector_name} | {r.access_level} | {_fmt(r.rhob_score)} | {ci} | "
            f"{_fmt(r.mean_auroc)} | {_fmt(r.mean_miss_rate)} | {_fmt(r.mean_tfd)} | "
            f"{_fmt(r.mean_fpr_at_k)} |"
        )
    return "\n".join(lines)


def _nan_to_none(value: float) -> Optional[float]:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return None
    return float(value)


def _safe(value: float) -> float:
    return -1.0 if (value is None or np.isnan(value)) else value
