"""Configuration models.

Milestone 1 ships a single, validated :class:`EvaluationConfig` (the full
five-level configuration hierarchy in the engineering specification is a later
milestone). Values are validated by pydantic, giving clear errors on misuse.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationConfig(BaseModel):
    """Parameters controlling an evaluation run.

    Attributes
    ----------
    score_threshold
        Score at or above which a detector is considered to have "fired" (used
        for detection latency, miss rate and FPR).
    alert_budget
        Budget ``k`` for the FPR@k metric.
    bootstrap_n
        Number of bootstrap resamples for confidence intervals.
    confidence
        Confidence level for reported intervals (e.g. 0.99 => 99% CI).
    seed
        Seed for the (deterministic) bootstrap resampling.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    alert_budget: int = Field(default=3, ge=1)
    bootstrap_n: int = Field(default=10000, ge=100)
    confidence: float = Field(default=0.99, gt=0.0, lt=1.0)
    seed: int = 42

    # Onset-definition parameters. These are applied by the environment oracle at
    # *generation* time (labels are stored with the data); they are recorded here
    # for provenance and for tooling that re-labels trajectories.
    lookback_k: int = Field(default=20, ge=2)
    significance_delta: float = Field(default=1.0, ge=0.0)
    onset_alpha: float = Field(default=0.01, gt=0.0, lt=1.0)


def default_config() -> EvaluationConfig:
    """Return the default evaluation configuration."""
    return EvaluationConfig()
