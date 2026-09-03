"""Reward Skewness must not pad a short horizon with a constant.

The detector needs ``baseline_window + test_window`` episodes (100 at the defaults).
Before this file existed, any shorter run silently returned ``0.5`` -- the same number
for every run on both sides -- and that constant was averaged into the published L0 row
as "at chance" on 24 of 33 families (``leaderboard/v5_replicated.json``, Reward
Skewness: mean 0.500, SD 0.000). Eleven of the twelve families that fail the nightly
admission smoke test run 40 or 60 episodes, so the shape panel was DEGENERATE on them
before a single rollout was scored.

A number the detector could not have varied is not a measurement. Convention 1 of the
paper says it is N/A, and the harness already has the machinery for that
(``missing_channels``); what was missing was the detector saying how many episodes of
the channel it needs.
"""

from __future__ import annotations

import numpy as np
import pytest

from rhob.detectors.l0_reward_skewness import RewardSkewnessDetector
from rhob.detectors.l0_reward_threshold import RewardThresholdDetector
from rhob.detectors.posthoc import RunData


def _run(n: int, rng: np.random.Generator) -> RunData:
    return RunData(
        proxy_rewards=rng.normal(0.5, 0.1, n),
        true_rewards=np.zeros(n),
        state_counts=None,
        behav_trace=None,
    )


def test_declares_the_horizon_it_needs():
    det = RewardSkewnessDetector()
    assert det.min_episodes == det.baseline_window + det.test_window == 100


@pytest.mark.parametrize("n", [40, 60, 99])
def test_a_short_run_is_not_measured_rather_than_scored_at_chance(n):
    score = RewardSkewnessDetector().classify(_run(n, np.random.default_rng(0)))
    assert np.isnan(score), f"got {score!r}; a run below min_episodes must not score"


@pytest.mark.parametrize("n", [100, 160, 200])
def test_a_long_run_scores_in_the_unit_interval(n):
    score = RewardSkewnessDetector().classify(_run(n, np.random.default_rng(0)))
    assert np.isfinite(score) and 0.0 <= score <= 1.0


def test_the_base_class_default_is_no_minimum():
    """A detector that never declared a horizon requirement keeps scoring everything."""
    assert RewardThresholdDetector().min_episodes == 0
