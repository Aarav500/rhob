"""A detector that cannot read a family's horizon produces an N/A cell, not a score.

One rule down from ``missing_channels``: ``required_channels`` says *which* signals a
detector needs, ``min_episodes`` says *how many episodes* of them. A cell that trips
either is excluded from aggregates with a reason, the way an absent channel already is.
Scoring it anyway would publish the detector's fallback constant as a property of the
family, which is exactly what happened to the Reward Skewness row -- and, until
2026-11-26, to eight other L0 rows, every L1 row on ``continuous_camping`` and three L2
rows (``tests/test_detectors/test_min_episodes.py``).
"""

from __future__ import annotations

import numpy as np

from rhob.detectors.l0_reward_skewness import RewardSkewnessDetector
from rhob.detectors.l0_reward_threshold import RewardThresholdDetector
from rhob.detectors.posthoc import RunData
from rhob.v3.benchmark import _evaluate_cell


def _runs(n_runs: int, n_episodes: int, shift: float, rng: np.random.Generator):
    return [
        RunData(
            proxy_rewards=rng.normal(0.5 + shift, 0.1, n_episodes),
            true_rewards=np.zeros(n_episodes),
            state_counts=None,
            behav_trace=None,
        )
        for _ in range(n_runs)
    ]


def test_short_horizon_is_not_applicable_for_a_detector_that_needs_more():
    rng = np.random.default_rng(0)
    a, b = _runs(10, 40, 0.2, rng), _runs(10, 40, 0.0, rng)
    auroc, mae, reason = _evaluate_cell(RewardSkewnessDetector(), a, b, [0] * 10, "L0", 40)
    assert np.isnan(auroc) and np.isnan(mae)
    assert reason and "40" in reason and "100" in reason, reason


def test_the_same_runs_score_for_a_detector_with_no_minimum():
    rng = np.random.default_rng(0)
    a, b = _runs(10, 40, 0.2, rng), _runs(10, 40, 0.0, rng)
    auroc, _, reason = _evaluate_cell(RewardThresholdDetector(), a, b, [0] * 10, "L0", 40)
    assert reason is None and np.isfinite(auroc)


def test_a_long_horizon_still_scores_the_same_detector():
    rng = np.random.default_rng(0)
    a, b = _runs(10, 200, 0.2, rng), _runs(10, 200, 0.0, rng)
    auroc, _, reason = _evaluate_cell(RewardSkewnessDetector(), a, b, [0] * 10, "L0", 200)
    assert reason is None and np.isfinite(auroc)


def test_an_ensemble_needs_every_members_horizon():
    """A member below its minimum returns a fallback, not a score (NaN for Skewness).

    Averaged in, that NaN used to fail the whole cell inside roc_auc_score. The ensemble
    declares the largest of its members' minimums, so the cell is not applicable instead,
    and it scores once the horizon covers every member.
    """
    from rhob.detectors import EnsembleDetector

    rng = np.random.default_rng(0)
    ens = EnsembleDetector([RewardSkewnessDetector(), RewardThresholdDetector()])
    assert ens.min_episodes == 100
    a, b = _runs(10, 40, 0.2, rng), _runs(10, 40, 0.0, rng)
    auroc, _, reason = _evaluate_cell(ens, a, b, [0] * 10, "L0", 40)
    assert np.isnan(auroc) and reason and "100" in reason, reason
    a, b = _runs(10, 100, 0.2, rng), _runs(10, 100, 0.0, rng)
    auroc, _, reason = _evaluate_cell(ens, a, b, [0] * 10, "L0", 100)
    assert reason is None and np.isfinite(auroc)
