"""Eight L0 detectors return a constant below their windows and declare no horizon.

Reward Skewness did this until it declared ``min_episodes`` (``test_l0_reward_skewness.py``).
These eight still do: on a run shorter than the length listed in
``UNDECLARED_HORIZON_FALLBACKS`` their ``classify`` returns one constant for every run
(0.5, or 0.0 for Reward CUSUM), so on a family that short every pair ties and the board
records an AUROC of 0.5 as if it had been measured. That is an open defect
(``ROADMAP.md``), not a design. These tests hold the table, which the docs and the
rendered access-level ladder quote, to what the detectors do:

* each listed detector declares no ``min_episodes``, returns one constant on runs one
  episode shorter than its listed length and at 40 episodes (the shortest horizon a
  registered family runs), and varies a little above its listed length;
* every other L0 detector on the committed board, except Reward Skewness, varies at 40
  episodes, so no other detector has a fallback that binds on a registered family.

When a detector declares its horizon, the first test fails for it: take it out of the
table, re-score the boards (``scripts/merge_rescored_detector.py``) and update the L0
figures the docs quote.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rhob.detectors import (
    GradientReversalDetector,
    MaxPlateauDetector,
    RewardAutocorrelationDetector,
    RewardCUSUMDetector,
    RewardKDEDetector,
    RewardMLPDetector,
    RewardPeakDetector,
    RewardSkewnessDetector,
    RewardThresholdDetector,
    RewardTrendDetector,
    RewardVarianceRatioDetector,
    SpectralRewardDetector,
    VarianceWindowDetector,
)
from rhob.detectors.posthoc import RunData
from rhob.v3.leaderboard.access_summary import UNDECLARED_HORIZON_FALLBACKS

_BOARD = Path(__file__).resolve().parents[2] / "leaderboard" / "v5_leaderboard.json"

#: The shortest horizon a registered family runs (seven families run 40 episodes).
_SHORTEST_HORIZON = 40

_L0 = {
    cls().name: cls
    for cls in (
        GradientReversalDetector,
        MaxPlateauDetector,
        RewardAutocorrelationDetector,
        RewardCUSUMDetector,
        RewardKDEDetector,
        RewardMLPDetector,
        RewardPeakDetector,
        RewardSkewnessDetector,
        RewardThresholdDetector,
        RewardTrendDetector,
        RewardVarianceRatioDetector,
        SpectralRewardDetector,
        VarianceWindowDetector,
    )
}


def _runs(n_episodes: int, n_runs: int = 12, seed: int = 0) -> list[RunData]:
    """Runs that differ in level, noise and shape, so a real statistic tells them apart.

    Half are V-shaped and half inverted, with a different amplitude each, so a slope
    statistic such as Gradient Reversal's sees both signs.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_episodes)
    v = np.abs(t - n_episodes / 2) / max(n_episodes / 2, 1)
    runs = []
    for i in range(n_runs):
        amplitude = (0.2 + 0.05 * i) * (1 if i % 2 else -1)
        proxy = 0.5 + 0.1 * i + amplitude * v + rng.normal(0, 0.02 + 0.01 * i, n_episodes)
        runs.append(RunData(proxy, np.zeros(n_episodes), None, None))
    return runs


def _detector(name: str):
    det = _L0[name]()
    if hasattr(det, "fit"):
        # Fitted on runs long enough for its window, so its constant below the window
        # is the length guard and not an untrained model's default.
        det.fit(_runs(120, seed=1)[:6], _runs(120, seed=2)[6:], epochs=3)
    return det


def _distinct(det, n_episodes: int) -> set[float]:
    return {round(float(det.classify(r)), 12) for r in _runs(n_episodes)}


def test_the_table_names_l0_detectors_on_the_board():
    board = json.loads(_BOARD.read_text(encoding="utf-8"))["results"]
    on_board = {name for name, info in board.items() if info["access_level"] == "L0"}
    assert on_board == set(_L0)
    assert set(UNDECLARED_HORIZON_FALLBACKS) <= on_board
    assert "Reward Skewness" not in UNDECLARED_HORIZON_FALLBACKS


@pytest.mark.parametrize("name", sorted(UNDECLARED_HORIZON_FALLBACKS))
def test_a_listed_detector_returns_a_constant_below_its_window(name):
    length = UNDECLARED_HORIZON_FALLBACKS[name]
    det = _detector(name)
    assert int(getattr(det, "min_episodes", 0) or 0) == 0, (
        f"{name} now declares min_episodes; take it out of UNDECLARED_HORIZON_FALLBACKS "
        "and re-score the boards"
    )
    constant = {0.0} if name == "Reward CUSUM" else {0.5}
    assert _distinct(det, length - 1) == constant
    if _SHORTEST_HORIZON < length:
        assert _distinct(det, _SHORTEST_HORIZON) == constant
    assert len(_distinct(det, length + 1)) > 1, f"{name} is constant above {length} too"


def test_no_other_l0_detector_falls_back_at_any_registered_horizon():
    for name in sorted(set(_L0) - set(UNDECLARED_HORIZON_FALLBACKS) - {"Reward Skewness"}):
        det = _detector(name)
        assert int(getattr(det, "min_episodes", 0) or 0) == 0, name
        assert len(_distinct(det, _SHORTEST_HORIZON)) > 1, name


def test_reward_skewness_is_the_one_that_declares_its_horizon():
    assert RewardSkewnessDetector().min_episodes == 100
