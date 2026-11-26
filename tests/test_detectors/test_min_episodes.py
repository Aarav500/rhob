"""Every detector on the board declares the run length its score needs.

A detector whose ``classify`` returns a constant below some length (a window that does
not fit, a model ``fit`` could not train) used to return that constant to the harness,
and the boards recorded it as an AUROC of 0.5 on every family too short for it: until
2026-11-26, eight L0 detectors, every L1 detector on ``continuous_camping`` and three L2
detectors, 104 of the per-family entries behind the L0 mean alone. Now each declares
``min_episodes``, and ``rhob.v3.benchmark._evaluate_cell`` marks a shorter cell not
applicable instead of scoring it (``tests/test_v3/test_benchmark_horizon_guard.py``).

These tests hold each declaration to the detector's own ``classify``, run the way
``_evaluate_cell`` runs it: every run in a cell has the same length, a trainable detector
is fitted on runs of that length, and ``observe_cell`` sees the cell first.

* one episode below ``min_episodes``, every run gets the same score;
* at ``min_episodes``, the scores differ, so the declaration is not stricter than the
  guard it stands for;
* a detector that declares no minimum varies on the shortest runs it can get.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from rhob.detectors import (
    AngularMomentumDetector,
    BehavioralThresholdDetector,
    BimodalOccupancyDetector,
    CentroidDriftDetector,
    CentroidTrackerDetector,
    FeatureConsistencyDetector,
    FeatureMagnitudeDetector,
    GradientReversalDetector,
    MaxPlateauDetector,
    OccupancyPolarizationDetector,
    PerfectFeatureOracleDetector,
    RewardAutocorrelationDetector,
    RewardCUSUMDetector,
    RewardFeatureCorrelationDetector,
    RewardKDEDetector,
    RewardMLPDetector,
    RewardPeakDetector,
    RewardSkewnessDetector,
    RewardThresholdDetector,
    RewardTrendDetector,
    RewardVarianceRatioDetector,
    SpectralRewardDetector,
    StateCoverageRateDetector,
    StateDivergenceDetector,
    StateFrequencyAnomalyDetector,
    TrajectoryMLPDetector,
    TransitionEntropyDetector,
    TrueRewardOracleDetector,
    VarianceWindowDetector,
    VisitationEntropyTrendDetector,
)
from rhob.detectors.posthoc import RunData

_BOARD = Path(__file__).resolve().parents[2] / "leaderboard" / "v5_leaderboard.json"

#: The shortest horizon a registered family runs (seven families run 40 episodes).
_SHORTEST_HORIZON = 40

#: Every detector class on the committed board, built as the board builds it
#: (``scripts/v5_leaderboard_and_transfer.py``: default constructor arguments).
_CLASSES = (
    RewardThresholdDetector,
    RewardCUSUMDetector,
    RewardVarianceRatioDetector,
    SpectralRewardDetector,
    RewardPeakDetector,
    RewardAutocorrelationDetector,
    RewardSkewnessDetector,
    RewardTrendDetector,
    VarianceWindowDetector,
    MaxPlateauDetector,
    GradientReversalDetector,
    RewardMLPDetector,
    RewardKDEDetector,
    StateFrequencyAnomalyDetector,
    CentroidDriftDetector,
    OccupancyPolarizationDetector,
    BimodalOccupancyDetector,
    TransitionEntropyDetector,
    StateCoverageRateDetector,
    VisitationEntropyTrendDetector,
    StateDivergenceDetector,
    BehavioralThresholdDetector,
    AngularMomentumDetector,
    CentroidTrackerDetector,
    FeatureMagnitudeDetector,
    FeatureConsistencyDetector,
    RewardFeatureCorrelationDetector,
    TrajectoryMLPDetector,
    TrueRewardOracleDetector,
    PerfectFeatureOracleDetector,
)
_BY_NAME = {cls().name: cls for cls in _CLASSES}

#: What each detector declares with its default arguments. Written out, not read back
#: from the detectors, so a change to a declaration has to change this table too, and
#: with it the boards (``scripts/merge_rescored_detector.py``) and the docs that quote
#: them.
_DECLARED = {
    "Reward Threshold": 0,
    "Reward CUSUM": 51,
    "Reward Variance Ratio": 21,
    "Spectral Reward": 24,
    "Reward Peak": 100,
    "Reward Autocorrelation": 100,
    "Reward Skewness": 100,
    "Reward Trend": 100,
    "Variance Window": 100,
    "Max Plateau": 50,
    "Gradient Reversal": 60,
    "Reward MLP": 50,
    "Reward KDE": 22,
    "State Frequency Anomaly": 100,
    "Centroid Drift": 100,
    "Occupancy Polarization": 100,
    "Bimodal Occupancy": 100,
    "Transition Entropy": 101,
    "State Coverage Rate": 50,
    "Visitation Entropy Trend": 50,
    "State Divergence": 50,
    "Behavioral Threshold": 0,
    "Angular Momentum": 0,
    "Centroid Tracker": 0,
    "Feature Magnitude": 50,
    "Feature Consistency": 100,
    "Reward-Feature Correlation": 50,
    "Trajectory MLP": 2,
    "True Reward Oracle": 0,
    "Perfect Feature Oracle": 0,
}

_N_BINS = 10


def _runs(n_episodes: int, n_runs: int = 12, seed: int = 0) -> list[RunData]:
    """Runs that differ on every channel, so a real statistic tells them apart.

    The proxy is V-shaped or inverted with a different level, amplitude and noise per
    run, as are the true reward and the behavioural trace. ``state_counts`` is sparse,
    so bins are discovered at different times, with a visitation centre that drifts at
    a different speed per run.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_episodes)
    v = np.abs(t - n_episodes / 2) / max(n_episodes / 2, 1)
    bins = np.arange(_N_BINS)
    runs = []
    for i in range(n_runs):
        sign = 1 if i % 2 else -1
        amplitude = (0.2 + 0.05 * i) * sign
        noise = 0.02 + 0.01 * i
        proxy = 0.5 + 0.1 * i + amplitude * v + rng.normal(0, noise, n_episodes)
        true = 0.3 + 0.05 * i - amplitude * v + rng.normal(0, noise, n_episodes)
        behav = 0.05 * (i - n_runs / 2) + 0.5 * amplitude * v + rng.normal(0, noise, n_episodes)
        centre = (i % _N_BINS) + sign * (0.02 + 0.01 * i) * t
        rate = (0.05 + 0.02 * i) * (1.0 + 4.0 * np.exp(-0.5 * (bins - centre[:, None] % _N_BINS) ** 2))
        counts = rng.poisson(rate).astype(float)
        runs.append(RunData(proxy, true, counts, behav))
    return runs


def _scores(det, n_episodes: int) -> set[object]:
    """The distinct scores ``det`` gives one cell of runs of this length.

    Fitted and observed the way ``_evaluate_cell`` does it, on the cell's own runs. A NaN
    is one value (Reward Skewness returns NaN below its horizon), not a different value
    per run.
    """
    runs = _runs(n_episodes)
    if hasattr(det, "fit"):
        kwargs = {"epochs": 3} if isinstance(det, (RewardMLPDetector, TrajectoryMLPDetector)) else {}
        det.fit(runs[::2], runs[1::2], **kwargs)
    det.observe_cell(runs)
    out = set()
    for run in runs:
        score = float(det.classify(run))
        out.add("nan" if math.isnan(score) else round(score, 12))
    return out


def test_the_table_is_the_board():
    board = json.loads(_BOARD.read_text(encoding="utf-8"))["results"]
    assert set(board) == set(_BY_NAME) == set(_DECLARED)


@pytest.mark.parametrize("name", sorted(_DECLARED))
def test_each_detector_declares_what_the_table_says(name):
    assert int(_BY_NAME[name]().min_episodes) == _DECLARED[name]


@pytest.mark.parametrize("name", sorted(n for n, k in _DECLARED.items() if k >= 2))
def test_a_declared_detector_is_constant_below_its_minimum_and_varies_at_it(name):
    needed = _DECLARED[name]
    below = _scores(_BY_NAME[name](), needed - 1)
    assert len(below) == 1, f"{name} varies at {needed - 1} episodes: {sorted(map(str, below))}"
    at = _scores(_BY_NAME[name](), needed)
    assert len(at) > 1, f"{name} is still constant at its declared {needed} episodes"


@pytest.mark.parametrize("name", sorted(n for n, k in _DECLARED.items() if k < 2))
def test_a_detector_with_no_minimum_varies_on_the_shortest_runs(name):
    for n_episodes in (2, _SHORTEST_HORIZON):
        assert len(_scores(_BY_NAME[name](), n_episodes)) > 1, (name, n_episodes)


@pytest.mark.parametrize(
    "det, needed",
    [
        (RewardPeakDetector(baseline_window=30, peak_window=20), 50),
        (RewardCUSUMDetector(baseline_window=10), 11),
        (GradientReversalDetector(window_size=12), 24),
        (RewardMLPDetector(window_size=8), 8),
        (TransitionEntropyDetector(baseline_window=13, test_window=17), 31),
        (StateDivergenceDetector(baseline_episodes=10, steady_window=25), 10),
        (FeatureConsistencyDetector(baseline_window=20, test_window=10), 30),
        (RewardFeatureCorrelationDetector(test_window=15), 15),
    ],
    ids=lambda x: x if isinstance(x, int) else x.name,
)
def test_the_declaration_follows_the_constructor_arguments(det, needed):
    """Scripts build some detectors with smaller windows (``evaluate_detectors.py``)."""
    assert det.min_episodes == needed
    assert len(_scores(det, needed - 1)) == 1
    assert len(_scores(det, needed)) > 1
