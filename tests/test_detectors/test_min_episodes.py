"""Every detector on the board, and every external baseline, declares the run length its
score needs.

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

The five external baselines (``rhob.detectors.external_baselines``) are on no board (the
HVTA bridge scores them, ``external/hvta/score_detectors.py``), so they have their own
table, held to the same rule. A completeness guard keeps both tables honest: every
``PosthocDetector`` the package defines is exported, and every exported one that can be
built without arguments is in one of them, so a new detector cannot ship undeclared.
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
import pkgutil
from pathlib import Path

import numpy as np
import pytest

import rhob.detectors
from rhob.detectors import (
    AngularMomentumDetector,
    ARResidualDetector,
    BehavioralThresholdDetector,
    BimodalOccupancyDetector,
    BOCPDDetector,
    CentroidDriftDetector,
    CentroidTrackerDetector,
    FeatureConsistencyDetector,
    FeatureMagnitudeDetector,
    GradientReversalDetector,
    IsolationForestDetector,
    MaxPlateauDetector,
    OccupancyPolarizationDetector,
    PageHinkleyDetector,
    PCAReconstructionDetector,
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
from rhob.detectors.external_baselines import __all__ as _EXTERNAL_EXPORTS
from rhob.detectors.posthoc import PosthocDetector, RunData

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

#: The external baselines, built with default arguments as ``external/hvta`` builds them.
_EXTERNAL = (
    PageHinkleyDetector,
    IsolationForestDetector,
    ARResidualDetector,
    PCAReconstructionDetector,
    BOCPDDetector,
)
_EXTERNAL_BY_NAME = {cls().name: cls for cls in _EXTERNAL}

#: What each external baseline declares with its default arguments, written out for the
#: same reason as ``_DECLARED``: a change to one has to change this table, and with it
#: the HVTA artifacts that score the detector and the docs that list the values.
_EXTERNAL_DECLARED = {
    "Page-Hinkley Test": 3,  # a running mean and two deviations
    "Isolation Forest": 2,  # a standard deviation and a range of the late window
    "AR(p) Residual": 6,  # order (5) lagged values plus the one they predict
    "PCA Reconstruction": 100,  # steady_window rows of state_counts
    "Bayesian Online Changepoint Detection": 5,  # the length the recursion runs from
}

_ALL_BY_NAME = {**_BY_NAME, **_EXTERNAL_BY_NAME}
_ALL_DECLARED = {**_DECLARED, **_EXTERNAL_DECLARED}

#: The exported detector classes that need constructor arguments, so cannot be in a
#: table. Named rather than skipped on any ``TypeError``, so a new detector cannot leave
#: the tables by requiring an argument.
_NEEDS_ARGUMENTS = {
    "PosthocDetector",  # the abstract base
    "EnsembleDetector",  # needs members; derives max(member minimums) itself
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


def test_the_external_table_is_the_external_baselines():
    """Every class ``rhob.detectors.external_baselines`` exports, none on the board."""
    assert {cls.__name__ for cls in _EXTERNAL} == set(_EXTERNAL_EXPORTS)
    assert set(_EXTERNAL_BY_NAME) == set(_EXTERNAL_DECLARED)
    assert not set(_EXTERNAL_DECLARED) & set(_DECLARED)


def test_every_shipped_posthoc_detector_is_in_a_table():
    """A detector that is not in a table is not held to its declaration, so none may ship.

    Two halves. Every ``PosthocDetector`` subclass defined anywhere under
    ``rhob.detectors`` is exported in ``rhob.detectors.__all__`` (so the second half sees
    it), and every exported one is either in ``_DECLARED`` / ``_EXTERNAL_DECLARED`` or
    named in ``_NEEDS_ARGUMENTS``.
    """
    defined = set()
    for info in pkgutil.walk_packages(rhob.detectors.__path__, "rhob.detectors."):
        module = importlib.import_module(info.name)
        for obj in vars(module).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, PosthocDetector)
                and obj.__module__ == module.__name__
            ):
                defined.add(obj)
    exported = {
        obj
        for obj in (getattr(rhob.detectors, attr) for attr in rhob.detectors.__all__)
        if inspect.isclass(obj) and issubclass(obj, PosthocDetector)
    }
    assert sorted(c.__qualname__ for c in defined - exported) == []

    tabled, unbuildable = set(), set()
    for cls in exported:
        if cls.__name__ in _NEEDS_ARGUMENTS:
            with pytest.raises(TypeError):
                cls()
            unbuildable.add(cls.__name__)
            continue
        tabled.add(cls().name)
    assert unbuildable == _NEEDS_ARGUMENTS
    assert sorted(tabled - set(_ALL_DECLARED)) == []
    assert sorted(set(_ALL_DECLARED) - tabled) == []


@pytest.mark.parametrize("name", sorted(_ALL_DECLARED))
def test_each_detector_declares_what_the_table_says(name):
    assert int(_ALL_BY_NAME[name]().min_episodes) == _ALL_DECLARED[name]


@pytest.mark.parametrize("name", sorted(n for n, k in _ALL_DECLARED.items() if k >= 2))
def test_a_declared_detector_is_constant_below_its_minimum_and_varies_at_it(name):
    needed = _ALL_DECLARED[name]
    below = _scores(_ALL_BY_NAME[name](), needed - 1)
    assert len(below) == 1, f"{name} varies at {needed - 1} episodes: {sorted(map(str, below))}"
    at = _scores(_ALL_BY_NAME[name](), needed)
    assert len(at) > 1, f"{name} is still constant at its declared {needed} episodes"


@pytest.mark.parametrize("name", sorted(n for n, k in _ALL_DECLARED.items() if k < 2))
def test_a_detector_with_no_minimum_varies_on_the_shortest_runs(name):
    for n_episodes in (2, _SHORTEST_HORIZON):
        assert len(_scores(_ALL_BY_NAME[name](), n_episodes)) > 1, (name, n_episodes)


@pytest.mark.parametrize("cls", _EXTERNAL, ids=lambda cls: cls.__name__)
def test_an_external_baseline_guard_reads_its_declaration(cls):
    """Raise the declaration by one and the score goes constant at the old minimum.

    So the guard in ``classify`` (and ``fit``) is the declared ``min_episodes``, not a
    literal that happens to equal it and could drift from it.
    """

    class Stricter(cls):
        @property
        def min_episodes(self) -> int:
            return super().min_episodes + 1

    needed = cls().min_episodes
    assert Stricter().min_episodes == needed + 1
    assert len(_scores(cls(), needed)) > 1
    assert len(_scores(Stricter(), needed)) == 1


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
        (PageHinkleyDetector(delta=0.1, threshold=1.0), 3),
        (IsolationForestDetector(window=50), 2),
        (ARResidualDetector(order=3, window=50), 4),
        (ARResidualDetector(order=1), 2),
        (PCAReconstructionDetector(n_components=2, steady_window=50), 50),
        (BOCPDDetector(hazard=0.01), 5),
    ],
    ids=lambda x: x if isinstance(x, int) else x.name,
)
def test_the_declaration_follows_the_constructor_arguments(det, needed):
    """Scripts build some detectors with smaller windows (``evaluate_detectors.py``), and
    ``tests/test_detectors/test_external_baselines.py`` builds the external baselines
    with non-default arguments."""
    assert det.min_episodes == needed
    assert len(_scores(det, needed - 1)) == 1
    assert len(_scores(det, needed)) > 1
