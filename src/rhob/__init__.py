"""RHOB -- the Reward-Hacking Onset Benchmark.

A researcher should go from ``pip install rhob`` to a full evaluation report in
under ten lines::

    import rhob

    class MyDetector(rhob.Detector):
        name = "my_detector"
        id = "user/my_detector"
        access_level = rhob.AccessLevel.L1

        def reset(self):
            self.history = []

        def step(self, obs):
            self.history.append(obs.reward)
            return my_score(self.history)

    trajectories = rhob.load_dataset("rhob_gridworld.h5")
    report = rhob.evaluate(MyDetector(), trajectories)
    print(report.rhob_score)

This Milestone 1 release ships the vertical slice: the GridWorld-Wireheading
environment, the Random and CUSUM baselines, the full metrics pipeline, HDF5
storage, and the evaluation runner.
"""

from rhob._version import __version__

# --- Core types ---
from rhob.core.access import AccessFilter
from rhob.core.exceptions import (
    ContractViolationError,
    DetectorError,
    RHOBError,
    ScoreBoundsError,
)
from rhob.core.onset import OnsetLabel
from rhob.core.trajectory import Observation, Timestep, Trajectory
from rhob.core.types import AccessLevel, HackingType, Tier

# --- Config ---
from rhob.config import EvaluationConfig, default_config

# --- Detectors ---
from rhob.detectors.base import AbstractDetector
from rhob.detectors.base import AbstractDetector as Detector  # friendly alias
from rhob.detectors.baselines.cusum import CUSUMDetector
from rhob.detectors.baselines.random_detector import RandomDetector

# --- Environments ---
from rhob.environments.base import AbstractEnvironment
from rhob.environments.base import AbstractEnvironment as Environment  # friendly alias
from rhob.environments.oracle import OnsetOracle
from rhob.environments.registry import (
    get_environment,
    get_environment_card,
    list_environments,
    register_environment,
)
from rhob.environments.tier1.gridworld_wireheading import GridWorldWireheading

# --- Data ---
from rhob.data.storage import load_dataset, load_trajectory, save_dataset, save_trajectory

# --- Evaluation ---
from rhob.evaluation import metrics
from rhob.evaluation.report import EvaluationReport, results_table
from rhob.evaluation.runner import EvaluationRunner, compare, evaluate

__all__ = [
    "__version__",
    # core
    "AccessLevel",
    "HackingType",
    "Tier",
    "Trajectory",
    "Timestep",
    "Observation",
    "OnsetLabel",
    "AccessFilter",
    "RHOBError",
    "DetectorError",
    "ContractViolationError",
    "ScoreBoundsError",
    # config
    "EvaluationConfig",
    "default_config",
    # detectors
    "Detector",
    "AbstractDetector",
    "RandomDetector",
    "CUSUMDetector",
    # environments
    "Environment",
    "AbstractEnvironment",
    "OnsetOracle",
    "GridWorldWireheading",
    "register_environment",
    "get_environment",
    "get_environment_card",
    "list_environments",
    # data
    "load_dataset",
    "save_dataset",
    "load_trajectory",
    "save_trajectory",
    # evaluation
    "evaluate",
    "compare",
    "EvaluationRunner",
    "EvaluationReport",
    "results_table",
    "metrics",
]
