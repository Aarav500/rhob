"""Shared infrastructure for the continuous-control tier (Pairs 02-04).

A single reflection-symmetrised DQN camping policy generates trajectories for all
three difficulty levels; difficulty is set by the eval-time attractor separation
``d`` (and spread ``sigma_a``). See ``docs/difficulty_spectrum.md``.
"""

from rhob.environments.continuous.config import DQN_DEFAULT, ContinuousConfig, DQNConfig
from rhob.environments.continuous.onset_label import label_onset
from rhob.environments.continuous.point_mass import PointMassPair
from rhob.environments.continuous.rollout import ContinuousRunLog, generate_run

__all__ = [
    "ContinuousConfig",
    "DQNConfig",
    "DQN_DEFAULT",
    "PointMassPair",
    "generate_run",
    "ContinuousRunLog",
    "label_onset",
]
