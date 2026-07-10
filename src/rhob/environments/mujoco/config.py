"""Configuration shared by every MuJoCo-tier family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MuJoCoConfig:
    """Immutable parameters for one MuJoCo-tier matched pair.

    Attributes:
        env_id: Gymnasium environment id, e.g. ``"HalfCheetah-v5"``.
        n_episodes: Episodes per run.
        horizon: Steps per episode (capped at the env's own max if shorter).
        action_noise_std: Std of Gaussian noise added to every scripted action.
        calibration_seed: Fixed seed used only for the one-time proxy-matching
            calibration search (not used for actual rollouts).
        calibration_tol: Acceptable absolute gap between the two variants' mean
            proxy after calibration.
        calibration_seeds: Number of seeds averaged per calibration probe.
        extra: Family-specific parameters (e.g. goal separation), for provenance.
    """

    env_id: str
    n_episodes: int = 60
    horizon: int = 200
    action_noise_std: float = 0.05
    calibration_seed: int = 999
    calibration_tol: float = 0.05
    calibration_seeds: int = 8
    extra: dict = None

    def __post_init__(self):
        if self.extra is None:
            object.__setattr__(self, "extra", {})
