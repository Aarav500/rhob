"""Configuration for the continuous-control tier (Pairs 02-04).

The three difficulty levels share one navigation/camping policy and one MDP; they
differ only in the eval-time target distribution: the attractor separation ``d``
(and, if needed, the intrinsic spread ``sigma_a``). Difficulty is therefore an
*intrinsic, designed* property -- the stochastic spread of the camping target --
not an artifact of how well the agent learned (see ``docs/difficulty_spectrum.md``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContinuousConfig:
    """Immutable parameters for a continuous-tier environment pair.

    Attributes:
        separation: Distance ``d`` between the two variants' attractor centres.
            Smaller ``d`` -> harder (more overlap) -> lower L2.
        sigma_a: Std of the per-run attractor location (intrinsic difficulty).
        box: Side length of the square arena ``[0, box]^2``.
        center: Arena centre and start cell; the true objective for the hacking
            variant. On the reflection axis so the two variants are mirror images.
        radius: Gaussian-bump reward radius (camping tightness scale).
        damping: Velocity damping (``v' = damping*v + force_gain*force``).
        force_gain: Force effect per step.
        pos_noise: Std of per-step position noise.
        pre_activation_speed_penalty: Coefficient of the pre-activation ``-c*||v||``
            reward that keeps the agent slow/centred before the exploit appears.
        horizon: Steps per episode.
        n_episodes: Episodes per run.
        activation_episode: Episode at which the attractor is revealed (the onset).
        steady_window: Trailing episodes used for admission features.
        camp_radius_factor: Camping counts steps within ``camp_radius_factor*radius``
            of the attractor.
    """

    separation: float
    sigma_a: float = 0.5
    box: float = 10.0
    center: float = 5.0
    radius: float = 1.0
    damping: float = 0.8
    force_gain: float = 0.1
    pos_noise: float = 0.05
    pre_activation_speed_penalty: float = 0.1
    horizon: int = 150
    n_episodes: int = 40
    activation_episode: int = 10
    steady_window: int = 25
    camp_radius_factor: float = 1.0


# DQN training hyperparameters (shared navigation/camping policy).
@dataclass(frozen=True)
class DQNConfig:
    """Hyperparameters for the shared camping policy."""

    hidden: int = 64
    lr: float = 1e-3
    gamma: float = 0.99
    buffer_size: int = 20000
    batch_size: int = 128
    warmup: int = 256
    target_update_steps: int = 500
    train_episodes: int = 250
    train_horizon: int = 150
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    target_min: float = 1.5
    target_max: float = 8.5


DQN_DEFAULT = DQNConfig()
