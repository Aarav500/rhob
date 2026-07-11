# src/rhob/environments/rlhf_rm/config.py
"""Configuration shared by every RLHF-RM-tier family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RLHFConfig:
    """Immutable parameters for one RLHF-RM-tier matched pair.

    Attributes:
        response_dim: Dimensionality of the synthetic response-feature space.
        n_episodes: Episodes (independent policy-optimization runs) per rollout.
        n_steps: Policy-gradient steps per episode.
        batch_size: Responses sampled per policy-gradient step.
        sigma: Fixed (non-optimized) std of the policy's Gaussian over response space.
        beta: KL-penalty coefficient (policy vs. reference).
        step_size: Learning rate for the policy-gradient ascent step on mu.
        n_pref_pairs: Number of preference pairs used to fit the reward model.
        label_noise_std: Std of Gaussian noise added to preference logits before
            thresholding into a binary label (models annotator disagreement).
        calibration_seed: Fixed seed used only for the one-time proxy-matching
            calibration search (not used for actual rollouts).
        calibration_tol: Acceptable absolute gap between the two variants' mean
            proxy after calibration.
        calibration_seeds: Number of seeds averaged per calibration probe.
        extra: Family-specific parameters, for provenance.
    """

    response_dim: int = 8
    n_episodes: int = 60
    n_steps: int = 40
    batch_size: int = 32
    sigma: float = 1.0
    beta: float = 0.1
    step_size: float = 0.05
    n_pref_pairs: int = 500
    label_noise_std: float = 0.3
    calibration_seed: int = 999
    calibration_tol: float = 0.05
    calibration_seeds: int = 8
    extra: dict = None

    def __post_init__(self):
        if self.extra is None:
            object.__setattr__(self, "extra", {})
