# src/rhob/environments/rlhf_rm/rollout.py
"""Policy-optimization episode loop for RLHF-RM families.

Each "episode" is one independent run of policy-gradient ascent against a fitted
reward model, with a KL penalty back to a fixed reference policy -- the RLHF-RM
analogue of one MuJoCo rollout. Per-step proxy/true/behavioral signals are logged the
same way MuJoCo families do via ``proxy_fn``/``true_fn``/``behav_fn``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from rhob.detectors.posthoc import RunData
from rhob.environments.rlhf_rm.config import RLHFConfig
from rhob.environments.rlhf_rm.preference import true_reward

# (mu, batch of sampled responses, rm_weights) -> per-step scalar contribution
StepMetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


def run_rlhf_episode(
    config: RLHFConfig,
    rm_weights: np.ndarray,
    mu_0: np.ndarray,
    proxy_fn: StepMetricFn,
    true_fn: StepMetricFn,
    behav_fn: StepMetricFn,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Run one policy-optimization episode, returning (mean_proxy, mean_true, mean_behav)."""
    mu = mu_0.copy()
    proxy_sum = true_sum = behav_sum = 0.0
    for _ in range(config.n_steps):
        batch = rng.normal(mu, config.sigma, size=(config.batch_size, config.response_dim))
        scores = batch @ rm_weights  # proxy: reward-model score
        # Policy-gradient ascent on mu (REINFORCE-style, since sigma is fixed):
        # mu += step_size * mean((score - baseline) * (batch - mu)) / sigma^2, plus a
        # KL-penalty gradient term pulling mu back toward mu_0 (KL between two
        # same-sigma Gaussians is proportional to ||mu - mu_0||^2 / (2 sigma^2), whose
        # gradient w.r.t. mu is (mu - mu_0) / sigma^2).
        baseline = scores.mean()
        grad = np.mean((scores - baseline)[:, None] * (batch - mu), axis=0) / (config.sigma**2)
        kl_grad = (mu - mu_0) / (config.sigma**2)
        mu = mu + config.step_size * (grad - config.beta * kl_grad)

        proxy_sum += proxy_fn(mu, batch, rm_weights)
        true_sum += true_fn(mu, batch, rm_weights)
        behav_sum += behav_fn(mu, batch, rm_weights)
    n = config.n_steps
    return proxy_sum / n, true_sum / n, behav_sum / n


def generate_rlhf_rundata(
    config: RLHFConfig,
    rm_weights: np.ndarray,
    mu_0: np.ndarray,
    proxy_fn: StepMetricFn,
    true_fn: StepMetricFn,
    behav_fn: StepMetricFn,
    seed: int,
) -> RunData:
    """Roll out ``config.n_episodes`` independent policy-optimization episodes.

    ``state_counts`` (L1) is intentionally left ``None`` -- this setting has no
    discrete state space to histogram (see the design spec's "Detector-suite
    implications" section), same rationale as the MuJoCo families.
    """
    rng = np.random.default_rng(seed)
    proxy = np.zeros(config.n_episodes)
    true = np.zeros(config.n_episodes)
    behav = np.zeros(config.n_episodes)
    for ep in range(config.n_episodes):
        p, t, b = run_rlhf_episode(config, rm_weights, mu_0, proxy_fn, true_fn, behav_fn, rng)
        proxy[ep] = p
        true[ep] = t
        behav[ep] = b
    return RunData(proxy_rewards=proxy, true_rewards=true, state_counts=None, behav_trace=behav)


def default_proxy_fn(mu: np.ndarray, batch: np.ndarray, rm_weights: np.ndarray) -> float:
    """Mean fitted-RM score of the current batch -- the standard proxy for every family."""
    return float((batch @ rm_weights).mean())


def default_true_fn(mu: np.ndarray, batch: np.ndarray, rm_weights: np.ndarray) -> float:
    """Mean true reward of the current batch -- the standard oracle signal for every family."""
    return float(true_reward(batch).mean())
