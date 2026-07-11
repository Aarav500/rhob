"""Generic MuJoCo episode-loop runner and numerical proxy-matching calibration.

Every MuJoCo-tier family supplies its own action/proxy/true/behavioral functions and
calls :func:`generate_mujoco_rundata` to turn them into a :class:`RunData`. Families
whose two variants' matched proxy can't be solved in closed form (real MuJoCo dynamics
have no closed-form reward model, unlike e.g. ``reward_channel_tampering``'s algebraic
``_solve_bonus``) use :func:`calibrate_scale` -- a deterministic binary search over one
scalar control parameter -- as the honest numerical equivalent.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from rhob.detectors.posthoc import RunData
from rhob.environments.calibration import calibrate_scale  # noqa: F401 -- re-exported
from rhob.environments.mujoco.config import MuJoCoConfig

# (step_index, horizon, np.random.Generator) -> action array
ActionFn = Callable[[int, int, np.random.Generator], np.ndarray]
# (env, info dict from env.step, native_reward) -> per-step scalar contribution
StepMetricFn = Callable[[object, dict, float], float]


def _make_env(env_id: str, **kwargs):
    import gymnasium as gym

    return gym.make(env_id, **kwargs)


def run_mujoco_episode(
    env,
    action_fn: ActionFn,
    proxy_fn: StepMetricFn,
    true_fn: StepMetricFn,
    behav_fn: StepMetricFn,
    horizon: int,
    rng: np.random.Generator,
    action_noise_std: float,
) -> tuple[float, float, float]:
    """Run one episode, returning (mean_proxy, mean_true, mean_behav) per-step averages."""
    if horizon <= 0:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    proxy_sum = true_sum = behav_sum = 0.0
    for t in range(horizon):
        action = action_fn(t, horizon, rng)
        noise = rng.normal(0.0, action_noise_std, size=action.shape)
        action = np.clip(action + noise, -1.0, 1.0).astype(np.float32)
        _obs, reward, terminated, truncated, info = env.step(action)
        proxy_sum += proxy_fn(env, info, float(reward))
        true_sum += true_fn(env, info, float(reward))
        behav_sum += behav_fn(env, info, float(reward))
        if terminated or truncated:
            break
    n = t + 1
    return proxy_sum / n, true_sum / n, behav_sum / n


def generate_mujoco_rundata(
    config: MuJoCoConfig,
    action_fn: ActionFn,
    proxy_fn: StepMetricFn,
    true_fn: StepMetricFn,
    behav_fn: StepMetricFn,
    seed: int,
    env_kwargs: dict | None = None,
) -> RunData:
    """Roll out ``config.n_episodes`` episodes and build a :class:`RunData`.

    ``state_counts`` (L1) is intentionally left ``None`` -- MuJoCo's state
    dimensionality (11-17+ dims for the tasks used here) makes any naive fixed-bin
    histogram either wrong or an undocumented design decision; see the "Detector-suite
    implications" section of the design spec for why this is a stated limitation, not
    silently worked around.
    """
    env = _make_env(config.env_id, **(env_kwargs or {}))
    rng = np.random.default_rng(seed)
    proxy = np.zeros(config.n_episodes)
    true = np.zeros(config.n_episodes)
    behav = np.zeros(config.n_episodes)
    try:
        for ep in range(config.n_episodes):
            p, t, b = run_mujoco_episode(
                env, action_fn, proxy_fn, true_fn, behav_fn,
                config.horizon, rng, config.action_noise_std,
            )
            proxy[ep] = p
            true[ep] = t
            behav[ep] = b
    finally:
        env.close()
    return RunData(proxy_rewards=proxy, true_rewards=true, state_counts=None, behav_trace=behav)
