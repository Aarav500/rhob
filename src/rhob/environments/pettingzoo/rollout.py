# src/rhob/environments/pettingzoo/rollout.py
"""Generic mpe2/PettingZoo parallel-env episode-loop runner.

Every PettingZoo-tier family supplies per-agent action functions and
proxy/true/behavioral aggregation functions operating on the full per-agent
observation/reward dicts, and calls :func:`generate_pettingzoo_rundata` to turn them
into a :class:`RunData` -- the same per-episode-scalar convention every other family
(MuJoCo, RLHF-RM) already uses, so the admission gate and detector suite work
unmodified.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from rhob.detectors.posthoc import RunData
from rhob.environments.pettingzoo.config import PettingZooConfig

# (agent_id, t, horizon, obs, rng) -> action array for that agent
ActionFn = Callable[[str, int, int, np.ndarray, np.random.Generator], np.ndarray]
# (env, obs_dict, reward_dict, info_dict, t, horizon) -> per-step scalar contribution,
# aggregated across all agents by the family (e.g. sum/mean of a specific agent's
# reward, or a function of the full joint observation). ``t``/``horizon`` let a family
# with phase-based mechanisms (e.g. an initial individual-pursuit phase followed by a
# collusion phase) compute metrics that are aware of which phase the current step
# falls in, rather than only ever averaging uniformly over the whole episode.
StepMetricFn = Callable[[object, dict, dict, dict, int, int], float]


def run_pettingzoo_episode(
    env,
    action_fns: dict[str, ActionFn],
    proxy_fn: StepMetricFn,
    true_fn: StepMetricFn,
    behav_fn: StepMetricFn,
    horizon: int,
    rng: np.random.Generator,
    action_noise_std: float,
) -> tuple[float, float, float]:
    """Run one episode, returning (mean_proxy, mean_true, mean_behav) per-step averages.

    ``action_fns`` maps each agent id to its own action function -- different agents
    in a family may follow genuinely different policies (e.g. the hacking variant's
    "free-rider" agent vs. its honest teammates), so this is a dict, not a single
    shared function like MuJoCo's single-agent ``ActionFn``.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    obs, _info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    proxy_sum = true_sum = behav_sum = 0.0
    t = 0
    for t in range(horizon):
        actions = {}
        for agent in env.agents:
            action_space = env.action_space(agent)
            raw = action_fns[agent](agent, t, horizon, obs[agent], rng)
            noise = rng.normal(0.0, action_noise_std, size=raw.shape)
            actions[agent] = np.clip(
                raw + noise, action_space.low, action_space.high
            ).astype(np.float32)
        obs, rewards, terms, truncs, infos = env.step(actions)
        proxy_sum += proxy_fn(env, obs, rewards, infos, t, horizon)
        true_sum += true_fn(env, obs, rewards, infos, t, horizon)
        behav_sum += behav_fn(env, obs, rewards, infos, t, horizon)
        if not env.agents or all(terms.values()) or all(truncs.values()):
            break
    n = t + 1
    return proxy_sum / n, true_sum / n, behav_sum / n


def generate_pettingzoo_rundata(
    config: PettingZooConfig,
    action_fns: dict[str, ActionFn],
    proxy_fn: StepMetricFn,
    true_fn: StepMetricFn,
    behav_fn: StepMetricFn,
    seed: int,
) -> RunData:
    """Roll out ``config.n_episodes`` episodes and build a :class:`RunData`.

    ``state_counts`` (L1) is intentionally left ``None`` -- multi-agent joint state
    spaces have no single natural per-family fixed-bin histogram representation any
    more than MuJoCo's high-dimensional continuous states did; see the design spec's
    "Detector-suite implications" section for why this is a stated limitation, not
    silently worked around.
    """
    env = config.env_factory()
    rng = np.random.default_rng(seed)
    proxy = np.zeros(config.n_episodes)
    true = np.zeros(config.n_episodes)
    behav = np.zeros(config.n_episodes)
    try:
        for ep in range(config.n_episodes):
            p, t, b = run_pettingzoo_episode(
                env, action_fns, proxy_fn, true_fn, behav_fn,
                config.horizon, rng, config.action_noise_std,
            )
            proxy[ep] = p
            true[ep] = t
            behav[ep] = b
    finally:
        env.close()
    return RunData(proxy_rewards=proxy, true_rewards=true, state_counts=None, behav_trace=behav)
