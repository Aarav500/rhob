"""Training rollout for Pair 01: turns an environment + seed into a run log.

Trains a tabular Q-learning agent on one Pair 01 variant and records the
per-episode signals needed for onset labelling and the admission gate:

* ``proxy`` / ``true`` -- per-episode return under each reward channel,
* ``frac_tile2``       -- fraction of steps on the second tile (behavioural onset),
* ``behav``            -- the L2 behavioural feature ``mean(row) - mean(col)``,
                          which separates the transposed variants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rhob.agents.tabular_q import TabularQLearner
from rhob.environments.pair_01._mdp import Pair01Env
from rhob.environments.pair_01.config import DEFAULT, Pair01Config


@dataclass(frozen=True)
class RunLog:
    """Per-episode signals recorded from one training run.

    Each array has length ``n_episodes``.
    """

    seed: int
    proxy: np.ndarray
    true: np.ndarray
    frac_tile2: np.ndarray
    behav: np.ndarray


def epsilon_schedule(episode: int, config: Pair01Config = DEFAULT) -> float:
    """Exploration rate for a given episode.

    Linear decay from ``epsilon_start`` to ``epsilon_end``, overridden by a short
    higher-exploration *burst* immediately after activation so that the discovery
    of ``tile2`` (and hence the onset) is fast and temporally localised.
    """
    start, end = config.activation_episode, config.activation_episode + config.activation_burst
    if start <= episode < end:
        return config.epsilon_activation
    frac = episode / max(1, config.n_episodes - 1)
    return config.epsilon_start + frac * (config.epsilon_end - config.epsilon_start)


def generate_run(env: Pair01Env, seed: int, config: Pair01Config = DEFAULT) -> RunLog:
    """Train the agent on ``env`` for one seed and return the recorded signals.

    Args:
        env: A configured Pair 01 variant.
        seed: Seed for the agent's RNG (determines the whole run).
        config: Shared Pair 01 configuration.

    Returns:
        A :class:`RunLog` of per-episode signals.
    """
    agent = TabularQLearner(
        env.n_states,
        env.n_actions,
        alpha=config.alpha,
        gamma=config.gamma,
        q_init=config.q_init,
        seed=seed,
    )
    n = config.n_episodes
    proxy = np.empty(n, dtype=np.float64)
    true = np.empty(n, dtype=np.float64)
    frac2 = np.empty(n, dtype=np.float64)
    behav = np.empty(n, dtype=np.float64)

    for ep in range(n):
        env.set_episode(ep)
        eps = epsilon_schedule(ep, config)
        state = env.reset()
        p_ret = t_ret = 0.0
        row_sum = col_sum = on_tile2 = 0.0
        for _ in range(config.horizon):
            action = agent.act(state, eps)
            next_state, pr, tr, _done, info = env.step(action)
            agent.update(state, action, pr, next_state)
            r, c = info["cell"]
            row_sum += r
            col_sum += c
            on_tile2 += float(env.tile2_active and info["cell"] == env.tile2)
            p_ret += pr
            t_ret += tr
            state = next_state
        proxy[ep] = p_ret
        true[ep] = t_ret
        frac2[ep] = on_tile2 / config.horizon
        behav[ep] = (row_sum - col_sum) / config.horizon
    return RunLog(seed=seed, proxy=proxy, true=true, frac_tile2=frac2, behav=behav)
