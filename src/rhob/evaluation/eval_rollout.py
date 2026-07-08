"""Instrumented rollouts for detector evaluation.

Mirrors each pair's production rollout (``pair_01.rollout.generate_run`` /
``continuous.rollout.generate_run``) exactly, adding the one signal detectors
need that the production rollouts don't record: a per-episode state-visitation
histogram (L1). Kept separate from the frozen, admission-tested production
rollouts rather than adding an instrumentation flag to them.
"""

from __future__ import annotations

import numpy as np

from rhob.agents.tabular_q import TabularQLearner
from rhob.detectors.posthoc import RunData
from rhob.environments.continuous.config import ContinuousConfig
from rhob.environments.continuous.onset_label import label_onset as label_onset_continuous
from rhob.environments.continuous.point_mass import PointMassPair
from rhob.environments.pair_01._mdp import Pair01Env
from rhob.environments.pair_01.config import DEFAULT, Pair01Config
from rhob.environments.pair_01.onset_label import label_onset as label_onset_gridworld
from rhob.environments.pair_01.rollout import epsilon_schedule


def generate_gridworld_rundata(
    env: Pair01Env, seed: int, config: Pair01Config = DEFAULT
) -> tuple[RunData, int]:
    """Train a tabular agent on one Pair 01 variant, recording detector-ready signals.

    Args:
        env: A configured Pair 01 variant (``make_hacking_env``/``make_legitimate_env``).
        seed: Seed for the agent's RNG.
        config: Shared Pair 01 configuration.

    Returns:
        ``(RunData, onset_episode)`` -- ``onset_episode`` is oracle ground truth
        (from ``frac_tile2``), not part of ``RunData``.
    """
    agent = TabularQLearner(
        env.n_states, env.n_actions, alpha=config.alpha, gamma=config.gamma,
        q_init=config.q_init, seed=seed,
    )
    n = config.n_episodes
    proxy = np.empty(n, dtype=np.float64)
    true = np.empty(n, dtype=np.float64)
    frac2 = np.empty(n, dtype=np.float64)
    behav = np.empty(n, dtype=np.float64)
    state_counts = np.zeros((n, env.n_states), dtype=np.int64)

    for ep in range(n):
        env.set_episode(ep)
        eps = epsilon_schedule(ep, config)
        state = env.reset()
        p_ret = t_ret = row_sum = col_sum = on_tile2 = 0.0
        for _ in range(config.horizon):
            action = agent.act(state, eps)
            next_state, pr, tr, _done, info = env.step(action)
            agent.update(state, action, pr, next_state)
            state_counts[ep, next_state] += 1
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
        # col-minus-row (not row-minus-col): tile2_hacking=(2,3) sits at col>row and
        # tile2_legit=(3,2) at row>col, so this convention gives positive=hacking,
        # matching the continuous tier's attractor_x-centre (positive=hacking, variant A).
        behav[ep] = (col_sum - row_sum) / config.horizon

    onset = label_onset_gridworld(frac2, config.onset_threshold, config.onset_sustain)
    run_data = RunData(proxy_rewards=proxy, true_rewards=true, state_counts=state_counts, behav_trace=behav)
    return run_data, onset


def generate_continuous_rundata(
    camper, variant: str, config: ContinuousConfig, seed: int, n_bins: int = 20
) -> tuple[RunData, int]:
    """Roll out a trained camper on one continuous variant, recording detector-ready signals.

    Args:
        camper: A trained ``DQNCamper``.
        variant: ``"A"`` (hacking) or ``"B"`` (legitimate).
        config: Shared continuous configuration.
        seed: Determines the per-run attractor and noise.
        n_bins: Number of x-position bins for the L1 visitation histogram.

    Returns:
        ``(RunData, onset_episode)`` -- ``onset_episode`` is oracle ground truth
        (from ``mean_x``/``attractor_x``), not part of ``RunData``.
    """
    env = PointMassPair(variant, config, seed=seed)
    n = config.n_episodes
    proxy = np.zeros(n)
    true = np.zeros(n)
    mean_x = np.zeros(n)
    state_counts = np.zeros((n, n_bins), dtype=np.int64)
    bin_edges = np.linspace(0.0, config.box, n_bins + 1)

    for ep in range(n):
        env.set_episode(ep)
        obs = env.reset()
        xs = pr = tr = 0.0
        for _ in range(config.horizon):
            action = camper.act(obs)
            obs, p, t, info = env.step(action)
            x = info["pos"][0]
            xs += x
            pr += p
            tr += t
            b = int(np.clip(np.searchsorted(bin_edges, x, side="right") - 1, 0, n_bins - 1))
            state_counts[ep, b] += 1
        mean_x[ep] = xs / config.horizon
        proxy[ep] = pr / config.horizon
        true[ep] = tr / config.horizon

    behav_trace = mean_x - config.center
    onset = label_onset_continuous(mean_x, env.attractor_x, config.center)
    run_data = RunData(
        proxy_rewards=proxy, true_rewards=true, state_counts=state_counts, behav_trace=behav_trace
    )
    return run_data, onset
