"""Shared per-step rollout primitive for the sequence-generation setting.

Structurally mirrors ``rlhf_rm/rollout.py``'s pattern (no external env/``step()``
call needed -- token sampling and scoring both happen directly in the rollout
loop), not ``mujoco``/``pettingzoo``'s ``gym.step()``-based ones.

``state_counts`` (L1) is intentionally left ``None`` -- this setting's vocabulary
is its own state space, dimensioned differently from every other family's, so
there's no natural shared fixed-bin histogram representation, matching the same
documented limitation as the RLHF-RM/MuJoCo/PettingZoo settings.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from rhob.detectors.posthoc import RunData
from rhob.environments.sequence_gen.config import SequenceGenConfig

# (t, horizon, tokens_emitted_so_far, rng) -> next token id. ``tokens_emitted_so_far``
# has length t (does not include the token being chosen at step t).
ActionFn = Callable[[int, int, np.ndarray, np.random.Generator], int]

# (tokens_so_far_including_current, t, horizon) -> per-step scalar contribution.
# ``tokens_so_far_including_current`` has length t+1 (includes the token just
# emitted at step t), letting a metric function see the token it's scoring.
StepMetricFn = Callable[[np.ndarray, int, int], float]

# True-reward step functions have a different, fixed signature (grammar_log_prob_step
# in grammar.py) since they need the *pre-emission* array indexed at t, not a
# growing slice -- kept as a distinct type so a family can't accidentally pass a
# StepMetricFn where a true-reward function is expected.
TrueStepFn = Callable[[np.ndarray, int], float]


def run_sequence_episode(
    config: SequenceGenConfig,
    action_fn: ActionFn,
    proxy_fn: StepMetricFn,
    true_fn: TrueStepFn,
    behav_fn: StepMetricFn,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Run one episode, returning (mean_proxy, mean_true, mean_behav) per-step averages."""
    horizon = config.horizon
    if horizon <= 0:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    tokens = np.zeros(horizon, dtype=np.int64)
    proxy_sum = true_sum = behav_sum = 0.0
    for t in range(horizon):
        token = action_fn(t, horizon, tokens[:t], rng)
        tokens[t] = token
        proxy_sum += proxy_fn(tokens[: t + 1], t, horizon)
        true_sum += true_fn(tokens, t)
        behav_sum += behav_fn(tokens[: t + 1], t, horizon)
    return proxy_sum / horizon, true_sum / horizon, behav_sum / horizon


def generate_sequence_rundata(
    config: SequenceGenConfig,
    action_fn: ActionFn,
    proxy_fn: StepMetricFn,
    true_fn: TrueStepFn,
    behav_fn: StepMetricFn,
    seed: int,
) -> RunData:
    """Roll out ``config.n_episodes`` episodes and build a :class:`RunData`."""
    rng = np.random.default_rng(seed)
    proxy_rewards = np.zeros(config.n_episodes)
    true_rewards = np.zeros(config.n_episodes)
    behav_trace = np.zeros(config.n_episodes)
    for i in range(config.n_episodes):
        proxy_rewards[i], true_rewards[i], behav_trace[i] = run_sequence_episode(
            config, action_fn, proxy_fn, true_fn, behav_fn, rng
        )
    return RunData(
        proxy_rewards=proxy_rewards,
        true_rewards=true_rewards,
        state_counts=None,
        behav_trace=behav_trace,
    )
