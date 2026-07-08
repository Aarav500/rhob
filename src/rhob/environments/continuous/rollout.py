"""Roll out a trained camper on a continuous pair variant to produce a run log."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rhob.environments.continuous.config import ContinuousConfig
from rhob.environments.continuous.point_mass import PointMassPair


@dataclass(frozen=True)
class ContinuousRunLog:
    """Per-episode signals from one continuous run (length ``n_episodes``)."""

    variant: str
    seed: int
    attractor_x: float
    proxy: np.ndarray
    true: np.ndarray
    mean_x: np.ndarray
    camp_frac: np.ndarray


def generate_run(camper, variant: str, config: ContinuousConfig, seed: int) -> ContinuousRunLog:
    """Roll out ``camper`` on one variant and record per-episode signals.

    The camper is a *pre-trained* policy (see :func:`rhob.agents.dqn.train_camper`);
    each episode is a greedy rollout. Pre-onset the observed target is the arena
    centre; at ``activation_episode`` the (stochastic) attractor is revealed and the
    agent switches to camping it -- the onset.

    Args:
        camper: A trained ``DQNCamper`` (or any object with ``act(obs) -> int``).
        variant: ``"A"`` (hacking) or ``"B"`` (legitimate).
        config: Shared continuous configuration.
        seed: Determines the per-run attractor and noise.

    Returns:
        A :class:`ContinuousRunLog`.
    """
    env = PointMassPair(variant, config, seed=seed)
    n = config.n_episodes
    proxy = np.zeros(n)
    true = np.zeros(n)
    mean_x = np.zeros(n)
    camp = np.zeros(n)
    for ep in range(n):
        env.set_episode(ep)
        obs = env.reset()
        xs = pr = tr = 0.0
        near = 0
        for _ in range(config.horizon):
            action = camper.act(obs)
            obs, p, t, info = env.step(action)
            xs += info["pos"][0]
            pr += p
            tr += t
            near += int(info["camping"])
        mean_x[ep] = xs / config.horizon
        proxy[ep] = pr / config.horizon
        true[ep] = tr / config.horizon
        camp[ep] = near / config.horizon
    return ContinuousRunLog(
        variant=variant,
        seed=seed,
        attractor_x=env.attractor_x,
        proxy=proxy,
        true=true,
        mean_x=mean_x,
        camp_frac=camp,
    )
