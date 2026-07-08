"""DQN camping agent with reflection symmetrisation (continuous tier).

A single navigation/camping policy is trained to reach and hold an *observed*
target. Because the target is part of the observation, the same policy serves all
difficulty levels -- only the eval-time target distribution changes.

The greedy policy is symmetrised about the arena's x-axis: ``Q_sym(s) = Q(s) +
Q(reflect s)[mirror]``. Tight camping makes the proxy reward nearly constant, so a
tiny left/right asymmetry in the learned net would otherwise dominate L0; exact
symmetrisation keeps the two variants' proxy matched (L0 at chance).

torch is required only for this module and is imported lazily with a clear error.
"""

from __future__ import annotations

import math

import numpy as np

from rhob.agents.replay_buffer import ReplayBuffer
from rhob.environments.continuous.config import DQN_DEFAULT, DQNConfig
from rhob.environments.continuous.point_mass import (
    FORCES,
    MIRROR_ACTION,
    N_ACTIONS,
    observe,
    reflect_obs,
)

try:  # torch is an optional dependency (continuous tier only)
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The continuous tier requires PyTorch. Install it with `pip install torch` "
        "(or `pip install rhob[continuous]`)."
    ) from exc

_MIRROR = list(MIRROR_ACTION)


class _QNet(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, x):  # type: ignore[no-untyped-def]
        return self.net(x)


class DQNCamper:
    """A trained, reflection-symmetrised camping policy.

    Use :func:`train_camper` to construct one. :meth:`act` returns the greedy,
    symmetrised action for an observation.
    """

    def __init__(self, net: "_QNet", box: float) -> None:
        self._net = net
        self._box = box

    def act(self, observation: np.ndarray) -> int:
        """Greedy, x-reflection-symmetrised action for ``observation``."""
        sr = reflect_obs(observation)
        with torch.no_grad():
            q1 = self._net(torch.from_numpy(observation)).numpy()
            q2 = self._net(torch.from_numpy(sr)).numpy()[_MIRROR]
        return int((q1 + q2).argmax())


def train_camper(config: DQNConfig = DQN_DEFAULT, box: float = 10.0, seed: int = 0) -> DQNCamper:
    """Train the shared navigation/camping policy on random targets.

    The task: from the arena centre, reach and hold a target drawn uniformly in
    ``[target_min, target_max] x {centre}`` each episode. Reward is a Gaussian bump
    at the target minus a small speed penalty (so the agent brakes and camps).

    Args:
        config: DQN hyperparameters.
        box: Arena side length.
        seed: Determinism seed (torch + numpy).

    Returns:
        A trained :class:`DQNCamper`.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net, target = _QNet(config.hidden), _QNet(config.hidden)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=config.lr)
    buf = ReplayBuffer(config.buffer_size, seed=seed)
    center = box / 2.0
    step_ct = 0

    for ep in range(config.train_episodes):
        eps = max(
            config.epsilon_end,
            config.epsilon_start
            - (config.epsilon_start - config.epsilon_end) * ep / config.train_episodes,
        )
        tx = float(rng.uniform(config.target_min, config.target_max))
        ty = center
        x = y = center
        vx = vy = 0.0
        for _ in range(config.train_horizon):
            s = observe(x, y, vx, vy, tx, ty, box)
            if rng.random() < eps:
                a = int(rng.integers(N_ACTIONS))
            else:
                with torch.no_grad():
                    a = int(net(torch.from_numpy(s)).argmax())
            fx, fy = FORCES[a]
            vx, vy = 0.8 * vx + 0.1 * fx, 0.8 * vy + 0.1 * fy
            x = min(box, max(0.0, x + vx + rng.normal(0, 0.05)))
            y = min(box, max(0.0, y + vy + rng.normal(0, 0.05)))
            bump = math.exp(-((x - tx) ** 2 + (y - ty) ** 2) / 2.0)
            r = bump - 0.05 * math.hypot(vx, vy)
            buf.push(s, a, r, observe(x, y, vx, vy, tx, ty, box))
            step_ct += 1
            if len(buf) >= config.warmup:
                bs, ba, br, bns = buf.sample(config.batch_size)
                bs_t = torch.from_numpy(bs)
                with torch.no_grad():
                    tgt = (
                        torch.from_numpy(br)
                        + config.gamma * target(torch.from_numpy(bns)).max(1).values
                    )
                pred = net(bs_t).gather(1, torch.from_numpy(ba).unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if step_ct % config.target_update_steps == 0:
                target.load_state_dict(net.state_dict())
    net.eval()
    return DQNCamper(net, box)
