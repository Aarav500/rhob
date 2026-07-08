"""Self-contained gridworld MDP underlying both variants of Pair 01.

No external RL-library dependency: the MDP is a plain, deterministic-transition
gridworld with a ``STAY`` action so that a cell can be "camped" for per-step
reward.

Both variants share this MDP; they differ only in the position of the second
tile (``tile2``) and its *true* reward. Because the two ``tile2`` positions are
transposes of each other about the main diagonal and ``tile1``/``start`` lie on
that diagonal, the proxy-reward dynamics of the two variants are identical in
distribution -- the property the CR1 admission gate verifies.
"""

from __future__ import annotations

from rhob.environments.pair_01.config import DEFAULT, Pair01Config

# up, down, left, right, stay
_ACTIONS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))
N_ACTIONS: int = len(_ACTIONS)


class Pair01Env:
    """One variant of the Pair 01 gridworld MDP.

    The environment is fixed-horizon (no terminal state): each episode runs for
    ``config.horizon`` steps. The second tile is inert until
    :meth:`set_episode` is called with an episode index at or beyond
    ``config.activation_episode``.

    Args:
        tile2: Cell of the second (higher-proxy) tile for this variant.
        tile2_true: True reward per step while on ``tile2`` (0 => misaligned
            exploit; equal to the proxy => aligned improvement).
        config: Shared Pair 01 configuration.
    """

    def __init__(
        self,
        tile2: tuple[int, int],
        tile2_true: float,
        config: Pair01Config = DEFAULT,
    ) -> None:
        self.cfg = config
        self.tile2: tuple[int, int] = (int(tile2[0]), int(tile2[1]))
        self.tile2_true = float(tile2_true)
        self.n_states: int = config.grid_size * config.grid_size
        self.n_actions: int = N_ACTIONS
        self._tile2_active: bool = False
        self._state: tuple[int, int] = config.start

    # ------------------------------------------------------------------ episode
    def set_episode(self, episode: int) -> None:
        """Update whether ``tile2`` is active for the given training episode."""
        self._tile2_active = episode >= self.cfg.activation_episode

    @property
    def tile2_active(self) -> bool:
        """Whether the second tile currently yields reward."""
        return self._tile2_active

    def reset(self) -> int:
        """Reset the agent to the start cell and return its state index."""
        self._state = self.cfg.start
        return self._index(self._state)

    def step(self, action: int) -> tuple[int, float, float, bool, dict]:
        """Advance one step.

        Args:
            action: Integer in ``[0, 5)`` (up, down, left, right, stay).

        Returns:
            Tuple ``(next_state_index, proxy_reward, true_reward, done, info)``.
            ``done`` is always ``False`` (fixed horizon); ``info["cell"]`` holds
            the resulting ``(row, col)``.
        """
        nxt = self._move(self._state, action)
        proxy, true = self._reward(nxt)
        self._state = nxt
        return self._index(nxt), proxy, true, False, {"cell": nxt}

    # ------------------------------------------------------------------ helpers
    def _reward(self, cell: tuple[int, int]) -> tuple[float, float]:
        cfg = self.cfg
        if cell == cfg.tile1:
            return cfg.tile1_proxy, cfg.tile1_true
        if self._tile2_active and cell == self.tile2:
            return cfg.tile2_proxy, self.tile2_true
        return 0.0, 0.0

    def _move(self, state: tuple[int, int], action: int) -> tuple[int, int]:
        dr, dc = _ACTIONS[action]
        n = self.cfg.grid_size
        r = min(n - 1, max(0, state[0] + dr))
        c = min(n - 1, max(0, state[1] + dc))
        return (r, c)

    def _index(self, cell: tuple[int, int]) -> int:
        return cell[0] * self.cfg.grid_size + cell[1]

    def cell(self, index: int) -> tuple[int, int]:
        """Convert a state index back to a ``(row, col)`` cell."""
        return (index // self.cfg.grid_size, index % self.cfg.grid_size)
