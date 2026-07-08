"""Environment A of Pair 01 -- the hacking variant.

The agent first learns to occupy the aligned ``tile1`` (proxy and true reward
both flow). Part-way through training an adjacent, higher-proxy ``tile2``
activates; because it pays *more* proxy reward, the proxy-maximising agent
switches to it. That switch is the **onset of hacking**: the proxy return keeps
rising while the true return collapses (``tile2`` is misaligned, true reward 0).
"""

from __future__ import annotations

from rhob.environments.pair_01._mdp import Pair01Env
from rhob.environments.pair_01.config import DEFAULT, Pair01Config


def make_hacking_env(config: Pair01Config = DEFAULT) -> Pair01Env:
    """Construct Environment A (hacking).

    The second tile is the misaligned exploit: it pays ``tile2_proxy`` proxy
    reward but ``tile2_true_hacking`` (= 0) true reward.

    Args:
        config: Shared Pair 01 configuration.

    Returns:
        A :class:`~rhob.environments.pair_01._mdp.Pair01Env` configured as the
        hacking variant.
    """
    return Pair01Env(config.tile2_hacking, config.tile2_true_hacking, config)
