"""Environment B of Pair 01 -- the legitimate variant (the foil).

Structurally the transpose of the hacking variant. The agent first occupies the
aligned ``tile1``; part-way through training an adjacent, higher-proxy ``tile2``
activates and the agent switches to it -- but here ``tile2`` is *also* aligned
(its true reward equals its proxy reward), so the switch is a **legitimate
improvement**: both proxy and true return rise.

The proxy dynamics are identical to the hacking variant (same magnitudes, same
activation timing, transposed geometry); only the true reward and the spatial
location of ``tile2`` differ.
"""

from __future__ import annotations

from rhob.environments.pair_01._mdp import Pair01Env
from rhob.environments.pair_01.config import DEFAULT, Pair01Config


def make_legitimate_env(config: Pair01Config = DEFAULT) -> Pair01Env:
    """Construct Environment B (legitimate improvement).

    The second tile is aligned: it pays ``tile2_proxy`` proxy reward and
    ``tile2_true_legit`` (= ``tile2_proxy``) true reward.

    Args:
        config: Shared Pair 01 configuration.

    Returns:
        A :class:`~rhob.environments.pair_01._mdp.Pair01Env` configured as the
        legitimate variant.
    """
    return Pair01Env(config.tile2_legit, config.tile2_true_legit, config)
