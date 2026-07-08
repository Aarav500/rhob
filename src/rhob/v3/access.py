"""Access-level enforcement: structurally prevent a detector from reading above its level.

Every detector-facing :class:`~rhob.detectors.posthoc.RunData` is passed through
:func:`restrict`, which nulls the fields the detector is not entitled to. The oracle
``true_rewards`` channel is nulled at L0--L2: it exists only for scoring. ``L3`` is the
one exception, deliberately: it is not a real access level a practical detector could
occupy, but a *ceiling* measurement -- an oracle given the true reward directly, used
only to bound how well detection could ever do, never to claim a practical result.
"""

from __future__ import annotations

import numpy as np

from rhob.detectors.posthoc import RunData

_LEVELS = ("L0", "L1", "L2", "L3")


def restrict(run: RunData, level: str) -> RunData:
    """Return a copy of ``run`` exposing only the signals allowed at ``level``.

    Args:
        run: The full run data.
        level: ``"L0"`` (reward only), ``"L1"`` (+ state histograms), ``"L2"``
            (+ behavioral trace), or ``"L3"`` (oracle ceiling: + true reward).
    """
    if level not in _LEVELS:
        raise ValueError(f"level must be one of {_LEVELS}, got {level!r}")
    empty = np.array([])
    state_counts = run.state_counts if level in ("L1", "L2", "L3") else None
    behav_trace = run.behav_trace if level in ("L2", "L3") else None
    true_rewards = run.true_rewards if level == "L3" else empty
    return RunData(
        proxy_rewards=run.proxy_rewards,
        true_rewards=true_rewards,
        state_counts=state_counts,
        behav_trace=behav_trace,
    )
