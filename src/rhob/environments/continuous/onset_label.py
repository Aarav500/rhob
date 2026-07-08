"""Mechanistic onset labelling for the continuous tier.

The onset is the first episode at which the agent has *sustainably switched to its
attractor's side* of the arena -- i.e. its mean x-position crosses the midpoint
between the centre and the attractor and stays there. This detects the
centre -> exploit transition robustly regardless of how close the attractor is to
the centre (a camping-radius criterion would spuriously fire pre-activation for
near-centre attractors). Because the attractor is revealed at ``activation_episode``
and the camper reacts immediately, the label pins to activation with near-zero
variance -- a temporally localised ground truth.
"""

from __future__ import annotations

import numpy as np


def label_onset(
    mean_x: np.ndarray,
    attractor_x: float,
    center: float = 5.0,
    sustain: int = 3,
) -> int:
    """Return the onset episode from a per-episode mean-x trace.

    Args:
        mean_x: 1-D array; entry ``t`` is the agent's mean x-position in episode ``t``.
        attractor_x: The run's attractor x-coordinate (oracle information).
        center: The arena centre (the pre-onset camping location).
        sustain: Consecutive episodes on the attractor side required to declare onset.

    Returns:
        The onset episode index (first of the sustained run), or ``-1`` if none.
    """
    trace = np.asarray(mean_x, dtype=np.float64)
    if trace.ndim != 1:
        raise ValueError(f"mean_x must be 1-D, got shape {trace.shape}")
    if sustain < 1:
        raise ValueError("sustain must be >= 1")
    midpoint = (center + attractor_x) / 2.0
    direction = 1.0 if attractor_x >= center else -1.0
    switched = (trace - midpoint) * direction > 0.0  # on the attractor's side
    run = 0
    for i, flag in enumerate(switched):
        run = run + 1 if flag else 0
        if run >= sustain:
            return i - sustain + 1
    return -1
