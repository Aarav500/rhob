"""Reference agents used to validate benchmark environments.

Agents are *not* part of the detection benchmark; they only generate the training
trajectories that detectors are later scored on. Kept minimal and dependency-free.
"""

from rhob.agents.tabular_q import TabularQLearner

__all__ = ["TabularQLearner"]
