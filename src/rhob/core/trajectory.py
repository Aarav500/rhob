"""Trajectory and observation data models.

A :class:`Trajectory` is a time-indexed record of a single training run. Each
index ``t`` corresponds to one *training step* (here, one training episode of
the RL agent). At every step we record:

* ``reward_proxy[t]`` -- the proxy return the agent obtained (the observable,
  possibly-misspecified reward). Available at access level L1+.
* ``reward_true[t]``  -- the true return (oracle-only; used for labelling, never
  shown to detectors).
* ``policy_features[t]`` -- a behavioural summary of the policy at step ``t``
  (e.g. state-visitation frequencies). This is the L2 signal.

This "training-curve" representation is the substrate of the entire benchmark:
onset detection asks *when, during training,* the proxy and true objectives
diverge, so the natural time axis is the training step, not the within-episode
environment step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import numpy as np

from rhob.core.onset import OnsetLabel
from rhob.core.types import AccessLevel, HackingType


@dataclass(frozen=True, eq=False)
class Observation:
    """Immutable, access-filtered view of a single training step.

    Detectors consume a stream of :class:`Observation` objects. The object is
    *frozen* so a detector cannot mutate it, and fields above the detector's
    declared access level are ``None`` -- this prevents information leakage by
    construction (see :class:`rhob.core.access.AccessFilter`).
    """

    t: int
    access_level: AccessLevel
    reward: float  # proxy reward, always available (L1)
    policy_features: Optional[np.ndarray] = None  # L2 behavioural summary

    @property
    def has_features(self) -> bool:
        """Whether L2 behavioural features are available at this access level."""
        return self.policy_features is not None


@dataclass
class Timestep:
    """A single training step of a trajectory (full, unfiltered)."""

    t: int
    reward_proxy: float
    reward_true: Optional[float] = None
    policy_features: Optional[np.ndarray] = None


@dataclass
class Trajectory:
    """A complete recorded training run.

    Attributes
    ----------
    environment_id
        The environment that produced this run.
    seed
        Random seed used for generation.
    algorithm
        Training algorithm used (e.g. ``"tabular_q_learning"``).
    is_hacking_run
        Whether this run was configured with an accessible exploit. Clean runs
        (``False``) must never trigger a detector.
    reward_proxy, reward_true
        Arrays of shape ``(T,)`` giving the per-step proxy and true returns.
    policy_features
        Array of shape ``(T, F)`` giving the per-step behavioural summary, or
        ``None`` if not recorded.
    onset_label
        The ground-truth onset annotation, or ``None`` if no hacking occurred.
    hacking_type
        Category of hacking this environment exhibits.
    """

    environment_id: str
    seed: int
    algorithm: str
    is_hacking_run: bool
    reward_proxy: np.ndarray
    reward_true: np.ndarray
    policy_features: Optional[np.ndarray] = None
    onset_label: Optional[OnsetLabel] = None
    hacking_type: Optional[HackingType] = None
    generation_timestamp: str = ""
    config_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.reward_proxy = np.asarray(self.reward_proxy, dtype=np.float64)
        self.reward_true = np.asarray(self.reward_true, dtype=np.float64)
        if self.reward_proxy.shape != self.reward_true.shape:
            raise ValueError(
                "reward_proxy and reward_true must have the same shape, got "
                f"{self.reward_proxy.shape} vs {self.reward_true.shape}"
            )
        if self.reward_proxy.ndim != 1:
            raise ValueError("reward arrays must be 1-D (shape (T,))")
        if self.policy_features is not None:
            self.policy_features = np.asarray(self.policy_features, dtype=np.float64)
            if self.policy_features.shape[0] != self.total_steps:
                raise ValueError(
                    "policy_features must have leading dimension T="
                    f"{self.total_steps}, got {self.policy_features.shape}"
                )
        if self.onset_label is not None and self.onset_label.onset_step >= self.total_steps:
            raise ValueError(
                f"onset_step {self.onset_label.onset_step} is outside trajectory "
                f"of length {self.total_steps}"
            )

    @property
    def total_steps(self) -> int:
        return int(self.reward_proxy.shape[0])

    def __len__(self) -> int:
        return self.total_steps

    @property
    def onset_step(self) -> Optional[int]:
        return None if self.onset_label is None else self.onset_label.onset_step

    def binary_labels(self) -> np.ndarray:
        """Per-step ground-truth labels for classification metrics.

        Returns an integer array of shape ``(T,)`` where entry ``t`` is ``1`` if
        the run is a hacking run and ``t >= onset_step`` (hacking is active),
        else ``0``. Clean runs are all-zero.
        """
        labels = np.zeros(self.total_steps, dtype=np.int64)
        if self.is_hacking_run and self.onset_label is not None:
            labels[self.onset_label.onset_step :] = 1
        return labels

    def iter_observations(self, access_level: AccessLevel) -> Iterator[Observation]:
        """Yield access-filtered observations, one per training step."""
        # Local import avoids a circular dependency (access -> trajectory).
        from rhob.core.access import AccessFilter

        filt = AccessFilter(access_level)
        has_feats = self.policy_features is not None
        for t in range(self.total_steps):
            feats = self.policy_features[t] if has_feats else None
            yield filt.filter(
                t=t,
                reward_proxy=float(self.reward_proxy[t]),
                policy_features=feats,
            )
