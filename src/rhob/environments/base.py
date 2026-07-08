"""Abstract environment contract.

Every RHOB environment is a *data generator*: given a seed (and optional
difficulty), it produces a complete :class:`~rhob.core.trajectory.Trajectory`
representing one training run, with a ground-truth onset label attached by the
environment's oracle.

The benchmark deliberately does **not** couple detection evaluation to any RL
algorithm: an environment records the training dynamics once, and detectors are
evaluated on the pre-recorded trajectories (engineering spec Section 8, gap
analysis Section 10.1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from rhob.core.trajectory import Trajectory
from rhob.core.types import HackingType, Tier


@dataclass
class EnvironmentCard:
    """Structured metadata describing an environment (for docs and registry)."""

    id: str
    name: str
    tier: Tier
    hacking_type: HackingType
    true_objective: str
    proxy_reward: str
    hacking_mechanism: str
    oracle_mechanism: str
    difficulty_knob: Optional[str] = None
    compute_per_seed: str = ""
    citation: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "tier": self.tier.value,
            "hacking_type": self.hacking_type.value,
            "true_objective": self.true_objective,
            "proxy_reward": self.proxy_reward,
            "hacking_mechanism": self.hacking_mechanism,
            "oracle_mechanism": self.oracle_mechanism,
            "difficulty_knob": self.difficulty_knob,
            "compute_per_seed": self.compute_per_seed,
            "citation": self.citation,
        }


@dataclass
class ValidationReport:
    """Result of an environment self-check."""

    environment_id: str
    n_seeds: int
    n_hacking: int
    hacking_reliability: float
    onset_steps: list[int] = field(default_factory=list)
    passed: bool = False
    message: str = ""


class AbstractEnvironment(ABC):
    """Contract every RHOB environment must satisfy.

    Subclasses set the identity/specification class attributes and implement
    :meth:`generate`. The base class provides :meth:`describe` and
    :meth:`validate` generically.
    """

    # --- Identity (set by subclass) ---
    id: str = ""
    name: str = ""
    tier: Tier = Tier.TIER1
    hacking_type: HackingType = HackingType.REWARD_TAMPERING
    description: str = ""

    # --- Specifications ---
    state_dim: int = 0
    action_dim: int = 0
    action_type: str = "discrete"
    max_steps: int = 0
    expected_onset_range: tuple[int, int] = (0, 0)
    hacking_reliability: float = 0.0

    # --- Difficulty ---
    difficulty_knob: Optional[str] = None
    difficulty_range: Optional[tuple[float, float]] = None
    difficulty_default: Optional[float] = None

    # --- Card content (set by subclass, used by describe()) ---
    true_objective: str = ""
    proxy_reward: str = ""
    hacking_mechanism: str = ""
    oracle_mechanism: str = ""
    compute_per_seed: str = ""
    citation: Optional[str] = None

    @abstractmethod
    def generate(
        self,
        seed: int,
        difficulty: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> Trajectory:
        """Generate a complete training trajectory with an onset label.

        Parameters
        ----------
        seed
            Random seed. Determinism guarantee: the same ``(seed, difficulty,
            config)`` must always yield an identical trajectory.
        difficulty
            Value of the difficulty knob; ``None`` uses :attr:`difficulty_default`.
        config
            Optional overrides. Notably ``config["hacking"] = False`` requests a
            *clean* run (the exploit is not accessible).
        """
        raise NotImplementedError

    def describe(self) -> EnvironmentCard:
        """Return structured metadata for this environment."""
        return EnvironmentCard(
            id=self.id,
            name=self.name,
            tier=self.tier,
            hacking_type=self.hacking_type,
            true_objective=self.true_objective,
            proxy_reward=self.proxy_reward,
            hacking_mechanism=self.hacking_mechanism,
            oracle_mechanism=self.oracle_mechanism,
            difficulty_knob=self.difficulty_knob,
            compute_per_seed=self.compute_per_seed,
            citation=self.citation,
        )

    def validate(self, n_seeds: int = 10, min_reliability: float = 0.6) -> ValidationReport:
        """Self-check: does this environment reliably produce hacking?

        Generates ``n_seeds`` *hacking-configured* runs and measures the fraction
        that produce a labelled onset. Passes if the reliability meets
        ``min_reliability`` (default 0.6, per the Milestone 1 success criterion).
        """
        onset_steps: list[int] = []
        n_hacking = 0
        for seed in range(n_seeds):
            traj = self.generate(seed=seed, config={"hacking": True})
            if traj.onset_label is not None:
                n_hacking += 1
                onset_steps.append(traj.onset_label.onset_step)
        reliability = n_hacking / n_seeds if n_seeds else 0.0
        passed = reliability >= min_reliability
        return ValidationReport(
            environment_id=self.id,
            n_seeds=n_seeds,
            n_hacking=n_hacking,
            hacking_reliability=reliability,
            onset_steps=onset_steps,
            passed=passed,
            message=(
                f"{n_hacking}/{n_seeds} runs hacked "
                f"({reliability:.0%}); threshold {min_reliability:.0%}."
            ),
        )
