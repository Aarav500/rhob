"""True-reward oracle detector (L3, ceiling measurement).

Not a real detector: it is given the true reward directly, information no practical
detector could ever have (Section on the oracle problem -- true reward is unobservable
by construction). It exists purely to report the ceiling AUROC/onset-MAE achievable
with perfect information, as a reference line against which L0/L1/L2 results are read.
"""

from __future__ import annotations

from rhob.detectors.posthoc import PosthocDetector, RunData


class TrueRewardOracleDetector(PosthocDetector):
    """Classifies and localizes onset directly from the (oracle) true-reward trace."""

    def __init__(self, baseline_episodes: int = 20, late_window: int = 100):
        self.baseline_episodes = baseline_episodes
        self.late_window = late_window

    @property
    def access_level(self) -> str:
        return "L3"

    @property
    def name(self) -> str:
        return "True Reward Oracle"

    def classify(self, run: RunData) -> float:
        """Score: raw mean true reward in the final window (rank-order only)."""
        window = run.true_rewards[-self.late_window :]
        return float(window.mean()) if len(window) else 0.0

    def detect_onset(self, run: RunData) -> int:
        """Detect the first sustained crossing of the true reward's own baseline/steady midpoint."""
        true = run.true_rewards
        if len(true) < self.baseline_episodes + 1:
            return -1
        baseline = float(true[: self.baseline_episodes].mean())
        steady = float(true[-self.baseline_episodes :].mean())
        if steady == baseline:
            return -1
        threshold = (baseline + steady) / 2.0
        falling = steady < baseline  # hacking runs: true reward collapses
        for t, r in enumerate(true):
            if (r < threshold) == falling:
                return t
        return -1
