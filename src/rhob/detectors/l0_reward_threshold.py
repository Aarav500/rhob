"""Reward threshold detector (L0, trivial baseline)."""

from __future__ import annotations

from rhob.detectors.posthoc import PosthocDetector, RunData


class RewardThresholdDetector(PosthocDetector):
    """Simplest baseline: classify by late-training mean reward.

    Self-calibrating (no absolute reward-scale assumption): the onset
    threshold is the midpoint between each run's own early-training baseline
    and late-training steady state, so the detector applies unchanged across
    pairs whose proxy magnitudes differ by orders of magnitude (gridworld
    returns are raw per-episode sums; continuous returns are per-step means).
    """

    def __init__(self, baseline_episodes: int = 20, late_window: int = 100):
        self.baseline_episodes = baseline_episodes
        self.late_window = late_window

    @property
    def access_level(self) -> str:
        return "L0"

    @property
    def name(self) -> str:
        return "Reward Threshold"

    def classify(self, run: RunData) -> float:
        """Score: raw mean proxy reward in the final window (rank-order only)."""
        window = run.proxy_rewards[-self.late_window :]
        return float(window.mean()) if len(window) else 0.0

    def detect_onset(self, run: RunData) -> int:
        """Detect the first sustained crossing of the run's own baseline/steady midpoint."""
        proxy = run.proxy_rewards
        if len(proxy) < self.baseline_episodes + 1:
            return -1
        baseline = float(proxy[: self.baseline_episodes].mean())
        steady = float(proxy[-self.baseline_episodes :].mean())
        if steady == baseline:
            return -1
        threshold = (baseline + steady) / 2.0
        rising = steady > baseline
        for t, r in enumerate(proxy):
            if (r > threshold) == rising:
                return t
        return -1
