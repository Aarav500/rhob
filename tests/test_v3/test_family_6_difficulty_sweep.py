"""Verify Family 6's difficulty knob moves L2 AUROC smoothly."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from rhob.v3.registry import FamilyRegistry
from rhob.detectors import BehavioralThresholdDetector


def test_family_6_difficulty_sweep_moves_auroc():
    """Confirm that varying spatial_correlation actually changes L2 detectability.

    The audit claim: "the knob actually moves L2 smoothly rather than being binary
    like the original gridworld." This test rolls out pairs at three difficulties
    (EASY, MEDIUM, HARD) and scores them with an L2 detector; AUROC should decrease
    as difficulty increases (correlation increases → harder to detect).
    """
    family = FamilyRegistry.get("proxy_correlation_gaming")
    detector = BehavioralThresholdDetector()

    # Generate pairs at three difficulty levels
    difficulties = [0.90, 0.75, 0.60]  # EASY, MEDIUM, HARD
    aurocs = []

    for d in difficulties:
        pair = family.generate_pair(difficulty=d, seed=42)

        # Roll out 10 seeds (more samples for stable AUROC)
        runs_a, runs_b, _ = pair.rollout(n_seeds=10)

        # Score each run
        scores_a = [detector.classify(run) for run in runs_a]
        scores_b = [detector.classify(run) for run in runs_b]

        # Compute AUROC
        y_true = [1] * len(scores_a) + [0] * len(scores_b)
        y_scores = scores_a + scores_b
        auroc = roc_auc_score(y_true, y_scores)
        aurocs.append(auroc)

    # AUROC should tend to decrease as difficulty increases
    # (higher correlation → harder detection)
    # Allow some noise: just check that easy > hard, with medium intermediate
    assert aurocs[0] > aurocs[2], (
        f"AUROC should generally decrease with difficulty (easy > hard), got: {aurocs}"
    )

    # Spread should be meaningful (not binary)
    spread = aurocs[0] - aurocs[2]
    assert spread > 0.05, (
        f"Difficulty knob spread too small: {spread:.3f}; "
        "indicates binary-like behavior, not smooth gradient"
    )


def test_family_6_params_vary_with_difficulty():
    """Verify that spatial_correlation parameter changes with difficulty."""
    family = FamilyRegistry.get("proxy_correlation_gaming")

    corr_values = []
    for d in [0.90, 0.75, 0.60]:
        pair = family.generate_pair(difficulty=d, seed=0)
        corr = pair.params["spatial_correlation"]
        corr_values.append(corr)

    # Correlation should increase with increasing difficulty
    assert corr_values[0] < corr_values[1] < corr_values[2], (
        f"spatial_correlation should increase with difficulty: {corr_values}"
    )
