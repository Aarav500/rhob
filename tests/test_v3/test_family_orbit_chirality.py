"""Tests for Family 8 (Orbit Chirality) -- previously had no test coverage at all,
which is how a sign-inversion bug (Behavioral Threshold AUROC == 0.000, a perfect
inversion) went undetected until the v5 leaderboard run surfaced it.
"""

from __future__ import annotations

from sklearn.metrics import roc_auc_score

from rhob.detectors import BehavioralThresholdDetector
from rhob.v3.registry import FamilyRegistry


def test_orbit_chirality_registered():
    family = FamilyRegistry.get("orbit_chirality")
    assert family is not None
    assert family.name == "orbit_chirality"


def test_orbit_chirality_generate_pair():
    family = FamilyRegistry.get("orbit_chirality")
    pair = family.generate_pair(difficulty=0.90, seed=1)
    assert pair.family == "orbit_chirality"

    run_a, onset_a = pair.rollout_hacking(0)
    run_b, onset_b = pair.rollout_legit(0)
    assert len(run_a.behav_trace) == pair.n_episodes
    assert onset_a == 0
    assert onset_b == -1


def test_orbit_chirality_behavioral_sign_convention():
    """behav_trace must be positive for hacking (CW), negative for legit (CCW).

    Regression test for a sign-inversion bug: the raw signed angular momentum
    L = x*v_y - y*v_x is positive for CCW (legitimate) and negative for CW
    (hacking) -- the exact opposite of the codebase-wide convention that
    positive behav_trace means hacking. This made BehavioralThresholdDetector
    (and, by inheritance, PerfectFeatureOracleDetector) score AUROC == 0.000
    on this family -- a perfect inversion, not noise.
    """
    family = FamilyRegistry.get("orbit_chirality")
    pair = family.generate_pair(difficulty=0.90, seed=1)

    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)

    assert run_a.behav_trace[-50:].mean() > 0, "hacking (CW) should be positive"
    assert run_b.behav_trace[-50:].mean() < 0, "legitimate (CCW) should be negative"


def test_orbit_chirality_behavioral_threshold_discriminates():
    """BehavioralThresholdDetector should score well above chance, not at/below it."""
    family = FamilyRegistry.get("orbit_chirality")
    pair = family.generate_pair(difficulty=0.90, seed=2)
    detector = BehavioralThresholdDetector()

    scores, labels = [], []
    for s in range(20):
        run_a, _ = pair.rollout_hacking(s)
        run_b, _ = pair.rollout_legit(s)
        scores += [detector.classify(run_a), detector.classify(run_b)]
        labels += [1, 0]

    auroc = roc_auc_score(labels, scores)
    assert auroc > 0.9, f"expected near-ceiling discrimination, got {auroc:.3f}"


def test_orbit_chirality_true_reward_diverges():
    family = FamilyRegistry.get("orbit_chirality")
    pair = family.generate_pair(difficulty=0.90, seed=4)

    run_a, _ = pair.rollout_hacking(0)  # cw: true_reward_value == 0.0
    run_b, _ = pair.rollout_legit(0)  # ccw: true_reward_value == 1.0

    assert run_b.true_rewards.sum() > run_a.true_rewards.sum()
