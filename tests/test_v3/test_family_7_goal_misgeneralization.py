"""Tests for GoalMisgeneralization family."""

import numpy as np

from rhob.v3.registry import FamilyRegistry

from admission_helpers import assert_admitted


def test_admitted():
    """Full 5-criterion admission check via AdmissionGate.

    Regression test: an earlier revision scaled the hacking variant's movement
    speed by an extra "commitment=0.9" factor the legitimate variant's movement
    didn't have, so the two variants converged to their respective goals at
    different rates. That broke "proxy matched by construction" (mean L0 AUROC
    was ~0.73, not noise) but the single-seed, +/-20%-tolerance sum comparison
    in test_proxy_matched below was too weak to catch it.
    """
    fam = FamilyRegistry.get("goal_misgeneralization")
    assert_admitted(fam)


def test_family_registered():
    """Family is discoverable."""
    fam = FamilyRegistry.get("goal_misgeneralization")
    assert fam.name == "goal_misgeneralization"


def test_difficulty_range():
    """Range is sensible."""
    fam = FamilyRegistry.get("goal_misgeneralization")
    lo, hi = fam.difficulty_range()
    assert lo < hi
    assert 0.5 < lo < 1.0
    assert 0.5 < hi < 1.0


def test_proxy_matched():
    """Proxy totals are similar between hacking and legitimate."""
    fam = FamilyRegistry.get("goal_misgeneralization")
    pair = fam.generate_pair(0.90, seed=0)

    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)

    proxy_a = run_a.proxy_rewards.sum()
    proxy_b = run_b.proxy_rewards.sum()

    # Allow ±20% variance
    assert abs(proxy_a - proxy_b) / max(abs(proxy_a), abs(proxy_b)) < 0.2


def test_true_reward_diverges():
    """True rewards diverge (hacking low, legitimate high)."""
    fam = FamilyRegistry.get("goal_misgeneralization")
    pair = fam.generate_pair(0.90, seed=0)

    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)

    true_a = run_a.true_rewards.sum()
    true_b = run_b.true_rewards.sum()

    assert true_a < true_b


def test_feature_anti_symmetry():
    """Behavioral feature is anti-symmetric (hacking positive, legit negative)."""
    fam = FamilyRegistry.get("goal_misgeneralization")
    pair = fam.generate_pair(0.90, seed=0)

    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)

    feature_a = np.mean(run_a.behav_trace[-50:])
    feature_b = np.mean(run_b.behav_trace[-50:])

    # Hacking should have positive feature, legitimate negative
    assert feature_a > feature_b


def test_difficulty_sweep():
    """Difficulty knob affects feature separation."""
    fam = FamilyRegistry.get("goal_misgeneralization")

    separations = []
    for d in [0.98, 0.80, 0.60]:
        pair = fam.generate_pair(d, seed=0)

        run_a, _ = pair.rollout_hacking(0)
        run_b, _ = pair.rollout_legit(0)

        feat_a = np.mean(run_a.behav_trace[-50:])
        feat_b = np.mean(run_b.behav_trace[-50:])

        sep = abs(feat_a - feat_b)
        separations.append(sep)

    # Separation should increase with difficulty
    assert separations[0] < separations[1] < separations[2]


def test_deterministic():
    """Same seed produces same rollout."""
    fam = FamilyRegistry.get("goal_misgeneralization")
    pair1 = fam.generate_pair(0.80, seed=42)
    pair2 = fam.generate_pair(0.80, seed=42)

    run1a, _ = pair1.rollout_hacking(0)
    run2a, _ = pair2.rollout_hacking(0)

    assert np.allclose(run1a.proxy_rewards, run2a.proxy_rewards)
    assert np.allclose(run1a.true_rewards, run2a.true_rewards)
