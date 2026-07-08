"""Tests for DistributionalShift family."""

import numpy as np
from rhob.v3.registry import FamilyRegistry


def test_family_registered():
    """Family is discoverable."""
    fam = FamilyRegistry.get("distributional_shift")
    assert fam.name == "distributional_shift"


def test_difficulty_range():
    """Range is sensible."""
    fam = FamilyRegistry.get("distributional_shift")
    lo, hi = fam.difficulty_range()
    assert lo < hi
    assert 0.5 < lo < 1.0
    assert 0.5 < hi < 1.0


def test_proxy_matched():
    """Proxy totals are similar between hacking and legitimate."""
    fam = FamilyRegistry.get("distributional_shift")
    pair = fam.generate_pair(0.90, seed=0)

    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)

    proxy_a = run_a.proxy_rewards.sum()
    proxy_b = run_b.proxy_rewards.sum()

    # Allow ±20% variance
    assert abs(proxy_a - proxy_b) / max(abs(proxy_a), abs(proxy_b)) < 0.2


def test_true_reward_diverges():
    """True rewards diverge (hacking low, legitimate high)."""
    fam = FamilyRegistry.get("distributional_shift")
    pair = fam.generate_pair(0.90, seed=0)

    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)

    true_a = run_a.true_rewards.sum()
    true_b = run_b.true_rewards.sum()

    assert true_a < true_b


def test_deterministic():
    """Same seed produces same rollout."""
    fam = FamilyRegistry.get("distributional_shift")
    pair1 = fam.generate_pair(0.80, seed=42)
    pair2 = fam.generate_pair(0.80, seed=42)

    run1a, _ = pair1.rollout_hacking(0)
    run2a, _ = pair2.rollout_hacking(0)

    assert np.allclose(run1a.proxy_rewards, run2a.proxy_rewards)
    assert np.allclose(run1a.true_rewards, run2a.true_rewards)
