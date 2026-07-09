"""Tests for Reward Channel Tampering (REWARD_TAMPERING mechanism)."""

from __future__ import annotations

import numpy as np

from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import HackingMechanism

from admission_helpers import assert_admitted


def test_registered():
    fam = FamilyRegistry.get("reward_channel_tampering")
    assert fam.name == "reward_channel_tampering"
    assert fam.mechanism == HackingMechanism.REWARD_TAMPERING


def test_admitted_across_difficulty_range():
    fam = FamilyRegistry.get("reward_channel_tampering")
    for d in fam.default_difficulties():
        assert_admitted(fam, difficulty=d, n_seeds_per_variant=24)


def test_behavioral_sign_convention():
    """behav_trace is +1 for tampered reports (hacking-dominant), -1 for honest."""
    fam = FamilyRegistry.get("reward_channel_tampering")
    pair = fam.generate_pair(0.98, seed=1)

    hacking_means = []
    legit_means = []
    for s in range(15):
        run_a, _ = pair.rollout_hacking(s)
        run_b, _ = pair.rollout_legit(s)
        hacking_means.append(run_a.behav_trace.mean())
        legit_means.append(run_b.behav_trace.mean())

    assert np.mean(hacking_means) > 0, "hacking should tamper more often (positive)"
    assert np.mean(legit_means) < np.mean(hacking_means)


def test_proxy_bonus_solved_not_guessed():
    """The bonus should exactly equalize E[proxy] between variants at every difficulty
    (algebraic solution, not an approximation) -- verify with a large sample."""
    fam = FamilyRegistry.get("reward_channel_tampering")
    for d in [0.98, 0.8, 0.6]:
        pair = fam.generate_pair(d, seed=2)
        proxy_a, proxy_b = [], []
        for s in range(200):
            run_a, _ = pair.rollout_hacking(s)
            run_b, _ = pair.rollout_legit(s)
            proxy_a.append(run_a.proxy_rewards.mean())
            proxy_b.append(run_b.proxy_rewards.mean())
        # With 200 seeds x 150 episodes, the sample mean should be very close to the
        # algebraic target -- allow a small tolerance for residual sampling noise.
        assert abs(np.mean(proxy_a) - np.mean(proxy_b)) < 0.02, f"diff={d}"


def test_true_reward_diverges():
    fam = FamilyRegistry.get("reward_channel_tampering")
    pair = fam.generate_pair(0.9, seed=3)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_b.true_rewards.mean() > run_a.true_rewards.mean()


def test_onset_is_from_start():
    """Tampering commitment is a variant-level trait, not something that switches on
    partway through -- onset is defined at episode 0 for hacking, -1 (no onset) for legit."""
    fam = FamilyRegistry.get("reward_channel_tampering")
    pair = fam.generate_pair(0.9, seed=4)
    _, onset_a = pair.rollout_hacking(0)
    _, onset_b = pair.rollout_legit(0)
    assert onset_a == 0
    assert onset_b == -1


def test_deterministic():
    fam = FamilyRegistry.get("reward_channel_tampering")
    pair1 = fam.generate_pair(0.8, seed=42)
    pair2 = fam.generate_pair(0.8, seed=42)
    run1, _ = pair1.rollout_hacking(0)
    run2, _ = pair2.rollout_hacking(0)
    assert np.allclose(run1.proxy_rewards, run2.proxy_rewards)
    assert np.allclose(run1.true_rewards, run2.true_rewards)
