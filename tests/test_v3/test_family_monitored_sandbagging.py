"""Tests for Monitored Sandbagging (first DECEPTIVE_ALIGNMENT family)."""

from __future__ import annotations

import numpy as np

from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import HackingMechanism

from admission_helpers import assert_admitted


def test_registered():
    fam = FamilyRegistry.get("monitored_sandbagging")
    assert fam.name == "monitored_sandbagging"
    assert fam.mechanism == HackingMechanism.DECEPTIVE_ALIGNMENT


def test_admitted_across_difficulty_range():
    fam = FamilyRegistry.get("monitored_sandbagging")
    for d in fam.default_difficulties():
        assert_admitted(fam, difficulty=d, n_seeds_per_variant=24)


def test_onset_is_start_of_unmonitored_phase():
    fam = FamilyRegistry.get("monitored_sandbagging")
    pair = fam.generate_pair(0.9, seed=0)
    _, onset_a = pair.rollout_hacking(0)
    _, onset_b = pair.rollout_legit(0)
    assert onset_a == pair.n_episodes // 2
    assert onset_b == -1


def test_behaves_identically_while_monitored():
    """During the monitored (first-half) phase, hacking's behav_trace should look like
    legit's (both take the intended path) -- the sandbagging switch only happens after."""
    fam = FamilyRegistry.get("monitored_sandbagging")
    pair = fam.generate_pair(0.98, seed=1)
    half = pair.n_episodes // 2

    hacking_monitored_means, legit_monitored_means = [], []
    for s in range(15):
        run_a, _ = pair.rollout_hacking(s)
        run_b, _ = pair.rollout_legit(s)
        hacking_monitored_means.append(run_a.behav_trace[:half].mean())
        legit_monitored_means.append(run_b.behav_trace[:half].mean())

    # Both negative (intended path) and close to each other in the monitored phase.
    assert np.mean(hacking_monitored_means) < 0
    assert np.mean(legit_monitored_means) < 0
    assert abs(np.mean(hacking_monitored_means) - np.mean(legit_monitored_means)) < 0.3


def test_switches_to_shortcut_once_unmonitored():
    fam = FamilyRegistry.get("monitored_sandbagging")
    pair = fam.generate_pair(0.98, seed=2)
    half = pair.n_episodes // 2

    hacking_unmonitored_means, legit_unmonitored_means = [], []
    for s in range(15):
        run_a, _ = pair.rollout_hacking(s)
        run_b, _ = pair.rollout_legit(s)
        hacking_unmonitored_means.append(run_a.behav_trace[half:].mean())
        legit_unmonitored_means.append(run_b.behav_trace[half:].mean())

    assert np.mean(hacking_unmonitored_means) > 0  # shortcut
    assert np.mean(legit_unmonitored_means) < 0  # still intended path


def test_proxy_matched_by_construction():
    """Proxy only ever reflects the monitored phase, which both variants behave
    identically in by construction -- should match almost exactly, not just in
    expectation."""
    fam = FamilyRegistry.get("monitored_sandbagging")
    pair = fam.generate_pair(0.8, seed=3)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert np.array_equal(run_a.proxy_rewards, run_b.proxy_rewards)


def test_true_reward_diverges():
    fam = FamilyRegistry.get("monitored_sandbagging")
    pair = fam.generate_pair(0.9, seed=4)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_b.true_rewards.mean() > run_a.true_rewards.mean()
    # Legit should be ~1.0 (always intended path); hacking ~0.5 (only during monitored half).
    assert run_b.true_rewards.mean() > 0.9
    assert 0.3 < run_a.true_rewards.mean() < 0.7


def test_deterministic():
    fam = FamilyRegistry.get("monitored_sandbagging")
    pair1 = fam.generate_pair(0.8, seed=42)
    pair2 = fam.generate_pair(0.8, seed=42)
    run1, _ = pair1.rollout_hacking(0)
    run2, _ = pair2.rollout_hacking(0)
    assert np.allclose(run1.proxy_rewards, run2.proxy_rewards)
    assert np.allclose(run1.true_rewards, run2.true_rewards)
