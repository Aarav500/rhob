"""Tests for RLHF KL-Penalty Gaming (REWARD_SHAPING mechanism)."""

from __future__ import annotations

import rhob.v3.families.rlhf_kl_penalty_gaming  # noqa: F401 -- self-import until Task 7
from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism

from admission_helpers import assert_admitted


def test_registered():
    fam = FamilyRegistry.get("rlhf_kl_penalty_gaming")
    assert fam.name == "rlhf_kl_penalty_gaming"
    assert fam.mechanism == HackingMechanism.REWARD_SHAPING
    assert fam.complexity == EnvironmentComplexity.SEQUENTIAL


def test_admitted_at_easiest_difficulty():
    fam = FamilyRegistry.get("rlhf_kl_penalty_gaming")
    lo, hi = fam.difficulty_range()
    assert_admitted(fam, difficulty=hi, n_seeds_per_variant=10)


def test_true_reward_diverges():
    fam = FamilyRegistry.get("rlhf_kl_penalty_gaming")
    pair = fam.generate_pair(fam.difficulty_range()[1], seed=1)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_b.true_rewards.mean() > run_a.true_rewards.mean()
