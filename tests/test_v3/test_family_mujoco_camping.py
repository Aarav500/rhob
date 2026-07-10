"""Tests for MuJoCo Camping Exploit (CAMPING_EXPLOIT mechanism, HalfCheetah-v5)."""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism

from admission_helpers import assert_admitted


def test_registered():
    fam = FamilyRegistry.get("mujoco_camping")
    assert fam.name == "mujoco_camping"
    assert fam.mechanism == HackingMechanism.CAMPING_EXPLOIT
    assert fam.complexity == EnvironmentComplexity.CONTINUOUS_COMPLEX


def test_admitted_at_easiest_difficulty():
    fam = FamilyRegistry.get("mujoco_camping")
    lo, hi = fam.difficulty_range()
    assert_admitted(fam, difficulty=hi, n_seeds_per_variant=10)


def test_true_reward_diverges():
    fam = FamilyRegistry.get("mujoco_camping")
    pair = fam.generate_pair(fam.difficulty_range()[1], seed=1)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_b.true_rewards.mean() > run_a.true_rewards.mean()


def test_onset_convention():
    fam = FamilyRegistry.get("mujoco_camping")
    pair = fam.generate_pair(fam.difficulty_range()[1], seed=2)
    _, onset_a = pair.rollout_hacking(0)
    _, onset_b = pair.rollout_legit(0)
    assert onset_a == 0
    assert onset_b == -1
