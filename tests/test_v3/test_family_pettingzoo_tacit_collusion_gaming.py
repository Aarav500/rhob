# tests/test_v3/test_family_pettingzoo_tacit_collusion_gaming.py
"""Tests for PettingZoo Tacit Collusion Gaming (CAMPING_EXPLOIT mechanism)."""

from __future__ import annotations

import pytest

pytest.importorskip("mpe2")

import rhob.v3.families.pettingzoo_tacit_collusion_gaming  # noqa: F401 -- self-import until Task 7
from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism

from admission_helpers import assert_admitted


def test_registered():
    fam = FamilyRegistry.get("pettingzoo_tacit_collusion_gaming")
    assert fam.name == "pettingzoo_tacit_collusion_gaming"
    assert fam.mechanism == HackingMechanism.CAMPING_EXPLOIT
    assert fam.complexity == EnvironmentComplexity.MULTI_AGENT


def test_admitted_at_easiest_difficulty():
    fam = FamilyRegistry.get("pettingzoo_tacit_collusion_gaming")
    lo, hi = fam.difficulty_range()
    assert_admitted(fam, difficulty=hi, n_seeds_per_variant=10)


def test_true_reward_diverges():
    fam = FamilyRegistry.get("pettingzoo_tacit_collusion_gaming")
    pair = fam.generate_pair(fam.difficulty_range()[1], seed=1)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_b.true_rewards.mean() > run_a.true_rewards.mean()
