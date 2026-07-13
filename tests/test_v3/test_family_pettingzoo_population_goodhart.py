# tests/test_v3/test_family_pettingzoo_population_goodhart.py
"""Tests for PettingZoo Population-Level Goodhart (PROXY_GAMING mechanism)."""

from __future__ import annotations

import pytest

pytest.importorskip("mpe2")

import rhob.v3.families.pettingzoo_population_goodhart  # noqa: F401 -- self-import until Task 7
from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism


def test_registered():
    fam = FamilyRegistry.get("pettingzoo_population_goodhart")
    assert fam.name == "pettingzoo_population_goodhart"
    assert fam.mechanism == HackingMechanism.PROXY_GAMING
    assert fam.complexity == EnvironmentComplexity.MULTI_AGENT


def test_true_reward_diverges():
    fam = FamilyRegistry.get("pettingzoo_population_goodhart")
    pair = fam.generate_pair(fam.difficulty_range()[1], seed=1)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_b.true_rewards.mean() > run_a.true_rewards.mean()


def test_behav_fn_ranks_hacking_above_legit():
    """Regression test for the sign-convention bug found during construction:
    an earlier draft's behav_fn (raw free-rider velocity magnitude, unnegated)
    ranked legit ABOVE hacking, the opposite of BehavioralThresholdDetector's
    "higher score = more hacking-like" convention."""
    fam = FamilyRegistry.get("pettingzoo_population_goodhart")
    pair = fam.generate_pair(fam.difficulty_range()[1], seed=1)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_a.behav_trace.mean() > run_b.behav_trace.mean()
