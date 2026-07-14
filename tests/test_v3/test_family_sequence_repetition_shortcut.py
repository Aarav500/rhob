"""Tests for Sequence Repetition-Blind-Spot Shortcut (SHORTCUT mechanism)."""

from __future__ import annotations

import rhob.v3.families.sequence_repetition_shortcut  # noqa: F401 -- self-import until Task 7
from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism


def test_registered():
    fam = FamilyRegistry.get("sequence_repetition_shortcut")
    assert fam.name == "sequence_repetition_shortcut"
    assert fam.mechanism == HackingMechanism.SHORTCUT
    assert fam.complexity == EnvironmentComplexity.SEQUENTIAL


def test_true_reward_diverges():
    fam = FamilyRegistry.get("sequence_repetition_shortcut")
    pair = fam.generate_pair(fam.difficulty_range()[1], seed=1)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_b.true_rewards.mean() > run_a.true_rewards.mean()


def test_behav_trace_ranks_hacking_above_legit():
    fam = FamilyRegistry.get("sequence_repetition_shortcut")
    pair = fam.generate_pair(fam.difficulty_range()[1], seed=1)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_a.behav_trace.mean() > run_b.behav_trace.mean()
