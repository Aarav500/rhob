"""Tests for Sequence Keyword-Stuffing Gaming (PROXY_GAMING mechanism)."""

from __future__ import annotations

import rhob.v3.families.sequence_keyword_stuffing  # noqa: F401 -- self-import until Task 7
from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism


def test_registered():
    fam = FamilyRegistry.get("sequence_keyword_stuffing")
    assert fam.name == "sequence_keyword_stuffing"
    assert fam.mechanism == HackingMechanism.PROXY_GAMING
    assert fam.complexity == EnvironmentComplexity.SEQUENTIAL


def test_true_reward_diverges():
    fam = FamilyRegistry.get("sequence_keyword_stuffing")
    pair = fam.generate_pair(fam.difficulty_range()[1], seed=1)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_b.true_rewards.mean() > run_a.true_rewards.mean()


def test_behav_trace_ranks_hacking_above_legit():
    """Regression-style check for the exact sign-convention bug found in
    pettingzoo_population_goodhart this session: verify directly, not assume,
    that behav_trace ranks hacking ABOVE legit (higher = more hacking-like)."""
    fam = FamilyRegistry.get("sequence_keyword_stuffing")
    pair = fam.generate_pair(fam.difficulty_range()[1], seed=1)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_a.behav_trace.mean() > run_b.behav_trace.mean()
