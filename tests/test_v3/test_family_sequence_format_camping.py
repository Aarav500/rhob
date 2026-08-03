"""Tests for Sequence Format-Compliance Camping (CAMPING_EXPLOIT mechanism)."""

from __future__ import annotations

import pytest

import rhob.v3.families.sequence_format_camping  # noqa: F401 -- self-import until Task 7
from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism


# `slow` module: deselected from the default `pytest tests/` run and executed
# by the nightly `admission-slow` CI job (see [tool.pytest.ini_options] in
# pyproject.toml). Nothing here is skipped -- it runs on a different schedule.
# Every test below rolls out the family, and hidden-Markov sequence rollouts cost ~9s each and sit
# behind a 90-165s proxy calibration per difficulty.
pytestmark = pytest.mark.slow


def test_registered():
    fam = FamilyRegistry.get("sequence_format_camping")
    assert fam.name == "sequence_format_camping"
    assert fam.mechanism == HackingMechanism.CAMPING_EXPLOIT
    assert fam.complexity == EnvironmentComplexity.SEQUENTIAL


def test_true_reward_diverges():
    fam = FamilyRegistry.get("sequence_format_camping")
    pair = fam.generate_pair(fam.default_difficulties()[0], seed=1)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_b.true_rewards.mean() > run_a.true_rewards.mean()


def test_behav_trace_ranks_hacking_above_legit():
    fam = FamilyRegistry.get("sequence_format_camping")
    pair = fam.generate_pair(fam.default_difficulties()[0], seed=1)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_a.behav_trace.mean() > run_b.behav_trace.mean()
