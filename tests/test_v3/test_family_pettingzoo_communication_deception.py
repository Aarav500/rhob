# tests/test_v3/test_family_pettingzoo_communication_deception.py
"""Tests for PettingZoo Communication-Channel Deception (DECEPTIVE_ALIGNMENT mechanism)."""

from __future__ import annotations

import pytest

pytest.importorskip("mpe2")

import rhob.v3.families.pettingzoo_communication_deception  # noqa: F401 -- self-import until Task 7
from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism

from admission_helpers import assert_smoke_admissible


# `slow` module: deselected from the default `pytest tests/` run and executed
# by the nightly `admission-slow` CI job (see [tool.pytest.ini_options] in
# pyproject.toml). Nothing here is skipped -- it runs on a different schedule.
# Every test below rolls out the family, and MPE multi-agent rollouts sit behind a proxy
# calibration run once per difficulty.
pytestmark = pytest.mark.slow


def test_registered():
    fam = FamilyRegistry.get("pettingzoo_communication_deception")
    assert fam.name == "pettingzoo_communication_deception"
    assert fam.mechanism == HackingMechanism.DECEPTIVE_ALIGNMENT
    assert fam.complexity == EnvironmentComplexity.MULTI_AGENT


def test_smoke_admissible_at_every_scored_difficulty():
    """Reduced-power screen at every difficulty the benchmark scores -- not certification.

    Two separate things were wrong with the test this replaces. It screened
    ``difficulty_range()[1]`` == 0.95, which ``default_difficulties()`` never returns, so
    the tiers the benchmark actually evaluates were covered by nothing. And it called what
    it did "admitted" while running what is now the shipped 576-rollout certification
    design -- which does not fit in CI. What this screen proves, and what it does not, is
    in ``admission_helpers``; the +/-0.10 certification is ``scripts/admission_ledger.py``'s.
    """
    assert_smoke_admissible(FamilyRegistry.get("pettingzoo_communication_deception"))


def test_true_reward_diverges():
    fam = FamilyRegistry.get("pettingzoo_communication_deception")
    pair = fam.generate_pair(fam.default_difficulties()[0], seed=1)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_b.true_rewards.mean() > run_a.true_rewards.mean()
