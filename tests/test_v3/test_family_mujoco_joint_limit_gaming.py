"""Tests for MuJoCo Joint-Limit Gaming (REWARD_SHAPING mechanism, Ant-v5)."""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

import rhob.v3.families.mujoco_joint_limit_gaming  # noqa: F401 -- triggers
# @FamilyRegistry.register; not yet imported by families/__init__.py (Task 6's job).
from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism

from admission_helpers import (
    assert_smoke_admissible_at,
    difficulty_id,
    scored_difficulties,
)


# `slow` module: deselected from the default `pytest tests/` run and executed
# by the nightly `admission-slow` CI job (see [tool.pytest.ini_options] in
# pyproject.toml). Nothing here is skipped -- it runs on a different schedule.
# Every test below rolls out the family, and MuJoCo physics rollouts cost 0.2-1.9s each and sit
# behind a proxy calibration of 9-230s per difficulty.
pytestmark = pytest.mark.slow


def test_registered():
    fam = FamilyRegistry.get("mujoco_joint_limit_gaming")
    assert fam.name == "mujoco_joint_limit_gaming"
    assert fam.mechanism == HackingMechanism.REWARD_SHAPING
    assert fam.complexity == EnvironmentComplexity.CONTINUOUS_COMPLEX


@pytest.mark.parametrize(
    "difficulty", scored_difficulties("mujoco_joint_limit_gaming"), ids=difficulty_id
)
def test_smoke_admissible_at_scored_difficulty(difficulty):
    """Reduced-power screen at every difficulty the benchmark scores -- not certification.

    Two separate things were wrong with the test this replaces. It screened
    ``difficulty_range()[1]`` == 0.95, which ``default_difficulties()`` never returns, so
    the tiers the benchmark actually evaluates were covered by nothing. And it called what
    it did "admitted" while running what is now the shipped 576-rollout certification
    design -- which does not fit in CI. What this screen proves, and what it does not, is
    in ``admission_helpers``; the +/-0.10 certification is ``scripts/admission_ledger.py``'s.
    """
    assert_smoke_admissible_at(FamilyRegistry.get("mujoco_joint_limit_gaming"), difficulty)


def test_true_reward_diverges():
    fam = FamilyRegistry.get("mujoco_joint_limit_gaming")
    pair = fam.generate_pair(fam.default_difficulties()[0], seed=1)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_b.true_rewards.mean() > run_a.true_rewards.mean()
