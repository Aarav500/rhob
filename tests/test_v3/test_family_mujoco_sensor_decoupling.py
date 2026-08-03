"""Tests for MuJoCo Sensor-Channel Decoupling (REWARD_TAMPERING mechanism, Walker2d-v5)."""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

import rhob.v3.families.mujoco_sensor_decoupling  # noqa: F401 -- triggers
# @FamilyRegistry.register; not yet imported by families/__init__.py (Task 6's job).
# LESSON FROM TASKS 2-4: without this self-import, @FamilyRegistry.register never
# fires and every test below fails with KeyError: unknown family -- do not omit it.
from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism

from admission_helpers import assert_smoke_admissible


# `slow` module: deselected from the default `pytest tests/` run and executed
# by the nightly `admission-slow` CI job (see [tool.pytest.ini_options] in
# pyproject.toml). Nothing here is skipped -- it runs on a different schedule.
# Every test below rolls out the family, and MuJoCo physics rollouts cost 0.2-1.9s each and sit
# behind a proxy calibration of 9-230s per difficulty.
pytestmark = pytest.mark.slow


def test_registered():
    fam = FamilyRegistry.get("mujoco_sensor_decoupling")
    assert fam.name == "mujoco_sensor_decoupling"
    assert fam.mechanism == HackingMechanism.REWARD_TAMPERING
    assert fam.complexity == EnvironmentComplexity.CONTINUOUS_COMPLEX


def test_smoke_admissible_at_every_scored_difficulty():
    """Reduced-power screen at every difficulty the benchmark scores -- not certification.

    Two separate things were wrong with the test this replaces. It screened
    ``difficulty_range()[1]`` == 0.95, which ``default_difficulties()`` never returns, so
    the tiers the benchmark actually evaluates were covered by nothing. And it called what
    it did "admitted" while running what is now the shipped 576-rollout certification
    design -- which does not fit in CI. What this screen proves, and what it does not, is
    in ``admission_helpers``; the +/-0.10 certification is ``scripts/admission_ledger.py``'s.
    """
    assert_smoke_admissible(FamilyRegistry.get("mujoco_sensor_decoupling"))


def test_true_reward_diverges():
    fam = FamilyRegistry.get("mujoco_sensor_decoupling")
    pair = fam.generate_pair(fam.default_difficulties()[0], seed=1)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_b.true_rewards.mean() > run_a.true_rewards.mean()
