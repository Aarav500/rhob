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

from admission_helpers import assert_admitted


def test_registered():
    fam = FamilyRegistry.get("mujoco_sensor_decoupling")
    assert fam.name == "mujoco_sensor_decoupling"
    assert fam.mechanism == HackingMechanism.REWARD_TAMPERING
    assert fam.complexity == EnvironmentComplexity.CONTINUOUS_COMPLEX


def test_admitted_at_easiest_difficulty():
    fam = FamilyRegistry.get("mujoco_sensor_decoupling")
    lo, hi = fam.difficulty_range()
    assert_admitted(fam, difficulty=hi, n_seeds_per_variant=10)


def test_true_reward_diverges():
    fam = FamilyRegistry.get("mujoco_sensor_decoupling")
    pair = fam.generate_pair(fam.difficulty_range()[1], seed=1)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_b.true_rewards.mean() > run_a.true_rewards.mean()
