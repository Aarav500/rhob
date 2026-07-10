"""Tests for MuJoCo Goal Misgeneralization (GOAL_MISGENERALIZATION mechanism, Reacher-v5)."""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

import numpy as np

import rhob.v3.families.mujoco_goal_misgeneralization  # noqa: F401 -- triggers
# @FamilyRegistry.register; not yet imported by families/__init__.py (Task 6's job).
from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism

from admission_helpers import assert_admitted


def test_registered():
    fam = FamilyRegistry.get("mujoco_goal_misgeneralization")
    assert fam.name == "mujoco_goal_misgeneralization"
    assert fam.mechanism == HackingMechanism.GOAL_MISGENERALIZATION
    assert fam.complexity == EnvironmentComplexity.CONTINUOUS_COMPLEX


def test_admitted_at_easiest_difficulty():
    fam = FamilyRegistry.get("mujoco_goal_misgeneralization")
    lo, hi = fam.difficulty_range()
    assert_admitted(fam, difficulty=hi, n_seeds_per_variant=15)


def test_behavioral_sign_convention():
    fam = FamilyRegistry.get("mujoco_goal_misgeneralization")
    pair = fam.generate_pair(fam.difficulty_range()[1], seed=1)
    hacking_means, legit_means = [], []
    for s in range(10):
        run_a, _ = pair.rollout_hacking(s)
        run_b, _ = pair.rollout_legit(s)
        hacking_means.append(run_a.behav_trace.mean())
        legit_means.append(run_b.behav_trace.mean())
    assert np.mean(hacking_means) > np.mean(legit_means)
