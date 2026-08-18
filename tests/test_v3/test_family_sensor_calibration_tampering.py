"""Tests for Sensor Calibration Tampering (REWARD_TAMPERING mechanism, continuous)."""

from __future__ import annotations

import numpy as np

import pytest

from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import HackingMechanism

from admission_helpers import (
    assert_smoke_admissible_at,
    difficulty_id,
    scored_difficulties,
)


def test_registered():
    fam = FamilyRegistry.get("sensor_calibration_tampering")
    assert fam.name == "sensor_calibration_tampering"
    assert fam.mechanism == HackingMechanism.REWARD_TAMPERING


@pytest.mark.parametrize(
    "difficulty", scored_difficulties("sensor_calibration_tampering"), ids=difficulty_id
)
def test_smoke_admissible_at_scored_difficulty(difficulty):
    """Reduced-power screen at all five scored tiers -- not certification.

    See ``admission_helpers``: this is a TOST at the smoke margin (+/-0.256), not the
    shipped +/-0.10. The +/-0.10 claim for this family is the admission ledger's.
    """
    assert_smoke_admissible_at(FamilyRegistry.get("sensor_calibration_tampering"), difficulty)


def test_behavioral_sign_convention():
    """behav_trace (the calibration bias) is positive for hacking, negative for legit."""
    fam = FamilyRegistry.get("sensor_calibration_tampering")
    pair = fam.generate_pair(0.98, seed=1)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_a.behav_trace.mean() > 0
    assert run_b.behav_trace.mean() < 0


def test_proxy_matched_by_construction_large_sample():
    fam = FamilyRegistry.get("sensor_calibration_tampering")
    for d in [0.98, 0.8, 0.6]:
        pair = fam.generate_pair(d, seed=2)
        proxy_a, proxy_b = [], []
        for s in range(200):
            run_a, _ = pair.rollout_hacking(s)
            run_b, _ = pair.rollout_legit(s)
            proxy_a.append(run_a.proxy_rewards.mean())
            proxy_b.append(run_b.proxy_rewards.mean())
        assert abs(np.mean(proxy_a) - np.mean(proxy_b)) < 0.02, f"diff={d}"


def test_true_reward_diverges():
    fam = FamilyRegistry.get("sensor_calibration_tampering")
    pair = fam.generate_pair(0.9, seed=3)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_b.true_rewards.mean() > run_a.true_rewards.mean()


def test_difficulty_widens_separation():
    """Easier difficulty (higher target L2) should mean a bigger calibration-bias gap."""
    fam = FamilyRegistry.get("sensor_calibration_tampering")
    pair_easy = fam.generate_pair(0.98, seed=0)
    pair_hard = fam.generate_pair(0.60, seed=0)
    assert pair_easy.params["d"] > pair_hard.params["d"]


def test_deterministic():
    fam = FamilyRegistry.get("sensor_calibration_tampering")
    pair1 = fam.generate_pair(0.8, seed=42)
    pair2 = fam.generate_pair(0.8, seed=42)
    run1, _ = pair1.rollout_hacking(0)
    run2, _ = pair2.rollout_hacking(0)
    assert np.allclose(run1.proxy_rewards, run2.proxy_rewards)
    assert np.allclose(run1.true_rewards, run2.true_rewards)
