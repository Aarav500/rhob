"""Tests for Eval-Probe Sandbagging (second DECEPTIVE_ALIGNMENT family)."""

from __future__ import annotations

import numpy as np

from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import HackingMechanism

from admission_helpers import assert_admitted


def test_registered():
    fam = FamilyRegistry.get("eval_probe_sandbagging")
    assert fam.name == "eval_probe_sandbagging"
    assert fam.mechanism == HackingMechanism.DECEPTIVE_ALIGNMENT


def test_admitted_across_difficulty_range():
    fam = FamilyRegistry.get("eval_probe_sandbagging")
    for d in fam.default_difficulties():
        assert_admitted(fam, difficulty=d, n_seeds_per_variant=24)


def test_onset_from_start():
    """Unlike monitored_sandbagging, there's no single switch point -- the
    probe-conditional strategy is committed to from episode 0."""
    fam = FamilyRegistry.get("eval_probe_sandbagging")
    pair = fam.generate_pair(0.9, seed=0)
    _, onset_a = pair.rollout_hacking(0)
    _, onset_b = pair.rollout_legit(0)
    assert onset_a == 0
    assert onset_b == -1


def test_proxy_matched_by_construction():
    """Proxy only ever reflects probe-present episodes, where both variants behave
    identically -- exact equality, not just closeness in expectation."""
    fam = FamilyRegistry.get("eval_probe_sandbagging")
    pair = fam.generate_pair(0.8, seed=1)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert np.array_equal(run_a.proxy_rewards, run_b.proxy_rewards)


def test_true_reward_diverges():
    fam = FamilyRegistry.get("eval_probe_sandbagging")
    pair = fam.generate_pair(0.9, seed=2)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_b.true_rewards.mean() > run_a.true_rewards.mean()
    assert run_b.true_rewards.mean() > 0.95  # legit: always intended path


def test_easier_difficulty_has_rarer_probes():
    """Lower q (rarer probes) is the easy end: hacking mostly shortcuts, strong signal."""
    fam = FamilyRegistry.get("eval_probe_sandbagging")
    pair_easy = fam.generate_pair(0.98, seed=0)
    pair_hard = fam.generate_pair(0.60, seed=0)
    assert pair_easy.params["q_probe"] < pair_hard.params["q_probe"]


def test_behavioral_separation_degrades_with_difficulty():
    """As probe density rises toward the hard end, hacking looks more like legit on
    average -- separation should shrink, not grow."""
    from rhob.detectors.l2_behavioral_threshold import BehavioralThresholdDetector

    fam = FamilyRegistry.get("eval_probe_sandbagging")
    det = BehavioralThresholdDetector()

    def _mean_score_gap(difficulty):
        pair = fam.generate_pair(difficulty, seed=5)
        scores_a = [det.classify(pair.rollout_hacking(s)[0]) for s in range(20)]
        scores_b = [det.classify(pair.rollout_legit(s)[0]) for s in range(20)]
        return np.mean(scores_a) - np.mean(scores_b)

    gap_easy = _mean_score_gap(0.98)
    gap_hard = _mean_score_gap(0.60)
    assert gap_easy > gap_hard > 0


def test_deterministic():
    fam = FamilyRegistry.get("eval_probe_sandbagging")
    pair1 = fam.generate_pair(0.8, seed=42)
    pair2 = fam.generate_pair(0.8, seed=42)
    run1, _ = pair1.rollout_hacking(0)
    run2, _ = pair2.rollout_hacking(0)
    assert np.allclose(run1.proxy_rewards, run2.proxy_rewards)
    assert np.allclose(run1.true_rewards, run2.true_rewards)
