"""Tests for the toy RLHF Reward-Model Overoptimization family."""

from __future__ import annotations

import numpy as np
import pytest

from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import HackingMechanism

from admission_helpers import assert_smoke_admissible


def test_registered():
    fam = FamilyRegistry.get("rlhf_reward_model_overopt")
    assert fam.name == "rlhf_reward_model_overopt"
    assert fam.mechanism == HackingMechanism.RM_OVEROPTIMIZATION


@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEGENERATE proxy, not a matched one. ``test_proxy_matched_exactly`` below pins "
        "why: every episode's proxy is one of two values that the RM fit makes equal, so "
        "the stream is a single constant to within one ULP (relative dispersion 1.6e-16, "
        "twelve orders of magnitude under the gate's informativeness floor) and all four "
        "L0 statistics order 0.0% of the cross-variant run pairs at every scored tier "
        "(0.98/0.9/0.8/0.7/0.6). proxy_matched and proxy_distribution_matched come back "
        "DEGENERATE, which is not a pass. This is the constant-proxy route, the same one "
        "as distributional_shift and physics_exploitation -- monitored_sandbagging and "
        "shortcut_exploitation are degenerate for the other reason, a varied proxy that "
        "is identical in every run. The fix either way is an informative-but-matched "
        "proxy in the family, not a change to the gate or to this test."
    ),
)
def test_smoke_admissible_at_every_scored_difficulty():
    """Reduced-power screen at all five scored tiers -- not certification.

    Expected to fail: see the xfail reason.
    """
    assert_smoke_admissible(FamilyRegistry.get("rlhf_reward_model_overopt"))


def test_rm_fit_learns_both_coefficients():
    """The fitted RM should assign positive weight to both the genuine-quality and
    spurious features -- a real fit result, not a hardcoded blind spot."""
    fam = FamilyRegistry.get("rlhf_reward_model_overopt")
    pair = fam.generate_pair(0.9, seed=7)
    assert pair.params["rm_quality_coef"] > 0
    assert pair.params["rm_spurious_coef"] > 0


def test_proxy_matched_exactly():
    """RM_score(BLIND) == RM_score(GENUINE) by algebraic construction from the fit,
    so proxy should match to floating-point precision, not just approximately."""
    fam = FamilyRegistry.get("rlhf_reward_model_overopt")
    for d in [0.98, 0.8, 0.6]:
        pair = fam.generate_pair(d, seed=1)
        run_a, _ = pair.rollout_hacking(0)
        run_b, _ = pair.rollout_legit(0)
        # Every episode's proxy value is one of exactly two values (rm_score_blind,
        # rm_score_genuine), which must be equal.
        assert len(set(np.round(run_a.proxy_rewards, 8)) | set(np.round(run_b.proxy_rewards, 8))) == 1


def test_true_reward_diverges():
    fam = FamilyRegistry.get("rlhf_reward_model_overopt")
    pair = fam.generate_pair(0.9, seed=2)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_b.true_rewards.mean() > run_a.true_rewards.mean()


def test_behavioral_sign_convention():
    fam = FamilyRegistry.get("rlhf_reward_model_overopt")
    pair = fam.generate_pair(0.98, seed=3)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_a.behav_trace.mean() > 0  # hacking mostly picks blind (+1)
    assert run_b.behav_trace.mean() < 0  # legit mostly picks genuine (-1)


def test_proxy_matching_holds_regardless_of_commitment_gap():
    """Proxy stays matched at every difficulty even though the commitment
    probabilities (tau_hack, tau_legit) change -- because both clusters' RM
    scores are equal, a weighted average of equal numbers is unaffected by
    the weights."""
    fam = FamilyRegistry.get("rlhf_reward_model_overopt")
    for d in [0.98, 0.6]:
        pair = fam.generate_pair(d, seed=9)
        proxy_a, proxy_b = [], []
        for s in range(30):
            run_a, _ = pair.rollout_hacking(s)
            run_b, _ = pair.rollout_legit(s)
            proxy_a.append(run_a.proxy_rewards.mean())
            proxy_b.append(run_b.proxy_rewards.mean())
        assert abs(np.mean(proxy_a) - np.mean(proxy_b)) < 1e-9


def test_deterministic():
    fam = FamilyRegistry.get("rlhf_reward_model_overopt")
    pair1 = fam.generate_pair(0.8, seed=42)
    pair2 = fam.generate_pair(0.8, seed=42)
    run1, _ = pair1.rollout_hacking(0)
    run2, _ = pair2.rollout_hacking(0)
    assert np.allclose(run1.proxy_rewards, run2.proxy_rewards)
    assert np.allclose(run1.true_rewards, run2.true_rewards)
    assert pair1.params == pair2.params
