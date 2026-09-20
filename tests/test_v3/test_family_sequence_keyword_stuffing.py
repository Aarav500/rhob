"""Tests for Sequence Keyword-Stuffing Gaming (PROXY_GAMING mechanism)."""

from __future__ import annotations

import numpy as np
import pytest

import rhob.v3.families.sequence_keyword_stuffing  # noqa: F401 -- self-import until Task 7
from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism


# `slow` module: deselected from the default `pytest tests/` run and executed
# by the nightly `admission-slow` CI job (see [tool.pytest.ini_options] in
# pyproject.toml). Nothing here is skipped -- it runs on a different schedule.
# Every test below rolls out the family, and hidden-Markov sequence rollouts cost ~9s each and sit
# behind a 90-165s proxy calibration per difficulty.
pytestmark = pytest.mark.slow


def test_registered():
    fam = FamilyRegistry.get("sequence_keyword_stuffing")
    assert fam.name == "sequence_keyword_stuffing"
    assert fam.mechanism == HackingMechanism.PROXY_GAMING
    assert fam.complexity == EnvironmentComplexity.SEQUENTIAL


def test_true_reward_diverges():
    fam = FamilyRegistry.get("sequence_keyword_stuffing")
    pair = fam.generate_pair(fam.default_difficulties()[0], seed=1)
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    assert run_b.true_rewards.mean() > run_a.true_rewards.mean()


def test_behav_trace_ranks_hacking_above_legit():
    """behav_trace must rank hacking ABOVE legit (higher = more hacking-like).

    Averaged over seeds, because the claim is about the family and not about one draw.
    Measured at difficulty 0.90 over 100 independent seed pairs: hacking's mean
    behav_trace is 0.03704 (SD 0.00528), legit's is 0.03079 (SD 0.00491), and the paired
    gap is +0.00625 with a per-draw SD of 0.00722 -- larger than the effect itself, so
    the sign of a SINGLE draw flips on about 22% of seeds (78/100 favoured hacking,
    paired t = 8.66). The population direction holds at every tier (+0.00525 / +0.00637
    / +0.00751 at 0.70 / 0.80 / 0.90).

    This test asserted the population claim on one seed and had been failing in the
    nightly at 0.0267 vs 0.0323. The single-seed form was never viable; it looked viable
    because commit 6fbeb2f moved the tier from 0.95 to 0.90 and added this module's
    ``slow`` marker in the same change, so the default ``pytest tests/`` run collects
    nothing here (exit 5) and no local run could see the inversion it introduced.

    The seed is NOT re-picked to one that passes. With the sign flipping on 22% of draws,
    a seed chosen after seeing which one passes would encode that search rather than the
    family's property -- the failure class this benchmark exists to study, and one this
    repository has already paid for once (headline transfer 0.994 -> 0.508 once the
    behavioural sign was randomized). 15 seeds matches the existing precedent in
    ``test_family_reward_channel_tampering`` and puts the flake rate near 1 in 2,500.
    """
    fam = FamilyRegistry.get("sequence_keyword_stuffing")
    pair = fam.generate_pair(fam.default_difficulties()[0], seed=1)

    hacking_means = []
    legit_means = []
    for s in range(15):
        run_a, _ = pair.rollout_hacking(s)
        run_b, _ = pair.rollout_legit(s)
        hacking_means.append(run_a.behav_trace.mean())
        legit_means.append(run_b.behav_trace.mean())

    assert np.mean(hacking_means) > np.mean(legit_means), (
        f"hacking {np.mean(hacking_means):.5f} vs legit {np.mean(legit_means):.5f} over "
        f"15 seeds; the measured population gap is +0.00625, so a failure here is a real "
        f"regression rather than an unlucky draw"
    )
