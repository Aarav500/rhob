"""Tests for the core type system."""

from __future__ import annotations

from rhob.core.types import TIER_WEIGHTS, AccessLevel, HackingType, Tier


def test_access_levels_form_total_order():
    assert AccessLevel.L1 < AccessLevel.L2 < AccessLevel.L3 < AccessLevel.L4
    assert AccessLevel.L2 >= AccessLevel.L1
    assert int(AccessLevel.L1) == 1 and int(AccessLevel.L4) == 4


def test_access_level_comparison_used_by_filter():
    # The access filter relies on >= comparisons being meaningful.
    assert (AccessLevel.L1 >= AccessLevel.L2) is False
    assert (AccessLevel.L3 >= AccessLevel.L2) is True


def test_hacking_type_values():
    assert HackingType.REWARD_TAMPERING.value == "reward_tampering"
    assert HackingType("proxy_gaming") is HackingType.PROXY_GAMING
    assert len(list(HackingType)) == 6


def test_tier_values():
    assert Tier.TIER1.value == "tier1"
    assert Tier("adversarial") is Tier.ADVERSARIAL


def test_tier_weights_increase_with_difficulty():
    assert (
        TIER_WEIGHTS[Tier.TIER1]
        < TIER_WEIGHTS[Tier.TIER2]
        < TIER_WEIGHTS[Tier.TIER3]
        < TIER_WEIGHTS[Tier.ADVERSARIAL]
    )
