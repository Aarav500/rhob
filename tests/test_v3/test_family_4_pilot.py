"""Pilot test for Family 4 (Shortcut Exploitation)."""

from __future__ import annotations


from rhob.v3.registry import FamilyRegistry


def test_family_4_registered():
    """Family 4 is registered."""
    family = FamilyRegistry.get("shortcut_exploitation")
    assert family is not None
    assert family.name == "shortcut_exploitation"


def test_family_4_difficulty_range():
    """Difficulty range is correct."""
    family = FamilyRegistry.get("shortcut_exploitation")
    lo, hi = family.difficulty_range()
    assert lo < hi
    assert 0.5 < lo < 1.0
    assert 0.5 < hi < 1.0


def test_family_4_generate_pair():
    """Can generate pairs at various difficulties."""
    family = FamilyRegistry.get("shortcut_exploitation")
    for d in [0.98, 0.80, 0.60]:
        pair = family.generate_pair(difficulty=d, seed=42)
        assert pair.family == "shortcut_exploitation"
        assert pair.difficulty == d
        assert "detour_efficiency" in pair.params


def test_family_4_proxy_matched():
    """Proxy rewards are matched between variants."""
    family = FamilyRegistry.get("shortcut_exploitation")
    pair = family.generate_pair(difficulty=0.90, seed=42)

    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)

    # Both variants take symmetric detours, so proxy costs should be equal
    proxy_a = run_a.proxy_rewards.sum()
    proxy_b = run_b.proxy_rewards.sum()

    assert abs(proxy_a - proxy_b) < 0.1, (
        f"Proxy should be matched: hacking={proxy_a:.1f}, legit={proxy_b:.1f}"
    )


def test_family_4_true_reward_diverges():
    """True rewards diverge: hacking=0, legitimate=full."""
    family = FamilyRegistry.get("shortcut_exploitation")
    pair = family.generate_pair(difficulty=0.90, seed=42)

    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)

    true_a = run_a.true_rewards.sum()
    true_b = run_b.true_rewards.sum()

    # Hacking gets 0 (glitch detour), legitimate gets full reward
    assert true_a < true_b, (
        f"True reward should diverge: hacking={true_a}, legit={true_b}"
    )


def test_family_4_behavioral_divergence():
    """Behavioral feature diverges: detour_used marker."""
    family = FamilyRegistry.get("shortcut_exploitation")
    pair = family.generate_pair(difficulty=0.90, seed=42)

    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)

    # Hacking uses D_A (feature=+1), legitimate uses D_B (feature=-1)
    # At least one non-zero entry should exist in each
    assert (run_a.behav_trace != 0).any(), "Hacking should mark detour use"
    assert (run_b.behav_trace != 0).any(), "Legitimate should mark detour use"


def test_family_4_difficulty_sweep_moves_auroc():
    """Verify difficulty knob moves behavioral separation smoothly."""
    family = FamilyRegistry.get("shortcut_exploitation")

    difficulties = [0.98, 0.80, 0.60]  # EASY to HARD
    separations = []

    for d in difficulties:
        pair = family.generate_pair(difficulty=d, seed=42)
        run_a, _ = pair.rollout_hacking(0)
        run_b, _ = pair.rollout_legit(0)

        # Measure behavioral separation (difference in mean behav_trace)
        sep = abs(run_a.behav_trace.mean() - run_b.behav_trace.mean())
        separations.append(sep)

    # Separation should decrease with difficulty
    assert separations[0] > separations[2], (
        f"Behavioral separation should decrease with difficulty (easy > hard), "
        f"got: {[f'{s:.3f}' for s in separations]}"
    )


def test_family_4_efficiency_varies():
    """Detour efficiency changes with difficulty."""
    family = FamilyRegistry.get("shortcut_exploitation")

    eff_values = []
    for d in [0.98, 0.80, 0.60]:
        pair = family.generate_pair(difficulty=d, seed=0)
        eff = pair.params["detour_efficiency"]
        eff_values.append(eff)

    # Efficiency should increase with increasing difficulty
    # (harder difficulty = less efficient shortcut = subtler difference)
    assert eff_values[0] < eff_values[1] < eff_values[2], (
        f"Efficiency should increase with difficulty: {eff_values}"
    )
