"""Tests for Pair 01 reward computation."""

from __future__ import annotations

from rhob.environments.pair_01 import DEFAULT, make_hacking_env, make_legitimate_env


def test_tile1_is_aligned_in_both_variants():
    for env in (make_hacking_env(), make_legitimate_env()):
        env.set_episode(0)
        assert env._reward(DEFAULT.tile1) == (DEFAULT.tile1_proxy, DEFAULT.tile1_true)


def test_empty_cell_gives_no_reward():
    env = make_hacking_env()
    env.set_episode(DEFAULT.activation_episode)
    assert env._reward((0, 1)) == (0.0, 0.0)


def test_tile2_inactive_before_activation():
    env = make_hacking_env()
    env.set_episode(DEFAULT.activation_episode - 1)
    assert env._reward(DEFAULT.tile2_hacking) == (0.0, 0.0)


def test_hacking_tile2_is_misaligned_when_active():
    env = make_hacking_env()
    env.set_episode(DEFAULT.activation_episode)
    # proxy paid, true reward is zero -> exploit.
    assert env._reward(DEFAULT.tile2_hacking) == (DEFAULT.tile2_proxy, DEFAULT.tile2_true_hacking)
    assert DEFAULT.tile2_true_hacking == 0.0


def test_legit_tile2_is_aligned_when_active():
    env = make_legitimate_env()
    env.set_episode(DEFAULT.activation_episode)
    # proxy and true both paid -> legitimate improvement.
    assert env._reward(DEFAULT.tile2_legit) == (DEFAULT.tile2_proxy, DEFAULT.tile2_true_legit)
    assert DEFAULT.tile2_true_legit == DEFAULT.tile2_proxy


def test_proxy_magnitudes_match_across_variants():
    """The proxy reward on each tile is identical between the two variants."""
    h, ell = make_hacking_env(), make_legitimate_env()
    h.set_episode(DEFAULT.activation_episode)
    ell.set_episode(DEFAULT.activation_episode)
    assert h._reward(h.cfg.tile1)[0] == ell._reward(ell.cfg.tile1)[0]
    assert h._reward(h.tile2)[0] == ell._reward(ell.tile2)[0]
