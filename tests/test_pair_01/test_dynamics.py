"""Tests for Pair 01 MDP dynamics and rollout determinism."""

from __future__ import annotations

import numpy as np

from rhob.environments.pair_01 import DEFAULT, generate_run, make_hacking_env, make_legitimate_env


def test_reset_returns_start_index():
    env = make_hacking_env()
    assert env.reset() == env._index(DEFAULT.start)


def test_movement_and_bounds():
    env = make_hacking_env()
    env.reset()  # at (0, 0)
    # Up and left are blocked at the corner -> stay.
    idx, *_ = env.step(0)  # up
    assert env.cell(idx) == (0, 0)
    idx, *_ = env.step(2)  # left
    assert env.cell(idx) == (0, 0)
    # Down then right move.
    idx, *_ = env.step(1)  # down
    assert env.cell(idx) == (1, 0)
    idx, *_ = env.step(3)  # right
    assert env.cell(idx) == (1, 1)


def test_stay_action_keeps_cell():
    env = make_hacking_env()
    env.reset()
    idx, *_ = env.step(4)  # stay
    assert env.cell(idx) == (0, 0)


def test_index_cell_roundtrip():
    env = make_hacking_env()
    for i in range(env.n_states):
        assert env._index(env.cell(i)) == i


def test_tile2_activation_schedule():
    env = make_hacking_env()
    env.set_episode(DEFAULT.activation_episode - 1)
    assert env.tile2_active is False
    env.set_episode(DEFAULT.activation_episode)
    assert env.tile2_active is True


def test_rollout_is_deterministic_given_seed():
    a = generate_run(make_hacking_env(), seed=7)
    b = generate_run(make_hacking_env(), seed=7)
    assert np.array_equal(a.proxy, b.proxy)
    assert np.array_equal(a.true, b.true)
    assert np.array_equal(a.frac_tile2, b.frac_tile2)
    assert np.array_equal(a.behav, b.behav)


def test_variants_have_transposed_tiles():
    h = make_hacking_env()
    ell = make_legitimate_env()
    assert h.tile2 == (DEFAULT.tile2_hacking[0], DEFAULT.tile2_hacking[1])
    assert ell.tile2 == (h.tile2[1], h.tile2[0])  # transpose
