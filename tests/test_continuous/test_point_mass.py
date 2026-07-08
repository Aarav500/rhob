"""Tests for the continuous point-mass dynamics and rewards."""

from __future__ import annotations

import numpy as np

from rhob.environments.continuous.config import ContinuousConfig
from rhob.environments.continuous.point_mass import PointMassPair


def test_dynamics_deterministic_given_seed():
    cfg = ContinuousConfig(separation=1.0)
    a, b = PointMassPair("A", cfg, seed=3), PointMassPair("A", cfg, seed=3)
    assert a.attractor_x == b.attractor_x
    a.set_episode(cfg.activation_episode)
    b.set_episode(cfg.activation_episode)
    a.reset()
    b.reset()
    for action in (0, 3, 5, 8, 1, 4):
        oa, pa, ta, _ = a.step(action)
        ob, pb, tb, _ = b.step(action)
        assert np.array_equal(oa, ob) and pa == pb and ta == tb


def test_attractor_inactive_before_activation():
    cfg = ContinuousConfig(separation=1.0)
    env = PointMassPair("A", cfg, seed=0)
    env.set_episode(cfg.activation_episode - 1)
    assert env.active is False
    tx, ty = env.observed_target()
    assert (tx, ty) == (cfg.center, cfg.center)  # observes centre pre-onset
    env.set_episode(cfg.activation_episode)
    assert env.active is True


def test_bump_is_one_on_attractor_and_decays():
    cfg = ContinuousConfig(separation=2.0, sigma_a=0.0)
    env = PointMassPair("A", cfg, seed=0)
    env._x, env._y = env.attractor_x, env.attractor_y
    assert env._bump(env.attractor_x, env.attractor_y) == 1.0
    assert env._bump(env.attractor_x + 2.0, env.attractor_y) < 0.2


def test_hacking_true_is_centre_based_legit_is_attractor_based():
    cfg = ContinuousConfig(separation=3.0, sigma_a=0.0, activation_episode=0)
    a = PointMassPair("A", cfg, seed=0)
    b = PointMassPair("B", cfg, seed=0)
    a.set_episode(0)
    b.set_episode(0)
    # Each agent sitting on its own attractor: the hacking variant's true objective
    # is the (distant) centre, so its true reward is strictly lower than the
    # legitimate variant's, whose true objective IS its attractor.
    a._x, a._y = a.attractor_x, a.attractor_y
    b._x, b._y = b.attractor_x, b.attractor_y
    true_a = a._bump(cfg.center, cfg.center)  # hacking: centre-based
    true_b = b._bump(b.attractor_x, b.attractor_y)  # legitimate: attractor-based
    assert true_b == 1.0
    assert true_a < true_b  # A (hacking) earns less true reward than B (legitimate)
