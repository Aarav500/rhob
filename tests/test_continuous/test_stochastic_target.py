"""Tests for the stochastic-target generation and its reflection symmetry."""

from __future__ import annotations

import numpy as np

from rhob.environments.continuous.config import ContinuousConfig
from rhob.environments.continuous.point_mass import PointMassPair


def test_variants_are_reflection_symmetric_in_expectation():
    cfg = ContinuousConfig(separation=1.0, sigma_a=0.5)
    a = np.array([PointMassPair("A", cfg, s).attractor_x for s in range(400)])
    b = np.array([PointMassPair("B", cfg, 1000 + s).attractor_x for s in range(400)])
    assert a.mean() > cfg.center > b.mean()  # A right, B left of centre
    # Offsets from centre are mirror images (sum ~ 0) and spreads match.
    assert abs((a.mean() - cfg.center) + (b.mean() - cfg.center)) < 0.15
    assert abs(a.std() - b.std()) < 0.15


def test_sigma_a_controls_spread():
    tight = ContinuousConfig(separation=1.0, sigma_a=0.1)
    wide = ContinuousConfig(separation=1.0, sigma_a=1.0)
    st = np.std([PointMassPair("A", tight, s).attractor_x for s in range(300)])
    sw = np.std([PointMassPair("A", wide, s).attractor_x for s in range(300)])
    assert sw > 3 * st  # larger sigma_a -> wider target spread


def test_separation_sets_mean_offset():
    for d in (0.3, 0.5, 1.0):
        cfg = ContinuousConfig(separation=d, sigma_a=0.5)
        a = np.mean([PointMassPair("A", cfg, s).attractor_x for s in range(400)])
        assert abs((a - cfg.center) - d / 2) < 0.1
