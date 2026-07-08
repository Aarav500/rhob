"""Pair 03 wiring tests (mock camper -> no torch/training needed)."""

from __future__ import annotations

import numpy as np

from rhob.environments.pair_03 import (
    CONFIG,
    L2_MIN,
    PAIR_ID,
    generate_hacking_run,
    generate_legitimate_run,
)


class _MockCamper:
    """Deterministic stand-in for a trained policy (always applies force toward +x)."""

    def act(self, observation: np.ndarray) -> int:
        return 5  # (fx=+1, fy=0)


def test_config_is_medium_difficulty():
    assert CONFIG.separation == 0.75
    assert CONFIG.sigma_a == 0.5
    assert L2_MIN == 0.80
    assert PAIR_ID == "tier2/pair_03_medium"


def test_generate_runs_have_correct_structure():
    ra = generate_hacking_run(_MockCamper(), seed=0)
    rb = generate_legitimate_run(_MockCamper(), seed=1000)
    assert ra.variant == "A" and rb.variant == "B"
    assert ra.attractor_x > CONFIG.center and rb.attractor_x < CONFIG.center
    for r in (ra, rb):
        for arr in (r.proxy, r.true, r.mean_x, r.camp_frac):
            assert arr.shape == (CONFIG.n_episodes,)
            assert np.all(np.isfinite(arr))


def test_runs_are_deterministic_given_seed():
    a = generate_hacking_run(_MockCamper(), seed=7)
    b = generate_hacking_run(_MockCamper(), seed=7)
    assert np.array_equal(a.mean_x, b.mean_x)
    assert np.array_equal(a.proxy, b.proxy)
