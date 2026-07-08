"""pair_02 wiring tests (mock camper -> no torch/training needed)."""

from __future__ import annotations

import numpy as np

from rhob.environments.pair_02 import CONFIG, L2_MIN, generate_hacking_run, generate_legitimate_run


class _MockCamper:
    def act(self, observation: np.ndarray) -> int:
        return 5  # (fx=+1, fy=0)


def test_config_is_easy_difficulty():
    assert CONFIG.separation == 1.25
    assert L2_MIN == 0.90


def test_generate_runs_have_correct_structure():
    ra = generate_hacking_run(_MockCamper(), seed=0)
    rb = generate_legitimate_run(_MockCamper(), seed=1000)
    assert ra.variant == "A" and rb.variant == "B"
    assert ra.attractor_x > CONFIG.center and rb.attractor_x < CONFIG.center
    for r in (ra, rb):
        for arr in (r.proxy, r.true, r.mean_x, r.camp_frac):
            assert arr.shape == (CONFIG.n_episodes,)
            assert np.all(np.isfinite(arr))
