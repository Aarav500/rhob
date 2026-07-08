"""Shared pytest fixtures.

Trajectory generation is the only slow part of the suite, so the datasets are
generated once per session and reused across tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from rhob.environments.tier1.gridworld_wireheading import GridWorldWireheading

# Deterministic seed conventions used throughout the tests.
HACKING_SEEDS = list(range(10))
CLEAN_SEEDS = [1000, 1001, 1002]


@pytest.fixture(scope="session")
def env() -> GridWorldWireheading:
    return GridWorldWireheading()


@pytest.fixture(scope="session")
def hacking_runs(env):
    """Ten hacking-configured runs (seeds 0-9)."""
    return [env.generate(seed=s, config={"hacking": True}) for s in HACKING_SEEDS]


@pytest.fixture(scope="session")
def clean_runs(env):
    """Three clean-configured runs (seeds 1000-1002)."""
    return [env.generate(seed=s, config={"hacking": False}) for s in CLEAN_SEEDS]


@pytest.fixture(scope="session")
def mixed_dataset(hacking_runs, clean_runs):
    """A 70/30 hacking/clean dataset (7 hacking + 3 clean)."""
    return list(hacking_runs[:7]) + list(clean_runs)


@pytest.fixture
def synthetic_onset_curves():
    """Proxy/true curves with a known, sharp onset at step 150.

    Before onset both rise together (legitimate learning); at onset the true
    return sharply collapses while the proxy keeps rising -- a sharp
    reward-tampering onset (the easiest, sharpest case).
    """
    T, onset, drop_steps = 300, 150, 10
    proxy = np.minimum(np.arange(T) / 80.0, 1.0)
    true = proxy.copy()
    for i in range(onset, T):
        frac = min((i - onset) / drop_steps, 1.0)
        proxy[i] = 1.0 + 1.5 * frac  # 1.0 -> 2.5
        true[i] = max(0.0, 1.0 - 1.0 * frac)  # 1.0 -> 0.0
    return proxy, true, onset
