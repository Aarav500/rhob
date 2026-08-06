"""``orbit_chirality``'s difficulty knob does nothing, and this measures that.

``chirality_strength`` is computed in ``generate_pair``, passed to
``_run_orbit_episode``, and reported into ``MatchedPair.params`` -- from where it reaches
admission certificates and leaderboard rows. The episode function never reads it: its
body sets ``angular_velocity`` from ``strategy`` alone. Every tier of this family,
TRIVIAL through EXTREME, therefore produces bit-identical trajectories.

This test asserts the *current, broken* state on purpose. That is unusual enough to
justify:

* A comment claiming "this parameter is inert" is an assertion. This is a measurement,
  and it is the same channel-digest probe the difficulty-calibration audit uses --
  identical digests across tiers are proof the knob changed nothing, which is strictly
  stronger than an unmoved AUROC (an unmoved AUROC is equally consistent with a knob that
  works but saturates).
* It fails the moment someone wires the parameter up. That is the point. Repairing this
  family changes its rollouts, which invalidates the 20-draw replication in
  ``leaderboard/v5_replicated.json`` that includes it, and moots the paper's account of
  the defect. The failure message says so, so the repair is a deliberate versioned change
  rather than a silent one that quietly desynchronises three published artifacts.

If you are here because this test failed: good. Implement the mapping, regenerate the
replication, update the paper, then delete this file.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import rhob.v3.families  # noqa: F401  -- registers the families
from rhob.v3.registry import FamilyRegistry

_FAMILY = "orbit_chirality"
_N_SEEDS = 3


def _digest(pair) -> str:
    """SHA-256 over the three emitted channels of a fixed rollout of this pair.

    The RNG is seeded per run, so two pairs differing only in ``difficulty`` are
    comparable: any difference in the digest is attributable to the difficulty argument
    and nothing else.
    """
    runs_a, runs_b, _ = pair.rollout(_N_SEEDS, seed_base=0, randomize_sign=False)
    h = hashlib.sha256()
    for run in (*runs_a, *runs_b):
        for channel in ("proxy_rewards", "true_rewards", "behav_trace"):
            arr = getattr(run, channel, None)
            if arr is not None:
                h.update(np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).tobytes())
    return h.hexdigest()


def test_every_difficulty_tier_emits_identical_data():
    """The channel-digest probe: if the knob worked, these would differ."""
    fam = FamilyRegistry.get(_FAMILY)
    tiers = fam.default_difficulties()
    assert len(tiers) >= 2, f"{_FAMILY} must expose multiple tiers for this to mean anything"

    digests = {d: _digest(fam.generate_pair_at(d, seed=0)) for d in tiers}
    distinct = set(digests.values())

    assert len(distinct) == 1, (
        f"orbit_chirality now emits different data across difficulty tiers: {digests}.\n"
        f"That means chirality_strength has been wired into _run_orbit_episode. This is a\n"
        f"REPAIR, not a regression -- but it is a versioned one:\n"
        f"  1. leaderboard/v5_replicated.json and results/replication/replicate_*.json\n"
        f"     include this family and are now stale. Regenerate them.\n"
        f"  2. The paper documents this dead knob as a finding. Update that account.\n"
        f"  3. Delete this test.\n"
        f"Do not simply relax the assertion."
    )


def test_the_parameter_is_still_reported_into_params():
    """It varies in the artifact while changing nothing -- that is the defect's shape.

    Reported so the test fails if the value is quietly dropped from ``params`` instead of
    being made to work: removing it would change published certificates without fixing
    anything, and would erase the evidence for the finding.
    """
    fam = FamilyRegistry.get(_FAMILY)
    tiers = fam.default_difficulties()
    values = {d: fam.generate_pair_at(d, seed=0).params.get("chirality_strength") for d in tiers}

    assert all(v is not None for v in values.values()), (
        f"chirality_strength vanished from params: {values}. If it was removed because it "
        f"is inert, note that it is inert *and still published*, which is the finding; "
        f"removing it silently changes certificates that already shipped."
    )
    assert len(set(values.values())) > 1, (
        f"chirality_strength no longer varies across tiers: {values}"
    )


@pytest.mark.parametrize("channel", ["proxy_rewards", "true_rewards", "behav_trace"])
def test_each_channel_individually_is_tier_invariant(channel: str):
    """Per-channel, so a partial repair is localised rather than just 'something changed'."""
    fam = FamilyRegistry.get(_FAMILY)
    tiers = fam.default_difficulties()
    lo, hi = min(tiers), max(tiers)

    def channel_bytes(d: float) -> bytes:
        runs_a, _, _ = fam.generate_pair_at(d, seed=0).rollout(
            _N_SEEDS, seed_base=0, randomize_sign=False
        )
        return b"".join(
            np.ascontiguousarray(np.asarray(getattr(r, channel), dtype=np.float64)).tobytes()
            for r in runs_a
        )

    assert channel_bytes(lo) == channel_bytes(hi), (
        f"{channel} now differs between difficulty {lo} and {hi}. See this module's "
        f"docstring: this is a repair and requires regenerating the replication."
    )
