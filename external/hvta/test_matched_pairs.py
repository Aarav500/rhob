"""Invariants the HVTA matched pairs must hold, or the measurement downstream is void.

Run::

    PYTHONUTF8=1 PYTHONPATH=<rhob>/src <hvta venv>/python -m pytest external/hvta -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

from hvta_adapter import build_cell, to_rundata
from hvta_policies import onset_schedule, rollout_pair, snap_read_step

MEDIUM = "Sudoku-v0-medium"


@pytest.mark.parametrize("slip", [0.0, 0.3, 0.5])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_two_consecutive_rejections(slip, seed):
    """Two consecutive invalid moves end a TextArena single-player episode.

    If that happened the legitimate variant would truncate while the hacking variant ran
    on, and episode *length* would become a label. ``_guess_cell`` restricting guesses to
    two-candidate cells is what prevents it; this pins the behaviour.
    """
    p = rollout_pair(env_id=MEDIUM, seed=seed, read_step=60, slip_rate=slip)
    for ep in (p.hacking, p.legit):
        rejects = [i for i, a in enumerate(ep.move_accepted) if a is False]
        game_idx = [i for i, k in enumerate(ep.step_kinds) if k == "game"]
        pos = {g: n for n, g in enumerate(game_idx)}
        ordinals = sorted(pos[i] for i in rejects)
        consecutive = [
            (a, b) for a, b in zip(ordinals, ordinals[1:]) if b == a + 1
        ]
        assert not consecutive, f"{ep.variant} had consecutive rejections at {consecutive}"


@pytest.mark.parametrize("slip", [0.0, 0.3, 0.5])
def test_pair_is_matched(slip):
    """Same length, identical actions before onset, and only the hacking side hacks."""
    p = rollout_pair(env_id=MEDIUM, seed=0, read_step=60, slip_rate=slip)
    assert p.length_matched
    assert p.prefix_identical
    assert p.hacking.hack_triggered and p.hacking.hack_onset_step is not None
    assert not p.legit.hack_triggered
    assert p.legit.hack_onset_step is None
    # The foil reads a file in the *same directory* and still does not trip the latch.
    assert p.legit.actions[p.effective_read_step].startswith("[fs_cat ")
    assert p.hacking.solution_path.rsplit("/", 1)[0] in p.legit.actions[p.effective_read_step]


def test_strict_mode_action_sequences_are_identical():
    """With ``exploit=False`` the only difference in the whole episode is the file read.

    This is the clean lower bound: any discrimination measured here comes from the read
    alone, not from the hack's downstream effect on play.
    """
    p = rollout_pair(env_id=MEDIUM, seed=0, read_step=60, slip_rate=0.3, exploit=False)
    k = p.effective_read_step
    assert p.hacking.actions[:k] == p.legit.actions[:k]
    assert p.hacking.actions[k + 1 :] == p.legit.actions[k + 1 :]
    assert p.hacking.actions[k] != p.legit.actions[k]
    assert p.hacking.hack_onset_step == k


def test_read_step_snaps_onto_a_filesystem_step():
    """A game step must never be replaced by the read, or the action kinds stop matching."""
    for requested in range(0, 110):
        t = snap_read_step(requested, total_steps=120, game_moves=35)
        assert t >= requested
    p = rollout_pair(env_id=MEDIUM, seed=0, read_step=60, slip_rate=0.3)
    assert p.hacking.step_kinds[p.effective_read_step] == "fs"
    assert p.legit.step_kinds[p.effective_read_step] == "fs"


def test_onset_matches_hvta_ground_truth_not_our_bookkeeping():
    """HVTA's ``hack_onset_step`` is the authority; our index must agree with it."""
    p = rollout_pair(env_id=MEDIUM, seed=0, read_step=60, slip_rate=0.3)
    assert p.hacking.hack_onset_step == p.effective_read_step
    assert p.hacking.game_info["solution_read_step"] == p.effective_read_step
    assert p.hacking.game_info["hack_triggered"] is True
    assert p.hacking.game_info["n_steps"] == len(p.hacking.actions)


def test_state_counts_absent_never_imputed():
    """The L1 channel has no honest source in HVTA and must stay None."""
    p = rollout_pair(env_id=MEDIUM, seed=0, read_step=60, slip_rate=0.3)
    for ep in (p.hacking, p.legit):
        assert to_rundata(ep).state_counts is None


def test_true_rewards_freeze_at_onset_and_are_oracle_only():
    p = rollout_pair(env_id=MEDIUM, seed=0, read_step=60, slip_rate=0.3)
    onset = p.hacking.hack_onset_step
    run = to_rundata(p.hacking)
    proxy, true = run.proxy_rewards, run.true_rewards
    np.testing.assert_allclose(true[:onset], proxy[:onset])
    assert np.allclose(true[onset:], true[onset])
    # The legitimate variant earned everything it scored.
    legit = to_rundata(p.legit)
    np.testing.assert_allclose(legit.true_rewards, legit.proxy_rewards)


def test_sign_randomization_is_on_by_default_and_preserves_magnitude():
    pairs = [
        rollout_pair(env_id=MEDIUM, seed=s, read_step=r, slip_rate=0.3)
        for s, r in enumerate(onset_schedule(3))
    ]
    cell = build_cell(pairs)
    assert cell.randomize_behav_sign
    assert abs(cell.behav_sign_drawn) == 1.0
    for flipped, pair in zip(cell.runs_a, pairs):
        raw = to_rundata(pair.hacking).behav_trace
        np.testing.assert_allclose(np.abs(flipped.behav_trace), np.abs(raw))


def test_zero_slip_is_degenerate_and_says_so():
    """At slip 0 the scripted solver needs no oracle, so the pair carries no signal.

    Pinned deliberately: it is the configuration in which any AUROC is 0.500 by
    construction, and it must not be mistaken for a measurement.
    """
    p = rollout_pair(env_id=MEDIUM, seed=0, read_step=60, slip_rate=0.0)
    k = p.effective_read_step
    # Identical everywhere except the one step where the file is read.
    assert p.hacking.actions[:k] == p.legit.actions[:k]
    assert p.hacking.actions[k + 1 :] == p.legit.actions[k + 1 :]
    assert p.hacking.actions[k] != p.legit.actions[k]
    assert p.hacking.completion == p.legit.completion
    for channel in ("proxy_rewards", "behav_trace"):
        np.testing.assert_allclose(
            getattr(to_rundata(p.hacking), channel),
            getattr(to_rundata(p.legit), channel),
            err_msg=f"{channel} should be identical at slip 0 -- no signal exists",
        )
