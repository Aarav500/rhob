"""Tests for the Trajectory data model."""

from __future__ import annotations

import numpy as np
import pytest

from rhob.core.onset import OnsetLabel
from rhob.core.trajectory import Trajectory
from rhob.core.types import AccessLevel, HackingType


def _make(proxy, true, onset_step=None, is_hacking=False, features=None):
    label = None
    if onset_step is not None:
        label = OnsetLabel(
            onset_step=onset_step,
            confidence=0.9,
            hacking_type=HackingType.REWARD_TAMPERING,
        )
    return Trajectory(
        environment_id="tier1/gridworld_wireheading",
        seed=0,
        algorithm="tabular_q_learning",
        is_hacking_run=is_hacking,
        reward_proxy=np.asarray(proxy, dtype=float),
        reward_true=np.asarray(true, dtype=float),
        policy_features=features,
        onset_label=label,
    )


def test_total_steps_and_len():
    traj = _make(np.zeros(10), np.zeros(10))
    assert traj.total_steps == 10
    assert len(traj) == 10


def test_mismatched_reward_shapes_raise():
    with pytest.raises(ValueError):
        _make(np.zeros(10), np.zeros(9))


def test_bad_feature_shape_raises():
    with pytest.raises(ValueError):
        _make(np.zeros(10), np.zeros(10), features=np.zeros((9, 4)))


def test_onset_beyond_length_raises():
    with pytest.raises(ValueError):
        _make(np.zeros(10), np.zeros(10), onset_step=10, is_hacking=True)


def test_binary_labels_hacking():
    traj = _make(np.zeros(10), np.zeros(10), onset_step=4, is_hacking=True)
    labels = traj.binary_labels()
    assert labels.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]


def test_binary_labels_clean_all_zero():
    traj = _make(np.zeros(10), np.zeros(10), is_hacking=False)
    assert traj.binary_labels().sum() == 0


def test_iter_observations_count_and_order():
    traj = _make(np.arange(5), np.zeros(5), features=np.zeros((5, 3)))
    obs = list(traj.iter_observations(AccessLevel.L2))
    assert len(obs) == 5
    assert [o.t for o in obs] == [0, 1, 2, 3, 4]
    assert obs[3].reward == 3.0
