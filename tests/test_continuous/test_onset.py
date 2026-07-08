"""Tests for continuous-tier onset labelling (mean-x midpoint-crossing)."""

from __future__ import annotations

import numpy as np
import pytest

from rhob.environments.continuous.onset_label import label_onset


def test_labels_switch_to_right_attractor():
    mean_x = np.full(40, 5.0)  # camps centre pre-onset
    mean_x[10:] = 5.9  # switches to a right attractor at episode 10
    assert label_onset(mean_x, attractor_x=6.0, center=5.0, sustain=3) == 10


def test_labels_switch_to_left_attractor():
    mean_x = np.full(40, 5.0)
    mean_x[10:] = 4.1  # left attractor
    assert label_onset(mean_x, attractor_x=4.0, center=5.0, sustain=3) == 10


def test_near_centre_attractor_does_not_fire_pre_onset():
    # A near-centre attractor: camping the centre must NOT be counted as onset.
    mean_x = np.full(40, 5.0)
    mean_x[10:] = 5.3
    assert label_onset(mean_x, attractor_x=5.3, center=5.0, sustain=3) == 10


def test_no_switch_returns_minus_one():
    assert label_onset(np.full(40, 5.0), attractor_x=6.0, center=5.0) == -1


def test_deterministic_and_rejects_2d():
    mean_x = np.full(40, 5.0)
    mean_x[15:] = 5.8
    assert label_onset(mean_x, 6.0) == label_onset(mean_x, 6.0)
    with pytest.raises(ValueError):
        label_onset(np.zeros((4, 4)), 6.0)
