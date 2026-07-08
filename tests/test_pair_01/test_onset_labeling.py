"""Tests for the Pair 01 onset labelling function."""

from __future__ import annotations

import numpy as np
import pytest

from rhob.environments.pair_01 import label_onset


def test_labels_first_episode_of_sustained_switch():
    frac = np.zeros(100)
    frac[40:] = 0.9
    assert label_onset(frac, threshold=0.5, sustain=3) == 40


def test_no_switch_returns_minus_one():
    assert label_onset(np.zeros(100), threshold=0.5, sustain=3) == -1


def test_brief_spike_below_sustain_is_ignored():
    frac = np.zeros(100)
    frac[40:42] = 0.9  # only 2 consecutive, sustain requires 3
    assert label_onset(frac, threshold=0.5, sustain=3) == -1


def test_all_above_threshold_is_zero():
    assert label_onset(np.ones(100), threshold=0.5, sustain=3) == 0


def test_threshold_is_respected():
    frac = np.full(100, 0.4)
    frac[30:] = 0.6
    assert label_onset(frac, threshold=0.5, sustain=3) == 30
    assert label_onset(frac, threshold=0.7, sustain=3) == -1


def test_deterministic():
    frac = np.zeros(100)
    frac[50:] = 0.8
    assert label_onset(frac) == label_onset(frac)


def test_rejects_non_1d_input():
    with pytest.raises(ValueError):
        label_onset(np.zeros((10, 10)))
