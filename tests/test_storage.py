"""Tests for HDF5 trajectory storage."""

from __future__ import annotations

import numpy as np
import pytest

from rhob.data.storage import (
    load_dataset,
    load_trajectory,
    save_dataset,
    save_trajectory,
)


def _assert_equal(a, b):
    assert a.environment_id == b.environment_id
    assert a.seed == b.seed
    assert a.algorithm == b.algorithm
    assert a.is_hacking_run == b.is_hacking_run
    assert a.hacking_type == b.hacking_type
    assert np.array_equal(a.reward_proxy, b.reward_proxy)
    assert np.array_equal(a.reward_true, b.reward_true)
    if a.policy_features is None:
        assert b.policy_features is None
    else:
        assert np.array_equal(a.policy_features, b.policy_features)
    assert a.onset_step == b.onset_step


def test_roundtrip_exact(mixed_dataset, tmp_path):
    path = tmp_path / "ds.h5"
    save_dataset(mixed_dataset, path)
    loaded = load_dataset(path)
    assert len(loaded) == len(mixed_dataset)
    for a, b in zip(mixed_dataset, loaded):
        _assert_equal(a, b)


def test_onset_label_fields_preserved(hacking_runs, tmp_path):
    path = tmp_path / "one.h5"
    original = hacking_runs[0]
    save_trajectory(original, path)
    loaded = load_trajectory(path)
    assert loaded.onset_label is not None
    assert loaded.onset_label.onset_step == original.onset_label.onset_step
    assert abs(loaded.onset_label.confidence - original.onset_label.confidence) < 1e-12
    assert loaded.onset_label.hacking_type == original.onset_label.hacking_type


def test_clean_run_onset_is_none(clean_runs, tmp_path):
    path = tmp_path / "clean.h5"
    save_trajectory(clean_runs[0], path)
    loaded = load_trajectory(path)
    assert loaded.onset_label is None
    assert loaded.is_hacking_run is False


def test_metadata_preserved(hacking_runs, tmp_path):
    path = tmp_path / "meta.h5"
    save_trajectory(hacking_runs[0], path)
    loaded = load_trajectory(path)
    assert loaded.metadata["wirehead_cell"] == hacking_runs[0].metadata["wirehead_cell"]
    assert "activation_episode" in loaded.metadata


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.h5")
