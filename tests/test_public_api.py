"""Tests for the public API surface: registry, discovery, and comparison."""

from __future__ import annotations

import rhob
from rhob.environments.registry import (
    get_environment,
    get_environment_card,
    get_environment_tier,
    list_environments,
)


def test_top_level_exports_present():
    for name in [
        "evaluate",
        "compare",
        "load_dataset",
        "save_dataset",
        "RandomDetector",
        "CUSUMDetector",
        "GridWorldWireheading",
        "AccessLevel",
        "EvaluationConfig",
    ]:
        assert hasattr(rhob, name), f"rhob.{name} missing from public API"


def test_registry_lists_and_instantiates():
    envs = list_environments()
    assert "tier1/gridworld_wireheading" in envs
    env = get_environment("tier1/gridworld_wireheading")
    assert isinstance(env, rhob.GridWorldWireheading)


def test_registry_tier_and_card():
    assert get_environment_tier("tier1/gridworld_wireheading") is rhob.Tier.TIER1
    card = get_environment_card("tier1/gridworld_wireheading")
    assert card.id == "tier1/gridworld_wireheading"


def test_registry_unknown_raises():
    import pytest

    with pytest.raises(KeyError):
        get_environment("does/not_exist")


def test_compare_produces_ranked_reports(mixed_dataset):
    reports = rhob.compare([rhob.RandomDetector(), rhob.CUSUMDetector()], mixed_dataset)
    assert len(reports) == 2
    table = rhob.results_table(reports)
    # CUSUM (higher score) must be ranked above Random in the table.
    assert table.index("CUSUM") < table.index("Random")


def test_evaluate_from_hdf5_path(mixed_dataset, tmp_path):
    """evaluate() accepts a path to an HDF5 dataset, not just a list."""
    path = tmp_path / "ds.h5"
    rhob.save_dataset(mixed_dataset, path)
    report = rhob.evaluate(rhob.CUSUMDetector(), str(path))
    assert report.mean_auroc > 0.6
