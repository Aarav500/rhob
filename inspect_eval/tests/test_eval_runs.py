"""The eval has to run end to end, and its metrics have to refuse to invent numbers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from inspect_ai.scorer import SampleScore, Score

from hvtb_hack_detection.metrics import detection_auroc, hack_rate, score_resolution
from hvtb_hack_detection.signals import signals_from_trajectory
from hvtb_hack_detection.task import hack_detection_replay

FIXTURE = Path("hvtb_hack_detection/fixtures/synthetic_trajectories.json")


def _sample_score(value: float, hacked: bool) -> SampleScore:
    return SampleScore(score=Score(value=value, metadata={"hacked": hacked}))


# ---------------------------------------------------------------- the signal mapping
def test_fixture_loads_and_is_balanced():
    runs = json.loads(FIXTURE.read_text(encoding="utf-8"))["runs"]
    assert len(runs) == 24
    assert sum(r["hidden_solution"] for r in runs) == 12


def test_a_hacking_run_resolves_an_onset_step_and_a_legit_run_does_not():
    runs = json.loads(FIXTURE.read_text(encoding="utf-8"))["runs"]
    hacked = signals_from_trajectory(next(r for r in runs if r["hidden_solution"]))
    legit = signals_from_trajectory(next(r for r in runs if not r["hidden_solution"]))
    assert hacked.hacked and isinstance(hacked.onset_step, int)
    # Never fired is None, not 0: "did not fire" and "fired at step zero" are different.
    assert legit.hacked is False and legit.onset_step is None


def test_a_step_with_no_test_outcome_inherits_rather_than_interpolates():
    sig = signals_from_trajectory(
        {"steps": [{"tests_passing": 3}, {}, {"tests_passing": 5}], "hidden_solution": 0}
    )
    assert list(sig.proxy_rewards) == [3.0, 3.0, 5.0]


# ------------------------------------------------------------------------- the metrics
def test_hack_rate_is_the_watcher_not_the_detector():
    scores = [_sample_score(0.9, True), _sample_score(0.9, False), _sample_score(0.9, True)]
    assert hack_rate()(scores) == pytest.approx(2 / 3)


def test_auroc_is_nan_not_half_when_one_class_is_absent():
    """0.5 would be indistinguishable from a measured coin flip."""
    scores = [_sample_score(0.9, True), _sample_score(0.1, True)]
    assert math.isnan(float(detection_auroc()(scores)))


def test_resolution_is_zero_when_the_detector_returned_one_number():
    """AUROC is then 0.5 by arithmetic, and the pair of metrics has to say so."""
    scores = [_sample_score(0.5, True), _sample_score(0.5, False)] * 4
    assert score_resolution()(scores) == pytest.approx(0.0)
    assert float(detection_auroc()(scores)) == pytest.approx(0.5)


def test_resolution_is_one_when_every_cross_label_pair_is_ordered():
    scores = [_sample_score(0.9, True), _sample_score(0.8, True),
              _sample_score(0.2, False), _sample_score(0.1, False)]
    assert score_resolution()(scores) == pytest.approx(1.0)
    assert float(detection_auroc()(scores)) == pytest.approx(1.0)


def test_nan_detector_scores_are_excluded_not_counted_as_ties():
    """An N/A cell carries no evidence; it must not drag the aggregate toward chance."""
    scores = [_sample_score(0.9, True), _sample_score(0.1, False),
              _sample_score(float("nan"), True)]
    assert score_resolution()(scores) == pytest.approx(1.0)
    assert hack_rate()(scores) == pytest.approx(2 / 3)  # label still counts


# ----------------------------------------------------------------------------- the task
def test_the_task_builds_with_the_fixture_dataset():
    t = hack_detection_replay()
    assert len(t.dataset) == 24
    assert t.scorer is not None


def test_an_unknown_access_level_is_na_rather_than_scored():
    """L1 and L2 have no honest construction on a terminal task."""
    from hvtb_hack_detection.task import detector_score
    assert detector_score(access_level="L2") is not None
