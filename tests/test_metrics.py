"""Tests for the metrics module."""

from __future__ import annotations

import numpy as np

from rhob.core.types import Tier
from rhob.evaluation import metrics


# --- AUROC / AUPRC ---------------------------------------------------------
def test_auroc_perfect_and_worst():
    labels = np.array([0, 0, 1, 1])
    assert metrics.auroc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert metrics.auroc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0


def test_auroc_single_class_is_nan():
    assert np.isnan(metrics.auroc(np.array([0.5, 0.6]), np.array([0, 0])))


def test_auprc_no_positives_is_nan():
    assert np.isnan(metrics.auprc(np.array([0.5, 0.6]), np.array([0, 0])))


# --- Perfect / random detector (M0 contract) -------------------------------
def test_perfect_detector_auroc_and_latency():
    onset, T = 40, 100
    labels = np.zeros(T, dtype=int)
    labels[onset:] = 1
    scores = labels.astype(float)  # oracle-perfect predictions
    assert metrics.auroc(scores, labels) == 1.0
    assert metrics.detection_latency(scores, onset, T, threshold=0.5) == 0.0


def test_random_detector_auroc_near_half():
    rng = np.random.default_rng(0)
    onset, T = 500, 1000
    labels = np.zeros(T, dtype=int)
    labels[onset:] = 1
    scores = rng.random(T)
    assert 0.4 < metrics.auroc(scores, labels) < 0.6


# --- Detection step / latency ----------------------------------------------
def test_detection_step():
    assert metrics.detection_step(np.array([0.1, 0.2, 0.7, 0.9]), 0.5) == 2
    assert metrics.detection_step(np.array([0.1, 0.2, 0.3]), 0.5) is None


def test_latency_never_fires_is_inf():
    scores = np.zeros(100)
    assert metrics.detection_latency(scores, onset_step=40, total_steps=100) == float("inf")


def test_latency_clean_run_is_nan():
    scores = np.ones(100)
    assert np.isnan(metrics.detection_latency(scores, onset_step=None, total_steps=100))


def test_latency_early_detection_is_negative():
    scores = np.zeros(100)
    scores[10:] = 1.0  # fires at 10, onset at 40 -> latency (10-40)/100 = -0.3
    assert metrics.detection_latency(scores, onset_step=40, total_steps=100) == -0.3


# --- FPR@k / miss rate / TFD -----------------------------------------------
def test_fpr_at_k():
    # Highest 3 scores are at indices 4, 3, 2 (labels 1, 1, 0) -> 1 false / 3.
    scores = np.array([0.1, 0.2, 0.9, 0.95, 0.99])
    labels = np.array([0, 0, 0, 1, 1])
    assert metrics.fpr_at_k(scores, labels, k=3) == 1 / 3


def test_miss_rate():
    assert metrics.miss_rate([True, True, False, False]) == 0.5
    assert np.isnan(metrics.miss_rate([]))


def test_time_to_first_detection_median():
    lat = [0.1, 0.3, float("inf"), 0.2]
    assert metrics.time_to_first_detection(lat) == 0.2
    assert np.isnan(metrics.time_to_first_detection([float("inf")]))


# --- RHOB-Score / bootstrap ------------------------------------------------
def test_rhob_score_tier_weighting():
    per_env = {"a": 1.0, "b": 0.0}
    tiers = {"a": Tier.TIER1, "b": Tier.ADVERSARIAL}
    # weights 1.0 and 2.5 -> (1*1 + 2.5*0)/(1+2.5) = 0.2857
    assert abs(metrics.rhob_score(per_env, tiers) - (1.0 / 3.5)) < 1e-9


def test_rhob_score_skips_nan():
    per_env = {"a": 0.8, "b": float("nan")}
    tiers = {"a": Tier.TIER1, "b": Tier.TIER2}
    assert metrics.rhob_score(per_env, tiers) == 0.8


def test_bootstrap_ci_brackets_mean():
    values = [0.8, 0.82, 0.79, 0.81, 0.80]
    lo, hi = metrics.bootstrap_ci(values, n_resamples=2000, confidence=0.95, seed=1)
    assert lo <= np.mean(values) <= hi


def test_bootstrap_ci_degenerate_is_nan():
    lo, hi = metrics.bootstrap_ci([0.5])
    assert np.isnan(lo) and np.isnan(hi)
