"""End-to-end integration tests.

These encode the Milestone 1 success criteria: the environment reliably hacks,
the onset labels are consistent, CUSUM beats Random, and the whole pipeline is
deterministic and fast.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.stats import ttest_1samp

from rhob.config import EvaluationConfig
from rhob.detectors.baselines.cusum import CUSUMDetector
from rhob.detectors.baselines.random_detector import RandomDetector
from rhob.evaluation.runner import evaluate


def _hacking_aurocs(report):
    return [r.auroc for r in report.per_trajectory if r.is_hacking and not np.isnan(r.auroc)]


def test_gridworld_hacking_rate(hacking_runs):
    """>= 60% of runs exhibit hacking (Milestone 1 success criterion 1)."""
    n_hacked = sum(1 for t in hacking_runs if t.onset_label is not None)
    assert n_hacked >= 6, f"only {n_hacked}/10 runs hacked"


def test_onset_labels_consistent(env, hacking_runs):
    """Onset labels are self-consistent and land in the expected window."""
    lo, hi = env.expected_onset_range
    for traj in hacking_runs:
        assert env._oracle.validate_label(traj.reward_true, traj.onset_label)
        assert lo <= traj.onset_label.onset_step <= hi


def test_cusum_beats_random(mixed_dataset):
    """CUSUM AUROC is significantly above random (criterion 3 & 4)."""
    cusum = evaluate(CUSUMDetector(), mixed_dataset)
    random = evaluate(RandomDetector(), mixed_dataset)

    cusum_aurocs = _hacking_aurocs(cusum)
    assert np.mean(cusum_aurocs) > 0.55
    _, p = ttest_1samp(cusum_aurocs, 0.5)
    assert p < 0.05

    random_aurocs = _hacking_aurocs(random)
    assert 0.4 < np.mean(random_aurocs) < 0.6


def test_cusum_ranks_above_random(mixed_dataset):
    cusum = evaluate(CUSUMDetector(), mixed_dataset)
    random = evaluate(RandomDetector(), mixed_dataset)
    assert cusum.rhob_score > random.rhob_score
    assert cusum.mean_auroc > 0.6


def test_full_pipeline_deterministic(mixed_dataset):
    """Same inputs -> identical metrics (criterion: reproducibility)."""
    r1 = evaluate(CUSUMDetector(), mixed_dataset)
    r2 = evaluate(CUSUMDetector(), mixed_dataset)
    assert r1.rhob_score == r2.rhob_score
    assert r1.mean_auroc == r2.mean_auroc
    env_id = "tier1/gridworld_wireheading"
    assert r1.per_environment[env_id].auroc == r2.per_environment[env_id].auroc


def test_pipeline_is_fast(mixed_dataset):
    """Detection pipeline (excluding generation) is well under 5 minutes."""
    start = time.perf_counter()
    evaluate(CUSUMDetector(), mixed_dataset)
    evaluate(RandomDetector(), mixed_dataset)
    assert time.perf_counter() - start < 60.0


def test_report_serialization_roundtrip(mixed_dataset):
    report = evaluate(CUSUMDetector(), mixed_dataset)
    import json

    data = json.loads(report.to_json())
    assert data["aggregate"]["rhob_score"] is not None
    assert "tier1/gridworld_wireheading" in data["per_environment"]
    md = report.to_markdown()
    assert "RHOB-Score" in md


def test_config_is_respected(mixed_dataset):
    """A stricter threshold changes detection outcomes (config plumbed through)."""
    low = evaluate(CUSUMDetector(), mixed_dataset, EvaluationConfig(score_threshold=0.1))
    high = evaluate(CUSUMDetector(), mixed_dataset, EvaluationConfig(score_threshold=0.99))
    # A higher firing threshold cannot detect earlier than a lower one.
    assert high.mean_tfd >= low.mean_tfd or np.isnan(high.mean_tfd)
