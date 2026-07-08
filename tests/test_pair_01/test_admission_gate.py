"""Tests for the admission gate on controlled synthetic run logs."""

from __future__ import annotations

import numpy as np

from rhob.environments.pair_01 import DEFAULT
from rhob.environments.pair_01.rollout import RunLog
from rhob.evaluation.admission_gate import compute_admission


def _run(seed: int, proxy_level: float, behav_level: float, onset_ep: int) -> RunLog:
    n = DEFAULT.n_episodes
    proxy = np.full(n, 90.0)
    proxy[onset_ep:] = proxy_level
    behav = np.zeros(n)
    behav[onset_ep:] = behav_level
    frac2 = np.zeros(n)
    frac2[onset_ep:] = 1.0
    return RunLog(seed=seed, proxy=proxy, true=np.zeros(n), frac_tile2=frac2, behav=behav)


def _pop(proxy_level: float, behav_level: float) -> list[RunLog]:
    # Small deterministic spread in onset timing (well within the std threshold).
    return [_run(s, proxy_level, behav_level, 250 + (s % 5)) for s in range(20)]


def test_matched_pair_is_admitted():
    a = _pop(134.0, -1.0)  # hacking: same proxy, behaviour -1
    b = _pop(134.0, +1.0)  # legit:   same proxy, behaviour +1
    report = compute_admission(a, b)
    assert report.tv_pass
    assert report.l0_pass
    assert report.l2_pass
    assert report.onset_std_pass
    assert report.onset_detected_pass
    assert report.admitted


def test_rejected_when_proxy_reveals_variant():
    a = _pop(100.0, -1.0)
    b = _pop(150.0, +1.0)  # proxy differs -> reward reveals the variant
    report = compute_admission(a, b)
    assert not (report.tv_pass and report.l0_pass)
    assert not report.admitted


def test_rejected_when_behavior_does_not_separate():
    a = _pop(134.0, 0.0)
    b = _pop(134.0, 0.0)  # identical behaviour -> L2 cannot separate
    report = compute_admission(a, b)
    assert not report.l2_pass
    assert not report.admitted


def test_report_renders_table():
    report = compute_admission(_pop(134.0, -1.0), _pop(134.0, 1.0))
    text = str(report)
    assert "ADMISSION GATE: PAIR 01" in text
    assert "ADMITTED" in text
