"""The aggregator must refuse to publish an interval it cannot estimate.

``scripts/aggregate_replication.py`` has two guards, and a guard that has never been
observed to fire is indistinguishable from one that cannot. These tests drive both paths
with synthetic replicate files, so the refusal is verified rather than assumed.

The failure being guarded against is specific: if the rollout cache (or any other shared
state) serves one draw to every replicate, every detector returns an identical score, the
percentile bootstrap over those scores has zero width, and the artifact reports a "95%
CI" of [x, x]. That is not a tight measurement, it is the absence of one -- and it is far
more dangerous than a wide interval, because it reads as precision.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_AGG = _REPO / "scripts" / "aggregate_replication.py"

# Two detectors at different access levels, so the ladder has something to aggregate.
_DETECTORS = {
    "Reward Threshold": "L0",
    "Behavioral Threshold": "L2",
}
_FAMILIES = ["gridworld_camping", "continuous_camping"]


def _write_replicate(out_dir: Path, rep_id: int, auroc_by_detector: dict[str, float]) -> None:
    payload = {
        "replicate_id": rep_id,
        "layout_seed": rep_id,
        "seed_base": rep_id * 1_000_000,
        "n_seeds": 5,
        "families_evaluated": _FAMILIES,
        "results": {
            name: {
                "access_level": _DETECTORS[name],
                "overall_auroc": auroc,
                "per_family": {f: auroc for f in _FAMILIES},
                "cells": 2,
                "cells_measured": 2,
                "cells_not_applicable": 0,
                "not_applicable_families": [],
                "not_applicable_reasons": {},
            }
            for name, auroc in auroc_by_detector.items()
        },
    }
    (out_dir / f"replicate_{rep_id:03d}.json").write_text(json.dumps(payload))


def _run_aggregator(in_dir: Path, tmp_path: Path):
    return subprocess.run(
        [
            sys.executable, str(_AGG),
            "--in-dir", str(in_dir),
            "--out", str(tmp_path / "agg.json"),
            "--out-md", str(tmp_path / "agg.md"),
        ],
        capture_output=True, text=True, cwd=str(_REPO),
    )


def test_identical_replicates_are_refused(tmp_path: Path):
    """Every replicate scoring identically means the draws collapsed. Refuse to publish."""
    in_dir = tmp_path / "reps"
    in_dir.mkdir()
    for rep in range(5):
        _write_replicate(in_dir, rep, {"Reward Threshold": 0.5, "Behavioral Threshold": 0.975})

    proc = _run_aggregator(in_dir, tmp_path)

    assert proc.returncode != 0, (
        "aggregator accepted 5 bit-identical replicates and would have published "
        f"zero-width intervals as a result.\nstdout:\n{proc.stdout}"
    )
    assert "FATAL" in proc.stdout + proc.stderr
    assert not (tmp_path / "agg.json").exists(), "an artifact was written despite the refusal"


def test_varying_replicates_produce_a_nonzero_interval(tmp_path: Path):
    """The positive control: real variation must yield a real, non-degenerate interval."""
    in_dir = tmp_path / "reps"
    in_dir.mkdir()
    for rep, (l0, l2) in enumerate(
        [(0.48, 0.91), (0.52, 0.88), (0.50, 0.95), (0.55, 0.86), (0.47, 0.93)]
    ):
        _write_replicate(in_dir, rep, {"Reward Threshold": l0, "Behavioral Threshold": l2})

    proc = _run_aggregator(in_dir, tmp_path)
    assert proc.returncode == 0, f"aggregator failed on valid input:\n{proc.stderr}"

    out = json.loads((tmp_path / "agg.json").read_text())
    assert out["n_replicates"] == 5
    bt = out["detectors"]["Behavioral Threshold"]
    assert bt["ci_hi"] > bt["ci_lo"], "interval has zero width on genuinely varying input"
    assert bt["ci_lo"] < bt["mean"] < bt["ci_hi"]
    assert bt["sd"] > 0


def test_duplicate_draws_are_refused(tmp_path: Path):
    """Two replicates at the same draw are one sample counted twice, not two samples."""
    in_dir = tmp_path / "reps"
    in_dir.mkdir()
    for rep, auroc in enumerate([0.48, 0.52, 0.50]):
        _write_replicate(in_dir, rep, {"Reward Threshold": auroc, "Behavioral Threshold": 0.9})
    # Re-stamp replicate 2 onto replicate 1's draw.
    dup = json.loads((in_dir / "replicate_002.json").read_text())
    dup["layout_seed"], dup["seed_base"] = 1, 1_000_000
    (in_dir / "replicate_002.json").write_text(json.dumps(dup))

    proc = _run_aggregator(in_dir, tmp_path)
    assert proc.returncode != 0, "aggregator pooled two replicates that share a draw"
    assert "share a draw" in proc.stdout + proc.stderr


def test_zero_variance_detector_is_reported_not_fatal(tmp_path: Path):
    """A saturated detector is a real finding; only *global* collapse is fatal."""
    in_dir = tmp_path / "reps"
    in_dir.mkdir()
    for rep, l0 in enumerate([0.48, 0.52, 0.50, 0.55, 0.47]):
        # L2 saturates at 1.0 on every draw; L0 varies.
        _write_replicate(in_dir, rep, {"Reward Threshold": l0, "Behavioral Threshold": 1.0})

    proc = _run_aggregator(in_dir, tmp_path)
    assert proc.returncode == 0, f"a single saturated detector must not be fatal:\n{proc.stderr}"

    out = json.loads((tmp_path / "agg.json").read_text())
    assert "Behavioral Threshold" in out["degeneracy"]["detectors_with_zero_variance"]
    assert "Reward Threshold" not in out["degeneracy"]["detectors_with_zero_variance"]


@pytest.mark.parametrize("n", [1])
def test_single_replicate_gets_no_interval(tmp_path: Path, n: int):
    """One draw is the regime this study replaces. Report the point, refuse the interval."""
    in_dir = tmp_path / "reps"
    in_dir.mkdir()
    _write_replicate(in_dir, 0, {"Reward Threshold": 0.5, "Behavioral Threshold": 0.975})

    proc = _run_aggregator(in_dir, tmp_path)
    assert proc.returncode == 0, proc.stderr
    out = json.loads((tmp_path / "agg.json").read_text())
    bt = out["detectors"]["Behavioral Threshold"]
    assert bt["ci_lo"] is None and bt["ci_hi"] is None
    assert "single replicate" in bt.get("note", "")
