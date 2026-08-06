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

# One detector per access level, so the ladder and the adjacent-rung comparisons have
# something to aggregate.
_DETECTORS = {
    "Reward Threshold": "L0",
    "State Divergence": "L1",
    "Behavioral Threshold": "L2",
    "True Reward Oracle": "L3",
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


def test_adjacent_level_comparison_pairs_by_replicate_not_by_position(tmp_path: Path):
    """A gap in one rung must not shift the other rung's values onto the wrong draws.

    The comparison between two access levels is paired: each draw scores both rungs, and
    the difference is taken within a draw. Collecting each rung into a flat list and
    zipping breaks that the moment either rung is unscoreable on any replicate -- and
    breaks it *silently*, because when the gaps are equal in number the lists are still
    the same length and a length check still passes.

    Here L1 is unscoreable on replicate 1 and L2 on replicate 3. Correct pairing uses
    replicates {0, 2, 4} and gives a mean difference of 0.3933. Position-zipping would
    pair L1@rep2 with L2@rep1 and L1@rep4 with L2@rep4, giving 0.3875 off four
    "pairs", two of which compare different draws.
    """
    in_dir = tmp_path / "reps"
    in_dir.mkdir()
    l1 = {0: 0.50, 1: None, 2: 0.52, 3: 0.54, 4: 0.56}
    l2 = {0: 0.90, 1: 0.91, 2: 0.92, 3: None, 4: 0.94}
    for rep in range(5):
        _write_replicate(in_dir, rep, {
            "Reward Threshold": 0.50 + 0.01 * rep,
            "State Divergence": l1[rep],
            "Behavioral Threshold": l2[rep],
            "True Reward Oracle": 0.99,
        })

    proc = _run_aggregator(in_dir, tmp_path)
    assert proc.returncode == 0, proc.stderr
    sep = json.loads((tmp_path / "agg.json").read_text())["adjacent_level_separation"]
    l2_l1 = sep["L2_minus_L1_max"]

    assert l2_l1["n_paired"] == 3, (
        f"expected the 3 replicates scoring both rungs, got n_paired={l2_l1['n_paired']}"
    )
    assert l2_l1["mean"] == pytest.approx(0.39333, abs=1e-4), (
        f"mean difference {l2_l1['mean']} != 0.39333; 0.3875 means the rungs were "
        f"zipped by position and compared across different draws"
    )


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


def test_intervals_do_not_move_when_another_quantity_is_added():
    """A published interval must not shift because an unrelated statistic was added.

    Threading one shared Generator through every bootstrap makes each interval depend on
    how many calls preceded it. That is invisible until someone adds a reported quantity,
    at which point every interval after it in the call order moves while the underlying
    data is untouched. It happened here: adding the mean-column and unsupervised-partition
    ladders shifted L0's suite-mean interval from [0.493366, 0.499950] to
    [0.493327, 0.499889], and a paper citing the artifact printed the superseded pair.

    Seeding each call from its own name makes an interval a function of (its values, its
    name) alone, so this test can assert exact equality across a changed call sequence.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_agg", _AGG)
    agg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agg)

    values = [0.48, 0.52, 0.50, 0.55, 0.47, 0.61, 0.49]

    before = agg.bootstrap_ci(values, "ladder::L0::mean")
    # Simulate a later revision adding several unrelated reported quantities first.
    for k in ("unsup::L0::max", "unsup::L1::mean", "detector::Something", "all::L2_minus_L1::max"):
        agg.bootstrap_ci([0.1, 0.2, 0.3, 0.4], k)
    after = agg.bootstrap_ci(values, "ladder::L0::mean")

    assert before == after, (
        "the interval moved when other quantities were bootstrapped first; each call must "
        "seed from its own key, not from a shared generator"
    )

    # ...and distinct quantities must still get distinct resamples, or the keying is inert.
    other = agg.bootstrap_ci(values, "ladder::L1::mean")
    assert (other["ci_lo"], other["ci_hi"]) != (before["ci_lo"], before["ci_hi"]), (
        "two differently-named quantities produced identical intervals from identical "
        "values -- the per-key seeding is not actually varying the resample"
    )
