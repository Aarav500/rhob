"""``scripts/cross_family_transfer.py`` end to end on stub families, with no real rollouts.

The script's real run takes minutes of rollouts and fits (5m23s for the published design
on a 4-core box under the CI numerics pins), and until 2026-11-27 no test ran it at all --
so three of its defects went unnoticed: the pooled fit and the
held-out evaluation checked channels but not ``min_episodes``, a frozen model that could
not consume a family's histograms was scored at 0.5 (State Divergence's whole transfer
row was 1200 such imputations), and the Top-5 ensemble's "5-fold CV" train AUROC was
in-sample for its fitted member. Each is pinned here on synthetic families built to
trip it (``transfer_stubs``), together with the guard that keeps the 2026-07-09 artifact
from being overwritten.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from transfer_stubs import TEST_STUBS, TRAIN_STUBS, one_torch_thread, register_stubs

from rhob.detectors.l0_reward_mlp import RewardMLPDetector
from rhob.detectors.l1_state_divergence import StateDivergenceDetector
from rhob.detectors.l2_ensemble import EnsembleDetector
from rhob.detectors.l2_trajectory_mlp import TrajectoryMLPDetector
from rhob.detectors.posthoc import PosthocDetector, RunData
from rhob.v3.benchmark import Benchmark, _evaluate_cell, not_applicable_reason
from rhob.v3.leaderboard.transfer import load_transfer_results, transfer_under_sign_randomization

_REPO = Path(__file__).resolve().parents[2]
_HISTORICAL = _REPO / "leaderboard" / "cross_family_transfer.json"


@pytest.fixture(scope="module", autouse=True)
def _one_torch_thread():
    with one_torch_thread():
        yield


@pytest.fixture(scope="module")
def cft():
    path = _REPO / "scripts" / "cross_family_transfer.py"
    spec = importlib.util.spec_from_file_location("_cft_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stubs(monkeypatch):
    register_stubs(monkeypatch, {**TRAIN_STUBS, **TEST_STUBS})
    return {"train": list(TRAIN_STUBS), "test": list(TEST_STUBS)}


def _no_nan(text: str) -> dict:
    def refuse(token):
        raise AssertionError(f"artifact contains {token}")

    return json.loads(text, parse_constant=refuse)


# -------------------------------------------------------------- the shared N/A rule
def _l0_runs(n_runs: int, n_episodes: int) -> list[RunData]:
    rng = np.random.default_rng(0)
    return [
        RunData(rng.normal(0.5, 0.1, n_episodes), np.zeros(n_episodes), None, None)
        for _ in range(n_runs)
    ]


def test_not_applicable_reason_is_the_harness_rule():
    """The script and Benchmark must not be able to disagree about what is N/A."""
    short = _l0_runs(4, 40)
    reason = not_applicable_reason(RewardMLPDetector(), short, "L0")
    _, _, harness = _evaluate_cell(RewardMLPDetector(), short[:2], short[2:], [0, 0], "L0", 40)
    assert reason == harness and "horizon 40" in reason and "50" in reason

    no_state = _l0_runs(4, 80)
    reason = not_applicable_reason(StateDivergenceDetector(), no_state, "L1")
    _, _, harness = _evaluate_cell(
        StateDivergenceDetector(), no_state[:2], no_state[2:], [0, 0], "L1", 80
    )
    assert reason == harness and "state_counts" in reason

    assert not_applicable_reason(RewardMLPDetector(), _l0_runs(4, 80), "L0") is None


# ----------------------------------------------------------------------- pooled fit
def test_the_pooled_fit_leaves_out_a_family_shorter_than_the_detectors_windows(cft, stubs):
    a, b, skipped, reasons = cft.pooled_runs(stubs["train"], "L0", 3, RewardMLPDetector())
    assert skipped == ["stub_train_short"]
    assert "horizon 40" in reasons["stub_train_short"]
    assert len(a) == len(b) == 3 * 2  # the long family's 2 cells x 3 seeds

    # The Trajectory MLP needs 2 episodes, so the short family trains it.
    _, _, skipped, _ = cft.pooled_runs(stubs["train"], "L2", 3, TrajectoryMLPDetector())
    assert skipped == []
    # No gate, no exclusions.
    assert cft.pooled_runs(stubs["train"], "L0", 3)[2] == []


def test_a_family_without_the_channel_is_left_out_of_the_pool(cft, stubs):
    fams = ["stub_train_long", "stub_test_nostate"]
    _, _, skipped, reasons = cft.pooled_runs(fams, "L1", 3, StateDivergenceDetector())
    assert skipped == ["stub_test_nostate"]
    assert "state_counts" in reasons["stub_test_nostate"]


# -------------------------------------------------------------------- transfer step
def test_a_held_out_family_below_min_episodes_is_null_not_half(cft, stubs, monkeypatch):
    register_stubs(monkeypatch, {"stub_test_short": dict(n_episodes=40, n_bins=6)})
    det = RewardMLPDetector(seed=0)
    a, b, _, _ = cft.pooled_runs(["stub_train_long"], "L0", 4, det)
    det.fit(a, b)
    ev = cft.transfer_eval(det, ["stub_test_short", "stub_test_bins"], "L0", 4)
    assert ev.per_family["stub_test_short"] is None  # was 0.5 before 2026-11-27
    assert "horizon 40" in ev.reasons["stub_test_short"]
    assert ev.per_family["stub_test_bins"] is not None
    assert ev.not_applicable == ["stub_test_short"]
    assert ev.average == pytest.approx(ev.per_family["stub_test_bins"])


def test_a_frozen_model_that_cannot_consume_a_family_is_null_and_counted(cft, stubs):
    """State Divergence's centroids are 6-bin; the held-out histograms are 3-bin."""
    det = StateDivergenceDetector()
    a, b, _, _ = cft.pooled_runs(["stub_train_long"], "L1", 3, det)
    det.fit(a, b)
    ev = cft.transfer_eval(det, ["stub_test_bins", "stub_test_nostate"], "L1", 3)
    assert ev.per_family == {"stub_test_bins": None, "stub_test_nostate": None}
    assert ev.average is None
    # Every run of both cells raised: 2 cells x 2 variants x 3 seeds.
    assert ev.incompatible_runs == {"stub_test_bins": 12}
    assert "cannot consume" in ev.reasons["stub_test_bins"]
    assert "state_counts" in ev.reasons["stub_test_nostate"]


class _RaisesOnSomeRuns(PosthocDetector):
    """L0 stand-in that cannot consume every other run."""

    def __init__(self):
        self.calls = 0

    @property
    def access_level(self):
        return "L0"

    @property
    def name(self):
        return "half-compatible"

    def classify(self, run):
        self.calls += 1
        if self.calls % 2:
            raise ValueError("shape mismatch")
        return float(run.proxy_rewards.mean())

    def detect_onset(self, run):
        return -1


def test_a_partial_mismatch_is_not_scored_on_the_runs_that_happened_to_fit(cft, stubs):
    ev = cft.transfer_eval(_RaisesOnSomeRuns(), ["stub_test_bins"], "L0", 3)
    assert ev.per_family["stub_test_bins"] is None
    assert ev.incompatible_runs["stub_test_bins"] == 6  # half of 2 cells x 6 runs
    assert "3 of 6 runs" in ev.reasons["stub_test_bins"]


# ------------------------------------------------------------------------ ensemble
def test_the_ensembles_train_auroc_is_cross_validated(cft, stubs, monkeypatch):
    """``Benchmark.evaluate`` refits the member on every fold, out of fold."""
    assert not hasattr(EnsembleDetector, "fit")  # why the wrapper exists
    calls = []
    real_fit = TrajectoryMLPDetector.fit

    def counting_fit(self, runs_a, runs_b, *args, **kwargs):
        calls.append((len(runs_a), len(runs_b)))
        return real_fit(self, runs_a, runs_b, *args, **kwargs)

    monkeypatch.setattr(TrajectoryMLPDetector, "fit", counting_fit)
    result = Benchmark.evaluate(
        cft.build_ensemble(0), families=["stub_train_long"], n_seeds=5, verbose=False
    )
    assert len(result.scored_cells) == 2
    assert len(calls) == 2 * 5  # 2 cells x 5 folds
    assert all(n_a == 4 and n_b == 4 for n_a, n_b in calls)  # 4/5 of each variant


def test_the_fold_fitted_ensemble_leaves_the_transfer_score_unchanged(cft, stubs):
    """Only train scoring changes: the frozen ensemble is the one the old code built."""
    train, test = stubs["train"], stubs["test"]
    row, _ = cft.run_one(
        "Ensemble (Top 5)", cft.build_ensemble(1), "L2", 4, 4,
        train_families=train, test_families=test,
    )
    # The pre-2026-11-27 construction: fit the member on the pool, gate on the member.
    mlp = TrajectoryMLPDetector(seed=1)
    a, b, _, _ = cft.pooled_runs(train, "L2", 4, mlp)
    mlp.fit(a, b)
    old = EnsembleDetector(cft._stateless_l2_members() + [mlp], name="Ensemble (Top 5)")
    ev = cft.transfer_eval(old, test, "L2", 4)
    assert row["per_family_transfer"] == {k: cft._round_or_none(v) for k, v in ev.per_family.items()}
    # The ensemble's own minimum (50, Feature Magnitude) makes the short family N/A for
    # its train AUROC, but its member is still trained on it.
    assert row["train_families_excluded_from_pooled_fit"] == []
    assert row["train_cells_not_applicable"] == 2


# ----------------------------------------------------------------------- the script
def test_main_writes_a_shipped_configuration_artifact(cft, stubs, tmp_path):
    out = tmp_path / "transfer.json"
    code = cft.main([
        "--n-trials", "2", "--n-seeds-train", "4", "--n-seeds-test", "4",
        "--train-families", *stubs["train"], "--test-families", *stubs["test"],
        "--out", str(out),
    ])
    assert code == 0
    payload = _no_nan(out.read_text(encoding="utf-8"))
    assert payload["configs"] == {
        "shipped": {"randomize_behav_sign": True, "detector_infers_orientation": True}
    }
    assert payload["published_design"] is False  # reduced families and seeds
    assert set(payload["results"]) == set(cft.ROWS)
    assert set(payload["provenance"]["wall_clock_s"]) == set(cft.ROWS)

    rows = payload["results"]
    for row in rows.values():
        assert "test_not_applicable_reasons" in row and row["pooled_fit_error"] is None
    assert rows["Reward MLP"]["train_families_excluded_from_pooled_fit"] == ["stub_train_short"]
    assert rows["Reward MLP"]["train_cells_not_applicable"] == 2
    assert len(rows["Trajectory MLP"]["all_trial_transfer_aurocs"]) == 2

    sd = rows["State Divergence"]
    assert sd["avg_transfer_auroc"] is None and sd["generalization_gap_pct"] is None
    assert sd["test_families_not_applicable"] == stubs["test"]
    assert sd["per_family_incompatible_runs"] == {"stub_test_bins": 16}

    board = load_transfer_results(out)
    assert board.shipped_config() == "shipped"
    assert board.predates_sign_randomization is False
    assert board.applies_na_rules is True
    assert board.published_design is False
    current = {c.name: c for c in transfer_under_sign_randomization(board)}
    assert current["State Divergence"].rts is None
    assert current["State Divergence"].row.n_families_scored == 0
    assert not any(c.carried_over_as_invariant for c in current.values())


@pytest.mark.parametrize("target", [
    "leaderboard/cross_family_transfer.json",
    str(_HISTORICAL),
])
def test_main_refuses_to_overwrite_the_historical_artifact(cft, target, capsys):
    before = _HISTORICAL.read_bytes() if _HISTORICAL.exists() else None
    with pytest.raises(SystemExit) as exc:
        cft.main(["--out", target, "--n-trials", "1"])
    assert exc.value.code == 2
    assert "refusing to overwrite" in capsys.readouterr().err
    if before is not None:
        assert _HISTORICAL.read_bytes() == before


def test_the_default_output_is_the_v5_path_and_the_published_design(cft):
    args = cft.build_parser().parse_args([])
    assert args.out == Path("leaderboard/cross_family_transfer_v5.json")
    assert cft.published_design(args, cft.ROWS) is True
    assert cft.published_design(cft.build_parser().parse_args(["--n-trials", "1"]), cft.ROWS) is False
    assert cft.published_design(args, ["Reward MLP"]) is False
