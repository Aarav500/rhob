"""Which transfer measurement the Space's RTS panel serves, and how it labels it.

``space/app.py`` prefers ``leaderboard/cross_family_transfer_v5.json`` and falls back to
the 2026-08-03 sign-randomization measurement, labelled *pre-declaration* while its rows
predate the 2026-11-27 not-applicable rules; it never serves the 2026-07-09 artifact as
current. The page is a deployed dashboard with no other test, so these pin the choice
and the labels. ``gradio`` and ``pandas`` are not in the CI lock, so when absent they are
replaced by stand-ins: the module builds its UI at import time but these tests only call
the HTML builders.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from html import unescape
from pathlib import Path
from unittest import mock

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SIGN_RAND = _REPO / "docs" / "figures" / "sign_randomization_impact.json"
_REPLICATED = _REPO / "leaderboard" / "v5_replicated.json"

pytestmark = pytest.mark.skipif(
    not (_SIGN_RAND.exists() and _REPLICATED.exists()), reason="Space inputs not present"
)


@pytest.fixture
def app(monkeypatch):
    for name in ("gradio", "pandas"):
        try:
            __import__(name)
        except ImportError:
            monkeypatch.setitem(sys.modules, name, mock.MagicMock())
    spec = importlib.util.spec_from_file_location("_space_app_under_test", _REPO / "space" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    module._transfer_board.cache_clear()
    module._transfer_rows.cache_clear()


def _point(app, monkeypatch, current: Path | None, sign_rand: Path | None) -> None:
    monkeypatch.setattr(app, "_CURRENT_TRANSFER_FILE", current or Path("/nonexistent/v5.json"))
    monkeypatch.setattr(app, "_SIGN_RANDOMIZATION_FILE", sign_rand or Path("/nonexistent/sr.json"))
    app._transfer_board.cache_clear()
    app._transfer_rows.cache_clear()


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", unescape(html)))


def _v5(tmp_path: Path) -> Path:
    def row(level, avg, per_family, **extra):
        return {
            "access_level": level, "n_trials": 5, "train_auroc_mean": 0.7,
            "avg_transfer_auroc_mean": avg, "avg_transfer_auroc_std": 0.01,
            "per_family_transfer_mean": per_family,
            "test_families_not_applicable": [k for k, v in per_family.items() if v is None],
            "test_not_applicable_reasons": extra.pop("reasons", {}),
            "per_family_incompatible_runs": {}, "pooled_fit_error": None, **extra,
        }

    path = tmp_path / "cross_family_transfer_v5.json"
    path.write_text(json.dumps({
        "train_families": ["t"] * 6, "test_families": ["a", "b"], "n_trials": 5,
        "configs": {"shipped": {"randomize_behav_sign": True,
                                "detector_infers_orientation": True}},
        "provenance": {"generated_utc": "2026-11-28T00:00:00Z",
                       "git": {"short_commit": "abcdef123456", "dirty": False}},
        "results": {
            "Trajectory MLP": row("L2", 0.725, {"a": 0.7, "b": 0.75}),
            "Ensemble (Top 5)": row("L2", 0.542, {"a": 0.5, "b": 0.584}),
            "State Divergence": row(
                "L1", None, {"a": None, "b": None},
                reasons={"a": "the frozen model cannot consume this family's state_counts"},
            ),
        },
    }), encoding="utf-8")
    return path


def test_the_current_artifact_is_preferred_and_not_labelled_pre_declaration(
    app, monkeypatch, tmp_path
):
    _point(app, monkeypatch, _v5(tmp_path), _SIGN_RAND)
    text = _text(app._rts_html())
    assert "Pre-declaration measurement" not in text
    assert "cross_family_transfer_v5.json" in text and "2026-11-28" in text
    assert "State Divergence scored no held-out family" in text
    assert "reproduces with randomization off" not in text  # v5 has no baseline config
    assert app._best_transfer().name == "Trajectory MLP"


def test_the_sign_randomization_measurement_is_the_labelled_fallback(app, monkeypatch):
    _point(app, monkeypatch, None, _SIGN_RAND)
    board = app._transfer_board()
    assert board is not None and board.path == _SIGN_RAND
    text = _text(app._rts_html())
    if board.applies_na_rules:
        assert "Pre-declaration measurement" not in text
    else:
        assert "Pre-declaration measurement" in text
        assert "column of imputed scores" in text  # its flat 0.500 L1 row


def test_without_a_measurement_the_panel_shows_nothing_rather_than_the_old_figures(
    app, monkeypatch
):
    _point(app, monkeypatch, None, None)
    assert app._transfer_rows() == ()
    text = _text(app._rts_html())
    assert "not rendered on this build" in text
    assert "0.994" not in text


def test_contributors_are_sent_to_submissions_not_leaderboard():
    source = (_REPO / "space" / "app.py").read_text(encoding="utf-8")
    assert "Open a PR adding your submission under `leaderboard/`" not in source
    assert "under `submissions/`" in source
    assert "leaderboard/community/" in source


@pytest.mark.parametrize(
    "resampled, says, never",
    [
        (True, "layout seed and rollout seeds resampled together", "environment layout 0"),
        (False, "every draw at environment layout 0", "resampled together"),
        (None, "independent draws of the full benchmark", "resampled together"),
    ],
)
def test_the_provenance_panel_says_what_the_draws_resampled(app, monkeypatch, resampled, says, never):
    """Draws made before the layout-seed fix varied their rollout seeds only."""
    board = app._board()
    monkeypatch.setattr(type(board), "layouts_resampled", property(lambda self: resampled))
    text = _text(app._provenance_html())
    assert says in text
    assert never not in text


def test_the_committed_board_is_described_by_its_own_layout_flag(app):
    board = app._board()
    text = _text(app._provenance_html())
    if board.layouts_resampled is True:
        assert "layout seed and rollout seeds resampled together" in text
    else:
        assert "resampled together" not in text
