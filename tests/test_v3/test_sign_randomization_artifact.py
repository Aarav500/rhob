"""The committed sign-randomization Markdown says what its JSON says.

``docs/figures/sign_randomization_impact.md`` is the table the README, the explainer and
the Space link to; ``docs/figures/sign_randomization_impact.json`` is the measurement.
Both are written by ``scripts/measure_sign_randomization.py report`` (or ``ci``), but
nothing tied them together once committed: a hand edit or a partial regeneration of one
would leave the docs quoting numbers the artifact does not contain. These checks hold
for the 2026-08-03 files and for any regeneration, and the last one only for an artifact
that carries its per-cell AUROCs (written since 2026-11-27).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

_FIG = Path(__file__).resolve().parents[2] / "docs" / "figures"
_JSON = _FIG / "sign_randomization_impact.json"
_MD = _FIG / "sign_randomization_impact.md"

pytestmark = pytest.mark.skipif(
    not (_JSON.exists() and _MD.exists()), reason="sign-randomization artifact not present"
)


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def md() -> str:
    return _MD.read_text(encoding="utf-8")


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _md_row(md: str, name: str) -> list[str]:
    """The cells of the per-detector table row for ``name``."""
    section = md.split("## Per-detector L2 (and L3) AUROC", 1)[1].split("\n## ", 1)[0]
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == name:
            return cells
    raise AssertionError(f"no per-detector row for {name!r} in {_MD.name}")


def test_every_per_detector_row_matches_the_json(payload, md):
    for name, row in payload["per_detector"].items():
        cells = _md_row(md, name)
        for config in ("before", "before_infer", "blind", "after"):
            assert _fmt(row[config]) in cells, (name, config, cells)
        assert _fmt(row.get("delta_after_minus_before")) == cells[-1], (name, cells)


def test_the_ladder_rows_match_the_json(payload, md):
    for config in ("before", "after"):
        rung = payload["l2_access_ladder"][config]
        expected = (
            f"| `{config}` | {rung['n_detectors']} | {rung['mean_auroc']:.4f} | "
            f"{rung['sd_across_detectors']:.4f} | {rung['max_auroc']:.4f} | "
            f"{rung['best_detector']} |"
        )
        assert expected in md


def test_the_transfer_table_matches_the_json(payload, md):
    transfer = payload.get("transfer")
    if not transfer:
        pytest.skip("artifact has no transfer block")
    section = md.split("## Cross-family transfer (RTS)", 1)[1].split("\n## ", 1)[0]
    for config, rows in transfer["results"].items():
        for name, row in rows.items():
            rts = row.get("avg_transfer_auroc_mean", row.get("avg_transfer_auroc"))
            shown = "N/A" if rts is None else f"{rts:.3f}"
            line = next(ln for ln in section.splitlines() if ln.startswith(f"| {name} |"))
            assert f"**{shown}**" in line, (config, name, line)


def test_the_overall_aurocs_are_the_means_of_the_committed_cells(payload):
    if "cells" not in payload:
        pytest.skip("artifact predates the per-cell block (written since 2026-11-27)")
    columns = payload["cells_columns"]
    auroc_at = columns.index("auroc")
    for name, by_config in payload["cells"].items():
        row = payload["per_detector"][name]
        for config, cells in by_config.items():
            scored = [c[auroc_at] for c in cells if c[auroc_at] is not None]
            assert len(scored) == row["n_cells_scored"][config], (name, config)
            if scored:
                assert row[config] == pytest.approx(float(np.mean(scored)), abs=1e-6)
