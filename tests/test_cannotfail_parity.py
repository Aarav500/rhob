"""The standalone tool must agree with the gate it was lifted from.

``tools/cannotfail`` deliberately imports nothing from ``rhob``, so that someone else's
benchmark can install it without RHOB's dependencies. That makes it a copy, and a copy
can drift. These tests hold it to the original: bit-identical resolution on adversarial
inputs, identical design arithmetic, and the same verdicts on the 50 cells of the
committed admission ledger.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "cannotfail" / "src"))

import cannotfail  # noqa: E402
from cannotfail.power import ci_multiplier as cf_ci_multiplier  # noqa: E402
from cannotfail.resolution import Resolution  # noqa: E402
from rhob.v3 import admission_gate as gate  # noqa: E402

LEDGER = REPO / "admission" / "admission_ledger.json"


def _gate_resolution(pos, neg) -> float:
    pos, neg = list(map(float, pos)), list(map(float, neg))
    with np.errstate(invalid="ignore"):
        return gate._score_resolution([1] * len(pos) + [0] * len(neg), pos + neg)


def _cf_resolution(pos, neg) -> float:
    r = cannotfail.resolution(
        np.concatenate([np.asarray(pos, float), np.asarray(neg, float)]),
        [1] * len(pos) + [0] * len(neg),
    )
    return r.resolution


def _cases(rng):
    yield rng.normal(size=24), rng.normal(size=24)
    yield np.round(rng.normal(size=24), 1), np.round(rng.normal(size=24), 1)
    yield np.full(24, -13.069062904), np.full(24, -13.069062904)
    # orbit_chirality's Reward KDE dust: 6 values spanning 3.0e-10 around -13.069062904.
    dust = -13.069062904 + rng.choice(np.linspace(-1.5e-10, 1.5e-10, 6), size=48)
    yield dust[:24], dust[24:]
    base = rng.normal(scale=50.0, size=24)
    for k in (0.999999, 1.0, 1.000001, 2.0):
        yield base, base * (1.0 + gate._TIE_REL_TOL * k)
    yield np.concatenate([rng.normal(size=20), [np.nan] * 4]), rng.normal(size=24)
    yield np.zeros(24), np.zeros(24)


@pytest.mark.parametrize("seed", range(20))
def test_resolution_is_bit_identical_to_the_gate(seed):
    rng = np.random.default_rng(seed)
    for pos, neg in _cases(rng):
        g, c = _gate_resolution(pos, neg), _cf_resolution(pos, neg)
        assert (g == c) or (g != g and c != c), (g, c)


def test_the_tie_tolerances_are_the_gates():
    assert cannotfail.resolution.__module__  # imported, not shadowed
    from cannotfail.resolution import ABS_TOL, EQUIVALENCE_MARGIN, REL_TOL

    assert REL_TOL == gate._TIE_REL_TOL
    assert ABS_TOL == gate._TIE_ABS_TOL
    assert EQUIVALENCE_MARGIN == gate.EQUIVALENCE_MARGIN


@pytest.mark.parametrize("margin", [0.05, 0.10, 0.15])
@pytest.mark.parametrize("n_layouts", [4, 8, 12, 20])
@pytest.mark.parametrize("power", [0.8, 0.9])
@pytest.mark.parametrize("n_tests", [1, 4])
def test_required_seeds_match_the_gate(margin, n_layouts, power, n_tests):
    assert cannotfail.required_seeds_per_layout(
        margin=margin, n_layouts=n_layouts, power=power, n_tests=n_tests
    ) == gate.required_seeds_per_layout(
        margin=margin, n_layouts=n_layouts, power=power, n_equivalence_tests=n_tests
    )


@pytest.mark.parametrize("n_layouts", [2, 5, 12, 40])
def test_ci_multiplier_and_null_sd_match_the_gate(n_layouts):
    assert cf_ci_multiplier(n_layouts) == gate.ci_multiplier(n_layouts)
    for n in (3, 5, 24):
        assert cannotfail.mann_whitney_null_sd(n) == gate.mann_whitney_null_sd(n)


def _ledger_cells():
    return json.loads(LEDGER.read_text(encoding="utf-8"))["results"]


def test_verdicts_on_the_ledger_match_the_gates_outcomes():
    """Fed the gate's own measurements, the tool's verdict rule reproduces every outcome.

    The gate calls a cell DEGENERATE when the proxy is uninformative or the statistic's
    attainable range sits inside the margin. The tool's ``dust`` and ``pinned`` verdicts
    are those two halves, and ``fixed`` is the limiting case of the second.
    """
    cells = _ledger_cells()
    assert len(cells) == 50
    for cell in cells:
        m = cell["metrics"]["proxy_matched"]
        r = Resolution(
            n=2, n_distinct=2, n_nonfinite=0, n_pos=1, n_neg=1,
            resolution=m["statistic_resolution"],
            dispersion=m["proxy_informativeness"],
            margin=m["margin"],
            dispersion_floor=m["informativeness_floor"],
        )
        tool_degenerate = r.verdict in {"fixed", "dust", "pinned"}
        gate_degenerate = cell["outcomes"]["proxy_matched"] == "degenerate"
        assert tool_degenerate == gate_degenerate, (cell["family"], cell["difficulty"], r.verdict)
    assert sum(c["outcomes"]["proxy_matched"] == "degenerate" for c in cells) == 30


def test_onset_localizable_is_fixed_across_the_ledger():
    """The instance the audit package used to miss: one value, in all 50 cells."""
    stds = [c["metrics"]["onset_localizable"]["onset_std"] for c in _ledger_cells()]
    assert len(stds) == 50
    r = cannotfail.resolution(stds)
    assert r.n_distinct == 1
    assert r.verdict == "fixed"
    assert all(c["outcomes"]["onset_localizable"] == "pass" for c in _ledger_cells())
