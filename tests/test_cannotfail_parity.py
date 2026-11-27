"""The standalone tool must agree with the gate it was lifted from.

``tools/cannotfail`` deliberately imports nothing from ``rhob``, so that someone else's
benchmark can install it without RHOB's dependencies. That makes it a copy, and a copy
can drift. These tests hold it to the original: the gate's tolerance resolution bit for
bit on adversarial inputs, pairs exactly on the tolerance among them, with one documented
exception; identical design arithmetic;
the verdict rule against the gate's own cut at the boundary; and, on the committed
admission ledger, which cells the gate marks degenerate.

The ledger is checked twice. The first test feeds the gate's *measured* resolution and
informativeness, as committed, into the tool's verdict rule for all 123 cells. Those values
sit far from the cut (at most 0.0035, or at least 0.874), so on their own they check the
rule's direction, not where it cuts. The boundary tests below pin the cut. The second test
does not take the gate's word for the numbers. The ledger stores aggregates, not score
vectors, so it re-certifies each DEGENERATE family at the ledger's design and captures
every score vector and proxy stream the gate measures. The tool recomputes each
resolution and dispersion from those vectors and must reproduce the ledger's figures and
call the cell degenerate on its own. One of the six families runs in the fast suite and
the other five under ``-m slow``.

There is a third copy of the tie rule: ``inspect_eval/hvtb_hack_detection/metrics.py``
(``score_resolution``). It imports ``inspect_ai``, which this suite does not install, so
the tests at the end read that file with :mod:`ast` instead of importing it. They check
that its tolerances are the gate's, and they run its tie rule, compiled from that source,
against the gate. Like the gate, and unlike ``cannotfail``, it counts a finite pair whose
difference overflows as tied. A change to the gate's tolerance has to be applied to all
three, and these tests fail until it is.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "cannotfail" / "src"))

import cannotfail  # noqa: E402
import rhob.v3.families  # noqa: E402,F401  (registration side effects)
from cannotfail.power import ci_multiplier as cf_ci_multiplier  # noqa: E402
from cannotfail.resolution import ABS_TOL, REL_TOL, Resolution  # noqa: E402
from rhob.v3 import admission_gate as gate  # noqa: E402
from rhob.v3.registry import FamilyRegistry  # noqa: E402

LEDGER = REPO / "admission" / "admission_ledger.json"
INSPECT_METRICS = REPO / "inspect_eval" / "hvtb_hack_detection" / "metrics.py"


def _gate_resolution(pos, neg) -> float:
    pos, neg = list(map(float, pos)), list(map(float, neg))
    with np.errstate(invalid="ignore", over="ignore"):
        return gate._score_resolution([1] * len(pos) + [0] * len(neg), pos + neg)


def _cf_resolution(pos, neg) -> float:
    r = cannotfail.resolution(
        np.concatenate([np.asarray(pos, float), np.asarray(neg, float)]),
        [1] * len(pos) + [0] * len(neg),
    )
    return r.tolerance_resolution


def _overflows(pos, neg) -> bool:
    a = np.asarray(pos, float)[:, None]
    b = np.asarray(neg, float)[None, :]
    with np.errstate(over="ignore", invalid="ignore"):
        return bool((np.isfinite(a) & np.isfinite(b) & ~np.isfinite(a - b)).any())


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
    yield from _ON_THE_TOLERANCE


#: Pairs exactly on the tie tolerance, ``|a - b| == tol`` in float64, where a rule written
#: with ``<`` instead of ``<=`` parts from the gate: absolute (0 against 1e-12) and
#: relative (``1e-9 * 1e9`` rounds to exactly 1.0). Then one ulp further apart, which is
#: ordered, and a mix that ties half its pairs. Nothing above lands on the boundary: the
#: ``base * (1 + _TIE_REL_TOL * k)`` cases miss it by rounding.
_ON_THE_TOLERANCE = (
    (np.zeros(24), np.full(24, gate._TIE_ABS_TOL)),
    (np.full(24, 1e9), np.full(24, 1e9 - 1.0)),
    (np.full(24, 1e9), np.full(24, np.nextafter(1e9 - 1.0, 0.0))),
    (np.repeat([1e9, 0.0], 12), np.repeat([1e9 - 1.0, gate._TIE_ABS_TOL], 12)),
)


def test_the_boundary_cases_sit_exactly_on_the_tolerance():
    for (a, b), expected in zip(
        [(0.0, gate._TIE_ABS_TOL), (1e9, 1e9 - 1.0), (1e9, np.nextafter(1e9 - 1.0, 0.0))],
        [0.0, 0.0, 1.0],
    ):
        tol = max(gate._TIE_ABS_TOL, gate._TIE_REL_TOL * max(abs(a), abs(b)))
        assert (abs(a - b) == tol) == (expected == 0.0), (a, b)
        assert _gate_resolution([a], [b]) == expected, (a, b)
    assert _gate_resolution(*_ON_THE_TOLERANCE[-1]) == 0.5


@pytest.mark.parametrize("seed", range(20))
def test_resolution_is_bit_identical_to_the_gate(seed):
    rng = np.random.default_rng(seed)
    for pos, neg in _cases(rng):
        assert not _overflows(pos, neg)
        g, c = _gate_resolution(pos, neg), _cf_resolution(pos, neg)
        assert (g == c) or (g != g and c != c), (g, c)


def test_the_one_known_divergence_is_an_overflowing_difference():
    """The gate ties a finite pair whose difference overflows float64; the tool orders it.

    Found by review. The gate's matrix turns ``1e308 - (-1e308)`` into ``inf`` and its mask
    reads a non-finite difference as a tie. The pair is perfectly ordered, and a standard
    AUROC says so. The gate is left as it is -- no RHOB family produces scores near 1e308 --
    and the divergence is pinned here rather than hidden.
    """
    assert _gate_resolution([1e308], [-1e308]) == 0.0
    assert _cf_resolution([1e308], [-1e308]) == 1.0


def test_the_tie_tolerances_are_the_gates():
    assert cannotfail.resolution.__module__  # imported, not shadowed
    from cannotfail.resolution import EQUIVALENCE_MARGIN

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
    assert len(cells) == 123
    for cell in cells:
        m = cell["metrics"]["proxy_matched"]
        r = Resolution(
            n=2, n_distinct=2, n_nonfinite=0, n_pos=1, n_neg=1,
            exact_resolution=None,
            tolerance_resolution=m["statistic_resolution"],
            dispersion=m["proxy_informativeness"],
            margin=m["margin"],
            dispersion_floor=m["informativeness_floor"],
        )
        tool_degenerate = r.verdict in {"fixed", "dust", "pinned"}
        gate_degenerate = cell["outcomes"]["proxy_matched"] == "degenerate"
        assert tool_degenerate == gate_degenerate, (cell["family"], cell["difficulty"], r.verdict)
    assert sum(c["outcomes"]["proxy_matched"] == "degenerate" for c in cells) == 30


#: The two families whose onset label is read off what the trained agent does after an
#: activation episode the config sets. Every other family's onset label is the configured
#: schedule, so its ``onset_std`` is exactly 0.0.
_AGENT_ONSET_FAMILIES = {"continuous_camping", "gridworld_camping"}


def test_onset_localizable_on_the_shipped_design():
    """The instance the audit package used to miss, and what the full ledger shows.

    The 10-family ledger had ``onset_std`` 0.0 in all 50 cells, and the tool called it
    fixed: one value across the whole design. The shipped design is now the full ledger
    (2026-11-19, 123 cells), and there the statistic takes six distinct values, so the
    tool must not call it fixed: it is 0.0 in the 118 cells of the 31 families whose
    onset label is the configured schedule, and varies only in the 5 cells of the two
    families whose onset is read off the trained agent's behaviour. The tool must still
    call the statistic fixed on those 118 cells alone, which is the case it was written
    to catch. ``onset_localizable`` passes in every cell either way.
    """
    cells = _ledger_cells()
    assert len(cells) == 123
    assert all(c["outcomes"]["onset_localizable"] == "pass" for c in cells)

    r_all = cannotfail.resolution([c["metrics"]["onset_localizable"]["onset_std"] for c in cells])
    assert r_all.n_distinct == 6
    assert r_all.verdict == "ok"
    varying = {c["family"] for c in cells if c["metrics"]["onset_localizable"]["onset_std"] != 0.0}
    assert varying == _AGENT_ONSET_FAMILIES

    scheduled = [c for c in cells if c["family"] not in _AGENT_ONSET_FAMILIES]
    assert len(scheduled) == 118
    assert len({c["family"] for c in scheduled}) == 31
    r = cannotfail.resolution([c["metrics"]["onset_localizable"]["onset_std"] for c in scheduled])
    assert r.n_distinct == 1
    assert r.verdict == "fixed"


def _verdict_from(resolution, informativeness=1.0, margin=0.10, floor=1e-4):
    return Resolution(
        n=2, n_distinct=2, n_nonfinite=0, n_pos=1, n_neg=1,
        exact_resolution=None, tolerance_resolution=resolution,
        dispersion=informativeness, margin=margin, dispersion_floor=floor,
    ).verdict


@pytest.mark.parametrize("margin", [0.05, 0.10, 0.15])
def test_the_resolution_cut_is_the_gates(margin):
    """Either side of r/2 == margin, the tool and gate.is_degenerate must agree."""
    cut = 2.0 * margin
    for r in (np.nextafter(cut, 0.0), cut, np.nextafter(cut, 1.0), cut - 1e-9, cut + 1e-9):
        tool = _verdict_from(float(r), margin=margin) in {"fixed", "pinned"}
        assert tool == gate.is_degenerate(float(r), margin), r


def test_the_informativeness_cut_is_the_gates():
    floor = gate.PROXY_INFORMATIVENESS_FLOOR
    for i in (np.nextafter(floor, 0.0), floor, np.nextafter(floor, 1.0)):
        tool = _verdict_from(1.0, informativeness=float(i), floor=floor) == "dust"
        assert tool == gate.is_uninformative(float(i), floor), i


# --- The DEGENERATE cells, recomputed from the score vectors the gate measured ---------

_LEDGER_JSON: dict[str, Any] = json.loads(LEDGER.read_text(encoding="utf-8"))

#: The six families the ledger holds out of the L0 control.
_DEGENERATE_FAMILIES: list[str] = _LEDGER_JSON["summary"]["degenerate_families"]

#: The one the fast suite re-certifies; the other five run under ``-m slow`` (about 43 s
#: on a 4-core Xeon, 15 of it ``physics_exploitation``). This one takes about 4 s for its
#: 5 tiers. Its proxy is constant to within one ULP (dispersion 1.6e-16 to 2.4e-16,
#: depending on the CPU), so its cells are degenerate by informativeness. On some CPUs its
#: scores also carry rounding dust. On a Xeon here, one or two of each tier's 48 vectors
#: order most of their pairs (at most, all of them) by exact comparison and none by the
#: tie tolerance, and the tool calls those layouts dust or pinned rather than fixed. On
#: the CPU that certified the 0.98 tier they were exact ties (AUROC 0.5000, bootstrap SE
#: 0.0). That CPU dependence is why ``tests/test_v3/test_ledger_reproduces.py`` leaves
#: this family out. Nothing here reads an AUROC.
_FAST_DEGENERATE_FAMILY = "rlhf_reward_model_overopt"

_DEGENERATE_VERDICTS = frozenset({"fixed", "dust", "pinned"})


@contextmanager
def _capturing_the_gate() -> Iterator[list[tuple]]:
    """Record, in call order, every score vector the gate resolves and every proxy stream
    whose informativeness it measures, without changing what either returns."""
    events: list[tuple] = []
    score_resolution, proxy_informativeness = gate._score_resolution, gate.proxy_informativeness

    def resolution_spy(labels, scores):
        value = score_resolution(labels, scores)
        events.append(("resolution", np.asarray(labels, int), np.asarray(scores, float), value))
        return value

    def informativeness_spy(runs):
        value = proxy_informativeness(runs)
        streams = [np.asarray(getattr(r, "proxy_rewards", ()), float).ravel() for r in runs]
        events.append(("informativeness", np.concatenate(streams) if streams else np.empty(0),
                       value))
        return value

    with mock.patch.object(gate, "_score_resolution", resolution_spy), mock.patch.object(
        gate, "proxy_informativeness", informativeness_spy
    ):
        yield events


def _same(a: float, b: float) -> bool:
    """Bit-equal, or both NaN."""
    return a == b or (a != a and b != b)


def _mean_over_layouts(values: list[float]) -> float:
    """The gate's own reduction (``admission_gate._mean_over_layouts``): NaNs ignored."""
    vals = [v for v in values if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def _key(detector_name: str) -> str:
    return detector_name.lower().replace(" ", "_")


def _check_degenerate_cell_from_its_vectors(cell: dict, row: dict, events: list[tuple]) -> None:
    where = (cell["family"], cell["difficulty"])
    design = cell["design"]
    pm = cell["metrics"]["proxy_matched"]
    pdm = cell["metrics"]["proxy_distribution_matched"]
    margin, floor = pm["margin"], pm["informativeness_floor"]

    # The gate, re-run, still says what the ledger says.
    assert row["status"] == cell["status"] == "DEGENERATE", where
    assert row["outcomes"] == cell["outcomes"], where
    assert row["degenerate_criteria"] == cell["degenerate_criteria"], where

    # Which statistic each call belongs to. Per layout, certify() resolves the L0 mean
    # statistic, measures the proxy stream, then resolves each shape detector whose
    # windows fit the horizon, in PROXY_SHAPE_DETECTORS order.
    shapes = [
        name for name in design["shape_detectors"]
        if f"{_key(name)}_not_applicable_horizon" not in pdm
    ]
    per_layout = ["resolution", "informativeness"] + ["resolution"] * len(shapes)
    assert [e[0] for e in events] == per_layout * design["n_layouts"], where
    names = ["proxy_matched", None, *shapes]

    vectors: dict[str, list[tuple[np.ndarray, np.ndarray, Resolution]]] = {
        n: [] for n in names if n is not None
    }
    dispersions: list[float] = []
    for i in range(design["n_layouts"]):
        layout = events[i * len(per_layout):(i + 1) * len(per_layout)]
        for name, event in zip(names, layout):
            if name is None:
                _, stream, measured = event
                dispersion = cannotfail.dispersion(stream)
                assert _same(dispersion, measured), (where, i, dispersion, measured)
                dispersions.append(dispersion)
                continue
            _, labels, scores, measured = event
            assert not _overflows(scores[labels == 1], scores[labels == 0]), (where, name)
            r = cannotfail.resolution(scores, labels)
            assert _same(r.tolerance_resolution, measured), (where, name, i, r, measured)
            vectors[name].append((labels, scores, r))

    # The tool's figures, reduced over layouts as the gate reduces them, are the ledger's.
    informativeness = _mean_over_layouts(dispersions)
    assert _same(informativeness, row["metrics"]["proxy_matched"]["proxy_informativeness"])
    assert informativeness == pytest.approx(pm["proxy_informativeness"], rel=1e-6, abs=1e-12)
    uninformative = gate.is_uninformative(pm["proxy_informativeness"], floor)
    assert gate.is_uninformative(informativeness, floor) == uninformative, where

    committed = {"proxy_matched": pm["statistic_resolution"]}
    committed.update({name: pdm[f"{_key(name)}_resolution"] for name in shapes})
    for name, layouts in vectors.items():
        tolerance = _mean_over_layouts([r.tolerance_resolution for *_, r in layouts])
        exact = _mean_over_layouts([r.exact_resolution for *_, r in layouts])
        assert tolerance == pytest.approx(committed[name], abs=1e-12), (where, name)

        # And from those figures alone, the tool calls the statistic degenerate. The
        # counts come from the cell's pooled vector, the resolutions from the per-layout
        # comparisons the gate makes, and the dispersion from the proxy stream.
        pooled = cannotfail.resolution(
            np.concatenate([s for _, s, _ in layouts]),
            np.concatenate([lab for lab, _, _ in layouts]),
        )
        verdict = dataclasses.replace(
            pooled,
            exact_resolution=exact,
            tolerance_resolution=tolerance,
            dispersion=informativeness,
            margin=margin,
            dispersion_floor=floor,
        )
        assert verdict.verdict in _DEGENERATE_VERDICTS, (where, name, verdict.as_dict())
        if not uninformative:
            # Degenerate by resolution, so the scores alone have to show it.
            no_floor = dataclasses.replace(verdict, dispersion_floor=None)
            assert no_floor.verdict in _DEGENERATE_VERDICTS, (where, name, no_floor.as_dict())
        # Every layout on its own is degenerate too: no layout carries the cell.
        for _, _, r in layouts:
            assert r.verdict in _DEGENERATE_VERDICTS, (where, name, r.as_dict())


def test_the_degenerate_families_are_the_ledgers_six():
    cells = [c for c in _ledger_cells() if c["status"] == "DEGENERATE"]
    assert sorted({c["family"] for c in cells}) == _DEGENERATE_FAMILIES
    assert len(_DEGENERATE_FAMILIES) == 6 and len(cells) == 30
    assert _FAST_DEGENERATE_FAMILY in _DEGENERATE_FAMILIES
    # Both reasons a cell can be degenerate occur among the six, each family is one kind
    # at every tier, and the fast family is the informativeness kind.
    kinds = {
        family: {
            gate.is_uninformative(
                c["metrics"]["proxy_matched"]["proxy_informativeness"],
                c["metrics"]["proxy_matched"]["informativeness_floor"],
            )
            for c in cells
            if c["family"] == family
        }
        for family in _DEGENERATE_FAMILIES
    }
    assert all(len(k) == 1 for k in kinds.values()), kinds
    assert set().union(*kinds.values()) == {True, False}
    assert kinds[_FAST_DEGENERATE_FAMILY] == {True}


@pytest.mark.parametrize(
    "family",
    [
        pytest.param(f, id=f, marks=() if f == _FAST_DEGENERATE_FAMILY else pytest.mark.slow)
        for f in _DEGENERATE_FAMILIES
    ],
)
def test_degenerate_cells_recomputed_from_the_gates_score_vectors(family):
    """The tool, on the vectors the gate actually scored, reproduces the ledger.

    For every tier of ``family``: each per-layout resolution and dispersion the tool
    computes is bit-identical to the gate's; their means over layouts are the ledger's
    ``statistic_resolution``, per-detector ``*_resolution`` and ``proxy_informativeness``;
    and the tool calls the L0 mean statistic and every shape detector degenerate. When the
    gate's reason is resolution rather than informativeness, it does so from the scores
    alone, with no dispersion floor.
    """
    certifier = gate.AdmissionGate(**_LEDGER_JSON["provenance"]["gate"])
    fam = FamilyRegistry.get(family)
    cells = [c for c in _ledger_cells() if c["family"] == family]
    assert [c["difficulty"] for c in cells] == list(fam.default_difficulties())
    for cell in cells:
        with _capturing_the_gate() as events:
            row = certifier.certify(fam, difficulty=cell["difficulty"]).to_dict()
        _check_degenerate_cell_from_its_vectors(cell, row, events)


# --- The third copy: inspect_eval/hvtb_hack_detection/metrics.py, read with ast ----------


def _inspect_metrics_tree() -> ast.Module:
    return ast.parse(INSPECT_METRICS.read_text(encoding="utf-8"), filename=str(INSPECT_METRICS))


def _literal_assignments(tree: ast.Module) -> dict[str, Any]:
    """Every module-level ``NAME = <literal>`` (annotated or not), evaluated without running
    the module."""
    out: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if isinstance(target, ast.Name):
            try:
                out[target.id] = ast.literal_eval(value)
            except ValueError:
                continue
    return out


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    return fn


def _inspect_score_resolution():
    """``score_resolution``'s compute function, compiled from the file without importing it.

    The module imports ``inspect_ai`` at the top, which this suite does not install. So
    this keeps only what the metric runs on: the ``from __future__`` import, the literal
    constants, ``_labels_and_scores`` and ``score_resolution``. It swaps Inspect's
    ``@metric`` registration for the identity. What runs is the file's own tie rule and
    its own sample filtering, unmodified.
    """
    tree = _inspect_metrics_tree()
    keep = [
        n for n in tree.body
        if (isinstance(n, ast.ImportFrom) and n.module == "__future__")
        or (isinstance(n, ast.FunctionDef) and n.name in {"_labels_and_scores", "score_resolution"})
    ]
    literals = _literal_assignments(tree)
    namespace: dict[str, Any] = {"np": np, "metric": lambda fn: fn, **literals}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(INSPECT_METRICS), "exec"), namespace)
    return namespace["score_resolution"]()


def _samples(pos, neg) -> list[SimpleNamespace]:
    """Replay samples in the shape ``_labels_and_scores`` reads: ``score.value`` and the
    watcher label in ``score.metadata["hacked"]``."""
    return [
        SimpleNamespace(score=SimpleNamespace(value=float(v), metadata={"hacked": label}))
        for values, label in ((pos, True), (neg, False))
        for v in values
    ]


@pytest.fixture(scope="module")
def inspect_score_resolution():
    return _inspect_score_resolution()


def test_the_inspect_metric_tie_tolerances_are_the_gates():
    literals = _literal_assignments(_inspect_metrics_tree())
    assert literals["_TIE_REL_TOL"] == gate._TIE_REL_TOL == REL_TOL
    assert literals["_TIE_ABS_TOL"] == gate._TIE_ABS_TOL == ABS_TOL


def test_the_inspect_metric_tie_rule_reads_those_constants():
    """Equal constants mean nothing if the rule stops reading them.

    So the function that computes the resolution must name both, and must contain no
    tolerance-sized literal of its own that could have replaced one.
    """
    fn = _function(_inspect_metrics_tree(), "score_resolution")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert {"_TIE_REL_TOL", "_TIE_ABS_TOL"} <= names
    inlined = [
        n.value for n in ast.walk(fn)
        if isinstance(n, ast.Constant) and isinstance(n.value, float) and 0 < abs(n.value) < 1e-3
    ]
    assert not inlined, inlined


@pytest.mark.parametrize("seed", range(5))
def test_the_inspect_metric_resolution_is_bit_identical_to_the_gate(
    inspect_score_resolution, seed
):
    """On finite scores, the same adversarial cases the tool is held to above.

    The metric drops a sample whose score is not finite before it pairs anything (next
    test), so the gate is given the same finite scores.
    """
    rng = np.random.default_rng(seed)
    for pos, neg in _cases(rng):
        pos, neg = np.asarray(pos, float), np.asarray(neg, float)
        fpos, fneg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
        got = inspect_score_resolution(_samples(pos, neg))
        assert _same(got, _gate_resolution(fpos, fneg)), (got, _gate_resolution(fpos, fneg))


def test_the_inspect_metric_ties_an_overflowing_difference_like_the_gate(
    inspect_score_resolution,
):
    """The divergence pinned above for ``cannotfail`` is not one for the metric."""
    with np.errstate(over="ignore"):
        assert inspect_score_resolution(_samples([1e308], [-1e308])) == 0.0
    assert _gate_resolution([1e308], [-1e308]) == 0.0
    assert _cf_resolution([1e308], [-1e308]) == 1.0


def test_the_inspect_metric_drops_a_nonfinite_score_rather_than_tying_it(
    inspect_score_resolution,
):
    """Where the metric and the gate part ways, pinned so it is not mistaken for parity.

    ``_labels_and_scores`` discards a sample whose score is NaN or infinite, so the sample
    leaves the denominator. The gate keeps it and counts each of its pairs as tied.
    ``score_resolution``'s docstring says non-finite scores count as ties, but in the
    metric none ever reaches the tie rule. Two hacking runs and two clean runs, cleanly
    ordered, plus a NaN-scored hacking run: the metric orders 4 of 4 pairs, the gate 4 of 6.
    """
    pos, neg = [1.0, 2.0, float("nan")], [0.0, 0.5]
    assert inspect_score_resolution(_samples(pos, neg)) == 1.0
    assert _gate_resolution(pos, neg) == pytest.approx(4 / 6)
