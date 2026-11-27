"""Four registered families, re-certified at the ledger's own design, reproduce the ledger.

``admission/admission_ledger.json`` is the certification behind RHOB's L0 negative
control. Most of the suite only reads it: it checks the file against itself
(``test_admission_ledger_artifact.py``) and against the docs
(``test_ledger_doc_consistency.py``). ``tests/test_cannotfail_parity.py`` re-certifies the
six DEGENERATE families (two on every push, four nightly) and holds ``cannotfail`` to the
ledger's resolutions, informativeness, statuses and outcomes for them. This file covers
what that leaves: ADMITTED and NOT ADMITTED cells, and every number in a certificate.
Without it, a change to a detector, to a family or to the gate could leave the ledger
certifying code that no longer exists.

So this re-certifies four families, every tier the ledger scores, with an
``AdmissionGate`` built from the ledger's own ``provenance.gate`` block (12 layouts x 24
seeds per side, 2000 bootstrap resamples), and compares each certificate with its
committed cell. Together the four cover every status and every outcome value the ledger
records, with measured FAILs on two criteria, ``proxy_distribution_matched`` and
``behavioral_separated`` (see :data:`FAMILIES`). They are the cheapest families that do.
None calibrates a proxy, and 20 cells take about 17 s serially, both on the CI runners
(``timing_seconds``) and on a 4-core Xeon, so this runs in the default suite.

One (criterion, outcome) pair the ledger records is out of the four's reach:
``proxy_matched``'s FAIL, the L0 matched-proxy criterion's own failure branch (24 cells).
The cheapest family that has one, ``rlhf_feature_blindspot_gaming`` (FAIL at 0.80 only),
takes 325 s for its five tiers on the CI runners, too slow for every push. So
:data:`SLOW_CELLS` re-certifies that one cell under ``-m slow`` and the nightly
``admission-slow`` job covers the pair (about 220 s for the one tier, single-threaded, on
a 4-core Xeon).

What has to match, and how closely
----------------------------------
Statuses, outcomes, ``criteria``, ``degenerate_criteria`` and the design must match
exactly, and so must the ``details`` text once its numbers and the binding detector's
name are masked out, which pins the branch the gate took. The binding detector is the
shape detector whose interval reaches furthest outside the band, which makes it a
rank-derived choice like the numbers it is chosen from: it has to match exactly only
where the ledger's choice is clear by more than ``2 * RANK_ATOL``, and elsewhere may be
any detector within that of the worst gap (:func:`_admissible_bindings`). Every number in
``metrics`` must match within one of two tolerances.

The tolerances come from measurements, not from guesses. The ledger was certified on the CI lock (Python 3.12, numpy 2.5.4, scipy 1.18.1,
scikit-learn 1.9.1) on AMD EPYC and Intel Xeon runners. The fast suite runs unpinned on
Python 3.10, 3.11 and 3.12 (``.github/workflows/tests.yml``). Re-certifying these four
families on 2026-11-27 gave:

- CI lock on another Intel Xeon: every cell reproduces bit for bit.
- Python 3.11, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1: bit for bit except
  ``proxy_correlation_gaming``'s Reward Skewness, whose AUROC moved by 3.6e-4 and
  whose CI bound moved by 7.6e-4.
- Python 3.10, numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2: the same Skewness drift,
  plus about 1e-14 on the other continuous values.

The Skewness drift is a few near-tied scores ranked the other way under scipy < 1.18. A
rank statistic cannot move by a fraction of a pair: one cross-variant pair flipped in
one layout moves that layout's AUROC by 1/576 and the 12-layout mean by 1.4e-4. Every
rank-derived number therefore gets :data:`RANK_ATOL`, about 6x the largest drift
measured and 20x smaller than the 0.10 margin the outcomes are read against. Every other
number is continuous arithmetic and gets :data:`CONTINUOUS_RTOL`. That is rounding, not a
different computation.

Why not the cheaper-looking alternatives: ``eval_probe_sandbagging`` (4 s) moves its
Reward KDE AUROC by 3.6e-4 between two CPU models even on the lock, and
``rlhf_reward_model_overopt`` (4 s) has a proxy that is constant to within one ULP, so
its L0 AUROC is rounding error that moves between CPUs (0.5000 committed, 0.4931 here).
Both would be numerics tests, not ledger tests. ``tests/test_cannotfail_parity.py`` uses
the second exactly for that reason: nothing there reads an AUROC.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

import rhob.v3.families  # noqa: F401  (registration side effects)
from rhob.v3.admission_gate import CRITERIA, AdmissionGate, is_degenerate, is_uninformative
from rhob.v3.registry import FamilyRegistry

_LEDGER_PATH = Path(__file__).resolve().parents[2] / "admission" / "admission_ledger.json"
_LEDGER: dict[str, Any] = json.loads(_LEDGER_PATH.read_text(encoding="utf-8"))

#: The re-certified families, and what each one is here to cover.
FAMILIES: dict[str, str] = {
    "reward_channel_tampering": (
        "ADMITTED at 0.80, and NOT ADMITTED at its other four tiers on a measured "
        "proxy_distribution_matched FAIL: Reward KDE outside the band at 0.60, 0.70 and "
        "0.90, and Reward Skewness's interval clipping it by 0.0045 at 0.98"
    ),
    "proxy_correlation_gaming": (
        "ADMITTED at four tiers, and NOT ADMITTED at 0.60 on behavioral_separated "
        "(mean L2 AUROC 0.527 against the 0.60 floor), the ledger's only FAIL of that "
        "criterion"
    ),
    "shortcut_exploitation": (
        "DEGENERATE by resolution: a richly informative proxy (dispersion 3.0 to 4.4) "
        "that is the same in every run, so every cross-variant pair ties"
    ),
    "distributional_shift": (
        "DEGENERATE by informativeness: a proxy that is one number everywhere "
        "(dispersion exactly 0.0)"
    ),
}

#: Cells re-certified only under ``-m slow`` (the nightly ``admission-slow`` job), and why.
SLOW_CELLS: dict[tuple[str, float], str] = {
    ("rlhf_feature_blindspot_gaming", 0.8): (
        "NOT ADMITTED on a measured proxy_matched FAIL, the one (criterion, outcome) pair "
        "the ledger records that FAMILIES does not reach; also the only re-certified cell "
        "with a shape detector its horizon rules out (Reward Skewness at 60 episodes)"
    ),
}

#: Absolute tolerance on rank-derived numbers: AUROCs, their bootstrap SE and interval,
#: resolutions and the worst detector's gap to the band. See the module docstring.
RANK_ATOL = 5e-3
#: Relative tolerance on every other number in a certificate.
CONTINUOUS_RTOL = 1e-9
#: Absolute floor under CONTINUOUS_RTOL, for values that are exactly or nearly zero.
CONTINUOUS_ATOL = 1e-12

#: The criteria whose numbers are built from AUROCs, and the metric names among them
#: that are rank-derived. Their thresholds, counts and proxy informativeness are not.
_RANK_CRITERIA = frozenset({"proxy_matched", "proxy_distribution_matched", "behavioral_separated"})
_RANK_KEY = re.compile(r"(auroc|ci_lo|ci_hi|bootstrap_se|resolution|worst_detector_gap)$")

#: A number in a ``details`` string, masked before the strings are compared.
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
#: The binding detector in a ``proxy_distribution_matched`` detail, which the gate writes
#: as ``binding: <name> AUROC <mean>, CI [<lo>, <hi>]``. Also masked: see
#: :func:`_admissible_bindings` for how its name is checked instead.
_BINDING = re.compile(r"binding: (?P<name>[A-Za-z][A-Za-z ]*?) AUROC ")
_PDM = "proxy_distribution_matched"


def _cells(family: str) -> list[dict[str, Any]]:
    return [c for c in _LEDGER["results"] if c["family"] == family]


def _cell(family: str, difficulty: float) -> dict[str, Any]:
    (cell,) = [c for c in _cells(family) if c["difficulty"] == difficulty]
    return cell


def _key(detector_name: str) -> str:
    return detector_name.lower().replace(" ", "_")


def _binding(text: str) -> str | None:
    """The detector a detail names as binding, or None when it names none."""
    names = _BINDING.findall(text)
    assert len(names) <= 1, text
    return names[0] if names else None


def _shape_gaps(cert: dict[str, Any]) -> dict[str, float]:
    """How far each shape detector's interval reaches outside the band, for every detector
    eligible to bind.

    The gate's own arithmetic (``admission_gate._check_proxy_distribution_matched``) on
    the bounds it published. A detector the horizon rules out has no interval, and one
    that cannot resolve the proxy is not eligible, as in the gate. ``cert`` is a ledger
    cell or a re-certified row.
    """
    metrics = cert["metrics"][_PDM]
    margin = metrics["margin"]
    gaps: dict[str, float] = {}
    for name in cert["design"]["shape_detectors"]:
        key = _key(name)
        if f"{key}_ci_lo" not in metrics or is_degenerate(metrics[f"{key}_resolution"], margin):
            continue
        lo, hi = metrics[f"{key}_ci_lo"], metrics[f"{key}_ci_hi"]
        gaps[name] = max(0.5 - margin - lo, hi - (0.5 + margin))
    return gaps


def _admissible_bindings(cell: dict[str, Any]) -> set[str]:
    """The detectors a re-certification within tolerance of ``cell`` could name as binding.

    A gap is the larger of two interval bounds' distances to the band, so it moves by at
    most RANK_ATOL when the bounds do. The gate names the largest gap. So a detector whose
    committed gap is within ``2 * RANK_ATOL`` of the committed largest can overtake it
    with every number still inside its tolerance, and one further behind cannot. Where no
    other detector is that close, this is the committed binding alone.
    """
    gaps = _shape_gaps(cell)
    if not gaps:
        return set()
    worst = max(gaps.values())
    return {name for name, gap in gaps.items() if gap >= worst - 2 * RANK_ATOL}


def _ledger_gate() -> AdmissionGate:
    """The gate exactly as the ledger records it was configured."""
    return AdmissionGate(**_LEDGER["provenance"]["gate"])


def _is_rank_metric(criterion: str, key: str) -> bool:
    return criterion in _RANK_CRITERIA and _RANK_KEY.search(key) is not None


def _close(got: Any, want: Any, *, rank: bool) -> bool:
    if isinstance(want, bool) or not isinstance(want, (int, float)):
        return got == want
    if isinstance(got, bool) or not isinstance(got, (int, float)):
        return False
    got, want = float(got), float(want)
    if math.isnan(want) or math.isnan(got):
        return math.isnan(want) and math.isnan(got)
    if rank:
        return abs(got - want) <= RANK_ATOL
    return math.isclose(got, want, rel_tol=CONTINUOUS_RTOL, abs_tol=CONTINUOUS_ATOL)


def _masked(text: str) -> str:
    """A detail with its numbers and its binding detector's name taken out."""
    return _NUMBER.sub("#", _BINDING.sub("binding: <detector> AUROC ", text))


def _mismatches(row: dict[str, Any], cell: dict[str, Any]) -> list[str]:
    """Every way a re-certified row differs from its committed cell, beyond tolerance."""
    out: list[str] = []
    for key in ("family", "passed", "status", "criteria", "outcomes", "degenerate_criteria"):
        if row[key] != cell[key]:
            out.append(f"{key}: got {row[key]!r}, ledger {cell[key]!r}")
    if not math.isclose(row["difficulty"], cell["difficulty"], abs_tol=1e-9):
        out.append(f"difficulty: got {row['difficulty']!r}, ledger {cell['difficulty']!r}")
    if set(row["design"]) != set(cell["design"]):
        out.append(f"design keys: {sorted(set(row['design']) ^ set(cell['design']))}")
    for key in sorted(set(row["design"]) & set(cell["design"])):
        if not _close(row["design"][key], cell["design"][key], rank=False):
            out.append(f"design.{key}: got {row['design'][key]!r}, ledger {cell['design'][key]!r}")
    for criterion in CRITERIA:
        got, want = row["metrics"][criterion], cell["metrics"][criterion]
        if set(got) != set(want):
            out.append(f"{criterion} metric keys: {sorted(set(got) ^ set(want))}")
        for key in sorted(set(got) & set(want)):
            if not _close(got[key], want[key], rank=_is_rank_metric(criterion, key)):
                out.append(f"{criterion}.{key}: got {got[key]!r}, ledger {want[key]!r}")
        got_text, want_text = row["details"][criterion], cell["details"][criterion]
        if _masked(got_text) != _masked(want_text):
            out.append(f"{criterion} details:\n    got    {got_text}\n    ledger {want_text}")
            continue
        # The masked texts agree, so both name a binding detector or neither does. Only
        # proxy_distribution_matched names one (checked on the whole ledger below).
        got_binding, want_binding = _binding(got_text), _binding(want_text)
        if criterion != _PDM or got_binding is None:
            continue
        allowed = _admissible_bindings(cell)
        if got_binding not in allowed:
            out.append(
                f"{criterion} binding: got {got_binding!r}, ledger {want_binding!r}; the "
                f"detectors within 2 * RANK_ATOL of the ledger's worst gap are {sorted(allowed)}"
            )
        elif _shape_gaps(row).get(got_binding) != row["metrics"][criterion].get(
            "worst_detector_gap"
        ):
            out.append(
                f"{criterion} binding: {got_binding!r} is named, but its gap is not the "
                "re-certified row's own worst_detector_gap"
            )
    return out


_FAST_CELL_PARAMS = [
    pytest.param(cell["family"], cell["difficulty"], id=f"{cell['family']}@{cell['difficulty']}")
    for family in FAMILIES
    for cell in _cells(family)
]
_CELL_PARAMS = _FAST_CELL_PARAMS + [
    pytest.param(family, difficulty, id=f"{family}@{difficulty}", marks=pytest.mark.slow)
    for family, difficulty in SLOW_CELLS
]


def test_the_four_families_cover_every_status_and_outcome_value():
    """The point of choosing these four, checked against the ledger rather than asserted."""
    cells = [c for family in FAMILIES for c in _cells(family)]
    every_status = {c["status"] for c in _LEDGER["results"]}
    every_outcome = {o for c in _LEDGER["results"] for o in c["outcomes"].values()}
    assert every_status == {"ADMITTED", "NOT ADMITTED", "DEGENERATE"}
    assert {c["status"] for c in cells} == every_status
    assert {o for c in cells for o in c["outcomes"].values()} == every_outcome

    # A measured FAIL on more than one criterion, so a regression that fails everything
    # the same way cannot pass for a reproduction.
    failed = {k for c in cells for k, v in c["outcomes"].items() if v == "fail"}
    assert failed == {"proxy_distribution_matched", "behavioral_separated"}

    # Every (criterion, outcome) pair the ledger records but one, proxy_matched's FAIL,
    # which is too slow to re-certify on every push. SLOW_CELLS covers it nightly.
    def pairs(cs):
        return {(k, v) for c in cs for k, v in c["outcomes"].items()}

    assert pairs(_LEDGER["results"]) - pairs(cells) == {("proxy_matched", "fail")}
    slow = [_cell(family, difficulty) for family, difficulty in SLOW_CELLS]
    assert pairs(_LEDGER["results"]) == pairs(cells) | pairs(slow)

    # Both reasons a cell can be DEGENERATE: no signal in the proxy, or a signal that
    # never differs between runs.
    degenerate = [c for c in cells if c["status"] == "DEGENERATE"]
    reasons = {
        is_uninformative(
            c["metrics"]["proxy_matched"]["proxy_informativeness"],
            c["metrics"]["proxy_matched"]["informativeness_floor"],
        )
        for c in degenerate
    }
    assert reasons == {True, False}

    # Every tier the ledger scores for each family, and nothing else.
    for family in FAMILIES:
        tiers = [c["difficulty"] for c in _cells(family)]
        assert tiers == list(FamilyRegistry.get(family).default_difficulties()), family
    assert len(_FAST_CELL_PARAMS) == len(cells) == 20
    assert not set(SLOW_CELLS) & {(c["family"], c["difficulty"]) for c in cells}


def test_the_four_families_stay_cheap_enough_for_the_default_suite():
    """A replacement family that calibrates would quietly put minutes on every push."""
    seconds = {family: _LEDGER["timing_seconds"][family] for family in FAMILIES}
    assert sum(seconds.values()) < 20.0, seconds


def test_the_gate_is_built_at_the_ledgers_design():
    gate = _ledger_gate()
    design = _LEDGER["design"]
    assert gate.n_layouts == design["n_layouts"] == 12
    assert gate.min_seeds_per_layout == design["seeds_per_layout"] == 24
    assert gate.n_bootstrap == design["n_bootstrap"]
    assert gate.equivalence_margin == design["equivalence_margin"]
    assert gate.informativeness_floor == design["informativeness_floor"]
    assert gate.alpha == design["alpha"]
    assert gate.behavioral_floor == design["behavioral_floor"]


def test_the_tolerances_are_far_inside_the_decision_cuts():
    """A tolerance as wide as a cut would let a moved verdict hide inside it.

    The closest any of the 20 cells sits to an outcome's cut is reward_channel_tampering
    at 0.98, whose Reward Skewness interval leaves the band by 0.0045. That is about
    RANK_ATOL, but it is pinned by the outcome, which is compared exactly; the tolerance
    applies only to the number.

    Closer still is a choice that is not an outcome: which detector the detail names as
    binding. At 0.80 the same family names Reward KDE over Reward Skewness by 4.4e-4 of
    gap, and proxy_correlation_gaming names Reward Variance Ratio over Reward Skewness by
    3.4e-3 at every tier. Both are inside ``2 * RANK_ATOL``, and the first is smaller than
    the 7.6e-4 Skewness drift measured under scipy < 1.18, so in those six cells the name
    is checked against :func:`_admissible_bindings`, not exactly. At 0.98
    reward_channel_tampering's runner-up is 0.01005 behind, outside ``2 * RANK_ATOL``, so
    that binding is pinned.
    """
    assert RANK_ATOL * 20 <= _LEDGER["design"]["equivalence_margin"]
    assert CONTINUOUS_RTOL <= 1e-6
    # The keys that drifted under scipy < 1.18 are classed as rank-derived.
    for key in ("reward_skewness_auroc", "reward_skewness_ci_lo", "reward_skewness_ci_hi",
                "worst_detector_gap"):
        assert _is_rank_metric("proxy_distribution_matched", key), key
    # ...and the proxy's informativeness and the bootstrap of true rewards are not.
    assert not _is_rank_metric("proxy_matched", "proxy_informativeness")
    assert not _is_rank_metric("true_reward_diverges", "diff_ci_lo")
    # The cells where the binding detector is not pinned exactly are the ones named above.
    loose = {
        (c["family"], c["difficulty"])
        for family in FAMILIES
        for c in _cells(family)
        if len(_admissible_bindings(c)) > 1
    }
    assert loose == {("reward_channel_tampering", 0.8)} | {
        ("proxy_correlation_gaming", c["difficulty"]) for c in _cells("proxy_correlation_gaming")
    }, "update this docstring"


def test_details_masking_keeps_the_words():
    detail = "binding: Reward KDE AUROC 0.5301, CI [0.4867, 0.5735]; 1.6e-16 below 1e-04"
    assert _masked(detail) == "binding: <detector> AUROC #, CI [#, #]; # below #"
    assert _binding(detail) == "Reward KDE"
    assert _binding("binding: Reward Variance Ratio AUROC 0.5, CI [0.4, 0.6]") == (
        "Reward Variance Ratio"
    )
    assert _binding("all within [0.40, 0.60]") is None


def test_the_binding_rule_is_the_gates_on_every_ledger_cell():
    """:func:`_shape_gaps` is the gate's arithmetic, so the relaxed check below rests on it.

    On every one of the ledger's cells that names a binding detector, the largest gap it
    recomputes is the committed ``worst_detector_gap`` bit for bit, and belongs to the
    detector the detail names. Only ``proxy_distribution_matched`` names one.
    """
    named = 0
    for cell in _LEDGER["results"]:
        where = (cell["family"], cell["difficulty"])
        for criterion, text in cell["details"].items():
            assert criterion == _PDM or _binding(text) is None, (where, criterion)
        binding = _binding(cell["details"][_PDM])
        if binding is None:
            continue
        named += 1
        gaps = _shape_gaps(cell)
        worst = cell["metrics"][_PDM]["worst_detector_gap"]
        assert gaps[binding] == max(gaps.values()) == worst, (where, binding, gaps, worst)
        assert binding in _admissible_bindings(cell), where
    assert named > 0


def _rebound(cell: dict[str, Any], name: str, outward: float) -> dict[str, Any]:
    """``cell`` as a re-certification would publish it if ``name``'s interval reached
    ``outward`` further outside the band on its binding side, numbers and detail alike."""
    row = copy.deepcopy(cell)
    metrics = row["metrics"][_PDM]
    margin, key = metrics["margin"], _key(name)
    lo, hi = metrics[f"{key}_ci_lo"], metrics[f"{key}_ci_hi"]
    if 0.5 - margin - lo >= hi - (0.5 + margin):
        metrics[f"{key}_ci_lo"] = lo - outward
    else:
        metrics[f"{key}_ci_hi"] = hi + outward
    gaps = _shape_gaps(row)
    binding = max(gaps, key=gaps.__getitem__)
    metrics["worst_detector_gap"] = gaps[binding]
    row["details"][_PDM] = _BINDING.sub(f"binding: {binding} AUROC ", row["details"][_PDM])
    return row


def test_a_near_tied_binding_detector_may_change_within_tolerance():
    """The review's case: at reward_channel_tampering 0.80, Reward Skewness's gap is 4.4e-4
    behind Reward KDE's, so a drift far inside RANK_ATOL makes the gate name Skewness.
    Every number is within tolerance, so that is a reproduction, not a mismatch."""
    cell = _cell("reward_channel_tampering", 0.8)
    gaps = _shape_gaps(cell)
    assert _binding(cell["details"][_PDM]) == "Reward KDE"
    behind = gaps["Reward KDE"] - gaps["Reward Skewness"]
    assert 0 < behind < RANK_ATOL / 10
    row = _rebound(cell, "Reward Skewness", behind + 1e-4)
    assert _binding(row["details"][_PDM]) == "Reward Skewness"
    assert _mismatches(row, cell) == []


def test_a_binding_detector_outside_the_tolerance_is_a_mismatch():
    """Naming a detector that no in-tolerance drift could make binding is still caught, and
    so is a detail whose named detector is not the row's own largest gap."""
    caught = 0
    for family in FAMILIES:
        for cell in _cells(family):
            binding = _binding(cell["details"][_PDM])
            if binding is None:
                continue
            allowed = _admissible_bindings(cell)
            for name in set(_shape_gaps(cell)) - allowed:
                row = copy.deepcopy(cell)
                row["details"][_PDM] = _BINDING.sub(
                    f"binding: {name} AUROC ", row["details"][_PDM]
                )
                (problem,) = _mismatches(row, cell)
                assert "within 2 * RANK_ATOL" in problem, problem
                caught += 1
            for name in allowed - {binding}:
                row = copy.deepcopy(cell)
                row["details"][_PDM] = _BINDING.sub(
                    f"binding: {name} AUROC ", row["details"][_PDM]
                )
                (problem,) = _mismatches(row, cell)
                assert "own worst_detector_gap" in problem, problem
                caught += 1
    assert caught > 0


@pytest.fixture(scope="module")
def gate() -> AdmissionGate:
    return _ledger_gate()


@pytest.mark.parametrize(("family", "difficulty"), _CELL_PARAMS)
def test_recertifying_reproduces_the_committed_cell(gate, family, difficulty):
    cell = _cell(family, difficulty)
    assert cell["error"] is None
    row = gate.certify(FamilyRegistry.get(family), difficulty=difficulty).to_dict()
    problems = _mismatches(row, cell)
    why = FAMILIES.get(family) or SLOW_CELLS[(family, difficulty)]
    assert not problems, (
        f"{family} @ {difficulty} no longer reproduces admission/admission_ledger.json "
        f"({why}):\n  " + "\n  ".join(problems)
        + "\nIf the change is intended, the ledger has to be re-certified "
        "(regenerate.yml, artifacts: ledger), not this test loosened."
    )
