"""Four registered families, re-certified at the ledger's own design, reproduce the ledger.

``admission/admission_ledger.json`` is the certification behind RHOB's L0 negative
control, and the rest of the suite only reads it: it checks the file against itself
(``test_admission_ledger_artifact.py``), against the docs
(``test_ledger_doc_consistency.py``) and against ``cannotfail``
(``tests/test_cannotfail_parity.py``). Nothing re-ran the gate to see whether the
committed numbers are still what the code produces. A change to a detector, to a family
or to the gate could leave the ledger certifying code that no longer exists.

So this re-certifies four families, every tier the ledger scores, with an
``AdmissionGate`` built from the ledger's own ``provenance.gate`` block (12 layouts x 24
seeds per side, 2000 bootstrap resamples), and compares each certificate with its
committed cell. Together the four cover every status and every outcome the ledger
records (see :data:`FAMILIES`). They are the cheapest families that do. None calibrates
a proxy, and 20 cells take about 17 s serially, both on the CI runners
(``timing_seconds``) and on a 4-core Xeon, so this runs in the default suite.

What has to match, and how closely
----------------------------------
Statuses, outcomes, ``criteria``, ``degenerate_criteria`` and the design must match
exactly, and so must the ``details`` text once its numbers are masked out, which pins
the branch the gate took and the binding detector. Every number in ``metrics`` must
match within one of two tolerances. The tolerances come from measurements, not from
guesses. The ledger was certified on the CI lock (Python 3.12, numpy 2.5.4, scipy 1.18.1,
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
the second exactly for that reason.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

import rhob.v3.families  # noqa: F401  (registration side effects)
from rhob.v3.admission_gate import CRITERIA, AdmissionGate, is_uninformative
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


def _cells(family: str) -> list[dict[str, Any]]:
    return [c for c in _LEDGER["results"] if c["family"] == family]


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
    return _NUMBER.sub("#", text)


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
        if _masked(row["details"][criterion]) != _masked(cell["details"][criterion]):
            out.append(
                f"{criterion} details:\n    got    {row['details'][criterion]}\n"
                f"    ledger {cell['details'][criterion]}"
            )
    return out


_CELL_PARAMS = [
    pytest.param(cell["family"], cell["difficulty"], id=f"{cell['family']}@{cell['difficulty']}")
    for family in FAMILIES
    for cell in _cells(family)
]


def test_the_four_families_cover_every_status_and_outcome():
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
    assert len(_CELL_PARAMS) == len(cells) == 20


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

    The closest any of the 20 cells sits to a cut is reward_channel_tampering at 0.98,
    whose Reward Skewness interval leaves the band by 0.0045. That is about RANK_ATOL,
    but it is pinned by the outcome, which is compared exactly; the tolerance applies
    only to the number.
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


def test_details_masking_keeps_the_words():
    detail = "binding: Reward KDE AUROC 0.5301, CI [0.4867, 0.5735]; 1.6e-16 below 1e-04"
    assert _masked(detail) == "binding: Reward KDE AUROC #, CI [#, #]; # below #"


@pytest.fixture(scope="module")
def gate() -> AdmissionGate:
    return _ledger_gate()


@pytest.mark.parametrize(("family", "difficulty"), _CELL_PARAMS)
def test_recertifying_reproduces_the_committed_cell(gate, family, difficulty):
    (cell,) = [c for c in _cells(family) if c["difficulty"] == difficulty]
    assert cell["error"] is None
    row = gate.certify(FamilyRegistry.get(family), difficulty=difficulty).to_dict()
    problems = _mismatches(row, cell)
    assert not problems, (
        f"{family} @ {difficulty} no longer reproduces admission/admission_ledger.json "
        f"({FAMILIES[family]}):\n  " + "\n  ".join(problems)
        + "\nIf the change is intended, the ledger has to be re-certified "
        "(regenerate.yml, artifacts: ledger), not this test loosened."
    )
