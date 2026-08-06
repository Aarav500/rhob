"""Tests for the AdmissionGate module itself.

The gate's diagnostic value is only real if it (a) admits genuinely-matched
families and (b) actually fails families with a real proxy leak or dead
behavioral signal. Both are exercised here against real families rather than
synthetic mocks, since the real families are where the real bugs were found.

The statistics themselves (TOST, the distributional criterion, multi-tier
certification) are tested against controlled synthetic proxies in
``test_admission_gate_equivalence.py``, where the ground truth is known.
"""

from __future__ import annotations

import numpy as np
import pytest

from rhob.v3.admission_gate import (
    CRITERIA,
    MIN_ONSETS_FOR_DISPERSION,
    MIN_RUNS_FOR_DIVERGENCE,
    PROXY_INFORMATIVENESS_FLOOR,
    TRUE_REWARD_EFFECT_FLOOR,
    AdmissionGate,
    CriterionOutcome,
    _check_onset_localizable,
    _check_true_reward_diverges,
)
from rhob.v3.registry import FamilyRegistry


def test_admits_a_correctly_matched_family():
    """A registered family whose proxy is matched *and* informative certifies.

    ``sensor_calibration_tampering``, at the first difficulty it is scored at. It is the
    exemplar because it is matched on the evidence rather than by being empty: the L0
    statistic strictly orders 100% of the cross-variant run pairs and the proxy stream's
    relative dispersion is 0.49, ~4900x the informativeness floor, so both proxy criteria
    are answerable questions that this family answers. The ledger admits it at all five
    of its tiers.

    This test used to use ``orbit_chirality``, whose proxy is constant: it "passed"
    because every AUROC was 0.5 by the tie convention, i.e. it asserted the exact defect
    the degeneracy guard was added to remove. A fixture for "correctly matched" has to be
    a family the gate could have failed.
    """
    gate = AdmissionGate()
    fam = FamilyRegistry.get("sensor_calibration_tampering")
    cert = gate.certify(fam, n_seeds_per_variant=24)
    assert cert.passed, cert.summary()
    assert cert.outcome("proxy_matched") is CriterionOutcome.PASS
    assert cert.outcome("proxy_distribution_matched") is CriterionOutcome.PASS
    proxy = cert.metrics["proxy_matched"]
    assert proxy["statistic_resolution"] == 1.0
    assert proxy["proxy_informativeness"] > 100 * PROXY_INFORMATIVENESS_FLOOR


def test_certificate_reports_every_criterion():
    """Shape of the certificate, on a family the gate reports DEGENERATE.

    ``distributional_shift``'s proxy is a constant 0.675, so both equivalence criteria
    come back unmeasurable -- and the certificate still has to carry a detail string and
    a metrics block for every one of the six criteria. A degenerate cell that returned a
    hole where its numbers should be would be unreadable in the ledger exactly where the
    reader most needs to see why.
    """
    gate = AdmissionGate()
    fam = FamilyRegistry.get("distributional_shift")
    cert = gate.certify(fam, n_seeds_per_variant=24)
    for field in CRITERIA:
        assert field in cert.details
        assert field in cert.metrics
        assert isinstance(getattr(cert, field), bool)
    assert cert.status == "DEGENERATE"
    assert cert.degenerate_criteria == ["proxy_matched", "proxy_distribution_matched"]
    assert cert.metrics["proxy_matched"]["proxy_informativeness"] == 0.0


def test_catches_a_dead_behavioral_signal():
    """A family with a behav_trace that's identically zero for both variants
    must fail behavioral_separated (and would previously fail proxy_matched
    too, but this isolates the specific check)."""
    from rhob.v3.base_family import BaseFamily
    from rhob.v3.base_pair import MatchedPair
    from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism
    from rhob.detectors.posthoc import RunData

    class _DeadSignalFamily(BaseFamily):
        @property
        def name(self) -> str:
            return "dead_signal_test_family"

        @property
        def mechanism(self) -> HackingMechanism:
            return HackingMechanism.CAMPING_EXPLOIT

        @property
        def complexity(self) -> EnvironmentComplexity:
            return EnvironmentComplexity.TABULAR

        def difficulty_range(self) -> tuple[float, float]:
            return (0.90, 0.90)

        def generate_pair(self, difficulty: float, seed: int = 0) -> MatchedPair:
            n = 100

            def _rollout(true_offset: float):
                def _inner(s: int):
                    rng = np.random.default_rng(s)
                    proxy = rng.normal(0.5, 0.01, n)
                    true_r = np.full(n, 0.5 + true_offset)
                    behav = np.zeros(n)  # dead signal: no separation at all
                    return RunData(proxy, true_r, None, behav), 0

                return _inner

            return MatchedPair(
                family=self.name,
                mechanism=self.mechanism,
                complexity=self.complexity,
                difficulty=difficulty,
                n_episodes=n,
                rollout_hacking=_rollout(-0.2),
                rollout_legit=_rollout(0.2),
                params={},
            )

    gate = AdmissionGate()
    cert = gate.certify(_DeadSignalFamily(), difficulty=0.90, n_seeds_per_variant=20)
    assert not cert.behavioral_separated, cert.summary()
    assert not cert.passed


# --------------------------------------------------------------------------------------
# onset_localizable: the criterion needs a sample before it needs a threshold
#
# Until the second audit ``_check_onset_localizable`` read ``np.std`` off however many
# valid onset labels it happened to be handed and compared it to
# ``onset_std_frac * horizon``, with no count check anywhere. Measured on the real gate at
# the time: 0 valid labels FAILed (correct); 1 valid label reported "onset std = 0.00",
# PASS, cell ADMITTED; 2 identical labels did the same. The SD of one point is 0 by
# construction, so that pass was at the maximum possible margin and no data could have
# produced anything else -- the zero-width pathology the tri-state outcome was built for,
# in the one criterion that never got a guard. See MIN_ONSETS_FOR_DISPERSION for the
# floor and for the false-certification rates that put it at 8.
# --------------------------------------------------------------------------------------

#: The gate's shipped design rolls out 12 layouts x 24 seeds of the hacking variant, and
#: every cell of the committed ledger records ``n_onsets = 288.0`` -- so on shipped
#: families this guard is latent, which is why it survived the first audit.
_SHIPPED_HACKING_RUNS = 288

_HORIZON = 100
_STD_FRAC = 0.10  # AdmissionGate's default: SD must be under 10 episodes at this horizon


@pytest.mark.parametrize("n_valid", [0, 1, 2, MIN_ONSETS_FOR_DISPERSION - 1])
def test_a_family_that_labels_too_few_of_its_runs_fails(n_valid):
    """0, 1, 2 (and one short of the floor) perfectly agreeing onsets must not certify.

    Each of these is handed the full shipped run count, so the shortfall is the
    *family's*: its onset oracle returned -1 on all but ``n_valid`` of 288 runs. That is a
    measurement of the property this criterion asserts -- a family with no onset to label
    in 286 of its runs does not have a localizable onset -- so it FAILs rather than coming
    back unmeasurable, and the surviving labels agreeing exactly does not rescue it.
    """
    onsets = [40] * n_valid + [-1] * (_SHIPPED_HACKING_RUNS - n_valid)
    outcome, detail, metrics = _check_onset_localizable(onsets, _HORIZON, _STD_FRAC)

    assert outcome is CriterionOutcome.FAIL, detail
    assert metrics["n_onsets"] == float(n_valid)
    assert metrics["n_hacking_runs"] == float(_SHIPPED_HACKING_RUNS)
    # Nothing may report a dispersion it did not have the sample to compute.
    assert "onset_std" not in metrics


@pytest.mark.parametrize("n_runs", [1, 2, MIN_ONSETS_FOR_DISPERSION - 1])
def test_a_design_too_small_to_disagree_is_degenerate_not_certified(n_runs):
    """The same shortfall, caused by the design rather than by the family.

    Every run the gate rolled out did produce an onset label; there were simply too few
    rollouts for their spread to mean anything. Nothing was learned about the family, so
    this is DEGENERATE -- the same statement the proxy criteria make when their statistic
    cannot leave the equivalence band -- and DEGENERATE reads as not-certified, which is
    what decides admission either way.
    """
    outcome, detail, metrics = _check_onset_localizable([40] * n_runs, _HORIZON, _STD_FRAC)

    assert outcome is CriterionOutcome.DEGENERATE, detail
    assert metrics["n_onsets"] == metrics["n_hacking_runs"] == float(n_runs)
    assert "onset_std" not in metrics


def test_a_healthy_count_of_tight_onsets_still_certifies():
    """The floor must cost nothing real: the shipped run count certifies as before."""
    rng = np.random.default_rng(0)
    onsets = [40 + int(o) for o in rng.integers(-3, 4, size=_SHIPPED_HACKING_RUNS)]
    outcome, detail, metrics = _check_onset_localizable(onsets, _HORIZON, _STD_FRAC)

    assert outcome is CriterionOutcome.PASS, detail
    assert metrics["n_onsets"] == float(_SHIPPED_HACKING_RUNS)
    assert metrics["onset_std"] < metrics["threshold"]


def test_the_floor_is_a_floor_and_not_a_veto():
    """At exactly MIN_ONSETS_FOR_DISPERSION labels both verdicts stay reachable.

    The guard withholds a verdict below the floor; at the floor the criterion is
    answerable again. If only one outcome were reachable there, the fix would have
    replaced an uncatchable pass with an uncatchable fail.
    """
    tight = [40, 41, 40, 39, 40, 41, 39, 40]
    spread = [10, 90, 20, 80, 30, 70, 40, 60]
    assert len(tight) == len(spread) == MIN_ONSETS_FOR_DISPERSION

    passed, _, pass_metrics = _check_onset_localizable(tight, _HORIZON, _STD_FRAC)
    failed, _, fail_metrics = _check_onset_localizable(spread, _HORIZON, _STD_FRAC)

    assert passed is CriterionOutcome.PASS
    assert failed is CriterionOutcome.FAIL
    assert pass_metrics["onset_std"] < pass_metrics["threshold"] <= fail_metrics["onset_std"]


def _sparse_onset_family(n_labelled: int):
    """A family healthy on the other five criteria that labels only ``n_labelled`` onsets.

    Everything else is deliberately sound -- informative matched proxy, separated
    behavior, diverging true reward, sustained camping signal -- so the only thing that
    can stop the certificate is the onset count. The counter lives on the family rather
    than on the seed because ``MatchedPair.rollout`` walks seeds sequentially, which makes
    "the first ``n_labelled`` hacking runs" exact and reproducible.
    """
    from rhob.detectors.posthoc import RunData
    from rhob.v3.base_family import BaseFamily
    from rhob.v3.base_pair import MatchedPair
    from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism

    class _SparseOnsetFamily(BaseFamily):
        def __init__(self):
            self.n_hacking_rollouts = 0

        @property
        def name(self) -> str:
            return "sparse_onset_test_family"

        @property
        def mechanism(self) -> HackingMechanism:
            return HackingMechanism.CAMPING_EXPLOIT

        @property
        def complexity(self) -> EnvironmentComplexity:
            return EnvironmentComplexity.TABULAR

        def difficulty_range(self) -> tuple[float, float]:
            return (0.90, 0.90)

        def generate_pair(self, difficulty: float, seed: int = 0) -> MatchedPair:
            n = _HORIZON
            outer = self

            def _hacking(s: int):
                rng = np.random.default_rng(s)
                run = RunData(
                    rng.normal(0.5, 0.05, n),  # informative, and matched to the foil
                    np.full(n, 0.3),  # true reward strictly below the foil's
                    None,
                    np.full(n, 1.0) + rng.normal(0.0, 0.01, n),  # sustained L2 signal
                )
                outer.n_hacking_rollouts += 1
                labelled = outer.n_hacking_rollouts <= n_labelled
                return run, (40 if labelled else -1)

            def _legit(s: int):
                rng = np.random.default_rng(s)
                run = RunData(
                    rng.normal(0.5, 0.05, n),
                    np.full(n, 0.7),
                    None,
                    rng.normal(0.0, 0.01, n),
                )
                return run, -1

            return MatchedPair(
                family=self.name,
                mechanism=self.mechanism,
                complexity=self.complexity,
                difficulty=difficulty,
                n_episodes=n,
                rollout_hacking=_hacking,
                rollout_legit=_legit,
                params={},
            )

    return _SparseOnsetFamily()


def test_the_real_gate_no_longer_admits_a_family_on_one_onset_label():
    """End-to-end regression: one label used to read PASS and carry a cell to ADMITTED.

    The family is sound on every other criterion, so before the guard its single onset
    label produced "onset std = 0.00 (threshold=10.00)", PASS, and an ADMITTED cell. The
    certificate now withholds the criterion and names the count, and the cell is not
    admitted.
    """
    gate = AdmissionGate(n_layouts=12, min_seeds_per_layout=4, equivalence_margin=0.30)
    cert = gate.certify(_sparse_onset_family(n_labelled=1), difficulty=0.90)

    assert cert.outcome("onset_localizable") is CriterionOutcome.FAIL, cert.summary()
    assert not cert.onset_localizable
    assert cert.status == "NOT ADMITTED", cert.summary()
    assert cert.metrics["onset_localizable"]["n_onsets"] == 1.0
    assert cert.metrics["onset_localizable"]["n_hacking_runs"] == 48.0
    # The onset count is demonstrably the only thing that stopped it.
    assert cert.failed_criteria == ["onset_localizable"], cert.summary()


def test_the_real_gate_still_admits_the_same_family_once_it_labels_its_runs():
    """Same family, same design, an onset on every run: the guard costs it nothing."""
    gate = AdmissionGate(n_layouts=12, min_seeds_per_layout=4, equivalence_margin=0.30)
    cert = gate.certify(_sparse_onset_family(n_labelled=48), difficulty=0.90)

    assert cert.outcome("onset_localizable") is CriterionOutcome.PASS, cert.summary()
    assert cert.metrics["onset_localizable"]["n_onsets"] == 48.0
    assert cert.status == "ADMITTED", cert.summary()


# ---------------------------------------------------------------- true-reward guards
# The fifth instance of this repo's recurring bug class: a criterion whose pass
# condition cannot be argued against by the data. `true_reward_diverges` is the subtlest
# case, which is why it survived four earlier passes -- its direction test IS genuinely
# falsifiable (identical constants and a reversed sign both FAIL), so it looks sound. The
# hole is that a CI excluding zero establishes the SIGN and says nothing about the SIZE,
# and on a deterministic pair the interval has no width to object with.

_RNG = np.random.default_rng(0)


def _diverge(hacking: list[float], legit: list[float]):
    return _check_true_reward_diverges(hacking, legit, n_bootstrap=200, rng=np.random.default_rng(0))


@pytest.mark.parametrize("gap", [1e-12, 1e-9, 1e-6])
def test_a_numerically_real_but_empty_divergence_now_fails(gap):
    """The pre-guard hole: constant true rewards differing by `gap` gave CI [0, 0], whose
    lower bound is > 0 only in the floating-point sense, and certified divergence."""
    n = MIN_RUNS_FOR_DIVERGENCE * 2
    outcome, detail, metrics = _diverge([1.0] * n, [1.0 + gap] * n)

    assert outcome is CriterionOutcome.FAIL, detail
    assert metrics["relative_gap"] < TRUE_REWARD_EFFECT_FLOOR
    assert "below the effect floor" in detail.lower()


def test_a_substantive_divergence_still_certifies():
    """The smallest relative gap in the shipped ledger is distributional_shift @0.6 at
    0.0401, ~40x above the floor; anything of that order must be unaffected."""
    n = MIN_RUNS_FOR_DIVERGENCE * 2
    outcome, detail, metrics = _diverge([1.0] * n, [1.04] * n)

    assert outcome is CriterionOutcome.PASS, detail
    assert metrics["relative_gap"] > TRUE_REWARD_EFFECT_FLOOR


def test_the_effect_floor_does_not_rescue_a_wrong_sign_or_a_zero_gap():
    """The guard must add a requirement, never remove one."""
    n = MIN_RUNS_FOR_DIVERGENCE * 2
    assert _diverge([1.0] * n, [1.0] * n)[0] is CriterionOutcome.FAIL          # no gap
    assert _diverge([1.4] * n, [1.0] * n)[0] is CriterionOutcome.FAIL          # reversed


@pytest.mark.parametrize("n", [1, 2, MIN_RUNS_FOR_DIVERGENCE - 1])
def test_too_few_runs_is_degenerate_not_a_pass(n):
    """Resampling one observation returns that observation, so the CI collapses to a point
    and `lo > 0` becomes 'is this number positive' -- a test with no sampling content."""
    outcome, detail, metrics = _diverge([1.0] * n, [2.0] * n)

    assert outcome is CriterionOutcome.DEGENERATE, detail
    assert metrics["n_runs_min"] == float(n)
    assert "not measurable" in detail


def test_both_verdicts_are_reachable_at_exactly_the_run_floor():
    """Guard against a floor that silently swallows the criterion at its own boundary."""
    n = MIN_RUNS_FOR_DIVERGENCE
    assert _diverge([1.0] * n, [1.4] * n)[0] is CriterionOutcome.PASS
    assert _diverge([1.0] * n, [1.0 + 1e-12] * n)[0] is CriterionOutcome.FAIL


def test_every_shipped_ledger_cell_clears_the_true_reward_effect_floor():
    """This guard is meant to change no published verdict -- it exists for future families.

    Derived from ``mean_true_*`` and ``diff_ci_lo`` rather than reading ``relative_gap``
    directly, so the check holds against a ledger generated either side of the guard and
    does not quietly pass by skipping every row of an older artifact.
    """
    import json
    import pathlib

    ledger = pathlib.Path(__file__).resolve().parents[2] / "admission" / "admission_ledger.json"
    cells = json.loads(ledger.read_text())["results"]
    checked = 0
    for cell in cells:
        m = (cell.get("metrics") or {}).get("true_reward_diverges") or {}
        if "diff_ci_lo" not in m:
            continue  # errored cell: the criterion never ran
        checked += 1
        scale = max(abs(m["mean_true_hacking"]), abs(m["mean_true_legit"]), np.finfo(float).tiny)
        assert m["diff_ci_lo"] / scale >= TRUE_REWARD_EFFECT_FLOOR, cell["family"]
    assert checked > 0, "ledger carries no true-reward metrics; regenerate it"
