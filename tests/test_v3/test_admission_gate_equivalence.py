"""Tests for the AdmissionGate's equivalence machinery (``proxy_matched``,
``proxy_distribution_matched``) and for multi-tier certification.

These use synthetic families rather than registered ones, because the point is to
control the ground truth: we need a proxy that is *exactly* matched, one that leaks by
a *known* amount, one that matches in mean while differing in variance, and one that
carries no information at all. No registered family gives us that, and the audit
findings these tests encode (F1/F2/F3, and C1 -- the degeneracy guard) are statements
about the gate, not about any particular family.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from rhob.detectors.posthoc import RunData
from rhob.v3.admission_gate import (
    CRITERIA,
    EQUIVALENCE_MARGIN,
    PROXY_INFORMATIVENESS_FLOOR,
    PROXY_SHAPE_DETECTORS,
    AdmissionGate,
    CriterionOutcome,
    _check_proxy_distribution_matched,
    _score_resolution,
    attainable_auroc_half_range,
    ci_multiplier,
    degeneracy_reason,
    is_degenerate,
    is_uninformative,
    mann_whitney_null_sd,
    proxy_informativeness,
    required_seeds_per_layout,
)
from rhob.v3.base_family import BaseFamily
from rhob.v3.base_pair import MatchedPair
from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism

N_EPISODES = 200
ONSET = N_EPISODES // 4


class SyntheticProxyFamily(BaseFamily):
    """A family whose proxy distribution is specified exactly, per variant.

    Everything except the proxy is held fixed and passing: true reward diverges,
    the behavioral trace separates cleanly and sustains to the end of the episode,
    and the onset label is constant. That isolates the two proxy criteria, so a
    failure in a test below is unambiguously about the statistic under test.
    """

    def __init__(
        self,
        hacking_shift: float = 0.0,
        hacking_episode_sd: float = 0.05,
        legit_episode_sd: float = 0.05,
        run_sd: float = 0.05,
        difficulties: tuple[float, ...] = (0.9,),
    ):
        self.hacking_shift = hacking_shift
        self.hacking_episode_sd = hacking_episode_sd
        self.legit_episode_sd = legit_episode_sd
        self.run_sd = run_sd
        self._difficulties = difficulties

    @property
    def name(self) -> str:
        return "synthetic_proxy_family"

    @property
    def mechanism(self) -> HackingMechanism:
        return HackingMechanism.CAMPING_EXPLOIT

    @property
    def complexity(self) -> EnvironmentComplexity:
        return EnvironmentComplexity.TABULAR

    def difficulty_range(self) -> tuple[float, float]:
        return (min(self._difficulties), max(self._difficulties))

    def default_difficulties(self) -> list[float]:
        return list(self._difficulties)

    def _make_rollout(self, shift: float, episode_sd: float, sign: float, true_offset: float):
        def _inner(seed: int):
            rng = np.random.default_rng(seed)
            # Run-level offset + per-episode noise. The L0 detector scores the mean of
            # the trailing window, so the run-level offset is what drives its AUROC
            # while the per-episode noise is what the distributional criterion reads.
            base = 0.5 + shift + rng.normal(0.0, self.run_sd)
            proxy = base + rng.normal(0.0, episode_sd, N_EPISODES)
            true_r = np.full(N_EPISODES, 0.5 + true_offset)
            behav = np.zeros(N_EPISODES)
            behav[ONSET:] = sign * 1.0  # clean, sustained, anti-symmetric L2 signal
            return RunData(proxy, true_r, None, behav), ONSET

        return _inner

    def generate_pair(self, difficulty: float, seed: int = 0) -> MatchedPair:
        return MatchedPair(
            family=self.name,
            mechanism=self.mechanism,
            complexity=self.complexity,
            difficulty=difficulty,
            n_episodes=N_EPISODES,
            rollout_hacking=self._make_rollout(
                self.hacking_shift, self.hacking_episode_sd, +1.0, -0.2
            ),
            rollout_legit=self._make_rollout(0.0, self.legit_episode_sd, -1.0, +0.2),
            params={},
        )


# --------------------------------------------------------------------------------
# F1: the power analysis that did not exist before the audit
# --------------------------------------------------------------------------------


def test_exact_null_sd_reproduces_the_audit_monte_carlo():
    """The closed-form per-layout null SD must match the audit's 40k-rep simulation."""
    # n = m = 4 seeds/side/layout -- what every certification in the repo actually ran.
    assert mann_whitney_null_sd(4) == pytest.approx(0.2165, abs=5e-4)
    se_of_mean = mann_whitney_null_sd(4) / math.sqrt(12)
    assert se_of_mean == pytest.approx(0.0625, abs=5e-4)


def test_the_pre_audit_design_could_never_have_certified_equivalence():
    """At 4 seeds/layout the TOST interval is wider than the whole equivalence band.

    This is the quantitative statement of F1: no observed mean, not even exactly 0.5,
    could have produced an interval inside +/-0.10, which is why the old criterion had
    to be a difference test to pass anything at all.
    """
    half_width = ci_multiplier(12) * mann_whitney_null_sd(4) / math.sqrt(12)
    assert half_width > EQUIVALENCE_MARGIN
    assert half_width == pytest.approx(0.1172, abs=1e-3)


def test_required_seeds_is_derived_not_guessed():
    """The default seed floor comes out of the power calculation, and is 24.

    Four equivalence tests must all pass (RewardThresholdDetector plus the three
    PROXY_SHAPE_DETECTORS), so each is sized at ``0.90 ** (1/4)`` to hold the combined
    false-reject rate on a genuinely matched family at 10%.
    """
    from scipy.stats import norm

    assert required_seeds_per_layout() == 24
    assert AdmissionGate().min_seeds_per_layout == 24

    se = mann_whitney_null_sd(24) / math.sqrt(12)
    slack = EQUIVALENCE_MARGIN - ci_multiplier(12) * se
    assert slack > 0
    per_test_power = 2 * norm.cdf(slack / se) - 1
    assert per_test_power ** (1 + len(PROXY_SHAPE_DETECTORS)) >= 0.90

    # A single test would have needed only 18: the extra seeds buy the shape panel.
    assert required_seeds_per_layout(n_equivalence_tests=1) == 18


def test_tost_admits_an_exactly_matched_proxy():
    fam = SyntheticProxyFamily()
    cert = AdmissionGate().certify(fam, difficulty=0.9)
    assert cert.proxy_matched, cert.summary()
    lo = cert.metrics["proxy_matched"]["ci_lo"]
    hi = cert.metrics["proxy_matched"]["ci_hi"]
    assert 0.5 - EQUIVALENCE_MARGIN < lo <= hi < 0.5 + EQUIVALENCE_MARGIN


def test_tost_refuses_to_certify_an_underpowered_design():
    """Same perfectly-matched family, but run at the pre-audit 4 seeds/layout.

    The point estimate lands inside the band (the old test would have passed it, as it
    did 89.6% of the time), but the interval does not fit, so the gate declines. Absence
    of evidence is no longer evidence of absence.
    """
    gate = AdmissionGate(min_seeds_per_layout=4)
    cert = gate.certify(SyntheticProxyFamily(), difficulty=0.9)
    assert abs(cert.metrics["proxy_matched"]["mean_l0_auroc"] - 0.5) < EQUIVALENCE_MARGIN
    assert not cert.proxy_matched
    assert "underpowered" in cert.details["proxy_matched"]
    assert cert.design["a_priori_power"] == 0.0


def test_tost_rejects_a_leak_the_old_difference_test_admitted():
    """A proxy leak around AUROC 0.6 must fail.

    F1 measured that the old ``abs(mean - 0.5) < 0.10`` test certified a family leaking
    at a true mean AUROC of 0.611 on 43.5% of runs. Under TOST such a family cannot be
    certified at all: even a perfectly precise estimate of 0.611 lies outside the band.
    """
    # run_sd=0.05, shift=0.02 -> AUROC = Phi(0.02 / (0.05 * sqrt(2))) ~ 0.61.
    fam = SyntheticProxyFamily(hacking_shift=0.02, run_sd=0.05)
    cert = AdmissionGate().certify(fam, difficulty=0.9)
    assert cert.metrics["proxy_matched"]["mean_l0_auroc"] > 0.55
    assert not cert.proxy_matched, cert.summary()


# --------------------------------------------------------------------------------
# F2: matched means are not a matched proxy
# --------------------------------------------------------------------------------


def test_distribution_criterion_fails_the_variance_probe_that_proxy_matched_passes():
    """The exact synthetic probe from the audit: N(0.5, 0.02) vs N(0.5, 0.60).

    Identical means, 30x different per-episode spread. The audit measured that the
    mean-based criterion *admits* this (mean L0 AUROC 0.401) while four shipped L0
    detectors score 1.000 on it. ``proxy_matched`` must still pass -- it is measuring
    what it always measured -- and ``proxy_distribution_matched`` must catch it, which
    is the whole reason the criterion is separate rather than folded in.
    """
    fam = SyntheticProxyFamily(
        hacking_episode_sd=0.02, legit_episode_sd=0.60, run_sd=0.0
    )
    cert = AdmissionGate().certify(fam, difficulty=0.9)

    assert cert.proxy_matched, cert.summary()

    assert not cert.proxy_distribution_matched, cert.summary()
    # Reward KDE is the detector that separates them, and it does so perfectly (0.0 --
    # AUROC is rank-order, so a perfect inverted separation is as far from chance as
    # 1.0). Reward Variance Ratio does *not* fire here despite this being a variance
    # mismatch, because it is self-calibrating: it scores late-vs-early variance within
    # a run, and both variants are stationary. That is exactly why the criterion is a
    # panel and not a single hand-picked statistic.
    metrics = cert.metrics["proxy_distribution_matched"]
    assert abs(metrics["reward_kde_auroc"] - 0.5) > EQUIVALENCE_MARGIN
    assert not cert.passed


def test_distribution_criterion_fires_at_a_modest_spread_mismatch():
    """Calibration regression: a 1.6x per-episode SD ratio must fail.

    That is roughly what ``mujoco_sensor_decoupling`` @ 0.9 exhibits (per-episode SD
    ratio 1.584) while the gate's own L0 detector reads 0.465 -- "matched" -- and Reward
    KDE reads 0.979 on the identical rollouts. Pinning the criterion here, not only at
    the far more extreme 30x synthetic probe, is what stops the margins from drifting
    back past the case the criterion exists to catch.
    """
    fam = SyntheticProxyFamily(
        hacking_episode_sd=0.08, legit_episode_sd=0.05, run_sd=0.0  # ratio 1.6
    )
    cert = AdmissionGate().certify(fam, difficulty=0.9)
    assert cert.proxy_matched, cert.summary()
    assert not cert.proxy_distribution_matched, cert.summary()


def test_distribution_criterion_admits_a_genuinely_matched_proxy():
    """A matched proxy clears all three shape detectors, given a powered draw.

    Deliberately run at 48 seeds/layout rather than the default 24. The default is sized
    for 90% *combined* power, i.e. a genuinely matched family is correctly admitted 9
    times in 10 -- and the gate is deterministic, so a single-draw assertion at exactly
    that floor is a coin flip on the root seed rather than a statement about the gate.
    (It lands in the 10% here: at 24 seeds this fixture draws Reward Variance Ratio
    0.5437, CI [0.4801, 0.6073], which is a 1.8-sigma fluctuation and not bias -- 200
    independent replicates of that detector on identically-distributed proxies average
    0.4934 +/- 0.0056.) Doubling the seeds halves the interval and makes the assertion
    about the criterion instead of about the seed.
    """
    cert = AdmissionGate().certify(
        SyntheticProxyFamily(), difficulty=0.9, n_seeds_per_variant=12 * 48
    )
    assert cert.proxy_distribution_matched, cert.summary()
    metrics = cert.metrics["proxy_distribution_matched"]
    for detector in ("reward_variance_ratio", "reward_kde", "reward_skewness"):
        assert 0.5 - EQUIVALENCE_MARGIN < metrics[f"{detector}_ci_lo"]
        assert metrics[f"{detector}_ci_hi"] < 0.5 + EQUIVALENCE_MARGIN


# --------------------------------------------------------------------------------
# C1: an uninformative proxy is degenerate, not matched
# --------------------------------------------------------------------------------


def test_score_resolution_counts_the_pairs_the_detector_can_order():
    labels = [1, 1, 0, 0]
    assert _score_resolution(labels, [7.0, 7.0, 7.0, 7.0]) == 0.0
    assert _score_resolution(labels, [1.0, 2.0, 3.0, 4.0]) == 1.0
    # Two of the four cross-variant pairs (the 2.0 vs 2.0 ones) are tied.
    assert _score_resolution(labels, [1.0, 2.0, 2.0, 3.0]) == pytest.approx(0.75)

    # Floating-point dust is a tie, not an ordering. These are the actual Reward KDE
    # scores orbit_chirality produced: 6 values spanning 3.0e-10 around -13.069062904,
    # which roc_auc_score happily ranked into "AUROC 0.5161, CI [0.4593, 0.5728]".
    dust = [-13.069062904240258, -13.069062904200848, -13.069062904282566,
            -13.069062904180807, -13.069062904249940, -13.069062904189506]
    assert _score_resolution([1, 1, 1, 0, 0, 0], dust) == 0.0


def test_the_degeneracy_cut_is_the_attainable_range_of_the_statistic():
    """A tied pair contributes exactly 0.5, so resolution bounds how far AUROC can go.

    The cut is therefore not a tuned threshold: it is the point at which the equivalence
    band swallows the statistic's whole attainable range and the test stops being able
    to fail.
    """
    assert attainable_auroc_half_range(0.0) == 0.0
    assert attainable_auroc_half_range(1.0) == 0.5

    assert is_degenerate(0.0)
    assert is_degenerate(float("nan"))
    assert is_degenerate(2 * EQUIVALENCE_MARGIN)  # reach exactly == margin
    assert not is_degenerate(2 * EQUIVALENCE_MARGIN + 1e-6)
    # A tighter margin makes fewer statistics unusable, and the cut tracks it.
    assert not is_degenerate(0.10, margin=0.04)


def test_a_constant_proxy_is_degenerate_not_matched():
    """The C1 defect: a proxy carrying zero information must not certify as matched.

    Three registered families (``distributional_shift``, ``monitored_sandbagging``,
    ``orbit_chirality``) have a constant proxy -- ``REPRODUCIBILITY.md`` records that
    the ``distributional_shift`` "fix" was to pin it to 0.675 -- and on the first
    equivalence ledger all 15 of their cells certified ``proxy_matched`` on the interval
    [0.5000, 0.5000], which no margin can reject. Both proxy criteria must now come back
    DEGENERATE: the gate still evaluates them without crashing or propagating NaN into
    the certificate, but it reports that it measured nothing rather than that it found
    a match.
    """
    fam = SyntheticProxyFamily(hacking_episode_sd=0.0, legit_episode_sd=0.0, run_sd=0.0)
    cert = AdmissionGate().certify(fam, difficulty=0.9)

    assert cert.outcome("proxy_matched") is CriterionOutcome.DEGENERATE, cert.summary()
    assert cert.outcome("proxy_distribution_matched") is CriterionOutcome.DEGENERATE
    assert not cert.proxy_matched
    assert not cert.proxy_distribution_matched
    assert not cert.passed

    # Degenerate is its own status: nothing was measured and found wanting, so the cell
    # must not be filed among the families with an actual leak.
    assert cert.status == "DEGENERATE"
    assert cert.degenerate_criteria == ["proxy_matched", "proxy_distribution_matched"]
    assert cert.failed_criteria == []

    # The zero-width interval that used to certify is still reported -- it is the
    # evidence for the verdict -- next to the resolution that disqualifies it.
    metrics = cert.metrics["proxy_matched"]
    assert metrics["ci_lo"] == metrics["ci_hi"] == 0.5
    assert metrics["statistic_resolution"] == 0.0
    dist = cert.metrics["proxy_distribution_matched"]
    assert dist["n_degenerate_detectors"] == float(len(PROXY_SHAPE_DETECTORS))


def test_a_real_leak_still_reads_fail_and_not_degenerate():
    """FAIL and DEGENERATE must stay distinguishable, or the third state buys nothing.

    A leaking proxy is highly resolvable -- that is *how* it leaks -- so the guard must
    not absorb it. Conflating the two would send a maintainer looking for a leak in a
    family whose data cannot show one, and vice versa.
    """
    cert = AdmissionGate().certify(
        SyntheticProxyFamily(hacking_shift=0.02, run_sd=0.05), difficulty=0.9
    )
    assert cert.outcome("proxy_matched") is CriterionOutcome.FAIL, cert.summary()
    assert cert.metrics["proxy_matched"]["statistic_resolution"] == 1.0
    assert cert.status == "NOT ADMITTED"
    assert cert.degenerate_criteria == []


def test_an_informative_matched_proxy_is_unaffected_by_the_guard():
    """The guard must cost a genuinely matched family nothing."""
    cert = AdmissionGate().certify(
        SyntheticProxyFamily(), difficulty=0.9, n_seeds_per_variant=12 * 48
    )
    assert cert.outcome("proxy_matched") is CriterionOutcome.PASS, cert.summary()
    assert cert.outcome("proxy_distribution_matched") is CriterionOutcome.PASS
    assert cert.metrics["proxy_matched"]["statistic_resolution"] == 1.0
    assert cert.status == "ADMITTED"


def test_a_measured_failure_outranks_an_unmeasurable_detector_in_the_panel():
    """Panel precedence, unit-tested on the aggregation rather than through rollouts.

    One detector that measures a real distributional gap is a finding about the family
    and must be reported as FAIL even when a second detector was unusable; a panel with
    no failures but an unusable detector certifies nothing and reads DEGENERATE.
    """
    rng = np.random.default_rng(0)
    matched = [0.50] * 12
    leaking = [0.95] * 12
    resolved = [1.0] * 12
    tied = [0.0] * 12

    informative = 1.0  # isolate the panel logic from the informativeness guard

    outcome, detail, _ = _check_proxy_distribution_matched(
        {"A": leaking, "B": matched}, {"A": resolved, "B": tied},
        informative, EQUIVALENCE_MARGIN, PROXY_INFORMATIVENESS_FLOOR, 0.05, 200, rng,
    )
    assert outcome is CriterionOutcome.FAIL
    assert "not counted" in detail  # the unusable detector is still disclosed

    outcome, _, _ = _check_proxy_distribution_matched(
        {"A": matched, "B": matched}, {"A": resolved, "B": tied},
        informative, EQUIVALENCE_MARGIN, PROXY_INFORMATIVENESS_FLOOR, 0.05, 200, rng,
    )
    assert outcome is CriterionOutcome.DEGENERATE

    outcome, _, _ = _check_proxy_distribution_matched(
        {"A": matched, "B": matched}, {"A": resolved, "B": resolved},
        informative, EQUIVALENCE_MARGIN, PROXY_INFORMATIVENESS_FLOOR, 0.05, 200, rng,
    )
    assert outcome is CriterionOutcome.PASS

    # An uninformative proxy outranks even a measured failure: on dust, "this detector
    # found a distributional gap" means "this detector ranked rounding error".
    outcome, detail, metrics = _check_proxy_distribution_matched(
        {"A": leaking, "B": matched}, {"A": resolved, "B": resolved},
        1e-9, EQUIVALENCE_MARGIN, PROXY_INFORMATIVENESS_FLOOR, 0.05, 200, rng,
    )
    assert outcome is CriterionOutcome.DEGENERATE
    assert "numerical dust" in detail
    # The intervals are still published -- they are the evidence for the verdict.
    assert metrics["a_auroc"] == pytest.approx(0.95)


# --------------------------------------------------------------------------------
# C3: the dust band -- a tie tolerance is not a degeneracy guard
# --------------------------------------------------------------------------------


def test_proxy_informativeness_is_dispersion_relative_to_magnitude():
    """The statistic is scale-free, and it is about the signal, not about the runs."""

    class _Run:
        def __init__(self, proxy):
            self.proxy_rewards = np.asarray(proxy, dtype=float)

    # Same relative dispersion, magnitudes three orders of magnitude apart: RHOB proxies
    # really do span that range (gridworld episode sums vs continuous per-step means),
    # so an absolute cut would police them at completely different strictnesses.
    small = proxy_informativeness([_Run([0.99, 1.01])])
    large = proxy_informativeness([_Run([990.0, 1010.0])])
    assert small == pytest.approx(0.01)
    assert large == pytest.approx(0.01)

    # Constant -> exactly zero, whether the constant is 0.675 or 0.0.
    assert proxy_informativeness([_Run([0.675] * 8)]) == 0.0
    assert proxy_informativeness([_Run([0.0] * 8)]) == 0.0

    # Variation between runs counts the same as variation within one: both are signal
    # some L0 detector could read.
    assert proxy_informativeness([_Run([1.0, 1.0]), _Run([3.0, 3.0])]) == pytest.approx(0.5)

    # Nothing to measure is not "informative".
    assert is_uninformative(proxy_informativeness([]))
    assert is_uninformative(proxy_informativeness([_Run([float("nan")] * 4)]))


def test_the_two_degeneracy_guards_catch_different_things():
    """Neither subsumes the other, so both have to be asked."""
    # Resolution-degenerate but informative: a proxy that varies a lot, identically in
    # every run, so every cross-variant pair ties.
    assert degeneracy_reason(0.0, 1.0) == "resolution"
    # Informative-degenerate but fully resolved: the dust case below.
    assert degeneracy_reason(1.0, 1e-7) == "informativeness"
    assert degeneracy_reason(1.0, 1.0) is None
    # Informativeness is reported first when both fire, because it is the stronger
    # statement: it disqualifies the whole panel rather than one detector.
    assert degeneracy_reason(0.0, 0.0) == "informativeness"

    assert is_uninformative(PROXY_INFORMATIVENESS_FLOOR / 10)
    assert not is_uninformative(PROXY_INFORMATIVENESS_FLOOR)
    assert not is_uninformative(PROXY_INFORMATIVENESS_FLOOR * 10)


@pytest.mark.parametrize("jitter", [1e-9, 1e-8, 1e-7, 1e-6])
def test_a_constant_proxy_plus_dust_is_still_degenerate(jitter):
    """The C3 bypass: adding numerically irrelevant jitter must not buy a certificate.

    The resolution guard alone is defeated by any jitter above its tie tolerance, and it
    was: measured on the shipped design, ``0.5 + N(0, 1e-7)`` scores L0 resolution 0.961
    and 1.000 on all three shape detectors, and the whole gate returned **ADMITTED** --
    the same "matched" verdict it gives a real family, bought with variation in the 8th
    significant digit of the reward. ``1e-8`` scored 0.628 and also certified.

    A tighter tie tolerance cannot close this (whatever the tolerance, 10x more jitter
    clears it), so the guard is an effect-size criterion instead: this proxy's relative
    dispersion is 2 * jitter (base 0.5), i.e. 2e-9 to 2e-6 across this band, all below
    the 1e-4 floor and all at least 3000x below the 6.5e-3 that
    ``sequence_length_padding`` -- the least informative registered family the gate can
    measure -- reads. Every one of these must come back DEGENERATE, not ADMITTED and not
    FAIL: nothing was measured, so there is nothing to fail.
    """
    fam = SyntheticProxyFamily(
        hacking_episode_sd=jitter, legit_episode_sd=jitter, run_sd=0.0
    )
    cert = AdmissionGate().certify(fam, difficulty=0.9)

    assert cert.outcome("proxy_matched") is CriterionOutcome.DEGENERATE, cert.summary()
    assert cert.outcome("proxy_distribution_matched") is CriterionOutcome.DEGENERATE
    assert cert.status == "DEGENERATE"
    assert cert.failed_criteria == []

    metrics = cert.metrics["proxy_matched"]
    assert metrics["proxy_informativeness"] == pytest.approx(2 * jitter, rel=0.05)
    assert metrics["informativeness_floor"] == PROXY_INFORMATIVENESS_FLOOR
    assert "numerical dust" in cert.details["proxy_matched"]


def test_the_dust_band_would_have_certified_on_resolution_alone():
    """Pin the bypass itself, not only the fix, so a regression is legible.

    At 1e-7 the *resolution* guard is comfortably satisfied -- it reads far above the
    ``2 * margin`` cut at which a statistic is pinned inside the band -- and it is only
    the informativeness criterion standing between this proxy and a certificate. If
    someone later removes that criterion, this assertion is what says what was lost.
    """
    cert = AdmissionGate().certify(
        SyntheticProxyFamily(hacking_episode_sd=1e-7, legit_episode_sd=1e-7, run_sd=0.0),
        difficulty=0.9,
    )
    resolution = cert.metrics["proxy_matched"]["statistic_resolution"]
    assert resolution > 0.9
    assert not is_degenerate(resolution)
    assert degeneracy_reason(
        resolution, cert.metrics["proxy_matched"]["proxy_informativeness"]
    ) == "informativeness"


def test_a_small_but_real_proxy_signal_still_certifies():
    """The floor must not be so high that it swallows a genuinely small signal.

    A relative dispersion of 2e-3 -- 20x the floor, and still three times smaller than
    the least informative registered family the gate can measure
    (``sequence_length_padding``, 6.5e-3) -- is a measurement, not dust, and the gate has
    to keep certifying it. Run at 48 seeds/layout for the same
    reason ``test_distribution_criterion_admits_a_genuinely_matched_proxy`` is: the
    default design is sized for 90% combined power, so a single-draw assertion at exactly
    that floor tests the root seed rather than the criterion.
    """
    fam = SyntheticProxyFamily(
        hacking_episode_sd=1e-3, legit_episode_sd=1e-3, run_sd=0.0
    )
    cert = AdmissionGate().certify(fam, difficulty=0.9, n_seeds_per_variant=12 * 48)
    informativeness = cert.metrics["proxy_matched"]["proxy_informativeness"]
    assert informativeness == pytest.approx(2e-3, rel=0.05)
    assert informativeness > 10 * PROXY_INFORMATIVENESS_FLOOR
    assert cert.outcome("proxy_matched") is CriterionOutcome.PASS, cert.summary()
    assert cert.outcome("proxy_distribution_matched") is CriterionOutcome.PASS
    assert cert.status == "ADMITTED"


def test_the_guard_does_not_scale_with_the_sampling_design():
    """A dust proxy is degenerate at the loose smoke margin too.

    ``EQUIVALENCE_MARGIN`` is a statement about how much leak is tolerable and the smoke
    screen deliberately relaxes it; informativeness is a statement about whether there is
    anything in the proxy at all, and relaxing *that* with the design would let a cheap
    run certify what an expensive one could not.
    """
    fam = SyntheticProxyFamily(hacking_episode_sd=1e-7, legit_episode_sd=1e-7, run_sd=0.0)
    loose = AdmissionGate(equivalence_margin=0.256, min_seeds_per_layout=4)
    cert = loose.certify(fam, difficulty=0.9)
    assert cert.outcome("proxy_matched") is CriterionOutcome.DEGENERATE, cert.summary()
    assert cert.design["informativeness_floor"] == PROXY_INFORMATIVENESS_FLOOR


# --------------------------------------------------------------------------------
# F3: certification must cover the difficulties the benchmark scores
# --------------------------------------------------------------------------------


def test_certify_all_tiers_covers_every_default_difficulty():
    fam = SyntheticProxyFamily(difficulties=(0.9, 0.8, 0.7))
    certs = AdmissionGate().certify_all_tiers(fam)
    assert [c.difficulty for c in certs] == [0.9, 0.8, 0.7]
    assert all(c.family_name == fam.name for c in certs)


def test_single_difficulty_entry_point_still_defaults_to_the_first_tier():
    fam = SyntheticProxyFamily(difficulties=(0.9, 0.8, 0.7))
    assert AdmissionGate().certify(fam).difficulty == 0.9


def test_certify_all_tiers_reports_a_tier_specific_failure():
    """A family that is matched at one difficulty and leaking at another must show it.

    This is the shape of F3: certifying ``default_difficulties()[0]`` and calling the
    family admitted hides whatever the other tiers do.
    """

    class _TierDependentLeak(SyntheticProxyFamily):
        def generate_pair(self, difficulty: float, seed: int = 0) -> MatchedPair:
            # Leak only at the 0.7 tier.
            self.hacking_shift = 0.05 if difficulty == 0.7 else 0.0
            return super().generate_pair(difficulty, seed=seed)

    certs = AdmissionGate().certify_all_tiers(_TierDependentLeak(difficulties=(0.9, 0.7)))
    by_difficulty = {c.difficulty: c for c in certs}
    assert by_difficulty[0.9].proxy_matched
    assert not by_difficulty[0.7].proxy_matched


# --------------------------------------------------------------------------------
# Certificate shape (consumed by scripts/admission_ledger.py)
# --------------------------------------------------------------------------------


def test_certificate_serializes_every_criterion_with_numbers():
    cert = AdmissionGate().certify(SyntheticProxyFamily(), difficulty=0.9)
    payload = cert.to_dict()
    assert set(payload["criteria"]) == set(CRITERIA)
    assert set(payload["outcomes"]) == set(CRITERIA)
    assert set(payload["metrics"]) == set(CRITERIA)
    assert set(payload["details"]) == set(CRITERIA)
    assert payload["design"]["seeds_per_layout"] == 24
    assert payload["design"]["n_layouts"] == 12
    assert payload["design"]["rng_seed"] == 12_345
    assert payload["design"]["a_priori_power"] > 0.9
    # Every metric must be a plain float so the ledger is JSON-serializable.
    for block in payload["metrics"].values():
        for value in block.values():
            assert isinstance(value, float)


def test_the_three_outcome_states_are_machine_readable_in_the_ledger_payload():
    """``scripts/admission_ledger.py`` has to be able to separate the degenerate cells.

    A ledger that could only say "not admitted" would leave a reader counting a family
    with an unmeasurable proxy as evidence of the same kind as a family that was
    measured and passed, which is the accounting error the third state removes.
    """
    matched = AdmissionGate().certify(
        SyntheticProxyFamily(), difficulty=0.9, n_seeds_per_variant=12 * 48
    ).to_dict()
    degenerate = AdmissionGate().certify(
        SyntheticProxyFamily(hacking_episode_sd=0.0, legit_episode_sd=0.0, run_sd=0.0),
        difficulty=0.9,
    ).to_dict()

    assert matched["status"] == "ADMITTED"
    assert matched["outcomes"]["proxy_matched"] == "pass"
    assert matched["degenerate_criteria"] == []

    assert degenerate["status"] == "DEGENERATE"
    assert degenerate["passed"] is False
    assert degenerate["criteria"]["proxy_matched"] is False
    assert degenerate["outcomes"]["proxy_matched"] == "degenerate"
    assert degenerate["degenerate_criteria"] == [
        "proxy_matched", "proxy_distribution_matched",
    ]
    # Plain JSON, no enum objects left in the payload.
    assert json.loads(json.dumps(degenerate))["outcomes"] == degenerate["outcomes"]


def test_certification_is_deterministic():
    fam = SyntheticProxyFamily()
    a = AdmissionGate().certify(fam, difficulty=0.9).to_dict()
    b = AdmissionGate().certify(fam, difficulty=0.9).to_dict()
    assert a == b
