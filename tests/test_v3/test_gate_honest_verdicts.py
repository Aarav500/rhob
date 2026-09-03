"""The gate's two other honest verdicts, on constructions it used to get wrong.

``test_criteria_discriminate.py`` shows every criterion can return FAIL. This module
shows the gate can also say "I measured nothing" (DEGENERATE) and "this detector does
not apply here" (not applicable) in the two places it previously said something else:

* ``true_reward_diverges`` on arms that hold the same values reported FAIL -- "measured
  and found not to diverge" -- when the difference of means was 0 by arithmetic before
  any rollout. A counterfactual true reward that imports the matched twin's return is
  exactly this construction, and the port of HVTA's two-player channels shipped one.
* ``proxy_distribution_matched`` on a 40-episode family reported DEGENERATE because
  Reward Skewness, which needs 100 episodes, returned 0.5 on every run and tied on every
  pair. Eleven of the twelve families that fail the nightly smoke test run 40 or 60
  episodes; 33 of its 36 DEGENERATE cells were this.

Both are the harness describing itself and calling it a property of the family.
"""

from __future__ import annotations

import numpy as np
from adversarial_families import (
    DIFFICULTY,
    IdenticalTrueRewardFamily,
    ShortHorizonAdmissibleFamily,
    certification_gate,
)

from rhob.v3.admission_gate import CriterionOutcome


def test_the_identical_arms_fixture_actually_pairs_its_arms():
    """If base_pair's legit-seed offset ever changes, fail here, not in the gate test."""
    pair = IdenticalTrueRewardFamily().generate_pair_at(DIFFICULTY, seed=3)
    a, b, _ = pair.rollout(6, seed_base=4242, randomize_sign=False)
    assert all(np.array_equal(ra.true_rewards, rb.true_rewards) for ra, rb in zip(a, b))
    # ...and they are not a constant: within-arm variation is the whole point.
    assert np.std([r.true_rewards.mean() for r in a]) > 0.01


def test_identical_arms_are_reported_unmeasurable_not_false():
    cert = certification_gate().certify(IdenticalTrueRewardFamily(), difficulty=DIFFICULTY)
    assert cert.outcome("true_reward_diverges") is CriterionOutcome.DEGENERATE, cert.summary()
    assert cert.degenerate_criteria == ["true_reward_diverges"], cert.summary()
    assert cert.failed_criteria == [], cert.summary()
    assert cert.metrics["true_reward_diverges"]["arms_tied"] == 1.0


def test_a_shape_detector_that_does_not_fit_the_horizon_is_not_applicable():
    cert = certification_gate().certify(ShortHorizonAdmissibleFamily(), difficulty=DIFFICULTY)
    m = cert.metrics["proxy_distribution_matched"]
    assert m["n_not_applicable_detectors"] == 1.0, cert.summary()
    assert m["reward_skewness_not_applicable_horizon"] == 40.0, cert.summary()
    # It never ran, so it has no AUROC to report -- as opposed to a 0.5.
    assert "reward_skewness_auroc" not in m
    # The panel rests on the two detectors that fit, and is measured one way or the other.
    assert cert.outcome("proxy_distribution_matched") is not CriterionOutcome.DEGENERATE, (
        cert.summary()
    )
    assert "not applicable" in cert.details["proxy_distribution_matched"], cert.summary()
