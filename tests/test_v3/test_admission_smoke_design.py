"""The family tests' smoke design must stay honestly labelled and strictly weaker.

``tests/test_v3/admission_helpers.py`` runs the admission gate at a reduced design so it
can run in CI, and says so. Two things could quietly turn that into the pre-audit lie
again: the smoke margin drifting away from what the design can actually support, or the
smoke design creeping up to (or being mistaken for) the shipped certification design.
Both are cheap to assert, so they are asserted here rather than trusted to a comment.
"""

from __future__ import annotations

from rhob.v3.admission_gate import (
    EQUIVALENCE_MARGIN,
    PROXY_SHAPE_DETECTORS,
    TARGET_POWER,
    TOST_ALPHA,
    required_seeds_per_layout,
)

from admission_helpers import (
    SMOKE_LAYOUTS,
    SMOKE_MARGIN,
    SMOKE_SEEDS_PER_LAYOUT,
    smoke_equivalence_margin,
    smoke_gate,
)

N_TESTS = 1 + len(PROXY_SHAPE_DETECTORS)


def test_smoke_margin_is_exactly_what_the_smoke_design_supports():
    """SMOKE_MARGIN is ``required_seeds_per_layout`` inverted, not a chosen tolerance.

    At the smoke margin the gate's own power calculation asks for exactly
    SMOKE_SEEDS_PER_LAYOUT seeds; at anything tighter it asks for more than the smoke
    design has, which would make the screen a test that cannot pass.
    """
    assert (
        required_seeds_per_layout(SMOKE_MARGIN, SMOKE_LAYOUTS, TOST_ALPHA, TARGET_POWER, N_TESTS)
        == SMOKE_SEEDS_PER_LAYOUT
    )
    assert (
        required_seeds_per_layout(
            SMOKE_MARGIN * 0.999, SMOKE_LAYOUTS, TOST_ALPHA, TARGET_POWER, N_TESTS
        )
        > SMOKE_SEEDS_PER_LAYOUT
    )


def test_smoke_margin_is_strictly_looser_than_the_published_claim():
    """The screen must never be mistakable for certification.

    If these two ever met, the family tests would be claiming the benchmark's published
    +/-0.10 matched-proxy result on 96 rollouts -- exactly the pre-audit defect.
    """
    assert SMOKE_MARGIN > EQUIVALENCE_MARGIN
    assert 0.25 < SMOKE_MARGIN < 0.26, SMOKE_MARGIN


def test_certification_design_would_support_the_published_margin():
    """Same inversion applied to the shipped design returns the shipped margin.

    A consistency check on :func:`smoke_equivalence_margin` itself: fed the
    certification design (12 x 24) it must land on EQUIVALENCE_MARGIN, which is where
    ``required_seeds_per_layout`` derived that 24 from in the first place.
    """
    margin = smoke_equivalence_margin(n_layouts=12, seeds_per_layout=24)
    assert abs(margin - EQUIVALENCE_MARGIN) < 0.001, margin


def test_smoke_gate_is_configured_at_the_smoke_design():
    gate = smoke_gate()
    assert gate.n_layouts == SMOKE_LAYOUTS
    assert gate.min_seeds_per_layout == SMOKE_SEEDS_PER_LAYOUT
    assert gate.equivalence_margin == SMOKE_MARGIN
    # 96 rollouts per cell against the certification design's 576.
    assert 2 * gate.n_layouts * gate.min_seeds_per_layout == 96
