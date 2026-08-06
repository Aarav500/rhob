"""The supervised/unsupervised partition must track the code that does the scoring.

``rhob.v3.benchmark._evaluate_cell`` decides how a detector is scored with one predicate::

    if not hasattr(detector, "fit"):   # score directly
    ...                                # else: 5-fold CV, fitted on labels

``rhob.detectors.supervision`` reports which detectors fall on the fitted side, and the
replication aggregate publishes the access ladder twice -- once over all detectors and
once over the unsupervised ones only. That second ladder is not cosmetic: on the 20-draw
replication the paired L1-L0 step is ``+0.394`` over all detectors and ``-0.010`` without
the label-fitted ones, reversing sign and ceasing to separate. The whole published climb
from reward-only to state-visitation access is one label-fitted detector.

So the partition is load-bearing, and two ways of getting it wrong both end in a wrong
ladder rather than an error:

  * The derived set drifts from the scoring predicate -- a detector gains or loses ``fit``
    and quietly changes partition, moving a published rung.
  * ``partition_by_supervision`` fails to import the suite and reports *nothing* as
    label-fitted, which silently republishes the single-ladder view under a name that
    claims to be the unsupervised one.
"""

from __future__ import annotations

import pytest

from rhob.detectors.supervision import (
    KNOWN_LABEL_FITTED,
    is_label_fitted,
    label_fitted_detector_names,
    partition_by_supervision,
)


def test_derived_set_matches_the_pinned_set():
    """Membership change is a reviewable event, not a silent ladder shift."""
    derived = label_fitted_detector_names()
    assert derived == KNOWN_LABEL_FITTED, (
        f"the set of label-fitted detectors changed.\n"
        f"  gained: {sorted(derived - KNOWN_LABEL_FITTED)}\n"
        f"  lost:   {sorted(KNOWN_LABEL_FITTED - derived)}\n"
        f"This moves the published access ladder: a detector on this list is scored by "
        f"cross-validation on labels, and the unsupervised ladder excludes it. Update "
        f"KNOWN_LABEL_FITTED deliberately and regenerate leaderboard/v5_replicated.json."
    )


def test_the_predicate_is_the_scoring_paths_own():
    """is_label_fitted must agree with what _evaluate_cell actually branches on."""
    import inspect

    from rhob.v3 import benchmark

    src = inspect.getsource(benchmark._evaluate_cell)
    assert 'hasattr(detector, "fit")' in src, (
        "benchmark._evaluate_cell no longer branches on hasattr(detector, 'fit'). "
        "rhob.detectors.supervision.is_label_fitted reimplements that predicate and is "
        "now out of step with how detectors are actually scored."
    )


def test_every_named_detector_really_has_fit():
    """Guard against the pinned set naming a detector that is not actually fitted."""
    import importlib
    import inspect as _inspect
    import pkgutil

    import rhob.detectors as pkg

    by_name = {}
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name[:2] not in {"l0", "l1", "l2", "l3"}:
            continue
        try:
            module = importlib.import_module(f"rhob.detectors.{mod.name}")
        except Exception:  # pragma: no cover - optional deps
            continue
        for _, cls in _inspect.getmembers(module, _inspect.isclass):
            if cls.__module__ != module.__name__ or not hasattr(cls, "access_level"):
                continue
            try:
                by_name[cls().name] = cls
            except Exception:  # pragma: no cover
                continue

    for name in KNOWN_LABEL_FITTED:
        assert name in by_name, f"{name} is pinned as label-fitted but is not a shipped detector"
        assert is_label_fitted(by_name[name]()), f"{name} is pinned as label-fitted but has no fit()"


def test_partition_degrades_to_the_pinned_set_never_to_empty(monkeypatch):
    """If the suite cannot be imported, fall back -- do not report 'nothing is fitted'.

    Reporting an empty fitted set would republish the all-detector ladder under the
    'unsupervised only' label, which is worse than failing: it looks like the robustness
    check was run and passed.
    """
    import rhob.detectors.supervision as sup

    monkeypatch.setattr(
        sup, "label_fitted_detector_names", lambda: (_ for _ in ()).throw(ImportError("no deps"))
    )
    names = sorted(KNOWN_LABEL_FITTED) + ["Reward Threshold", "True Reward Oracle"]
    unsup, fitted = sup.partition_by_supervision(names)

    assert set(fitted) == KNOWN_LABEL_FITTED, "fallback did not use the pinned set"
    assert fitted, "partition reported NO label-fitted detectors -- the silent-failure mode"
    assert set(unsup) == {"Reward Threshold", "True Reward Oracle"}


@pytest.mark.parametrize("name", sorted(KNOWN_LABEL_FITTED))
def test_partition_places_each_pinned_detector_on_the_fitted_side(name: str):
    unsup, fitted = partition_by_supervision([name, "Reward Threshold"])
    assert name in fitted and name not in unsup
