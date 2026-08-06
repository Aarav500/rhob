"""Which detectors are fitted on labels, and why every aggregate has to say so.

A RHOB detector is scored one of two ways, decided by a single predicate in
``rhob.v3.benchmark._evaluate_cell``::

    if not hasattr(detector, "fit"):        # score each run directly
    ...                                     # else: 5-fold stratified CV, fit on labels

A detector exposing ``fit(runs_a, runs_b)`` is therefore handed the hacking/legitimate
labels for the training folds of every cell it is scored on. Cross-validation keeps it
honest -- it is never scored on runs it was fitted on -- so the number is not leaked and
is a legitimate measurement. But it answers a *different question* from an unsupervised
post-hoc detector's number:

    unsupervised: "does this fixed statistic separate the two variants?"
    label-fitted: "is the two variants' difference learnable from labelled examples?"

Stacking both into one per-access-level maximum answers neither. It reports whichever
question happened to produce the larger number at that level.

Why this is load-bearing here
-----------------------------
On the 20-draw replication the access ladder's best detector is label-fitted at L1
(State Divergence) and at L2 (Trajectory MLP), and unsupervised at L0 and L3. Removing
the five label-fitted detectors and re-deriving the ladder from the same draws:

===== ================= =====================
level all detectors     unsupervised only
===== ================= =====================
L0    0.5477            0.5437
L1    0.9416            0.5340
L2    0.9430            0.5872
L3    0.9746            0.9746
===== ================= =====================

The paired L1-L0 step goes from ``+0.394 [+0.384, +0.403]``, 20/20 draws, to
``-0.0097 [-0.028, +0.009]``, 5/20 draws -- it reverses sign and stops separating. The
entire published climb from reward-only to state-visitation access is one supervised
detector. That is a real result about RHOB, but it is not the result the ladder was
being read as, and a reader cannot recover it from a single-column table.

The paper this ships with previously applied a "supervised" caveat to Trajectory MLP
alone, using it to deflate L2 while L1's equally-supervised champion carried no such
mark -- an asymmetry that inflated L1 against L2 and made the L1/L2 comparison
incoherent. Reporting both partitions is what makes that comparison mean something.

Derived, not listed
-------------------
The set below is computed from the same ``hasattr(d, "fit")`` predicate the scoring path
uses, rather than hand-maintained, so a detector that gains or loses a ``fit`` method
cannot silently fall out of step with how it is actually scored.
``tests/test_v3/test_supervision_partition.py`` pins the current membership so such a
change is surfaced in review rather than absorbed into an aggregate.
"""

from __future__ import annotations

from functools import lru_cache

#: Membership as of the 2026-08 replication. Not the source of truth --
#: :func:`label_fitted_detector_names` is -- but pinned so a change shows up as a test
#: failure with a name attached instead of a silently shifted ladder.
KNOWN_LABEL_FITTED: frozenset[str] = frozenset(
    {
        "Reward MLP",              # L0
        "State Coverage Rate",     # L1
        "State Divergence",        # L1 -- the entire published L0->L1 step
        "Visitation Entropy Trend",  # L1
        "Trajectory MLP",          # L2
    }
)


def is_label_fitted(detector: object) -> bool:
    """Whether ``detector`` is scored by fitting on labels.

    This is the scoring path's own predicate, not a reimplementation of it: see
    ``rhob.v3.benchmark._evaluate_cell``.
    """
    return hasattr(detector, "fit")


@lru_cache(maxsize=1)
def label_fitted_detector_names() -> frozenset[str]:
    """Names of every shipped ``lN_*`` detector that is fitted on labels.

    Walks the ``rhob.detectors`` package rather than a hand-kept list, and applies the
    scoring path's own predicate, so the answer tracks the code. A class that gains a
    ``fit`` method is picked up here automatically; the pinned :data:`KNOWN_LABEL_FITTED`
    is what turns that into a visible test failure rather than a silent shift.

    ``fit`` is checked on the class, which is equivalent to the instance check in
    ``_evaluate_cell`` for a normally-defined method, so no detector need be constructed
    to classify it -- but the reported ``name`` does require an instance, and a detector
    that cannot be constructed with no arguments is skipped rather than guessed at.
    """
    import importlib
    import inspect
    import pkgutil

    import rhob.detectors as pkg

    found: set[str] = set()
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name[:2] not in {"l0", "l1", "l2", "l3"}:
            continue
        try:
            module = importlib.import_module(f"rhob.detectors.{mod.name}")
        except Exception:  # noqa: BLE001 -- an optional-dependency module is not a fatal error here
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue  # imported into the module, not defined by it
            if not (hasattr(cls, "access_level") and hasattr(cls, "fit")):
                continue
            try:
                found.add(cls().name)
            except Exception:  # noqa: BLE001 -- cannot name it, so cannot filter on it
                continue
    return frozenset(found)


def partition_by_supervision(detector_names: list[str]) -> tuple[list[str], list[str]]:
    """Split names into ``(unsupervised, label_fitted)``.

    Falls back to :data:`KNOWN_LABEL_FITTED` when the suite cannot be imported (for
    example when reading a committed artifact in an environment without the optional
    simulation dependencies), so an aggregate over JSON never silently reports every
    detector as unsupervised.
    """
    try:
        fitted = label_fitted_detector_names()
    except Exception:  # noqa: BLE001 -- see docstring: degrade to the pinned set, never to "none"
        fitted = KNOWN_LABEL_FITTED
    return (
        [n for n in detector_names if n not in fitted],
        [n for n in detector_names if n in fitted],
    )
