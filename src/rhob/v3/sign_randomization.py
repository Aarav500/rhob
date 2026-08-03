"""Evaluation-time randomization of the L2 behavioral feature's sign.

Why this exists
---------------
Until the 2026-08 audit, ``CONTRIBUTING.md`` required every family to emit a
``RunData.behav_trace`` that is anti-symmetric across the pair *with one fixed global
orientation*: positive = hacking, negative = legitimate.
:class:`~rhob.detectors.l2_behavioral_threshold.BehavioralThresholdDetector` scores
``behav_trace[-100:].mean()`` and the benchmark computes AUROC with label 1 = hacking,
so **the sign of the observation was the label**. Under that convention an L2 detector
that "transfers to an unseen hacking mechanism" has discovered the repository's house
style guide, not a property of reward hacking, and the headline RTS of 0.994 measured
convention-following rather than generalization. It is the L2-side twin of the L0
tautology the same audit removed.

The convention also explains a separate measured finding that had no other explanation:
L2-AUROC was pinned at exactly 1.000 across *every* difficulty tier in 12 of the 13
families whose knob was swept (knob span 0.000). The knob moves how *far* the two
variants separate; it cannot move a statistic whose sign already is the answer.

What this module does
---------------------
:func:`behav_sign` maps each ``(family, layout_seed)`` to +1 or -1, and
:meth:`~rhob.v3.base_pair.MatchedPair.rollout` multiplies **both** variants' traces by
that one number before any detector sees them. It is a coordinate flip of the axis the
pair is defined on, not a per-run perturbation:

- Applying it to both variants is what makes it de-biasing rather than destructive.
  Flipping the two variants independently would not hide the direction, it would
  dissolve the pair -- the anti-symmetry *is* the pair.
- Every magnitude is preserved exactly, so the admission gate's behavioral separation,
  the difficulty knob's effect, and each family's internal anti-symmetry are unchanged.
- What is not preserved is the global constant that let a detector read the label off
  the sign of a single number.

What this module does NOT do
----------------------------
It removes the *direction* freebie, and only that. ``behav_trace`` is still a
hand-authored feature: a contributor who knew which variant was the hacking one chose
the axis the family separates on, and the benchmark still hands that axis to L2
detectors. Rebuilding the L2 channel out of label-blind raw trajectory summaries is a
separate, deliberately deferred piece of work. Until it lands, an L2 number measured
under sign randomization means "this detector did not need the sign convention" -- it
does **not** mean "the L2 channel is label-blind".

One residual leak is worth naming precisely, because it is a consequence of the scope
line above rather than an oversight: at L2 a detector also sees ``state_counts``, and
that channel is *not* flipped (a histogram over a family's own state discretization has
no benchmark-wide sign to flip). A family whose histogram bins encode the same spatial
axis as its trace therefore still carries, in principle, enough information to recover
the family's original trace coordinate -- and with it the old convention. Nothing in
the repo does this today, and doing it deliberately would be reading the label by a
longer route, exactly as reading ``true_rewards`` would be. It closes properly only
when the L2 channel itself is rebuilt.

Why the mapping is deterministic
--------------------------------
Reproducibility is a hard requirement here (``REPRODUCIBILITY.md``): the same seed must
produce the same rollout, and the same benchmark run must produce the same leaderboard.
So the flip is a pure function of ``(family, layout_seed)`` via a keyed digest -- no RNG
state, no call ordering, no wall clock. Re-running the suite a year from now reproduces
the same signs.

Why the mapping is not a secret
-------------------------------
It cannot be. This is an open benchmark; any constant in this file is readable, and a
detector that imports :func:`behav_sign` and divides it back out will score 1.000 again.
That is not a hole this module can close with cryptography, and it is not a new class of
problem: ``RunData.true_rewards`` is likewise just an attribute, and what stops an L2
detector from reading it is :func:`~rhob.v3.access.restrict` plus the rule that doing so
is disqualifying, not an impossibility proof. The standard here is the same one:

    the *interface* must not hand a detector the direction, and a detector that goes
    around the interface to recover it is reading the label.

What the interface hands a post-hoc detector is a set of runs per cell, without labels
(see ``PosthocDetector.observe_cell``). Inferring orientation from that population is
legitimate; importing this module is not.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from rhob.detectors.posthoc import RunData

#: Domain-separation tag mixed into the digest. Not a secret (see the module docstring)
#: -- it exists so this mapping cannot collide with any other hash of a family name in
#: the repo, and so the published signs are stable under changes elsewhere.
_PERSONALIZATION = b"rhob-behav-sgn"


def behav_sign(family: str, layout_seed: int = 0) -> float:
    """The evaluation-time orientation of ``family``'s behavioral axis at this layout.

    ``+1.0`` leaves the family's authored coordinate alone; ``-1.0`` flips it. Both
    variants of a pair always get the same value -- see the module docstring for why a
    per-variant flip would be a different (and wrong) operation.

    Deterministic and process-independent: a digest, not :func:`hash`, whose per-process
    randomization for ``str`` would make the benchmark unreproducible across runs.
    """
    payload = f"{family}\x00{int(layout_seed)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8, person=_PERSONALIZATION).digest()
    return -1.0 if digest[-1] & 1 else 1.0


def apply_behav_sign(run: RunData, sign: float) -> RunData:
    """Return ``run`` with its behavioral trace expressed in the flipped coordinate.

    A copy: the caller's ``RunData`` is never mutated, so a cached un-randomized rollout
    stays un-randomized. A run whose ``behav_trace`` is ``None`` -- a family that emits
    no L2 channel at all, scored N/A by :func:`~rhob.v3.benchmark.missing_channels` --
    is returned unchanged, since there is no axis to flip.
    """
    if run.behav_trace is None:
        return run
    return replace(run, behav_trace=np.asarray(run.behav_trace, dtype=float) * sign)
