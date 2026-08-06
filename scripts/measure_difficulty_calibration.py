#!/usr/bin/env python
"""Measure the *achieved* L2-AUROC each family reaches at each *requested* difficulty.

Why this exists
---------------
``rhob.v3.base_family.BaseFamily`` documents its ``difficulty`` argument as "the *target*
behavioral separability (L2-AUROC)" (``src/rhob/v3/base_family.py:3-6``), and
``MatchedPair.difficulty`` repeats the claim (``src/rhob/v3/base_pair.py:31``). Every
tier label the benchmark publishes -- ``TRIVIAL = 0.98``, ``EASY = 0.90``, ... down to
``EXTREME = 0.60`` (``src/rhob/v3/taxonomy.py:44-48``) -- is a number on that scale.

Nothing in the repo has ever measured whether a family reaches the L2 it was asked for.
The admission gate records a ``mean_l2_auroc`` per cell, but only compares it to a fixed
floor of 0.60 (``admission_gate._check_behavioral_separated``); it never compares it to
the difficulty that cell was generated at. So the difficulty axis has been a *label*, not
a measurement, for the whole life of the benchmark.

This script closes that gap and nothing else. It rolls out every registered family at
every difficulty in ``default_difficulties()``, scores the same L2 detector the gate
scores, and publishes achieved-vs-requested. It deliberately does **not** plot a
dose-response curve: the reason a dose-response figure is not in this repo is precisely
what this measurement establishes, and drawing the curve first would have presented a
construction artifact as an empirical law.

What is measured, exactly
-------------------------
The design mirrors ``AdmissionGate.certify`` line for line, so the number here is
comparable to the one the admission ledger already publishes rather than being a new
statistic with its own conventions:

- ``n_layouts`` independent layouts per cell, each from its own ``generate_pair_at``
  seed, because a single layout can separate by chance while the family does not on
  average (``admission_gate.py:863-866``).
- ``seeds_per_layout`` rollout seeds per variant per layout, defaulting to the gate's own
  power-derived floor (:func:`~rhob.v3.admission_gate.required_seeds_per_layout`, 24 at
  the shipped margins).
- The layout seeds and rollout seed bases are drawn from ``np.random.default_rng`` at the
  gate's fixed root seed ``admission_gate._RNG_SEED``, in the gate's order (layout seed,
  then seed base, per layout). At ``n_layouts=12`` that reproduces the gate's draw
  exactly, so at the default design this script's ``achieved_l2_auroc`` must equal the
  ledger's ``behavioral_separated.mean_l2_auroc`` for the same cell. The
  ``gate_crosscheck`` block in the artifact checks that against
  ``admission/admission_ledger.json`` rather than asserting it.
- :class:`~rhob.detectors.l2_behavioral_threshold.BehavioralThresholdDetector`, scored
  through ``classify`` with **no** ``observe_cell`` call -- again the gate's behavior, so
  the detector keeps orientation ``+1`` and reads the family's authored coordinate.
- ``randomize_sign=False``. This measures a *family's* separability, not a detector's
  ability to orient itself in the benchmark's randomized coordinate, which is the same
  reason the gate certifies un-randomized (``admission_gate.py:887-893``: "letting it in
  here would make ``behavioral_separated`` measure BehavioralThresholdDetector's ability
  to orient itself instead of the family's separability"). The two are not
  interchangeable and mixing them would make the achieved-vs-requested error unreadable.
- Per-layout AUROC, then the unweighted mean across layouts -- ``_check_behavioral_separated``'s
  aggregation, NaN layouts dropped.

The single most diagnostic number is the **span**: ``max - min`` of a family's achieved
separation across its own tiers. A span of 0.000 means the difficulty argument moved
nothing a behavioral detector can see.

Two spans are reported, and only one of them answers the question:

``auroc_span``
    over the raw achieved AUROC.
``separation_span``
    over ``|AUROC - 0.5|``. This is the one to read. ``|AUROC - 0.5|`` is exactly the
    quantity invariant under the benchmark's behavioral sign flip
    (:mod:`rhob.v3.sign_randomization`), so it is the same number whether the evaluation
    randomizes the coordinate or not, and it is what a dose-response sweep would need to
    move. At ``randomize_sign=False`` the two coincide for every family that separates in
    its authored direction; they are both reported so that a family which does not is
    visible rather than silently averaged.

Why the knob fails, per family
------------------------------
A table of spans is not a finding, so every family is classified by *mechanism*, and the
classification is measured rather than asserted wherever it can be.

The load-bearing measurement is the **channel probe**. Layout 0 of a cell is generated at
the same layout seed and rolled out at the same seed base for every difficulty (the RNG
is re-seeded per cell, so the draw does not depend on the difficulty). Its runs are
therefore directly comparable across tiers, and this script hashes three channels of them
-- ``behav_trace``, ``true_rewards``, ``proxy_rewards``. If a family's ``behav_trace``
hash is identical at every tier, then the difficulty argument provably changed nothing an
L2 detector can read, whatever the source looks like. That is a proof of decoupling, not
an inference from a flat curve, and it separates the two ways of having a dead knob:

``structurally_decoupled``
    ``behav_trace`` is bit-identical across tiers. The knob went somewhere else (the
    ``true_rewards`` / ``proxy_rewards`` hashes say where) or nowhere at all. No amount of
    recalibration fixes this without changing the family.
``saturated``
    ``behav_trace`` does move with difficulty, but the achieved separation is pinned at
    the ceiling anyway: the knob is real and the operating range is wrong.

Two more classes are assigned from the source and from the tier count:

``closed_form_calibrated``
    the family inverts a closed-form separability law. Detected mechanically, by the
    presence of ``scipy.special.ndtri`` in the family's module -- exactly
    ``continuous_camping.py:15,38-41`` and ``sensor_calibration_tampering.py:25,38-41``,
    both of which invert ``L2 = Phi(d / (sqrt(2) sigma_a))``. These are the only two of
    the 33 that do; the other 31 map difficulty by a linear lerp
    ``t = (difficulty - lo) / (hi - lo)`` onto an internal parameter with no measured
    relationship to L2 at all.
``single_tier``
    ``default_difficulties()`` has one entry, so a span is not defined. Only
    ``gridworld_camping`` (a fixed L2=1.0 anchor, ``gridworld_camping.py:39-43``).

Everything else with a moving span is ``responsive``, and a family whose span is flat but
*not* at the ceiling is ``flat_off_ceiling`` -- a fourth failure mode that would otherwise
be mislabelled as saturation.

Cost, resumability and partial artifacts
----------------------------------------
The full grid is 123 cells over 33 families, and the per-cell cost ranges over three
orders of magnitude: the cheap pure-numpy families take a second or two at the default
12 x 24 design, while one MuJoCo pair takes ~81s to *construct* before a single rollout
happens (12 layouts x 3 tiers = 36 constructions per family). So:

- the artifact is rewritten after **every family**, and an interrupted run leaves a
  correctly-labelled partial artifact rather than nothing;
- a re-run **resumes**: families already present in the artifact are skipped unless
  ``--force``, so the expensive groups can be added incrementally in separate processes;
- ``--families`` / ``--difficulties`` / ``--n-layouts`` / ``--seeds`` scope and cheapen a
  run, and the design actually used is recorded in the artifact. Family blocks measured
  under a *different* design are dropped rather than tabulated alongside these ones, both
  on resume and on ``--merge-from``: two designs in one table is not one measurement;
- every family that was not measured is listed by name under ``families_not_measured``,
  the Markdown leads with the fact that it is partial, and ``--note`` records *why* a group
  was left out. **A partial artifact that says it is partial is fine; a complete-looking
  one that silently omits families is not.**

Every family's ``summary`` block -- spans, channel response, classification -- is a pure
function of its measured tiers and is recomputed on every write, including
``--render-only``. The expensive half (the rollouts) is cached across runs; the cheap half
(the reading of them) never is, so an artifact assembled over several days cannot end up
with some rows classified by the current rules and some by last week's.

Usage
-----
::

    # the cheap groups, at the gate's own design
    python scripts/measure_difficulty_calibration.py --families tabular,continuous

    # add another group later, into the same artifact (already-measured families are
    # skipped, so this resumes rather than repeats)
    python scripts/measure_difficulty_calibration.py --families sequence

    # groups run as separate concurrent processes, then folded into one artifact
    python scripts/measure_difficulty_calibration.py --families rlhf --out-dir /tmp/rlhf
    python scripts/measure_difficulty_calibration.py --merge-from /tmp/rlhf/difficulty_calibration.json \
        --render-only

    # re-render the Markdown from the JSON without measuring anything
    python scripts/measure_difficulty_calibration.py --render-only

``--families`` accepts family names, the group aliases below, or ``all``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Run against *this* checkout's sources, not whatever `rhob` happens to be installed --
# in a worktree the editable install resolves to a different tree entirely, and an
# artifact that records this commit's SHA has to have been produced by this commit's code.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import rhob.v3.families  # noqa: F401,E402  (import for registration side effects)
from rhob.detectors.l2_behavioral_threshold import BehavioralThresholdDetector  # noqa: E402

# ``_RNG_SEED`` and ``_safe_auroc`` are imported rather than re-derived on purpose. This
# script's entire claim is that its number *is* the gate's number, and a re-implementation
# of either -- a second root seed, or a second AUROC wrapper with its own degenerate-label
# handling -- would silently make it a different statistic that happens to be nearby. They
# are private to the gate because nothing else should be re-deriving them; this is the one
# caller whose correctness depends on being byte-identical to it.
from rhob.v3.admission_gate import (  # noqa: E402
    _RNG_SEED,
    _safe_auroc,
    required_seeds_per_layout,
)
from rhob.v3.base_family import BaseFamily  # noqa: E402
from rhob.v3.provenance import provenance_block  # noqa: E402
from rhob.v3.registry import FamilyRegistry  # noqa: E402

DEFAULT_OUT_DIR = _REPO_ROOT / "results" / "difficulty_calibration"

#: Layouts per cell. The gate's default, so the draw and therefore the statistic match.
DEFAULT_N_LAYOUTS = 12

#: Rollout seeds per variant per layout. The gate's power-derived floor (24 at the
#: shipped margins) rather than a number chosen here, for the same comparability reason.
DEFAULT_SEEDS_PER_LAYOUT = required_seeds_per_layout()

#: Name prefixes of the four expensive family groups, in the order they should be run
#: last. Used only by the ``--families`` group aliases; nothing about the measurement
#: depends on the grouping.
_GROUPS: dict[str, Callable[[str], bool]] = {
    "mujoco": lambda n: n.startswith("mujoco_"),
    "pettingzoo": lambda n: n.startswith("pettingzoo_"),
    "rlhf": lambda n: n.startswith("rlhf_"),
    "sequence": lambda n: n.startswith("sequence_"),
}
_GROUPS["cheap"] = lambda n: not any(f(n) for k, f in _GROUPS.items() if k != "cheap")

#: Widest separation span a family can show and still be called flat. Not a
#: measurement-noise threshold in disguise: at the default 12 x 24 design the achieved
#: separations measured here are either **exactly** 0.000 apart across a family's tiers
#: (the byte-identical case) or apart by far more than this, so no family's class turns
#: on the exact value. It exists so that a future family separated by float noise is not
#: reported as a live knob.
SPAN_TOLERANCE = 0.01

#: A family is "at the ceiling" when its mean achieved separation ``|AUROC - 0.5|`` is
#: within this of the maximum possible 0.5, i.e. AUROC within 0.01 of 0.0 or 1.0. Used
#: only to split a flat family into ``saturated`` (pinned at the ceiling, so the knob has
#: no headroom) and ``flat_off_ceiling`` (flat somewhere in the middle, which is a
#: different bug and must not be reported as saturation).
CEILING_TOLERANCE = 0.01

#: Where each structurally decoupled family's difficulty knob actually goes, read out of
#: the family source. The *classification* is a measurement (identical ``behav_trace``
#: digests across tiers, see :func:`_channel_hash`) and does not depend on this table; what
#: this adds is the reason, at file:line, so a maintainer can go and fix the family instead
#: of only knowing that it is broken. Every entry below was read at the cited lines.
#:
#: A family measured as decoupled and missing from this table is reported as
#: "source not audited" rather than being quietly given one of these explanations.
_KNOB_WIRING: dict[str, str] = {
    "distributional_shift": (
        "`difficulty` -> `shift_magnitude` (distributional_shift.py:68-69), which appears "
        "only in the true-reward line of each variant (`:104`, `:149`). The behavioural "
        "channel is a hardcoded constant in both: `+1.0/-1.0` by visited state for the "
        "hacking variant (`:115-116`) and a literal `0.0` for the legitimate one "
        "(`:157-158`), at every difficulty"
    ),
    "physics_exploitation": (
        "`difficulty` -> `crash_penalty` (physics_exploitation.py:110-111), used only in "
        "the true reward (`:126`). The behavioural channel is `signed_impact`, a function "
        "of the simulated arc and the brake flag alone (`:137-138`)"
    ),
    "orbit_chirality": (
        "`difficulty` -> `chirality_strength` (orbit_chirality.py:134-135), which is passed "
        "to `_run_orbit_episode` (`:139`, `:151`) and **never read in its body**: the name "
        "appears in the signature (`:26`) and the docstring (`:35`) and nowhere else in the "
        "function. The knob is a dead parameter -- it reaches no channel at all, which is "
        "why the true-reward and proxy digests are identical across tiers too"
    ),
}

#: Channels hashed per cell for the structural-decoupling probe. ``behav_trace`` is the
#: one the classification turns on -- it is the only channel an L2 detector reads
#: (:func:`rhob.v3.access.restrict` nulls it below L2 and exposes nothing else at it) --
#: and the other two are recorded to answer *where the knob went* when it did not go
#: there.
_PROBE_CHANNELS = ("behav_trace", "true_rewards", "proxy_rewards")


def _channel_hash(runs_a: list, runs_b: list, channel: str) -> str | None:
    """SHA-256 over one channel of a layout's rolled-out runs, both variants in order.

    Returns ``None`` when the channel is absent on every run (``behav_trace`` is optional
    on :class:`~rhob.detectors.posthoc.RunData`), which is a different statement from "the
    channel is constant" and must not be collapsed into one.

    Hashed rather than stored because the comparison this feeds is exact equality across
    difficulties and the arrays are ~200 floats per run x 576 runs per cell. A hash
    collision would have to be adversarial, and nothing here is adversarial; what matters
    is that the digest is published, so the claim "these two tiers produced byte-identical
    behavior" is checkable from the artifact.
    """
    digest = hashlib.sha256()
    seen = False
    for run in runs_a + runs_b:
        values = getattr(run, channel, None)
        if values is None:
            digest.update(b"<none>")
            continue
        seen = True
        digest.update(np.ascontiguousarray(np.asarray(values, dtype=float)).tobytes())
    return digest.hexdigest() if seen else None


def measure_cell(
    family: BaseFamily,
    difficulty: float,
    n_layouts: int,
    seeds_per_layout: int,
) -> dict[str, Any]:
    """Achieved L2-AUROC for one (family, difficulty), by the admission gate's design.

    The RNG is re-seeded from ``admission_gate._RNG_SEED`` for every cell, exactly as
    ``AdmissionGate.certify`` does, and the two integers per layout are drawn in the gate's
    order (layout seed, then rollout seed base). That is what makes layout ``i`` of this
    cell the same layout as layout ``i`` of the same family's *other* difficulties -- the
    property the channel probe depends on -- and what makes the result reproduce the
    ledger's ``mean_l2_auroc`` at ``n_layouts=12``.
    """
    rng = np.random.default_rng(_RNG_SEED)
    detector = BehavioralThresholdDetector()

    layout_aurocs: list[float] = []
    layout_seeds: list[int] = []
    seed_bases: list[int] = []
    probe: dict[str, str | None] = {}

    for layout_idx in range(n_layouts):
        layout_seed = int(rng.integers(0, 1_000_000))
        pair = family.generate_pair_at(difficulty, seed=layout_seed)
        seed_base = int(rng.integers(0, 1_000_000))
        # Un-randomized, and no `observe_cell`: the gate's configuration. See the module
        # docstring for why the randomized coordinate would answer a different question.
        runs_a, runs_b, _ = pair.rollout(
            seeds_per_layout, seed_base=seed_base, randomize_sign=False
        )
        labels = [1] * len(runs_a) + [0] * len(runs_b)
        scores = [detector.classify(r) for r in runs_a + runs_b]
        layout_aurocs.append(_safe_auroc(labels, scores))
        layout_seeds.append(layout_seed)
        seed_bases.append(seed_base)
        if layout_idx == 0:
            probe = {c: _channel_hash(runs_a, runs_b, c) for c in _PROBE_CHANNELS}

    valid = [a for a in layout_aurocs if a == a]
    achieved = float(np.mean(valid)) if valid else float("nan")
    return {
        "requested_difficulty": round(float(difficulty), 4),
        "achieved_l2_auroc": None if achieved != achieved else round(achieved, 6),
        "error_achieved_minus_requested": (
            None if achieved != achieved else round(achieved - float(difficulty), 6)
        ),
        "achieved_separation": (
            None if achieved != achieved else round(abs(achieved - 0.5), 6)
        ),
        "n_layouts_scored": len(valid),
        "sd_across_layouts": (
            None if len(valid) < 2 else round(float(np.std(valid, ddof=1)), 6)
        ),
        "min_layout_auroc": None if not valid else round(float(min(valid)), 6),
        "max_layout_auroc": None if not valid else round(float(max(valid)), 6),
        "per_layout_auroc": [None if a != a else round(float(a), 6) for a in layout_aurocs],
        "layout_seeds": layout_seeds,
        "rollout_seed_bases": seed_bases,
        "layout_0_channel_sha256": probe,
    }


def _uses_closed_form(family_name: str) -> str | None:
    """``"path:line,line"`` if this family's module inverts a closed-form L2 law, else None.

    Detected from the source rather than from a hand-maintained list, so a family that
    gains or loses a closed-form inversion cannot drift out of this classification: the
    marker is an import of ``scipy.special.ndtri``, which is the inverse normal CDF and has
    no other use anywhere in ``src/rhob/v3/families``. Verified over the whole tree: the
    only two hits are ``continuous_camping`` and ``sensor_calibration_tampering``, each of
    which uses it in exactly one function, ``difficulty_to_separation``, to invert
    ``L2 = Phi(d / (sqrt(2) sigma_a))``.
    """
    path = _REPO_ROOT / "src" / "rhob" / "v3" / "families" / f"{family_name}.py"
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    hits = [i for i, line in enumerate(lines, 1) if "ndtri" in line]
    if not hits:
        return None
    rel = path.relative_to(_REPO_ROOT).as_posix()
    return f"{rel}:{','.join(str(h) for h in hits)}"


def summarize_family(name: str, tiers: list[dict[str, Any]]) -> dict[str, Any]:
    """Spans, channel response and mechanism class for one family's measured tiers."""
    aurocs = [t["achieved_l2_auroc"] for t in tiers if t["achieved_l2_auroc"] is not None]
    seps = [t["achieved_separation"] for t in tiers if t["achieved_separation"] is not None]
    errors = [
        t["error_achieved_minus_requested"]
        for t in tiers
        if t["error_achieved_minus_requested"] is not None
    ]

    # A channel "varies with difficulty" iff layout 0 -- the same layout seed and the same
    # rollout seed base at every tier -- produced a different byte pattern at some tier.
    # `None` (channel absent) is one distinct value like any other; a family with no
    # behav_trace at all reports `False` here and is caught by the NaN AUROC path instead.
    channels: dict[str, bool | None] = {}
    for channel in _PROBE_CHANNELS:
        digests = {t["layout_0_channel_sha256"].get(channel) for t in tiers}
        channels[channel] = None if len(tiers) < 2 else (len(digests) > 1)

    auroc_span = round(max(aurocs) - min(aurocs), 6) if len(aurocs) > 1 else None
    separation_span = round(max(seps) - min(seps), 6) if len(seps) > 1 else None
    mean_separation = float(np.mean(seps)) if seps else float("nan")
    closed_form = _uses_closed_form(name)

    if len(tiers) < 2:
        klass, why = "single_tier", (
            "default_difficulties() has one entry, so a span is not defined; this family "
            "is a fixed anchor rather than a swept axis"
        )
    elif closed_form:
        # The class is about *how the family maps difficulty*, not about whether the map
        # came out right -- those are separate findings and collapsing them would hide the
        # more interesting one. A closed-form family can still be pinned at the ceiling,
        # and when it is, the reason says so with the measured numbers rather than letting
        # the flattering class name stand alone.
        klass = "closed_form_calibrated"
        why = (
            f"inverts a closed-form separability law ({closed_form}) rather than lerping "
            "an unrelated internal parameter"
        )
        if separation_span is not None and separation_span <= SPAN_TOLERANCE:
            why += (
                f" -- but the achieved separation still spans only {separation_span:.4f} "
                f"across its tiers at a mean |AUROC - 0.5| of {mean_separation:.4f}, so "
                "the law it inverts is not the statistic that gets scored"
            )
    elif channels["behav_trace"] is False:
        moved = [c for c in ("true_rewards", "proxy_rewards") if channels[c]]
        klass = "structurally_decoupled"
        why = (
            "the difficulty argument leaves behav_trace bit-identical at every tier "
            "(same layout seed, same rollout seeds), so no L2 detector can see it; it "
            + (f"moves {' and '.join(moved)} instead" if moved else "moves no channel at all")
        )
        why += ". Source: " + _KNOB_WIRING.get(
            name,
            "not audited -- the decoupling is measured, but nobody has traced the knob "
            "through this family's source to say where it goes",
        )
    elif separation_span is not None and separation_span <= SPAN_TOLERANCE:
        at_ceiling = mean_separation >= 0.5 - CEILING_TOLERANCE
        klass = "saturated" if at_ceiling else "flat_off_ceiling"
        why = (
            "behav_trace does move with difficulty, but the achieved separation is pinned "
            f"at the ceiling (mean |AUROC - 0.5| = {mean_separation:.4f} of a possible "
            "0.5), so the knob has no headroom to express itself in"
            if at_ceiling
            else (
                "behav_trace moves with difficulty but the achieved separation does not, "
                f"and it is not at the ceiling either (mean |AUROC - 0.5| = "
                f"{mean_separation:.4f}); the knob moves the wrong quantity"
            )
        )
    else:
        klass, why = "responsive", (
            f"achieved separation moves by {separation_span:.4f} across the family's tiers"
        )

    return {
        "n_tiers": len(tiers),
        "auroc_span": auroc_span,
        "separation_span": separation_span,
        "span_is_exactly_zero": None if separation_span is None else separation_span == 0.0,
        "mean_achieved_separation": (
            None if mean_separation != mean_separation else round(mean_separation, 6)
        ),
        "worst_error": (
            None if not errors else round(max(errors, key=abs), 6)
        ),
        "max_abs_error": None if not errors else round(max(abs(e) for e in errors), 6),
        "channel_varies_with_difficulty": channels,
        "closed_form_source": closed_form,
        "classification": klass,
        "classification_reason": why,
    }


def resolve_families(spec: str) -> list[str]:
    """``all``, a group alias, or a comma-separated list of names -> family names."""
    registered = FamilyRegistry.list_families()
    if spec == "all":
        return registered
    names: list[str] = []
    for token in (t.strip() for t in spec.split(",")):
        if not token:
            continue
        if token in _GROUPS:
            names.extend(n for n in registered if _GROUPS[token](n))
        else:
            FamilyRegistry.get(token)  # fail fast on a typo
            names.append(token)
    # Deduplicate while keeping the registry's stable order, so `--families cheap,mujoco`
    # and `--families mujoco,cheap` measure the same set in the same order.
    wanted = set(names)
    return [n for n in registered if n in wanted]


def gate_crosscheck(families: dict[str, Any], design: dict[str, Any]) -> dict[str, Any]:
    """Does this harness reproduce the admission ledger's own L2 number?

    At ``n_layouts=12`` and the gate's seed floor the draw here is the gate's draw, so
    every cell both artifacts contain must agree to the last digit. This is the check that
    makes the achieved-vs-requested table a re-reading of the gate's measurement rather
    than a second, differently-conventioned statistic that happens to be nearby.

    A mismatch is reported, never raised: the ledger is a committed artifact from an
    earlier commit and a family may legitimately have changed under it. The size of any
    disagreement is what the reader needs, not a crash.
    """
    ledger_path = _REPO_ROOT / "admission" / "admission_ledger.json"
    if not ledger_path.exists():
        return {"checked": False, "reason": f"{ledger_path.name} not found"}
    if (
        design["n_layouts"] != DEFAULT_N_LAYOUTS
        or design["seeds_per_layout"] != DEFAULT_SEEDS_PER_LAYOUT
    ):
        return {
            "checked": False,
            "reason": (
                "this run's design is not the gate's "
                f"({design['n_layouts']} x {design['seeds_per_layout']} vs "
                f"{DEFAULT_N_LAYOUTS} x {DEFAULT_SEEDS_PER_LAYOUT}), so the draws differ "
                "and the cells are not comparable"
            ),
        }
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger_cells = {
        (row["family"], round(float(row["difficulty"]), 4)): row["metrics"]
        .get("behavioral_separated", {})
        .get("mean_l2_auroc")
        for row in ledger["results"]
    }
    compared, exact, diffs = 0, 0, []
    for name, block in families.items():
        for tier in block.get("tiers", []):
            key = (name, tier["requested_difficulty"])
            theirs = ledger_cells.get(key)
            mine = tier["achieved_l2_auroc"]
            if theirs is None or mine is None:
                continue
            compared += 1
            # The ledger stores the raw float and this artifact stores it rounded to 6
            # decimals, so the ledger value is rounded the same way before differencing.
            # Otherwise every cell would show a spurious ~1e-7 disagreement that is this
            # script's own output formatting rather than a difference in the measurement.
            delta = round(mine - round(float(theirs), 6), 9)
            if delta == 0.0:
                exact += 1
            else:
                diffs.append({"family": name, "difficulty": key[1], "delta": delta})
    return {
        "checked": True,
        "ledger": ledger_path.relative_to(_REPO_ROOT).as_posix(),
        "ledger_commit": ledger["provenance"]["git"]["short_commit"],
        "n_cells_compared": compared,
        "n_exact": exact,
        "max_abs_delta": round(max((abs(d["delta"]) for d in diffs), default=0.0), 9),
        "differing_cells": diffs[:20],
    }


def build_summary(families: dict[str, Any], measured: list[str]) -> dict[str, Any]:
    """The headline counts: what the Markdown has to lead with and a test has to pin."""
    ok = {n: b for n, b in families.items() if b.get("status") == "measured"}
    multi = {n: b for n, b in ok.items() if b["summary"]["n_tiers"] > 1}
    zero = [n for n, b in multi.items() if b["summary"]["span_is_exactly_zero"]]
    by_class: dict[str, list[str]] = {}
    for name, block in ok.items():
        by_class.setdefault(block["summary"]["classification"], []).append(name)

    errors = [
        (n, b["summary"]["worst_error"])
        for n, b in ok.items()
        if b["summary"]["worst_error"] is not None
    ]
    worst = max(errors, key=lambda kv: abs(kv[1]), default=(None, None))
    all_cells = [
        (n, t["requested_difficulty"], t["error_achieved_minus_requested"])
        for n, b in ok.items()
        for t in b["tiers"]
        if t["error_achieved_minus_requested"] is not None
    ]
    worst_cell = max(all_cells, key=lambda c: abs(c[2]), default=(None, None, None))
    registered = FamilyRegistry.list_families()
    return {
        "n_families_registered": len(registered),
        "n_families_measured": len(ok),
        "n_families_errored": sum(1 for b in families.values() if b.get("status") == "error"),
        "n_cells_measured": sum(len(b["tiers"]) for b in ok.values()),
        "n_families_with_more_than_one_tier": len(multi),
        "n_families_separation_span_exactly_zero": len(zero),
        "families_separation_span_exactly_zero": sorted(zero),
        "n_by_classification": {k: len(v) for k, v in sorted(by_class.items())},
        "families_by_classification": {k: sorted(v) for k, v in sorted(by_class.items())},
        "worst_family_error": {"family": worst[0], "error": worst[1]},
        "worst_cell_error": {
            "family": worst_cell[0],
            "requested_difficulty": worst_cell[1],
            "error": worst_cell[2],
        },
        "families_not_measured": sorted(set(registered) - set(measured)),
        "complete": sorted(measured) == registered,
    }


def _plural(n: int, one: str, many: str) -> str:
    """Pick the singular or plural phrasing. The counts here are routinely 0, 1 or 30."""
    return one if n == 1 else many


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return format(float(value), spec)


#: One-line gloss per class, for the Markdown legend. Kept next to the classifier so a new
#: class cannot be introduced without a reader-facing explanation of it.
_CLASS_GLOSS = {
    "closed_form_calibrated": (
        "the family inverts a closed-form separability law rather than lerping an "
        "unrelated internal parameter"
    ),
    "structurally_decoupled": (
        "the difficulty argument leaves `behav_trace` **bit-identical** across every "
        "tier, so no L2 detector can respond to it at all"
    ),
    "saturated": (
        "`behav_trace` does move with difficulty, but the achieved separation is pinned "
        "at the ceiling, so the knob has no headroom"
    ),
    "flat_off_ceiling": (
        "`behav_trace` moves but the achieved separation does not, and it is not at the "
        "ceiling either -- the knob moves the wrong quantity"
    ),
    "responsive": "the achieved separation actually tracks the requested difficulty",
    "single_tier": "a fixed anchor with one difficulty; a span is undefined",
}


def render_markdown(payload: dict[str, Any]) -> str:
    """The committed report. Leads with the finding; the method comes after it."""
    s = payload["summary"]
    design = payload["design"]
    families = payload["families"]
    out: list[str] = []

    out.append("# RHOB difficulty calibration: requested vs achieved L2-AUROC")
    out.append("")
    out.append(
        "`BaseFamily.generate_pair(difficulty=...)` documents its argument as the "
        "**target behavioral separability (L2-AUROC)** "
        "(`src/rhob/v3/base_family.py:3-6`), and every named tier the benchmark "
        "publishes -- `TRIVIAL = 0.98` down to `EXTREME = 0.60` "
        "(`src/rhob/v3/taxonomy.py:44-48`) -- is a number on that scale. Until this "
        "artifact, nothing in the repo had ever measured whether a family reaches the L2 "
        "it was asked for. This is that measurement, and nothing else: there is "
        "deliberately no dose-response curve here, because what follows is the reason "
        "there cannot be one yet."
    )
    out.append("")

    if not s["complete"]:
        out.append(
            f"> **PARTIAL.** {s['n_families_measured']} of {s['n_families_registered']} "
            "registered families measured. Families absent from the tables below are "
            "**unmeasured**, which is not the same as calibrated. The unmeasured set is "
            "listed under [Not measured](#not-measured)."
        )
        out.append("")

    zero = s["n_families_separation_span_exactly_zero"]
    multi = s["n_families_with_more_than_one_tier"]
    counts = s["n_by_classification"]
    worst_cell = s["worst_cell_error"]
    n_decoupled = counts.get("structurally_decoupled", 0)
    n_saturated = counts.get("saturated", 0)
    out.append("## The finding")
    out.append("")
    out.append(
        f"- **{zero} of the {multi} measured families with more than one tier have a "
        f"separation span of exactly 0.000.** Sweeping their difficulty argument from one "
        "end of their advertised range to the other changes the achieved L2-AUROC by "
        "nothing whatsoever. The difficulty axis is a label on those families, not a knob."
    )
    out.append(
        f"- **{counts.get('closed_form_calibrated', 0)} of the "
        f"{s['n_families_measured']} measured families invert a closed-form separability "
        "law.** Every other one maps difficulty by an uncalibrated linear interpolation "
        "`t = (difficulty - lo) / (hi - lo)` onto an internal parameter whose relationship "
        "to L2 had never been measured -- until this table."
    )
    if worst_cell["family"]:
        out.append(
            f"- **The worst nominal-vs-achieved error is "
            f"{worst_cell['error']:+.4f}**, at `{worst_cell['family']}` requested "
            f"{worst_cell['requested_difficulty']:.2f}. A cell labelled "
            f"`{worst_cell['requested_difficulty']:.2f}` measures "
            f"{worst_cell['requested_difficulty'] + worst_cell['error']:.4f}."
        )
    out.append(
        f"- **{n_decoupled} {_plural(n_decoupled, 'family is', 'families are')} "
        "structurally decoupled**: the difficulty argument leaves the behavioral channel "
        "*bit-identical* across every tier -- measured, by hashing layout 0's "
        "`behav_trace` at the same layout seed and the same rollout seeds at each "
        "difficulty. Recalibration cannot fix these; the knob is wired to something an L2 "
        "detector cannot see, or to nothing at all."
    )
    out.append(
        f"- **{n_saturated} {_plural(n_saturated, 'family is', 'families are')} "
        "saturated**: the knob does move the behavioral channel, and the achieved "
        "separation is pinned at the ceiling anyway."
    )
    out.append("")
    out.append(
        "The consequence for the benchmark: **the difficulty axis cannot currently carry "
        "a dose-response result.** Plotting detector AUROC against nominal difficulty "
        "today would draw a flat line against a mislabelled x-axis and present a "
        "construction artifact as an empirical law."
    )
    out.append("")

    out.append("## Classification")
    out.append("")
    out.append("| class | families | what it means |")
    out.append("|---|---|---|")
    for klass, names in s["families_by_classification"].items():
        out.append(
            f"| `{klass}` | {len(names)} | {_CLASS_GLOSS.get(klass, '-')} |"
        )
    out.append("")
    out.append(
        "`structurally_decoupled` vs `saturated` is a measurement, not a judgement: it is "
        "decided by whether layout 0's `behav_trace` SHA-256 changes across tiers. Layout "
        "0 is generated at the same layout seed and rolled out at the same seed base at "
        "every difficulty (the RNG is re-seeded per cell), so an identical digest is proof "
        "that the difficulty argument changed nothing an L2 detector reads. The digests "
        "are in the JSON."
    )
    out.append("")

    out.append("## Design")
    out.append("")
    out.append(
        "Mirrors `AdmissionGate.certify` so the achieved number is the same statistic the "
        "admission ledger already publishes, not a new one:"
    )
    out.append("")
    out.append(f"- **{design['n_layouts']} independent layouts** per cell x "
               f"**{design['seeds_per_layout']} rollout seeds per variant per layout** "
               f"({design['n_layouts'] * design['seeds_per_layout']} runs per side per "
               "cell).")
    out.append(f"- Detector: **{design['detector']}**, scored through `classify` with no "
               "`observe_cell` call -- the gate's configuration, so the detector keeps "
               "orientation `+1` and reads the family's authored coordinate.")
    out.append("- `randomize_sign=False`. This measures the **family's** separability, "
               "not a detector's ability to orient itself in the benchmark's randomized "
               "coordinate. It is the same reason the gate certifies un-randomized "
               "(`src/rhob/v3/admission_gate.py:887-893`).")
    out.append("- Aggregation: AUROC per layout, then the unweighted mean across layouts, "
               "NaN layouts dropped -- `admission_gate._check_behavioral_separated`.")
    out.append(f"- Layout seeds and rollout seed bases drawn from "
               f"`np.random.default_rng({design['rng_seed']})` per cell, in the gate's "
               "order. Re-running on this commit reproduces every number below except the "
               "wall clock.")
    out.append("")
    out.append(
        "`separation span` is `max - min` of `|AUROC - 0.5|` across a family's tiers, and "
        "it is **the number to read**. `|AUROC - 0.5|` is invariant under the benchmark's "
        "behavioral sign flip (`rhob.v3.sign_randomization`), so it is the same quantity "
        "whether the evaluation randomizes the coordinate or not, and it is what a "
        "dose-response sweep would have to move. The raw `AUROC span` is reported beside "
        "it so that a family separating against its authored direction is visible."
    )
    out.append("")

    check = payload.get("gate_crosscheck") or {}
    out.append("### Cross-check against the admission ledger")
    out.append("")
    if not check.get("checked"):
        out.append(f"Not checked: {check.get('reason', 'n/a')}.")
    else:
        out.append(
            f"At this design the draw is the gate's draw, so every cell present in both "
            f"artifacts must agree exactly. Against `{check['ledger']}` (commit "
            f"`{check['ledger_commit']}`): **{check['n_exact']} of "
            f"{check['n_cells_compared']} cells identical**, largest disagreement "
            f"{check['max_abs_delta']:.2e}."
        )
        if check["differing_cells"]:
            out.append("")
            out.append("| family | difficulty | this run - ledger |")
            out.append("|---|---|---|")
            for row in check["differing_cells"]:
                out.append(
                    f"| {row['family']} | {row['difficulty']:.2f} | {row['delta']:+.6f} |"
                )
    out.append("")

    out.append("## Per family")
    out.append("")
    header = [
        "family", "class", "tiers", "achieved range", "AUROC span", "separation span",
        "worst error", "`behav_trace` varies", "seconds",
    ]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for name, block in families.items():
        if block.get("status") != "measured":
            out.append(
                f"| `{name}` | **{block.get('status', '?')}** | "
                + " | ".join(["-"] * (len(header) - 3))
                + f" | {_fmt(block.get('wall_clock_seconds'), '.1f')} |"
            )
            continue
        f = block["summary"]
        aurocs = [t["achieved_l2_auroc"] for t in block["tiers"]
                  if t["achieved_l2_auroc"] is not None]
        rng_txt = (
            f"{min(aurocs):.4f} - {max(aurocs):.4f}" if aurocs else "-"
        )
        out.append(
            f"| `{name}` | `{f['classification']}` | {f['n_tiers']} | {rng_txt} | "
            f"{_fmt(f['auroc_span'])} | **{_fmt(f['separation_span'])}** | "
            f"{_fmt(f['worst_error'], '+.4f')} | "
            f"{_fmt(f['channel_varies_with_difficulty']['behav_trace'])} | "
            f"{_fmt(block.get('wall_clock_seconds'), '.1f')} |"
        )
    out.append("")

    out.append("## Per cell: requested vs achieved")
    out.append("")
    out.append(
        "`error` is `achieved - requested`. A family whose knob worked would show this "
        "column near zero at every tier; a family whose knob is dead shows the requested "
        "column moving and the achieved column standing still."
    )
    out.append("")
    out.append(
        "| family | requested | achieved L2-AUROC | error | separation | SD across layouts |"
    )
    out.append("|---|---|---|---|---|---|")
    for name, block in families.items():
        if block.get("status") != "measured":
            continue
        for tier in block["tiers"]:
            out.append(
                f"| `{name}` | {tier['requested_difficulty']:.2f} | "
                f"{_fmt(tier['achieved_l2_auroc'], '.4f')} | "
                f"{_fmt(tier['error_achieved_minus_requested'], '+.4f')} | "
                f"{_fmt(tier['achieved_separation'], '.4f')} | "
                f"{_fmt(tier['sd_across_layouts'], '.4f')} |"
            )
    out.append("")

    out.append("## Where the knob went, for the families whose L2 cannot see it")
    out.append("")
    out.append(
        "For each structurally decoupled family, which channels layout 0 *did* respond to. "
        "A `no` in the `behav_trace` column is the decoupling; a `yes` under "
        "`true_rewards` or `proxy_rewards` says the knob is wired to the true-reward gap "
        "or to the proxy stream instead, neither of which an L2 detector reads. All three "
        "`no` means the knob is wired to nothing that reaches a rollout at all."
    )
    out.append("")
    out.append("| family | `behav_trace` | `true_rewards` | `proxy_rewards` | reason |")
    out.append("|---|---|---|---|---|")
    for name, block in families.items():
        if block.get("status") != "measured":
            continue
        f = block["summary"]
        if f["classification"] != "structurally_decoupled":
            continue
        ch = f["channel_varies_with_difficulty"]
        out.append(
            f"| `{name}` | {_fmt(ch['behav_trace'])} | {_fmt(ch['true_rewards'])} | "
            f"{_fmt(ch['proxy_rewards'])} | {f['classification_reason']} |"
        )
    out.append("")

    out.append("## Every family's classification, justified")
    out.append("")
    for name, block in families.items():
        if block.get("status") != "measured":
            continue
        f = block["summary"]
        source = f" Source: `{f['closed_form_source']}`." if f["closed_form_source"] else ""
        out.append(f"- **`{name}`** -- `{f['classification']}`: {f['classification_reason']}.{source}")
    out.append("")

    errored = {n: b for n, b in families.items() if b.get("status") == "error"}
    if errored:
        out.append("## Families that raised")
        out.append("")
        out.append(
            "Recorded rather than dropped: a family that could not be measured is "
            "unmeasured, and an artifact that omitted it would read as if it had been "
            "measured and found fine."
        )
        out.append("")
        for name, block in errored.items():
            out.append(f"- `{name}`: `{block['error']}`")
        out.append("")

    out.append("## Not measured")
    out.append("")
    missing = s["families_not_measured"]
    if payload.get("note"):
        out.append(payload["note"])
        out.append("")
    if not missing:
        out.append("None -- every registered family is in this artifact.")
    else:
        out.append(
            f"{len(missing)} of {s['n_families_registered']} registered families are not "
            "in this artifact. **They are unmeasured, not calibrated.** The measurement "
            "is resumable: re-running this script with `--families <name>` adds them to "
            "the same JSON without re-measuring anything already present."
        )
        out.append("")
        for name in missing:
            out.append(f"- `{name}`")
    out.append("")

    prov = payload["provenance"]
    git = prov["git"]
    out.append("## Provenance")
    out.append("")
    out.append(f"- Generated: {prov['generated_utc']}")
    out.append(
        f"- Commit: `{git['commit']}` (branch `{git['branch']}`, working tree "
        f"{'dirty' if git['dirty'] else 'clean'})"
    )
    out.append(f"- Python: {prov['python']} on {prov['platform']}")
    out.append(
        "- Packages: "
        + ", ".join(f"{k} {v if v else 'not installed'}" for k, v in sorted(prov["packages"].items()))
    )
    out.append(f"- Command: `{' '.join(prov['argv'])}`")
    out.append("")
    out.append("### Wall clock")
    out.append("")
    out.append(
        "Machine-dependent, and the only non-reproducible part of this artifact. Recorded "
        "so the cost of measuring the families that are still missing is knowable in "
        "advance rather than discovered."
    )
    out.append("")
    out.append("| family | seconds | cells | seconds/cell |")
    out.append("|---|---|---|---|")
    total = 0.0
    for name, block in families.items():
        secs = block.get("wall_clock_seconds")
        cells = len(block.get("tiers", []))
        if secs is None:
            continue
        total += secs
        per_cell = f"{secs / cells:.1f}" if cells else "-"
        out.append(f"| {name} | {secs:.1f} | {cells} | {per_cell} |")
    out.append(f"| **total** | **{total:.1f}** | **{s['n_cells_measured']}** | |")
    out.append("")
    return "\n".join(out)


def load_payload(json_path: Path) -> dict[str, Any]:
    """An existing artifact, or ``{}``.

    An interrupted run's artifact is a valid input to the next run: this is the resume
    path. A file that cannot be parsed is treated as absent rather than fatal -- the
    artifact is rewritten after every family, so a truncated one is the expected shape of
    "the process was killed mid-write" and must not block the re-run that fixes it.
    """
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_existing(json_path: Path) -> dict[str, Any]:
    """Previously measured families, for the resume and ``--merge-from`` paths."""
    return load_payload(json_path).get("families", {})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--families",
        default="all",
        help="'all', a group alias (%s), or a comma-separated list of family names"
        % ", ".join(sorted(_GROUPS)),
    )
    p.add_argument(
        "--difficulties",
        default=None,
        help="comma-separated difficulties overriding each family's default_difficulties()",
    )
    p.add_argument("--n-layouts", type=int, default=DEFAULT_N_LAYOUTS,
                   help=f"independent layouts per cell (default {DEFAULT_N_LAYOUTS}, the "
                        "gate's; lowering it makes the run cheaper and breaks the "
                        "cross-check against the ledger, which the artifact records)")
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS_PER_LAYOUT,
                   help=f"rollout seeds per variant per layout (default "
                        f"{DEFAULT_SEEDS_PER_LAYOUT}, the gate's power-derived floor)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--merge-from",
        type=Path,
        nargs="+",
        default=None,
        metavar="JSON",
        help="fold the family blocks of other difficulty_calibration.json files into this "
        "artifact before measuring. Family groups differ in cost by three orders of "
        "magnitude, so they are run as separate processes into separate --out-dirs; this "
        "is how those pieces become one artifact. Blocks measured under a different "
        "design are refused, not silently mixed in.",
    )
    p.add_argument(
        "--note",
        default=None,
        help="free text recorded in the artifact and rendered under 'Not measured'. For "
        "saying *why* a family group was left out -- a skip with a reason is a finding, a "
        "skip without one is a gap. Replaces any note already in the artifact.",
    )
    p.add_argument("--force", action="store_true",
                   help="re-measure families already present in the artifact")
    p.add_argument("--render-only", action="store_true",
                   help="re-render the Markdown from the existing JSON, measuring nothing")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "difficulty_calibration.json"
    md_path = args.out_dir / "DIFFICULTY_CALIBRATION.md"

    design = {
        "n_layouts": args.n_layouts,
        "seeds_per_layout": args.seeds,
        "n_runs_per_side_per_cell": args.n_layouts * args.seeds,
        "rng_seed": _RNG_SEED,
        "detector": BehavioralThresholdDetector().name,
        "detector_access_level": BehavioralThresholdDetector().access_level,
        "detector_observe_cell_called": False,
        "randomize_behav_sign": False,
        "aggregation": "unweighted mean of per-layout AUROC, NaN layouts dropped",
        "mirrors": "rhob.v3.admission_gate.AdmissionGate.certify",
        "difficulty_override": args.difficulties,
    }

    previous = load_payload(json_path)
    # A note survives runs that do not restate it: it explains a standing omission, and
    # resuming the measurement for one more family is not a reason to drop the explanation
    # of why the rest are still missing.
    note: str | None = args.note if args.note is not None else previous.get("note")
    families: dict[str, Any] = dict(previous.get("families", {}))
    # A family measured under a different design is not comparable to one measured under
    # this one, so it is re-measured rather than silently mixed into the same table.
    if not args.render_only:
        families = {
            n: b
            for n, b in families.items()
            if b.get("design", {}).get("n_layouts") == args.n_layouts
            and b.get("design", {}).get("seeds_per_layout") == args.seeds
        }

    for source in args.merge_from or []:
        incoming = load_existing(source)
        if not incoming:
            raise SystemExit(f"--merge-from {source}: no family blocks found")
        for name, block in incoming.items():
            got = (block.get("design", {}).get("n_layouts"),
                   block.get("design", {}).get("seeds_per_layout"))
            if got != (args.n_layouts, args.seeds):
                raise SystemExit(
                    f"--merge-from {source}: `{name}` was measured at "
                    f"{got[0]} layouts x {got[1]} seeds, this artifact is "
                    f"{args.n_layouts} x {args.seeds}. Cells measured under different "
                    "designs are not one measurement and must not share a table."
                )
            families[name] = block
        if not args.quiet:
            print(f"  merged {len(incoming)} families from {source}", flush=True)

    def write() -> dict[str, Any]:
        ordered = {n: families[n] for n in FamilyRegistry.list_families() if n in families}
        # Every family's summary -- spans, channel response, classification -- is a pure
        # function of its measured tiers, so it is recomputed on every write rather than
        # carried forward from whenever that family was rolled out. Otherwise a run that
        # resumes an artifact, or merges one, would publish a table where some rows were
        # classified by the current rules and some by whatever the rules were weeks ago,
        # with nothing in the file saying which. The expensive half (the rollouts) is
        # cached; the cheap half (the reading of them) is never stale.
        for name_, block_ in ordered.items():
            if block_.get("status") == "measured":
                block_["summary"] = summarize_family(name_, block_["tiers"])
        measured = [n for n, b in ordered.items() if b.get("status") == "measured"]
        payload = {
            "schema": "rhob.difficulty_calibration/1",
            "provenance": provenance_block(
                script="scripts/measure_difficulty_calibration.py"
            ),
            "design": design,
            "note": note,
            "summary": build_summary(ordered, measured),
            "gate_crosscheck": gate_crosscheck(ordered, design),
            "families": ordered,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(render_markdown(payload), encoding="utf-8")
        return payload

    if args.render_only:
        payload = write()
        print(f"[OK] re-rendered {md_path}", flush=True)
        return 0

    names = resolve_families(args.families)
    override = (
        None
        if args.difficulties is None
        else [float(d) for d in args.difficulties.split(",") if d.strip()]
    )
    todo = [n for n in names if args.force or n not in families]
    if not args.quiet:
        print(
            f"difficulty calibration: {len(todo)} families to measure "
            f"({len(names) - len(todo)} already in the artifact), "
            f"{args.n_layouts} layouts x {args.seeds} seeds/side per cell",
            flush=True,
        )

    for name in todo:
        family = FamilyRegistry.get(name)
        lo, hi = family.difficulty_range()
        tiers_wanted = override if override is not None else family.default_difficulties()
        tiers_wanted = [d for d in tiers_wanted if lo - 1e-9 <= d <= hi + 1e-9]
        started = time.perf_counter()
        try:
            tiers = [
                measure_cell(family, d, args.n_layouts, args.seeds) for d in tiers_wanted
            ]
            block: dict[str, Any] = {
                "status": "measured",
                "error": None,
                "difficulty_range": [round(lo, 4), round(hi, 4)],
                "requested_difficulties": [round(float(d), 4) for d in tiers_wanted],
                "design": dict(design),
                "tiers": tiers,
                "summary": summarize_family(name, tiers),
            }
        except Exception as exc:  # noqa: BLE001 -- a broken family is an artifact entry
            # An optional-dependency failure (mujoco, pettingzoo) or a genuinely broken
            # generator has to appear in the artifact, not silently shrink it.
            block = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "difficulty_range": [round(lo, 4), round(hi, 4)],
                "design": dict(design),
                "tiers": [],
            }
        block["wall_clock_seconds"] = round(time.perf_counter() - started, 3)
        families[name] = block
        write()  # flush after every family: an interrupted run must leave a usable artifact
        if not args.quiet:
            if block["status"] == "measured":
                f = block["summary"]
                print(
                    f"  {name:42s} span={_fmt(f['separation_span'])} "
                    f"class={f['classification']:24s} "
                    f"({block['wall_clock_seconds']:.1f}s, {len(block['tiers'])} tiers)",
                    flush=True,
                )
            else:
                print(f"  {name:42s} ERROR {block['error']}", flush=True)

    payload = write()
    s = payload["summary"]
    print(
        f"\n{s['n_families_separation_span_exactly_zero']} of "
        f"{s['n_families_with_more_than_one_tier']} multi-tier families have a separation "
        f"span of exactly 0.000; {s['n_families_measured']}/{s['n_families_registered']} "
        f"families measured. Wrote {json_path} and {md_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
