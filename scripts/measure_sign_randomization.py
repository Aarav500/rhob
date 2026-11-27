#!/usr/bin/env python
"""Measure what L2 sign randomization costs: the before/after numbers, honestly.

Until the 2026-08 audit, ``CONTRIBUTING.md`` fixed one global orientation for every
family's ``RunData.behav_trace`` (positive = hacking), ``BehavioralThresholdDetector``
returned the raw signed mean, and the benchmark computed AUROC with label 1 = hacking.
The sign of the observation *was* the label. :mod:`rhob.v3.sign_randomization` now draws
a per-``(family, layout_seed)`` orientation and withholds it. This script measures the
difference, on the same seeds, the same families and the same detectors.

Four configurations, and why all four are needed
------------------------------------------------
Two independent switches are in play -- whether the benchmark randomizes the sign, and
whether the detector infers its orientation from the unlabeled population
(:meth:`~rhob.detectors.posthoc.PosthocDetector.observe_cell`). Reporting a single
"before" and a single "after" would confound them, so all four corners are measured:

===============  ==========  ========  ====================================================
config           randomize   infer     what it is
===============  ==========  ========  ====================================================
``before``       off         off       the published pre-audit number: old convention scored
                                       by a detector that assumes it. This is the row that
                                       has to reproduce RTS 0.994 / L2 best 0.975 (the
                                       pre-audit board, ``c54a8cc``) before the other three
                                       mean anything.
``before_infer`` off         on        the shipped detector against the old convention --
                                       what the orientation hook costs when it is not needed.
``blind``        on          off       sign randomized, detector still assumes ``+1``: what
                                       the pre-audit detector actually knew. Its expectation
                                       over families is 0.5 by construction.
``after``        on          on        the shipped configuration. This is the honest number.
===============  ==========  ========  ====================================================

"infer off" is implemented by shadowing ``observe_cell`` with a no-op on the detector
instance, which is exactly the pre-audit detector: same ``classify``, no population hook.
Nothing in ``src/`` is modified or monkeypatched to produce it.

What is *not* measured here, because it cannot move
---------------------------------------------------
Sign randomization multiplies ``behav_trace`` and nothing else, and
:func:`rhob.v3.access.restrict` nulls ``behav_trace`` at L0 and L1. So every L0 and L1
number on the board is bit-identical in both modes, and the L0/L1 rungs of the
access-level ladder are unchanged by this work. That is an argument, not a measurement,
so the ``board`` subcommand also scores two cheap deterministic detectors (one L0, one
L1) in both modes and reports whether every cell matched exactly.

Subcommands
-----------
``board``     per-cell AUROC for every L2/L3 detector x 4 configs over the requested
              families, at the leaderboard's ``n_seeds``. Answers the L2 access-level
              ladder (item 1), the per-detector L2 deltas (item 2) and the difficulty
              axis (item 3) from one set of rollouts.
``transfer``  RTS, by calling ``scripts/cross_family_transfer.py``'s own ``run_rows`` in
              each configuration rather than reimplementing it (item 1).
``report``    turn the JSON artifacts into the committed JSON and Markdown table (item 4).
``ci``        the published design end to end, in one process: ``board`` over every
              family, the two companion boards, ``transfer`` in ``before``/``blind``/
              ``after``, then ``report``. This is what ``.github/workflows/regenerate.yml``
              runs; its options other than the output paths exist for smoke tests, and
              any of them set to a non-default value marks the artifact
              ``design.published_design = false``.

Run::

    python scripts/measure_sign_randomization.py ci \
        --out-json docs/figures/sign_randomization_impact.json \
        --out-md docs/figures/sign_randomization_impact.md

or the stages by hand::

    python scripts/measure_sign_randomization.py board --out results/sign_randomization/board.json
    python scripts/measure_sign_randomization.py transfer --out results/sign_randomization/transfer.json
    git show c54a8cc:leaderboard/v5_leaderboard.json > results/sign_randomization/pre_audit_board.json
    python scripts/measure_sign_randomization.py report --board results/sign_randomization/board_*.json \
        --transfer results/sign_randomization/transfer_*.json \
        --published-board results/sign_randomization/pre_audit_board.json

``report``'s ``--published-board`` has no default (``ci`` writes it out of git itself).
It used to be ``leaderboard/v5_leaderboard.json``, but that file was regenerated on
2026-11-19 with sign randomization on, so it is the ``after`` convention and a ``before``
column checked against it would be checked against the wrong thing. The pre-audit board
is the one at ``c54a8cc`` (identical to ``acd35c2``, the commit before the regeneration).
``--published-transfer`` defaults to ``leaderboard/cross_family_transfer.json``, the
2026-07-09 artifact that ``scripts/cross_family_transfer.py`` no longer writes (it writes
``leaderboard/cross_family_transfer_v5.json``); ``report`` refuses a published transfer
file that was measured with sign randomization on, since the ``before`` RTS is checked
against the pre-randomization figure.

``board`` can be run once per family group rather than once over all 33 families: the
groups are independent. ``report`` merges them, refusing to merge runs made at different
``n_seeds``, a family measured twice, or inputs from different commits or package
versions or from a tree not known to be clean (``--allow-dirty`` for a local smoke run).
Its output lands in ``docs/figures/`` -- ``results/`` is gitignored, and the merged file
is the one the docs cite. It carries every per-cell AUROC (``cells``: one
``[family, difficulty, auroc, na_reason]`` row per cell, per detector and
configuration), so nothing is lost by leaving the per-group files uncommitted.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# Run the checkout this file lives in, not whatever the editable install points at.
# In a git worktree those differ, and the failure is silent: the script imports a
# same-named module from the OTHER tree and publishes results attributed to this one.
# Enforced for every script by tests/test_scripts_run_their_own_checkout.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rhob.v3.families  # noqa: F401  -- importing registers every family
from rhob.detectors.l0_reward_cusum import RewardCUSUMDetector  # noqa: E402
from rhob.detectors.l1_coverage_rate import StateCoverageRateDetector  # noqa: E402
from rhob.detectors.l2_angular_momentum import AngularMomentumDetector  # noqa: E402
from rhob.detectors.l2_behavioral_threshold import BehavioralThresholdDetector  # noqa: E402
from rhob.detectors.l2_centroid_tracker import CentroidTrackerDetector  # noqa: E402
from rhob.detectors.l2_feature_consistency import FeatureConsistencyDetector  # noqa: E402
from rhob.detectors.l2_feature_magnitude import FeatureMagnitudeDetector  # noqa: E402
from rhob.detectors.l2_reward_feature_correlation import RewardFeatureCorrelationDetector  # noqa: E402
from rhob.detectors.l2_trajectory_mlp import TrajectoryMLPDetector  # noqa: E402
from rhob.detectors.l3_perfect_feature import PerfectFeatureOracleDetector  # noqa: E402
from rhob.detectors.l3_true_reward_oracle import TrueRewardOracleDetector  # noqa: E402
from rhob.detectors.posthoc import PosthocDetector  # noqa: E402
from rhob.v3.benchmark import Benchmark  # noqa: E402
from rhob.v3.leaderboard.transfer import load_transfer_results  # noqa: E402
from rhob.v3.provenance import provenance_block, sampling_block  # noqa: E402
from rhob.v3.registry import FamilyRegistry  # noqa: E402
from rhob.v3.sign_randomization import apply_behav_sign, behav_sign  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]

#: ``(randomize_behav_sign, detector_infers_orientation)`` per configuration.
CONFIGS: dict[str, tuple[bool, bool]] = {
    "before": (False, False),
    "before_infer": (False, True),
    "blind": (True, False),
    "after": (True, True),
}

#: The sampling design of the retired v3.2 board
#: (``git show b7324ec:scripts/populate_leaderboard_v32.py``). The v5 board is not at
#: this design: the pre-audit board's per-family values are matched by an ``n_seeds=5``
#: run (see :func:`_sensitivity`), and the board regenerated on 2026-11-19 records
#: ``sampling.n_seeds_per_variant = 5``.
BOARD_N_SEEDS = 10

#: Model-init trials of the published transfer measurement. Trial ``k`` seeds the
#: networks with ``k``, as ``scripts/cross_family_transfer.py`` does, so trials 0-2 here
#: are the first three of that script's five.
TRANSFER_N_TRIALS = 3

#: Where ``ci`` reads the pre-audit board and the pre-randomization RTS from by default.
PRE_AUDIT_BOARD_REV = "c54a8cc:leaderboard/v5_leaderboard.json"
PUBLISHED_TRANSFER = Path("leaderboard/cross_family_transfer.json")

#: Rollout seeds are ``range(n_seeds)`` for the hacking variant and ``1000 + range(n_seeds)``
#: for the legitimate one (``MatchedPair.rollout``), off one layout per family.
LEGIT_SEED_OFFSET = 1000
LAYOUT_SEEDS = [0]

#: The 7 L2 detectors the committed access-level ladder is computed over, plus the two
#: L3 rows. ``Perfect Feature Oracle`` is a relabelled duplicate of ``Behavioral
#: Threshold`` (``rhob.detectors.redundancy``) and is excluded from the L3 aggregate; it
#: is scored anyway because it inherits the orientation hook and is therefore one of the
#: rows this change moves.
L2_DETECTORS: dict[str, Callable[[], PosthocDetector]] = {
    "Behavioral Threshold": BehavioralThresholdDetector,
    "Angular Momentum": AngularMomentumDetector,
    "Centroid Tracker": CentroidTrackerDetector,
    "Feature Magnitude": FeatureMagnitudeDetector,
    "Feature Consistency": FeatureConsistencyDetector,
    "Reward-Feature Correlation": RewardFeatureCorrelationDetector,
    "Trajectory MLP": TrajectoryMLPDetector,
}
L3_DETECTORS: dict[str, Callable[[], PosthocDetector]] = {
    "True Reward Oracle": TrueRewardOracleDetector,
    "Perfect Feature Oracle": PerfectFeatureOracleDetector,
}
#: Deterministic, cheap, and defined over channels the flip cannot touch. Scored in both
#: modes purely to demonstrate the L0/L1 invariance rather than assert it.
INVARIANCE_DETECTORS: dict[str, Callable[[], PosthocDetector]] = {
    "Reward CUSUM": RewardCUSUMDetector,
    "State Coverage Rate": StateCoverageRateDetector,
}


def assume_orientation(detector: PosthocDetector) -> PosthocDetector:
    """Return ``detector`` with the unlabeled-population hook disabled.

    This is precisely the pre-audit detector: identical ``classify``, but never told
    about the cell it is scoring, so :class:`BehavioralThresholdDetector` keeps its
    ``+1`` default and reads the family's authored coordinate as if the convention still
    held. Shadowing the bound method with an instance attribute is enough --
    :func:`rhob.v3.benchmark.offer_population` finds it by ``getattr`` -- and it leaves
    ``src/`` untouched, so the "before" column is produced by the shipped code with one
    hook switched off rather than by a reconstruction of the old code.
    """
    detector.observe_cell = lambda runs: None  # type: ignore[method-assign]
    return detector


def build(factory: Callable[[], PosthocDetector], infer: bool) -> PosthocDetector:
    """Instantiate a detector for one configuration."""
    detector = factory()
    return detector if infer else assume_orientation(detector)


def cell_records(results) -> list[dict]:
    """One JSON-able record per evaluated cell.

    No onset column: :class:`~rhob.v3.benchmark.CellResult` has carried none since
    2026-09-02 (see its docstring), and reading ``c.onset_mae`` crashed the ``board``
    subcommand on its first cell until 2026-11-27. A cell whose AUROC is NaN although it
    was applicable (undefined, e.g. one variant only) is ``null`` like an N/A one, with
    no reason.
    """
    return [
        {
            "family": c.family,
            "difficulty": round(float(c.difficulty), 4),
            "auroc": (
                None
                if not c.applicable or np.isnan(c.discrimination_auroc)
                else round(float(c.discrimination_auroc), 6)
            ),
            "na_reason": c.na_reason,
        }
        for c in results.cells
    ]


def per_family_means(results) -> dict[str, float]:
    """Mean AUROC per family over that family's scored cells."""
    by_family: dict[str, list[float]] = {}
    for c in results.scored_cells:
        by_family.setdefault(c.family, []).append(float(c.discrimination_auroc))
    return {k: round(float(np.mean(v)), 6) for k, v in sorted(by_family.items())}


#: Every detector ``board`` can score, by name.
ALL_DETECTORS: dict[str, Callable[[], PosthocDetector]] = {
    **L2_DETECTORS,
    **L3_DETECTORS,
    **INVARIANCE_DETECTORS,
}


def derive_randomized_rollouts() -> int:
    """Fill each missing ``randomize_behav_sign=True`` rollout from its un-randomized twin.

    ``MatchedPair.rollout(..., randomize_sign=True)`` simulates exactly the runs it would
    with the flag off and then applies :func:`~rhob.v3.sign_randomization.apply_behav_sign`
    with ``behav_sign(family, layout_seed)`` to both variants (``rhob/v3/base_pair.py``).
    Doing that here, with those same two functions, on the cached un-randomized runs gives
    the identical arrays -- a multiplication by +/-1.0 -- without simulating every cell a
    second time, which is half of this script's rollout cost. Every key field other than
    the flag is kept, so nothing crosses draws. Returns how many entries were derived.

    Consequence for the invariance check: both modes then share one simulation, so the
    check verifies that the flip touches nothing but ``behav_trace`` and that ``restrict``
    hides it below L2 -- not that two simulations agree, which determinism already gives.
    """
    cache = Benchmark._rollout_cache
    derived = 0
    for key in list(cache):
        if key.randomize_behav_sign:
            continue
        twin = key._replace(randomize_behav_sign=True)
        if twin in cache:
            continue
        runs_a, runs_b, onsets_a = cache[key]
        sign = behav_sign(key.family, key.layout_seed)
        cache[twin] = (
            [apply_behav_sign(r, sign) for r in runs_a],
            [apply_behav_sign(r, sign) for r in runs_b],
            list(onsets_a),
        )
        derived += 1
    return derived


def run_board(args: argparse.Namespace) -> None:
    """Score every L2/L3 detector in all four configurations, per cell."""
    families = "all" if args.families == ["all"] else args.families
    if families != "all":
        unknown = sorted(set(families) - set(FamilyRegistry.list_families()))
        if unknown:
            raise SystemExit(f"board: unknown families {unknown}")
    detectors = dict(ALL_DETECTORS)
    if args.only:
        unknown = sorted(set(args.only) - set(detectors))
        if unknown:
            raise SystemExit(f"board: unknown detectors {unknown}; choose from {sorted(detectors)}")
        detectors = {k: v for k, v in detectors.items() if k in set(args.only)}

    out: dict[str, dict] = {}
    for config, (randomize, infer) in CONFIGS.items():
        if randomize:
            # The un-randomized configurations come first, so every cell they rolled out
            # is in the cache by now.
            derive_randomized_rollouts()
        for name, factory in detectors.items():
            # For a detector whose score does not depend on the feature's direction,
            # ``infer`` is a no-op (the base-class hook does nothing), so ``before`` and
            # ``blind`` would be the same run, as would ``before_infer`` and ``after``.
            # Such a detector is scored once per randomization mode, under the two config
            # names that differ in the mode: ``before`` (off) and ``after`` (on).
            if name not in ORIENTING and config in ("before_infer", "blind"):
                continue
            t0 = time.time()
            detector = build(factory, infer)
            results = Benchmark.evaluate(
                detector,
                families=families,
                difficulties="all",
                n_seeds=args.n_seeds,
                verbose=False,
                randomize_behav_sign=randomize,
            )
            entry = out.setdefault(
                name,
                {
                    "access_level": results.access_level,
                    "direction_dependent": name in ORIENTING,
                    "by_config": {},
                },
            )
            entry["by_config"][config] = {
                "overall_auroc": None
                if np.isnan(results.overall_auroc)
                else round(float(results.overall_auroc), 6),
                "n_cells_scored": len(results.scored_cells),
                "n_cells_na": len(results.na_cells),
                "n_families_scored": len({c.family for c in results.scored_cells}),
                "per_family": per_family_means(results),
                "cells": cell_records(results),
            }
            print(
                f"[{config:12s}] {name:28s} overall={results.overall_auroc:.4f} "
                f"({len(results.scored_cells)} cells, {time.time() - t0:.0f}s)",
                flush=True,
            )

    payload = {
        "measurement": "board",
        "n_seeds": args.n_seeds,
        "families": FamilyRegistry.list_families() if families == "all" else families,
        "configs": {k: {"randomize_behav_sign": r, "detector_infers_orientation": i}
                    for k, (r, i) in CONFIGS.items()},
        "drawn_signs": {
            f: behav_sign(f, 0)
            for f in (FamilyRegistry.list_families() if families == "all" else families)
        },
        "provenance": provenance_block(script="scripts/measure_sign_randomization.py"),
        "sampling": sampling_block(
            n_seeds=args.n_seeds,
            n_layouts=len(LAYOUT_SEEDS),
            layout_seeds=LAYOUT_SEEDS,
            rollout_seeds_hacking=list(range(args.n_seeds)),
            rollout_seeds_legit=[LEGIT_SEED_OFFSET + s for s in range(args.n_seeds)],
            n_replicates=1,
            # One simulation per cell; the randomized configurations read it through the
            # flip (derive_randomized_rollouts), so both modes score the identical runs.
            rollouts_shared_across_modes=True,
        ),
        "detectors": out,
    }
    _write(args.out, payload)


#: Detectors whose score depends on the behavioral feature's *direction*, i.e. the only
#: ones for which the ``infer`` switch can change anything. Read from the source, not
#: assumed: ``BehavioralThresholdDetector.classify`` and ``PerfectFeatureOracleDetector``
#: (its subclass) return the signed steady mean, and ``CentroidTrackerDetector`` returns
#: ``0.5 + 0.5*tanh(signed mean)``. The other four L2 detectors read ``abs`` (Angular
#: Momentum, Feature Magnitude), a standard-deviation ratio (Feature Consistency) or
#: ``|corr|`` (Reward-Feature Correlation), and ``TrueRewardOracleDetector`` never touches
#: ``behav_trace`` at all -- for those, the flip is a coordinate change their statistic
#: cannot see, which this run checks rather than assumes.
#:
#: ``Trajectory MLP`` is here for a different reason: it is *fitted*, cross-validated
#: within the cell, so its training folds' labels tell it the cell's direction. Whether
#: that makes it immune to the randomization is one of the findings this run produces,
#: so it is scored in all four configurations.
ORIENTING = {
    "Behavioral Threshold",
    "Centroid Tracker",
    "Trajectory MLP",
    "Perfect Feature Oracle",
}

#: The two companion boards of the published design: ``label -> (n_seeds, families)``.
#: Each anchors one figure published at another sample size (see :func:`_sensitivity`),
#: and only the direction-dependent detectors (:data:`ORIENTING`) are read from them, so
#: only those are scored.
COMPANIONS: dict[str, tuple[int, tuple[str, ...]]] = {
    # The 14 tabular and continuous families, at the pre-audit v5 board's n_seeds=5.
    "n5_v5board_reproduction": (5, (
        "continuous_camping", "distributional_shift", "eval_probe_sandbagging",
        "goal_misgeneralization", "gridworld_camping", "monitored_sandbagging",
        "novelty_farming", "orbit_chirality", "physics_exploitation",
        "proxy_correlation_gaming", "reward_channel_tampering", "rlhf_reward_model_overopt",
        "sensor_calibration_tampering", "shortcut_exploitation",
    )),
    # The 13 families of docs/l2_sign_randomization.md's 60-cell table, at n_seeds=20.
    "n20_doc_13family_subset": (20, (
        "distributional_shift", "eval_probe_sandbagging", "physics_exploitation",
        "proxy_correlation_gaming", "reward_channel_tampering", "novelty_farming",
        "shortcut_exploitation", "continuous_camping", "monitored_sandbagging",
        "sensor_calibration_tampering", "goal_misgeneralization", "gridworld_camping",
        "orbit_chirality",
    )),
}


@functools.lru_cache(maxsize=1)
def _load_transfer_module():
    """Import ``scripts/cross_family_transfer.py`` as a module (once per process).

    The RTS experiment is defined by that script; measuring RTS by reimplementing it
    would measure the reimplementation -- which is why the Top-5 ensemble is built by
    that script's ``build_ensemble`` rather than by a copy of it here. It is a script
    rather than a package module, so it is loaded by path.
    """
    path = Path(__file__).resolve().with_name("cross_family_transfer.py")
    spec = importlib.util.spec_from_file_location("_rhob_cross_family_transfer", path)
    if spec is None or spec.loader is None:  # pragma: no cover -- path is checked in
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The rows the published design measures in each configuration. The L0 and L1 rows are
#: measured once, in ``before``: :func:`rhob.v3.access.restrict` nulls ``behav_trace``
#: below L2, so they have nothing the flip could move (the board's invariance check
#: verifies that cell by cell) and :mod:`rhob.v3.leaderboard.transfer` carries them over.
TRANSFER_ROWS_BY_CONFIG: dict[str, tuple[str, ...]] = {
    "before": ("Reward MLP", "State Divergence", "Trajectory MLP", "Ensemble (Top 5)"),
    "before_infer": ("Trajectory MLP", "Ensemble (Top 5)"),
    "blind": ("Trajectory MLP", "Ensemble (Top 5)"),
    "after": ("Trajectory MLP", "Ensemble (Top 5)"),
}


def run_transfer(args: argparse.Namespace) -> None:
    """Measure RTS in each configuration, through cross_family_transfer's own code.

    Every configuration calls ``cross_family_transfer.run_rows`` -- the function that
    script's ``main`` calls -- with the configuration's randomization flag passed
    explicitly and, for an "infer off" configuration, every freshly built detector's
    population hook switched off (:func:`assume_orientation`). The scored code path
    (the seeds, the not-applicable rules, the pooled fit, the frozen-model transfer, the
    fold-fitted ensemble and the multi-trial aggregation) is the published one, so the
    ``after`` rows are the same measurement as ``leaderboard/cross_family_transfer_v5.json``'s
    at the same trials: trial ``k`` seeds the networks with ``k`` in both.
    """
    cft = _load_transfer_module()
    train_families = list(getattr(args, "train_families", None) or cft.TRAIN_FAMILIES)
    test_families = list(getattr(args, "test_families", None) or cft.TEST_FAMILIES)
    rows_by_config: dict[str, list[str]] = {
        config: [r for r in cft.ROWS if r in set(args.rows)] for config in args.configs
    }
    if getattr(args, "rows_by_config", None):
        rows_by_config = {c: list(args.rows_by_config[c]) for c in args.configs}

    results: dict[str, dict] = {}
    timings: dict[str, dict[str, float]] = {}
    for config in args.configs:
        randomize, infer = CONFIGS[config]
        print(f"\n########## config = {config} "
              f"(randomize={randomize}, infer={infer}) ##########", flush=True)
        if randomize:
            derive_randomized_rollouts()
        timings[config] = {}
        results[config] = cft.run_rows(
            rows_by_config[config],
            n_seeds_train=args.n_seeds_train,
            n_seeds_test=args.n_seeds_test,
            n_trials=args.n_trials,
            randomize_sign=randomize,
            prepare=None if infer else assume_orientation,
            train_families=train_families,
            test_families=test_families,
            timings=timings[config],
        )

    payload = {
        "measurement": "transfer",
        "train_families": train_families,
        "test_families": test_families,
        "n_seeds_train": args.n_seeds_train,
        "n_seeds_test": args.n_seeds_test,
        "n_trials": args.n_trials,
        "configs": {k: {"randomize_behav_sign": CONFIGS[k][0],
                        "detector_infers_orientation": CONFIGS[k][1]}
                    for k in args.configs},
        "rows_by_config": rows_by_config,
        "drawn_signs": {f: behav_sign(f, 0) for f in train_families + test_families},
        "cell_semantics": cft.CELL_SEMANTICS,
        "provenance": provenance_block(
            script="scripts/measure_sign_randomization.py", wall_clock_s=timings
        ),
        "results": results,
    }
    _write(args.out, payload)


def family_suite(family: str) -> str:
    """The coarse family group a name belongs to, for the difficulty-axis table."""
    for prefix in ("mujoco_", "pettingzoo_", "sequence_", "rlhf_"):
        if family.startswith(prefix):
            return prefix.rstrip("_")
    return FamilyRegistry.get(family).complexity.value


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _provenance_summary(provenance: Any) -> dict:
    """The parts of a provenance block two inputs must share to be merged."""
    prov = provenance if isinstance(provenance, dict) else {}
    git = prov.get("git") if isinstance(prov.get("git"), dict) else {}
    numerics = prov.get("numerics") if isinstance(prov.get("numerics"), dict) else {}
    return {
        "commit": git.get("commit"),
        "short_commit": git.get("short_commit"),
        "dirty": git.get("dirty"),
        "python": prov.get("python"),
        "packages": prov.get("packages"),
        "numerics_env": numerics.get("env"),
    }


def _uniform_provenance(inputs: list[tuple[Path, dict]], allow_dirty: bool = False) -> dict:
    """Refuse to merge inputs that are not one clean run's; return their shared summary.

    Mirrors ``scripts/admission_ledger.py --merge-shards``: every input must come from
    the same commit, Python, package versions and numerics pins, and from a tree known to
    be clean (``git.dirty`` exactly ``False``). The committed 2026-08-03 artifact is the
    counter-example this exists for: it was merged from files made at a dirty tree, at a
    commit that is not in this repository's history. ``allow_dirty`` admits a dirty or
    unknown tree for a local smoke run; it never admits a mixture.
    """
    shared: Optional[dict] = None
    first: Optional[Path] = None
    for path, payload in inputs:
        summary = _provenance_summary(payload.get("provenance"))
        if summary["dirty"] is not False and not allow_dirty:
            raise ValueError(
                f"{path}: not generated from a tree known to be clean "
                f"(provenance.git.dirty = {summary['dirty']!r}); refusing to merge it into a "
                "published artifact. Pass --allow-dirty for a local smoke run."
            )
        key = {k: v for k, v in summary.items() if k not in ("dirty", "short_commit")}
        if shared is None:
            shared, first = summary, path
            continue
        reference = {k: v for k, v in shared.items() if k not in ("dirty", "short_commit")}
        if key != reference:
            differs = {k: (reference[k], key[k]) for k in key if key[k] != reference[k]}
            raise ValueError(
                f"{path} and {first} are not from the same run (differs in "
                f"{sorted(differs)}): {differs}"
            )
        if summary["dirty"] is not False:
            shared = {**shared, "dirty": summary["dirty"]}
    return shared or {}


def _merge_boards(paths: list[Path], allow_dirty: bool = False) -> tuple[dict, dict]:
    """Merge several ``board`` artifacts (one per family group) into one.

    The board can be measured in per-family-group jobs; the cells are independent, so
    merging them is exact rather than an approximation. Returns ``(detectors, design)``,
    and raises if two jobs disagree about ``n_seeds`` (cells measured at different
    sample sizes must not be averaged into one ladder), if a family appears in two jobs
    (its cells would count twice), or if the jobs are not one clean run
    (:func:`_uniform_provenance`).
    """
    detectors: dict[str, dict] = {}
    families: list[str] = []
    owner: dict[str, Path] = {}
    n_seeds: set[int] = set()
    signs: dict[str, float] = {}
    loaded = [(Path(p), _load(p)) for p in paths]
    provenance = _uniform_provenance(loaded, allow_dirty=allow_dirty)
    for path, payload in loaded:
        n_seeds.add(int(payload["n_seeds"]))
        for fam in payload["families"]:
            if fam in owner:
                raise ValueError(
                    f"family {fam!r} is in both {owner[fam]} and {path}; its cells would be "
                    "counted twice"
                )
            owner[fam] = path
        families.extend(payload["families"])
        signs.update(payload.get("drawn_signs", {}))
        for name, entry in payload["detectors"].items():
            target = detectors.setdefault(
                name,
                {
                    "access_level": entry["access_level"],
                    "direction_dependent": entry["direction_dependent"],
                    "by_config": {},
                },
            )
            for config, block in entry["by_config"].items():
                merged = target["by_config"].setdefault(
                    config, {"cells": [], "n_cells_scored": 0, "n_cells_na": 0}
                )
                merged["cells"].extend(block["cells"])
                merged["n_cells_scored"] += block["n_cells_scored"]
                merged["n_cells_na"] += block["n_cells_na"]
    if len(n_seeds) != 1:
        raise ValueError(f"boards disagree on n_seeds: {sorted(n_seeds)}")
    # Recompute the aggregates over the merged cell list rather than averaging the
    # per-job means, which would weight a 1-family job like a 14-family one.
    for entry in detectors.values():
        for block in entry["by_config"].values():
            scored = [c["auroc"] for c in block["cells"] if c["auroc"] is not None]
            block["overall_auroc"] = (
                round(float(np.mean(scored)), 6) if scored else None
            )
            by_family: dict[str, list[float]] = {}
            for cell in block["cells"]:
                if cell["auroc"] is not None:
                    by_family.setdefault(cell["family"], []).append(cell["auroc"])
            block["per_family"] = {
                k: round(float(np.mean(v)), 6) for k, v in sorted(by_family.items())
            }
            block["n_families_scored"] = len(by_family)
    design = {
        "n_seeds": n_seeds.pop(),
        "families_measured": sorted(set(families)),
        "families_registered": FamilyRegistry.list_families(),
        "families_not_measured": sorted(
            set(FamilyRegistry.list_families()) - set(families)
        ),
        "drawn_signs_layout_0": signs,
        "board_files": [p.as_posix() for p, _ in loaded],
        "board_provenance": provenance,
    }
    return detectors, design


def _ladder(detectors: dict, config: str) -> dict:
    """Mean/max over the 7 L2 detectors of the committed access-level ladder."""
    scored = [
        (name, entry["by_config"][config]["overall_auroc"])
        for name, entry in detectors.items()
        if name in L2_DETECTORS and config in entry["by_config"]
        and entry["by_config"][config]["overall_auroc"] is not None
    ]
    scored.sort(key=lambda kv: -kv[1])
    if not scored:
        return {}
    values = [v for _, v in scored]
    mean = float(np.mean(values))
    return {
        "n_detectors": len(scored),
        "mean_auroc": round(mean, 4),
        "sd_across_detectors": round(float(np.std(values)), 4),
        "max_auroc": round(scored[0][1], 4),
        "best_detector": scored[0][0],
        "per_detector": {k: round(v, 4) for k, v in scored},
    }


def _difficulty_axis(detectors: dict, detector: str) -> dict:
    """Achieved AUROC per (family, requested difficulty), before vs after.

    ``span`` is ``max - min`` across a family's tiers: the earlier finding was that this
    was exactly 0.000 for 12 of 13 swept families, because a statistic whose sign is the
    label cannot respond to a knob that only moves how *far* the variants separate.

    ``separation_span`` is the same statistic computed on ``|AUROC - 0.5|``, which the
    coordinate flip leaves invariant. It is reported because it is the quantity a
    dose-response sweep would actually need to move, and it is identical in both modes by
    construction -- so the table shows directly whether randomization did anything to the
    knob or only to which end of the axis the detector points at.
    """
    entry = detectors.get(detector)
    if entry is None:
        return {}
    out: dict[str, dict] = {}
    for config in ("before", "after"):
        block = entry["by_config"].get(config)
        if block is None:
            continue
        for cell in block["cells"]:
            fam = out.setdefault(
                cell["family"],
                {
                    "suite": family_suite(cell["family"]),
                    "drawn_sign": behav_sign(cell["family"], 0),
                    "tiers": {},
                },
            )
            tier = fam["tiers"].setdefault(f"{cell['difficulty']:.2f}", {})
            tier[config] = cell["auroc"]
    for fam in out.values():
        for key in ("before", "after"):
            values = [t[key] for t in fam["tiers"].values() if t.get(key) is not None]
            fam[f"span_{key}"] = (
                round(float(max(values) - min(values)), 6) if values else None
            )
        sep = [
            abs(t["after"] - 0.5) for t in fam["tiers"].values()
            if t.get("after") is not None
        ]
        fam["separation_span"] = round(float(max(sep) - min(sep)), 6) if sep else None
        # |AUROC - 0.5| is preserved exactly by the flip, so a cell is oriented the same
        # way as the family's authored coordinate iff both sides of 0.5 agree.
        oriented = [
            (t["before"] - 0.5) * (t["after"] - 0.5) >= 0
            for t in fam["tiers"].values()
            if t.get("before") is not None and t.get("after") is not None
        ]
        fam["cells_oriented"] = f"{sum(oriented)}/{len(oriented)}" if oriented else None
        fam["tiers"] = dict(sorted(fam["tiers"].items(), key=lambda kv: -float(kv[0])))
    return dict(sorted(out.items()))


def _sensitivity(specs: list[str]) -> dict:
    """Secondary board runs at other sample sizes, for the two figures they anchor.

    The primary board here is measured at ``n_seeds=10``. Two published numbers were
    produced at other sample sizes and have to be checkable against their own design
    rather than against this one:

    - the pre-audit ``leaderboard/v5_leaderboard.json`` (``git show
      c54a8cc:leaderboard/v5_leaderboard.json``) -- reproduced by an ``n_seeds=5`` run,
      which is what its per-family values match.
    - ``docs/l2_sign_randomization.md``'s 60-cell table -- measured at ``n_seeds=20`` over
      13 families.

    Each spec is ``label=path``; the block reports every configuration of every
    direction-dependent detector so the comparison is not cherry-picked to one row.
    """
    out: dict[str, dict] = {}
    for spec in specs:
        label, _, raw = spec.partition("=")
        path = Path(raw)
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[label] = {
            "path": path.as_posix(),
            "n_seeds": payload["n_seeds"],
            "families": payload["families"],
            "n_families": len(payload["families"]),
            "detectors": {
                name: {
                    config: block["overall_auroc"]
                    for config, block in entry["by_config"].items()
                }
                for name, entry in payload["detectors"].items()
                if entry["direction_dependent"]
            },
            "n_cells": {
                name: {
                    config: block["n_cells_scored"]
                    for config, block in entry["by_config"].items()
                }
                for name, entry in payload["detectors"].items()
                if name == "Behavioral Threshold"
            },
        }
    return out


def _present(values) -> list[float]:
    """The values that are numbers (``None`` = not applicable, and ``0.0`` is a number)."""
    return [v for v in values if v is not None]


def _knob_summary(axis: dict) -> dict:
    """Whether the difficulty knob moves the achieved L2 separation, and where.

    The pre-audit finding was that L2-AUROC pinned at exactly 1.000 across every tier in
    almost every family, and the hypothesis under test was that the fixed sign convention
    caused it. The statistic that settles it is ``separation_span``: the range of
    ``|AUROC - 0.5|`` across a family's tiers, which the coordinate flip preserves
    exactly. If the knob were dead only because the sign was the label, this would open up
    under randomization -- it cannot, and the count below is by how much it does not.

    ``span_after`` is deliberately *not* the statistic used here. Orientation is a
    family-level inference that can come out differently at different tiers, so a family
    can show a span of 1.000 under ``after`` purely from two tiers being oriented
    oppositely, with the separation identical at both. That is an orientation error, not
    a knob response.
    """
    multi = {f: b for f, b in axis.items() if len(b["tiers"]) > 1}
    moving = {
        f: b["separation_span"] for f, b in multi.items() if (b["separation_span"] or 0) > 0
    }
    oriented_num = oriented_den = 0
    for block in axis.values():
        for tier in block["tiers"].values():
            if tier.get("before") is None or tier.get("after") is None:
                continue
            oriented_den += 1
            oriented_num += (tier["before"] - 0.5) * (tier["after"] - 0.5) >= 0
    # Broken out by family group because the published 63.3% was measured on the tabular
    # and continuous families alone -- the only ones cheap enough to sweep at the time --
    # and whether it holds on the four expensive groups is exactly the question a reader
    # of that figure needs answered.
    by_suite: dict[str, list[int]] = {}
    by_suite_auroc: dict[str, list[float]] = {}
    for block in axis.values():
        bucket = by_suite.setdefault(block["suite"], [0, 0])
        for tier in block["tiers"].values():
            if tier.get("before") is None or tier.get("after") is None:
                continue
            bucket[1] += 1
            bucket[0] += (tier["before"] - 0.5) * (tier["after"] - 0.5) >= 0
            by_suite_auroc.setdefault(block["suite"], []).append(tier["after"])

    return {
        "n_families_with_more_than_one_tier": len(multi),
        "n_families_separation_span_exactly_zero": len(multi) - len(moving),
        "families_whose_separation_span_is_nonzero": {
            f: round(v, 4) for f, v in sorted(moving.items(), key=lambda kv: -kv[1])
        },
        "separation_span_is_identical_before_and_after": all(
            b["span_before"] == b["separation_span"]
            for b in axis.values()
            if b["span_before"] is not None and b["separation_span"] is not None
            and min(_present(t.get("before") for t in b["tiers"].values()), default=0.0)
            >= 0.5
        ),
        "cells_oriented_as_the_authored_coordinate": f"{oriented_num}/{oriented_den}",
        "orientation_accuracy": round(oriented_num / oriented_den, 4) if oriented_den else None,
        "by_family_group": {
            suite: {
                "cells_oriented": f"{ok}/{n}",
                "orientation_accuracy": round(ok / n, 4),
                "mean_auroc_after": round(float(np.mean(by_suite_auroc[suite])), 4),
            }
            for suite, (ok, n) in sorted(by_suite.items())
            if n
        },
    }


def _by_cell(block: dict) -> dict[tuple[str, float], dict]:
    return {(c["family"], c["difficulty"]): c for c in block["cells"]}


def _invariance(detectors: dict) -> dict:
    """Whether every direction-free detector scored identically in both modes.

    Sign randomization multiplies ``behav_trace`` and nothing else, and
    :func:`rhob.v3.access.restrict` nulls that field below L2, so every L0/L1 cell and
    every cell of a magnitude-only L2 detector must be bit-identical with the flip on and
    off. Checked cell by cell rather than on the means, so a pair of compensating changes
    cannot hide.

    Only cells scored in *both* modes are compared: two not-applicable cells are not two
    identical measurements, and counting them so made the published "123/123 identical"
    count cells nothing had scored. A cell scored in one mode and not the other is listed
    in ``applicability_mismatches``, which must be empty -- applicability depends on the
    channels and the horizon, neither of which the flip touches.
    """
    report: dict[str, dict] = {}
    for name, entry in detectors.items():
        if entry["direction_dependent"]:
            continue
        before = entry["by_config"].get("before")
        after = entry["by_config"].get("after")
        if not before or not after:
            continue
        # Keyed by cell rather than zipped by position: the merged board concatenates
        # several jobs, and a positional comparison would quietly succeed on a
        # misalignment instead of reporting it.
        b_cells, a_cells = _by_cell(before), _by_cell(after)
        keys = sorted(set(b_cells) | set(a_cells))
        both = [
            k for k in keys
            if k in b_cells and k in a_cells
            and b_cells[k]["auroc"] is not None and a_cells[k]["auroc"] is not None
        ]
        applicability = [
            [k[0], k[1], (b_cells.get(k) or {}).get("auroc"), (a_cells.get(k) or {}).get("auroc")]
            for k in keys
            if ((b_cells.get(k) or {}).get("auroc") is None)
            != ((a_cells.get(k) or {}).get("auroc") is None)
        ]
        mismatched = [
            [k[0], k[1], b_cells[k]["auroc"], a_cells[k]["auroc"]]
            for k in both
            if b_cells[k]["auroc"] != a_cells[k]["auroc"]
        ]
        report[name] = {
            "access_level": entry["access_level"],
            "n_cells": len(keys),
            "n_scored": len(both),
            "n_na": len(keys) - len(both) - len(applicability),
            "n_identical": len(both) - len(mismatched),
            "mismatched_cells": mismatched[:10],
            "applicability_mismatches": applicability[:10],
            "n_applicability_mismatches": len(applicability),
        }
    return report


def _check_published_transfer(transfer_path: Path) -> None:
    """Refuse a "published" transfer file that is not the pre-randomization one.

    The ``before`` RTS is checked against the figure published before sign
    randomization. ``scripts/cross_family_transfer.py`` used to write that file's path,
    so a regenerated file there would silently be compared against a randomized
    measurement -- the trap ``--published-board`` already had.
    """
    if not transfer_path.exists():
        return
    board = load_transfer_results(transfer_path)
    if not board.predates_sign_randomization:
        raise SystemExit(
            f"report: --published-transfer {transfer_path} was measured with sign "
            "randomization on, so it is not the pre-randomization figure the `before` RTS "
            "is checked against. Use the 2026-07-09 artifact: "
            "git show 4728596:leaderboard/cross_family_transfer.json"
        )


def _reproduction_check(
    detectors: dict,
    board_path: Path,
    transfer_path: Path,
    board_label: Optional[str] = None,
) -> dict:
    """Does the ``before`` column reproduce the pre-audit artifacts?

    The ``after`` number is only worth reporting if the ``before`` one lands on the
    published figures it is supposed to be replacing, so this compares, family by family,
    the ``before`` column against the pre-audit board (``--published-board``; the
    ``leaderboard/v5_leaderboard.json`` of ``c54a8cc``) and the ``before`` RTS against
    the pre-randomization ``leaderboard/cross_family_transfer.json`` (2026-07-09).

    The ladder comparison is like for like: for each L2 detector, both sides are taken
    over the same families -- those this run scored for that detector *and* the board has
    a value for -- and both are weighted by this run's scored cells per family (the
    pre-audit board records only per-family means). Until 2026-11-27 this compared an
    unweighted family mean of the board over every family the run measured for *any*
    detector with this run's cell-weighted mean, which is not the same statistic.

    The pre-audit board records neither its ``n_seeds`` nor a sampling block (it predates
    both), so an exact match is not guaranteed and a mismatch is not by itself evidence of
    a regression -- ``max_abs_diff`` is reported per detector so the size of any
    disagreement is visible rather than hidden behind a boolean. A ``null`` family on
    either side (not applicable) is skipped, never compared.
    """
    if not board_path.exists():
        return {"published_board": f"{board_path} not found; not checked"}
    board = _load(board_path)
    published = board["results"]
    out: dict[str, dict] = {}
    for name, entry in detectors.items():
        block = entry["by_config"].get("before")
        if block is None or name not in published:
            continue
        their = published[name].get("per_family") or {}
        diffs = {
            fam: round(value - their[fam], 6)
            for fam, value in block["per_family"].items()
            if value is not None and their.get(fam) is not None
        }
        if not diffs:
            continue
        out[name] = {
            "n_families_compared": len(diffs),
            "n_exact": sum(1 for d in diffs.values() if d == 0),
            "max_abs_diff": round(max(abs(d) for d in diffs.values()), 6),
            "families_differing": {k: v for k, v in sorted(diffs.items()) if v != 0},
        }

    like_for_like: dict[str, dict] = {}
    for name in L2_DETECTORS:
        block = (detectors.get(name) or {}).get("by_config", {}).get("before")
        their = (published.get(name) or {}).get("per_family") or {}
        if block is None:
            continue
        scored = [c for c in block["cells"] if c["auroc"] is not None]
        counts = Counter(c["family"] for c in scored)
        fams = sorted(f for f in counts if their.get(f) is not None)
        if not fams:
            continue
        weight = sum(counts[f] for f in fams)
        theirs = sum(counts[f] * float(their[f]) for f in fams) / weight
        mine = float(np.mean([c["auroc"] for c in scored if c["family"] in set(fams)]))
        like_for_like[name] = {
            "n_families": len(fams),
            "n_cells": weight,
            "published": round(theirs, 4),
            "this_run_before": round(mine, 4),
        }
    ladder: dict = {}
    if like_for_like:
        pub = sorted(like_for_like.items(), key=lambda kv: -kv[1]["published"])
        mine = sorted(like_for_like.items(), key=lambda kv: -kv[1]["this_run_before"])
        ladder = {
            "n_detectors": len(like_for_like),
            "published_mean_auroc": round(
                float(np.mean([v["published"] for v in like_for_like.values()])), 4
            ),
            "this_run_mean_auroc": round(
                float(np.mean([v["this_run_before"] for v in like_for_like.values()])), 4
            ),
            "published_best_detector": pub[0][0],
            "published_max_auroc": pub[0][1]["published"],
            "this_run_best_detector": mine[0][0],
            "this_run_max_auroc": mine[0][1]["this_run_before"],
            "per_detector": dict(pub),
        }

    result: dict = {
        "published_board": board_label or Path(board_path).as_posix(),
        "published_board_note": (
            "the board records n_seeds_per_variant = "
            f"{board['sampling'].get('n_seeds_per_variant')}"
            if isinstance(board.get("sampling"), dict)
            else "the board records no sampling block; its n_seeds is not recoverable "
            "from the artifact"
        ),
        "per_family_vs_published": out,
        "l2_ladder_like_for_like": ladder,
    }
    if transfer_path.exists():
        published_rts = _load(transfer_path)["results"]
        result["published_transfer"] = Path(transfer_path).as_posix()
        result["published_rts"] = {
            row: value.get("avg_transfer_auroc_mean", value.get("avg_transfer_auroc"))
            for row, value in published_rts.items()
        }
    return result


def _rts(row: Optional[dict]) -> Optional[float]:
    """A transfer row's RTS, either shape; ``None`` when not run or not measurable."""
    if not row:
        return None
    return row.get("avg_transfer_auroc_mean", row.get("avg_transfer_auroc"))


def _rts_vs_published(payload: dict) -> dict[str, dict]:
    """``before`` RTS against the published one, per row, with the difference."""
    published = (payload.get("reproduction_check") or {}).get("published_rts") or {}
    before = ((payload.get("transfer") or {}).get("results") or {}).get("before") or {}
    out: dict[str, dict] = {}
    for row, value in published.items():
        mine = _rts(before.get(row))
        out[row] = {
            "published": value,
            "this_run_before": mine,
            "difference": (
                round(mine - value, 3) if mine is not None and value is not None else None
            ),
        }
    return out


def _merge_transfers(paths: list[Path], allow_dirty: bool = False) -> dict:
    """Merge per-configuration ``transfer`` artifacts into one block.

    Each configuration can be run as its own process -- they share no state and the
    experiment is hours long -- so the artifact may arrive in pieces. Merging refuses to
    proceed if two pieces disagree about the sampling design, since a ``before`` measured
    at one seed count and an ``after`` measured at another is not a comparison; if two
    pieces measured the same configuration; or if they are not one clean run
    (:func:`_uniform_provenance`).
    """
    merged: dict = {}
    loaded = [(Path(p), _load(p)) for p in paths]
    provenance = _uniform_provenance(loaded, allow_dirty=allow_dirty)
    for path, payload in loaded:
        design = {
            k: payload[k]
            for k in ("train_families", "test_families", "n_seeds_train",
                      "n_seeds_test", "n_trials")
        }
        if not merged:
            merged = {**design, "configs": {}, "results": {}}
        else:
            mismatch = {k: (merged[k], v) for k, v in design.items() if merged[k] != v}
            if mismatch:
                raise ValueError(f"{path} disagrees on the transfer design: {mismatch}")
        twice = sorted(set(merged["results"]) & set(payload["results"]))
        if twice:
            raise ValueError(f"{path} measures configuration(s) {twice} a second time")
        merged["configs"].update(payload["configs"])
        merged["results"].update(payload["results"])
        if payload.get("cell_semantics"):
            merged["cell_semantics"] = payload["cell_semantics"]
    merged["transfer_files"] = [p.as_posix() for p, _ in loaded]
    merged["transfer_provenance"] = provenance
    return merged


def _config_deltas(detectors: dict) -> dict:
    """Cell-level agreement between configuration pairs, for the direction-dependent rows.

    Three comparisons, each answering a different question:

    ``before`` vs ``blind``
        what the flip does to a detector that assumes the old convention.
    ``before_infer`` vs ``after``
        what the flip does to the shipped detector. Algebraically this should be *nothing*
        -- negating the feature negates the median split, which negates the inferred
        orientation, which cancels in the product -- so every cell where it is not zero is
        a cell where the orientation statistic was tied or within floating-point noise of
        a tie, and that count is worth reporting rather than assuming away.
    ``before`` vs ``before_infer``
        the cost of inferring the direction instead of being handed it, measured in the
        family's own coordinate with no randomization involved at all.

    ``n_cells`` counts the cells scored in both configurations; a not-applicable cell is
    not a cell that agreed.
    """
    pairs = [("before", "blind"), ("before_infer", "after"), ("before", "before_infer")]
    out: dict[str, dict] = {}
    for name, entry in detectors.items():
        if not entry["direction_dependent"]:
            continue
        row: dict[str, dict] = {}
        for left, right in pairs:
            lb, rb = entry["by_config"].get(left), entry["by_config"].get(right)
            if not lb or not rb:
                continue
            right_by_cell = {(c["family"], c["difficulty"]): c["auroc"] for c in rb["cells"]}
            both = [
                (c["family"], c["difficulty"], c["auroc"], right_by_cell[key])
                for c in lb["cells"]
                for key in [(c["family"], c["difficulty"])]
                if key in right_by_cell
                and c["auroc"] is not None and right_by_cell[key] is not None
            ]
            differing = [t for t in both if t[2] != t[3]]
            row[f"{left}_vs_{right}"] = {
                "n_cells": len(both),
                "n_differing": len(differing),
                "mean_left": lb["overall_auroc"],
                "mean_right": rb["overall_auroc"],
                "differing_cells": [
                    {"family": f, "difficulty": d, left: a, right: b}
                    for f, d, a, b in differing[:20]
                ],
            }
        out[name] = row
    return out


#: Columns of each row in the report's ``cells`` block.
CELL_COLUMNS = ["family", "difficulty", "auroc", "na_reason"]


def _cells_block(detectors: dict) -> dict:
    """Every per-cell AUROC, per detector and configuration, as compact rows."""
    return {
        name: {
            config: [
                [c["family"], c["difficulty"], c["auroc"], c.get("na_reason")]
                for c in sorted(block["cells"], key=lambda c: (c["family"], -c["difficulty"]))
            ]
            for config, block in entry["by_config"].items()
        }
        for name, entry in sorted(detectors.items())
    }


def _per_detector(detectors: dict) -> dict:
    """One row per detector: overall AUROC and scored/N-A counts per configuration."""
    per_detector: dict[str, dict] = {}
    for name, entry in sorted(detectors.items()):
        row: dict[str, Any] = {
            "access_level": entry["access_level"],
            "direction_dependent": entry["direction_dependent"],
        }
        counts: dict[str, dict[str, int]] = {
            "n_cells_scored": {}, "n_cells_na": {}, "n_families_scored": {},
        }
        for config in CONFIGS:
            block = entry["by_config"].get(config)
            if block is None and not entry["direction_dependent"]:
                # Scored once per randomization mode; ``blind``/``before_infer`` are the
                # same run as ``after``/``before`` for a detector with no orientation hook.
                block = entry["by_config"].get(
                    "after" if config == "blind" else "before"
                )
            row[config] = None if block is None else block["overall_auroc"]
            if block is not None:
                counts["n_cells_scored"][config] = block["n_cells_scored"]
                counts["n_cells_na"][config] = block["n_cells_na"]
                counts["n_families_scored"][config] = block.get(
                    "n_families_scored",
                    len({c["family"] for c in block["cells"] if c["auroc"] is not None}),
                )
        if row["before"] is not None and row["after"] is not None:
            row["delta_after_minus_before"] = round(row["after"] - row["before"], 4)
        row.update(counts)
        per_detector[name] = row
    return per_detector


def _is_published_design(design: dict, transfer: Optional[dict], companions: dict) -> bool:
    """Whether the inputs are the full published design, not a reduced run."""
    full_board = not design["families_not_measured"] and design["n_seeds"] == BOARD_N_SEEDS
    full_transfer = bool(transfer) and (
        set(transfer["results"]) == {"before", "blind", "after"}
        and transfer["n_trials"] == TRANSFER_N_TRIALS
        and transfer["n_seeds_train"] == 10
        and transfer["n_seeds_test"] == 20
        and all(
            set(transfer["results"][c]) == set(TRANSFER_ROWS_BY_CONFIG[c])
            for c in transfer["results"]
        )
    )
    return full_board and full_transfer and set(companions) == set(COMPANIONS)


def run_report(args: argparse.Namespace) -> None:
    """Merge the measurement artifacts into the committed JSON + Markdown table."""
    if args.published_board is None:
        raise SystemExit(
            "report: --published-board is required. The `before` column is checked against "
            "the pre-audit board, and leaderboard/v5_leaderboard.json stopped being that on "
            "2026-11-19, when it was regenerated with sign randomization on. Recover the "
            "pre-audit one with\n"
            "  git show c54a8cc:leaderboard/v5_leaderboard.json > pre_audit_board.json\n"
            "and pass --published-board pre_audit_board.json."
        )
    allow_dirty = bool(getattr(args, "allow_dirty", False))
    _check_published_transfer(args.published_transfer)
    # Every input -- boards, companions, transfers -- must be one clean run, not only
    # each kind among itself: a `before` board from one commit and an `after` RTS from
    # another would be two measurements presented as one comparison.
    companion_paths = [Path(spec.partition("=")[2]) for spec in (args.companion or [])]
    all_inputs = [Path(p) for p in args.board] + companion_paths + [
        Path(p) for p in (args.transfer or [])
    ]
    shared = _uniform_provenance([(p, _load(p)) for p in all_inputs], allow_dirty=allow_dirty)

    detectors, design = _merge_boards(args.board, allow_dirty=allow_dirty)
    if getattr(args, "require_all_families", False) and design["families_not_measured"]:
        raise SystemExit(
            "report: --require-all-families, but these registered families were not "
            f"measured: {design['families_not_measured']}"
        )
    design["configs"] = {
        k: {"randomize_behav_sign": r, "detector_infers_orientation": i}
        for k, (r, i) in CONFIGS.items()
    }
    design["input_provenance"] = shared

    axis = _difficulty_axis(detectors, "Behavioral Threshold")
    companions = _sensitivity(args.companion or [])
    transfer = _merge_transfers(args.transfer, allow_dirty=allow_dirty) if args.transfer else None
    design["published_design"] = _is_published_design(design, transfer, companions)
    payload: dict[str, Any] = {
        "measurement": "sign_randomization_impact",
        "design": design,
        "provenance": provenance_block(script="scripts/measure_sign_randomization.py"),
        "l2_access_ladder": {c: _ladder(detectors, c) for c in ("before", "after")},
        "per_detector": _per_detector(detectors),
        "difficulty_axis": axis,
        "knob_summary": _knob_summary(axis),
        "l0_l1_and_magnitude_invariance": _invariance(detectors),
        "config_deltas": _config_deltas(detectors),
        "sample_size_sensitivity": companions,
        "reproduction_check": _reproduction_check(
            detectors,
            args.published_board,
            args.published_transfer,
            board_label=getattr(args, "published_board_label", None),
        ),
        "cells_columns": CELL_COLUMNS,
        "cells": _cells_block(detectors),
    }
    if transfer is not None:
        payload["transfer"] = transfer
        payload["reproduction_check"]["rts_before_vs_published"] = _rts_vs_published(payload)
    _write(args.out_json, payload)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"[OK] wrote {args.out_md}", flush=True)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _span(values: list[float]) -> str:
    """``lo-hi`` over ``values`` (one value printed once), or ``n/a``."""
    values = _present(values)
    if not values:
        return "n/a"
    lo, hi = min(values), max(values)
    return f"{lo:.2f}" if f"{lo:.2f}" == f"{hi:.2f}" else f"{lo:.2f}-{hi:.2f}"


#: L2 detectors whose statistic is the feature's *signed* value (see :data:`ORIENTING`),
#: as opposed to Trajectory MLP, which is direction-dependent because it is fitted.
SIGN_READING = ("Behavioral Threshold", "Centroid Tracker")


def _rts_sentence(payload: dict) -> str:
    """The headline RTS bullet, comparing ``before`` with the published figure."""
    results = (payload.get("transfer") or {}).get("results") or {}
    before = _rts((results.get("before") or {}).get("Ensemble (Top 5)"))
    after = _rts((results.get("after") or {}).get("Ensemble (Top 5)"))
    if before is None and after is None:
        return ""
    check = (payload.get("reproduction_check") or {}).get("rts_before_vs_published") or {}
    row = check.get("Ensemble (Top 5)") or {}
    published, mine = row.get("published"), row.get("this_run_before")
    if published is None or mine is None:
        comparison = "The published pre-randomization figure was not compared."
    elif f"{published:.3f}" == f"{mine:.3f}":
        comparison = (
            f"With randomization off it reproduces the published {published:.3f} "
            "to three decimals."
        )
    else:
        comparison = (
            f"With randomization off it reads {mine:.3f}, against the published "
            f"{published:.3f} ({mine - published:+.3f}): the pre-randomization figure does "
            "not reproduce exactly."
        )
    return f"- **RTS (Ensemble Top-5) {_fmt(before)} -> {_fmt(after)}.** {comparison}"


def _sign_sentence(payload: dict) -> str:
    """Which L2 detectors moved, read off ``per_detector`` and the invariance check."""
    per = payload.get("per_detector") or {}
    inv = payload.get("l0_l1_and_magnitude_invariance") or {}
    signed = [n for n in SIGN_READING if n in per]
    magnitude = [
        n for n in L2_DETECTORS if n in per and not per[n]["direction_dependent"]
    ]
    if not signed or not magnitude:
        return ""
    unchanged = all(
        inv.get(n, {}).get("n_identical") == inv.get(n, {}).get("n_scored")
        and not inv.get(n, {}).get("n_applicability_mismatches")
        for n in magnitude
        if n in inv
    ) and all(n in inv for n in magnitude)
    after_state = (
        "identical after on every cell scored in both modes"
        if unchanged
        else "not identical after (see the invariance check)"
    )
    return (
        f"- **The L2 detectors that read the feature's *sign* "
        f"({', '.join(signed)}) scored {_span([per[n]['before'] for n in signed])} before "
        f"and {_span([per[n]['after'] for n in signed])} after.** The "
        f"{len(magnitude)} that read only a magnitude ({', '.join(magnitude)}) scored "
        f"{_span([per[n]['before'] for n in magnitude])} before and are {after_state}."
    )


def render_markdown(payload: dict) -> str:
    """The short committed table the docs cite and a test can pin.

    Every claim in it is computed from ``payload``. Until 2026-11-27 four were written
    into this function as text -- that the published RTS "reproduces exactly", which
    detectors scored ~0.97 and which 0.51-0.57, and that the orientation rule is "at or
    below a coin flip" -- and would have been printed whatever a re-run measured.
    """
    design = payload["design"]
    before_rung = payload["l2_access_ladder"].get("before") or {}
    after_rung = payload["l2_access_ladder"].get("after") or {}
    knob = payload.get("knob_summary") or {}
    lines: list[str] = [
        "# L2 sign randomization: measured before/after",
        "",
        "Generated by `scripts/measure_sign_randomization.py`; numbers come from",
        "`docs/figures/sign_randomization_impact.json`, which also carries every per-cell",
        "AUROC (`cells`). Every figure here is a measurement, not an estimate; anything that",
        "was not measured is named as such.",
        "",
        "## Headline",
        "",
    ]
    rts_line = _rts_sentence(payload)
    if rts_line:
        lines.append(rts_line)
    if before_rung and after_rung:
        lines.append(
            f"- **L2 suite mean {before_rung['mean_auroc']:.4f} -> "
            f"{after_rung['mean_auroc']:.4f}; best {before_rung['max_auroc']:.4f} "
            f"({before_rung['best_detector']}) -> {after_rung['max_auroc']:.4f} "
            f"({after_rung['best_detector']}).**"
        )
    sign_line = _sign_sentence(payload)
    if sign_line:
        lines.append(sign_line)
    if knob.get("n_families_with_more_than_one_tier"):
        zero = knob["n_families_separation_span_exactly_zero"]
        multi = knob["n_families_with_more_than_one_tier"]
        lines.append(
            "- **Difficulty knob.** `|AUROC - 0.5|` is invariant under the flip, so "
            f"randomization cannot revive a pinned knob: {zero} of {multi} multi-tier "
            "families have a separation span of exactly 0.000."
        )
    published_design = design.get("published_design")
    lines += [
        "",
        "## Design",
        "",
        f"- `n_seeds` = {design['n_seeds']} per variant per cell, layout seed 0, one draw"
        " (`n_replicates = 1`).",
        f"- Families measured: **{len(design['families_measured'])} of"
        f" {len(design['families_registered'])}**.",
    ]
    if published_design is False:
        lines.append(
            "- **Reduced design** (`design.published_design = false`): a smoke run, not "
            "the published measurement."
        )
    prov = design.get("input_provenance") or design.get("board_provenance") or {}
    if prov.get("short_commit"):
        dirty = prov.get("dirty")
        state = "clean tree" if dirty is False else f"tree dirty = {dirty}"
        lines.append(
            f"- Measured at commit `{prov['short_commit']}` ({state}), Python "
            f"{prov.get('python')}."
        )
    if design["families_not_measured"]:
        lines.append(
            "- Not measured: "
            + ", ".join(f"`{f}`" for f in design["families_not_measured"])
            + "."
        )
    lines += [
        "",
        "| config | `randomize_behav_sign` | detector infers orientation | what it is |",
        "| --- | --- | --- | --- |",
        "| `before` | off | no | the published pre-audit measurement |",
        "| `before_infer` | off | yes | shipped detector, old convention |",
        "| `blind` | on | no | what the pre-audit detector actually knew |",
        "| `after` | on | yes | the shipped configuration |",
        "",
        "## L2 access-level ladder",
        "",
        "| config | detectors | mean AUROC | SD across detectors | best AUROC | best detector |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for config in ("before", "after"):
        rung = payload["l2_access_ladder"].get(config) or {}
        if not rung:
            continue
        lines.append(
            f"| `{config}` | {rung['n_detectors']} | {rung['mean_auroc']:.4f} | "
            f"{rung['sd_across_detectors']:.4f} | {rung['max_auroc']:.4f} | "
            f"{rung['best_detector']} |"
        )
    lines += [
        "",
        "SD is the spread across the detectors at the level, not a confidence interval: "
        "the detectors share one evaluation draw, so these are not independent replicates. "
        "The detectors are not all averaged over the same cells: a detector that declares "
        "`min_episodes` is not applicable on families shorter than that, and the "
        "`cells scored` column below says how many each one measured. `Trajectory MLP` "
        "is fitted inside each cell under 5-fold cross-validation -- its training folds "
        "carry labels, so it learns the cell's direction from them. That is legitimate "
        "under the benchmark's rules and is why it barely moves, but where it is the "
        "`after` maximum, that maximum is a supervised result, not a statement that the "
        "unsupervised behavioral signal still separates.",
        "",
        "## Per-detector L2 (and L3) AUROC",
        "",
        "| detector | access | direction-dependent | cells scored | `before` | "
        "`before_infer` | `blind` | `after` | `after - before` |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, row in payload["per_detector"].items():
        scored = row.get("n_cells_scored") or {}
        na = row.get("n_cells_na") or {}
        n_after = scored.get("after", scored.get("before"))
        n_before = scored.get("before", n_after)
        cells = "n/a" if n_after is None else (
            f"{n_after}" if n_before == n_after else f"{n_before} / {n_after}"
        )
        if na.get("after"):
            cells += f" (+{na['after']} N/A)"
        lines.append(
            f"| {name} | {row['access_level']} | "
            f"{'yes' if row['direction_dependent'] else 'no'} | {cells} | "
            f"{_fmt(row['before'])} | {_fmt(row['before_infer'])} | {_fmt(row['blind'])} | "
            f"{_fmt(row['after'])} | {_fmt(row.get('delta_after_minus_before'))} |"
        )

    transfer = payload.get("transfer")
    if transfer:
        lines += [
            "",
            "## Cross-family transfer (RTS)",
            "",
            f"Train on {len(transfer['train_families'])} families, evaluate frozen on "
            f"{len(transfer['test_families'])} held out. "
            f"`n_seeds_train={transfer['n_seeds_train']}`, "
            f"`n_seeds_test={transfer['n_seeds_test']}`, "
            f"`n_trials={transfer['n_trials']}`: trial `k` seeds the networks with `k`, "
            "as `scripts/cross_family_transfer.py` does, so these trials are the first "
            f"{transfer['n_trials']} of the ones in `leaderboard/cross_family_transfer_v5.json` "
            "(the +/- across model inits is reported in the JSON, not here).",
            "",
            "The L0 and L1 rows are measured in `before` only, and marked `—` elsewhere "
            "rather than re-run: `restrict()` nulls `behav_trace` below L2, so their cells "
            "are bit-identical in both modes. That is not taken on trust -- the invariance "
            "table at the end of this file checks it cell by cell on the full board. "
            "`N/A` is a row no held-out family could be scored for; it is not 0.5.",
            "",
            "| detector | access | " + " | ".join(
                f"`{c}` train / RTS" for c in transfer["results"]
            ) + " |",
            "| --- | --- | " + " | ".join("---" for _ in transfer["results"]) + " |",
        ]
        rows = sorted({r for cfg in transfer["results"].values() for r in cfg})
        not_measurable: dict[str, str] = {}
        for row_name in rows:
            cells = []
            level = ""
            for cfg in transfer["results"]:
                entry = transfer["results"][cfg].get(row_name)
                if entry is None:
                    cells.append("—")
                    continue
                level = entry["access_level"]
                train = entry.get("train_auroc_mean", entry.get("train_auroc"))
                rts = _rts(entry)
                cells.append(f"{_fmt(train)} / **{'N/A' if rts is None else f'{rts:.3f}'}**")
                if rts is None:
                    reasons = entry.get("test_not_applicable_reasons") or {}
                    first = next(iter(dict.fromkeys(reasons.values())), "no reason recorded")
                    not_measurable.setdefault(row_name, first)
            lines.append(f"| {row_name} | {level} | " + " | ".join(cells) + " |")
        if not_measurable:
            lines.append("")
            for row_name, reason in not_measurable.items():
                lines.append(f"- {row_name}: no held-out family scored ({reason}).")

    repro = payload.get("reproduction_check") or {}
    ladder = repro.get("l2_ladder_like_for_like")
    if ladder:
        lines += [
            "",
            "## Reproduction of the pre-audit figures",
            "",
            "The `after` number is only worth anything if `before` lands on the published "
            "figures it replaces. Each row compares the pre-audit board with this run's "
            "`before` over the same families -- those this run scored for that detector "
            "and the board has a value for -- weighted by this run's scored cells per "
            "family on both sides:",
            "",
            "| detector | families | cells | pre-audit board | this run, `before` |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, row in ladder["per_detector"].items():
            lines.append(
                f"| {name} | {row['n_families']} | {row['n_cells']} | "
                f"{row['published']:.4f} | {row['this_run_before']:.4f} |"
            )
        lines += [
            f"| **L2 suite mean** | | | **{ladder['published_mean_auroc']:.4f}** | "
            f"**{ladder['this_run_mean_auroc']:.4f}** |",
            "",
            f"Pre-audit best at L2: **{ladder['published_best_detector']} "
            f"{ladder['published_max_auroc']:.4f}**; this run's `before` best: "
            f"**{ladder['this_run_best_detector']} {ladder['this_run_max_auroc']:.4f}**.",
        ]
    rts_check = repro.get("rts_before_vs_published") or {}
    if rts_check:
        lines += [
            "",
            "Published (2026-07-09, pre-randomization) RTS against this run's `before` RTS:",
            "",
            "| detector | published | this run, `before` | difference |",
            "| --- | --- | --- | --- |",
        ]
        for row, value in rts_check.items():
            diff = value["difference"]
            lines.append(
                f"| {row} | {_fmt(value['published'])} | {_fmt(value['this_run_before'])} | "
                f"{'n/a' if diff is None else f'{diff:+.3f}'} |"
            )

    lines += [
        "",
        "## Difficulty axis (Behavioral Threshold, per requested difficulty)",
        "",
        "`span` is `max - min` of the achieved AUROC across a family's tiers. "
        "`separation span` is the same over `|AUROC - 0.5|`, which the coordinate flip "
        "leaves invariant and which is therefore **the statistic that answers whether the "
        "knob moves**. `span after` is not: orientation is inferred per cell, so two tiers "
        "of one family can be oriented oppositely and produce a span of 1.000 at identical "
        "separation. Read `separation span`, not `span after`.",
        "",
    ]
    if knob:
        accuracy = knob.get("orientation_accuracy")
        verdict = (
            "not measured"
            if accuracy is None
            else ("at or below a coin flip" if accuracy <= 0.5 else "above a coin flip")
        )
        lines += [
            f"Of the **{knob['n_families_with_more_than_one_tier']}** measured families "
            f"with more than one tier, **{knob['n_families_separation_span_exactly_zero']}** "
            "have a separation span of exactly **0.000**. The families where the knob does "
            "move anything: "
            + (
                ", ".join(
                    f"`{f}` ({v:.3f})"
                    for f, v in knob["families_whose_separation_span_is_nonzero"].items()
                )
                or "none"
            )
            + ".",
            "",
            f"Cells oriented as the family's authored coordinate: "
            f"**{knob['cells_oriented_as_the_authored_coordinate']}** "
            f"({_fmt(accuracy)}) — the shipped orientation rule is {verdict} on the "
            "families measured here. Split by family group, because the 63.3% in "
            "`docs/l2_sign_randomization.md` (38 of 60) was measured on the tabular and "
            "continuous families alone:",
            "",
            "| family group | cells oriented | accuracy | mean AUROC `after` |",
            "| --- | --- | --- | --- |",
        ] + [
            f"| {suite} | {block['cells_oriented']} | {block['orientation_accuracy']:.3f} | "
            f"{block['mean_auroc_after']:.3f} |"
            for suite, block in knob.get("by_family_group", {}).items()
        ] + [""]
    lines += [
        "| family | suite | drawn sign | tiers | span `before` | span `after` | "
        "separation span | cells oriented |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for fam, block in payload["difficulty_axis"].items():
        lines.append(
            f"| `{fam}` | {block['suite']} | {block['drawn_sign']:+.0f} | "
            f"{len(block['tiers'])} | {_fmt(block['span_before'])} | "
            f"{_fmt(block['span_after'])} | {_fmt(block['separation_span'])} | "
            f"{block['cells_oriented'] or 'n/a'} |"
        )

    tiers = sorted(
        {t for block in payload["difficulty_axis"].values() for t in block["tiers"]},
        key=float,
        reverse=True,
    )
    if tiers:
        lines += [
            "",
            "### Per-tier detail (`before` / `after`)",
            "",
            "Requested difficulty is a target L2-AUROC, so a live knob would show these "
            "rows falling left to right.",
            "",
            "| family | " + " | ".join(f"L2\\*={t}" for t in tiers) + " |",
            "| --- | " + " | ".join("---" for _ in tiers) + " |",
        ]
        for fam, block in payload["difficulty_axis"].items():
            cells = []
            for tier in tiers:
                entry = block["tiers"].get(tier)
                if entry is None:
                    cells.append("—")
                else:
                    cells.append(
                        f"{_fmt(entry.get('before'))} / {_fmt(entry.get('after'))}"
                    )
            lines.append(f"| `{fam}` | " + " | ".join(cells) + " |")

    sens = payload.get("sample_size_sensitivity") or {}
    if sens:
        lines += [
            "",
            "## Other sample sizes",
            "",
            "Two published figures were produced at other sample sizes and are checked "
            "against their own design, not against this one.",
            "",
            "| run | `n_seeds` | families | detector | `before` | `before_infer` | "
            "`blind` | `after` |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for label, block in sens.items():
            for name, configs in block["detectors"].items():
                lines.append(
                    f"| {label} | {block['n_seeds']} | {block['n_families']} | {name} | "
                    + " | ".join(
                        _fmt(configs.get(c))
                        for c in ("before", "before_infer", "blind", "after")
                    )
                    + " |"
                )

    deltas = payload.get("config_deltas") or {}
    if deltas:
        lines += [
            "",
            "## Configuration deltas, cell by cell",
            "",
            "Only the direction-dependent detectors can move between configurations. "
            "`before_infer` vs `after` is the algebraic identity check: negating the "
            "feature negates the median split, which negates the inferred orientation, "
            "which cancels — so it should be zero everywhere except where the orientation "
            "statistic was tied. `cells` counts the cells scored in both configurations.",
            "",
            "| detector | comparison | cells | differing | mean left | mean right |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name, row in deltas.items():
            for comparison, block in row.items():
                lines.append(
                    f"| {name} | `{comparison.replace('_vs_', '` vs `')}` | "
                    f"{block['n_cells']} | {block['n_differing']} | "
                    f"{_fmt(block['mean_left'])} | {_fmt(block['mean_right'])} |"
                )

    inv = payload["l0_l1_and_magnitude_invariance"]
    if inv:
        lines += [
            "",
            "## Invariance check",
            "",
            "Every detector whose score does not depend on the feature's direction must "
            "score identically with randomization on and off. Checked cell by cell, over "
            "the cells scored in both modes; a cell not applicable in both is counted "
            "separately, and one scored in only one mode would be an applicability "
            "mismatch.",
            "",
            "| detector | access | cells | identical / scored in both | not applicable | "
            "applicability mismatches |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name, block in inv.items():
            scored = block.get("n_scored", block["n_cells"])
            lines.append(
                f"| {name} | {block['access_level']} | {block['n_cells']} | "
                f"{block['n_identical']}/{scored} | {block.get('n_na', 0)} | "
                f"{block.get('n_applicability_mismatches', 0)} |"
            )
    lines.append("")
    return "\n".join(lines)


def _dumps(obj: Any, level: int = 0) -> str:
    """JSON at indent 2, except that a list of at most four scalars stays on one line.

    That keeps the report's ``cells`` block -- several thousand
    ``[family, difficulty, auroc, na_reason]`` rows -- one row per line instead of six,
    and is otherwise ``json.dumps(indent=2)``. NaN is refused: a not-applicable value is
    ``null``.
    """
    pad, inner = "  " * level, "  " * (level + 1)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [
            f"{inner}{json.dumps(str(k))}: {_dumps(v, level + 1)}" for k, v in obj.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, (list, tuple)):
        if not obj:
            return "[]"
        if len(obj) <= 4 and not any(isinstance(x, (dict, list, tuple)) for x in obj):
            return "[" + ", ".join(json.dumps(x, allow_nan=False) for x in obj) + "]"
        return "[\n" + ",\n".join(f"{inner}{_dumps(v, level + 1)}" for v in obj) + "\n" + pad + "]"
    return json.dumps(obj, allow_nan=False)


def _write(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dumps(payload) + "\n", encoding="utf-8")
    print(f"\n[OK] wrote {path}", flush=True)


def _git_show(spec: str, dest: Path) -> Path:
    """Write ``git show <spec>`` to ``dest``; a clear error if the revision is missing."""
    try:
        blob = subprocess.run(
            ["git", "-C", str(_REPO), "show", spec],
            check=True, capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"") or b""
        raise SystemExit(
            f"ci: cannot read {spec} from git ({detail.decode(errors='replace').strip() or exc}). "
            "It needs the full history (actions/checkout with fetch-depth: 0), or pass "
            "--published-board with the pre-audit board written out by hand."
        ) from None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    return dest


#: ``ci`` options that define the published design, with their published values. Any
#: other value makes the artifact a reduced run (``design.published_design = false``).
CI_DEFAULTS: dict[str, Any] = {
    "families": ["all"],
    "only": None,
    "n_seeds": BOARD_N_SEEDS,
    "companions": True,
    "configs": ["before", "blind", "after"],
    "transfer_rows": None,
    "n_trials": TRANSFER_N_TRIALS,
    "n_seeds_train": 10,
    "n_seeds_test": 20,
    "train_families": None,
    "test_families": None,
}


def run_ci(args: argparse.Namespace) -> None:
    """The published design, end to end, in this process: board, companions, transfer, report.

    Stages write their intermediate artifacts under ``--work-dir`` (gitignored
    ``results/`` by default) and ``report`` reads them back, so the merged artifact is
    produced by exactly the code a by-hand run uses -- including every refusal in
    :func:`_merge_boards` and :func:`_uniform_provenance`. Rollouts are shared between
    stages through ``Benchmark._rollout_cache``: the transfer's train-family runs at
    ``n_seeds_train=10`` are the board's own cells.
    """
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    stage_s: dict[str, float] = {}

    published_board, board_label = args.published_board, None
    if published_board is None:
        published_board = _git_show(PRE_AUDIT_BOARD_REV, work / "pre_audit_board.json")
        board_label = f"git show {PRE_AUDIT_BOARD_REV}"
    _check_published_transfer(args.published_transfer)

    selected = None if args.families == ["all"] else set(args.families)
    t0 = time.time()
    board_path = work / "board.json"
    run_board(argparse.Namespace(
        families=args.families, n_seeds=args.n_seeds, only=args.only, out=board_path,
    ))
    stage_s["board"] = round(time.time() - t0, 1)

    companion_specs: list[str] = []
    if args.companions:
        for label, (n_seeds, fams) in COMPANIONS.items():
            fams = [f for f in fams if selected is None or f in selected]
            only = sorted(ORIENTING & set(args.only)) if args.only else sorted(ORIENTING)
            if not fams or not only:
                continue
            t0 = time.time()
            path = work / f"board_{label}.json"
            run_board(argparse.Namespace(families=fams, n_seeds=n_seeds, only=only, out=path))
            stage_s[f"companion_{label}"] = round(time.time() - t0, 1)
            companion_specs.append(f"{label}={path.as_posix()}")

    transfer_paths: list[Path] = []
    for config in args.configs:
        rows = [
            r for r in TRANSFER_ROWS_BY_CONFIG[config]
            if args.transfer_rows is None or r in set(args.transfer_rows)
        ]
        if not rows:
            continue
        t0 = time.time()
        path = work / f"transfer_{config}.json"
        run_transfer(argparse.Namespace(
            configs=[config], rows=rows, rows_by_config=None,
            n_seeds_train=args.n_seeds_train, n_seeds_test=args.n_seeds_test,
            n_trials=args.n_trials, train_families=args.train_families,
            test_families=args.test_families, out=path,
        ))
        stage_s[f"transfer_{config}"] = round(time.time() - t0, 1)
        transfer_paths.append(path)

    t0 = time.time()
    run_report(argparse.Namespace(
        board=[board_path],
        transfer=transfer_paths or None,
        companion=companion_specs or None,
        published_board=Path(published_board),
        published_board_label=board_label,
        published_transfer=args.published_transfer,
        out_json=args.out_json,
        out_md=args.out_md,
        allow_dirty=args.allow_dirty,
        require_all_families=args.families == ["all"],
    ))
    stage_s["report"] = round(time.time() - t0, 1)

    # A reduced ci run must not read as the published design even if every stage's
    # inputs happen to look complete (e.g. --only on the full family list).
    reduced = {k: getattr(args, k) for k, v in CI_DEFAULTS.items() if getattr(args, k) != v}
    payload = _load(args.out_json)
    payload["design"]["published_design"] = (
        not reduced and payload["design"].get("published_design", False)
    )
    payload["design"]["ci_reduced_options"] = reduced
    payload["design"]["ci_stage_wall_clock_s"] = {
        **stage_s, "total": round(time.time() - t_start, 1),
    }
    _write(args.out_json, payload)
    args.out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"[OK] ci finished in {time.time() - t_start:.0f}s: {stage_s}", flush=True)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    board = sub.add_parser("board", help="per-cell AUROC, all four configurations")
    board.add_argument("--families", nargs="+", default=["all"])
    board.add_argument("--n-seeds", type=int, default=BOARD_N_SEEDS)
    board.add_argument("--only", nargs="+", default=None, help="detector names to score")
    board.add_argument(
        "--out", type=Path, default=Path("results/sign_randomization/board.json")
    )
    board.set_defaults(func=run_board)

    transfer = sub.add_parser("transfer", help="RTS in each configuration")
    transfer.add_argument("--n-seeds-train", type=int, default=10)
    transfer.add_argument("--n-seeds-test", type=int, default=20)
    transfer.add_argument("--n-trials", type=int, default=TRANSFER_N_TRIALS)
    transfer.add_argument(
        "--configs", nargs="+", default=["before", "blind", "after"],
        choices=sorted(CONFIGS),
    )
    transfer.add_argument(
        "--rows", nargs="+",
        default=["Reward MLP", "State Divergence", "Trajectory MLP", "Ensemble (Top 5)"],
    )
    transfer.add_argument("--train-families", nargs="+", default=None, metavar="FAMILY",
                          help="Smoke testing only: replace the 6 train families.")
    transfer.add_argument("--test-families", nargs="+", default=None, metavar="FAMILY",
                          help="Smoke testing only: replace the 8 held-out families.")
    transfer.add_argument(
        "--out", type=Path, default=Path("results/sign_randomization/transfer.json")
    )
    transfer.set_defaults(func=run_transfer, rows_by_config=None)

    def add_report_inputs(p: argparse.ArgumentParser, *, board_required: bool) -> None:
        p.add_argument(
            "--published-board", type=Path, default=None,
            help="Pre-audit board the `before` column has to reproduce; "
                 "leaderboard/v5_leaderboard.json is no longer it (regenerated 2026-11-19 "
                 "with sign randomization on). "
                 + (
                     "Required: write it out with `git show "
                     f"{PRE_AUDIT_BOARD_REV} > <path>` and pass <path>."
                     if board_required
                     else f"Default: `git show {PRE_AUDIT_BOARD_REV}`, written under "
                     "--work-dir (needs the full git history)."
                 ),
        )
        p.add_argument(
            "--published-transfer", type=Path, default=PUBLISHED_TRANSFER,
            help="Pre-randomization RTS artifact the `before` RTS is compared with "
                 f"(default {PUBLISHED_TRANSFER}, the 2026-07-09 file). A file measured "
                 "with sign randomization on is refused.",
        )
        p.add_argument(
            "--out-json", type=Path,
            default=Path("docs/figures/sign_randomization_impact.json"),
        )
        p.add_argument(
            "--out-md", type=Path,
            default=Path("docs/figures/sign_randomization_impact.md"),
        )
        p.add_argument(
            "--allow-dirty", action="store_true",
            help="Local smoke runs only: accept inputs generated from a tree that is not "
                 "known to be clean. Inputs from different commits are refused regardless.",
        )

    report = sub.add_parser("report", help="merge the artifacts into JSON + Markdown")
    report.add_argument("--board", nargs="+", type=Path, required=True)
    report.add_argument("--transfer", nargs="+", type=Path, default=None)
    report.add_argument(
        "--companion", nargs="+", default=None, metavar="LABEL=PATH",
        help="Board runs at other sample sizes, reported alongside but never merged "
             "into the primary aggregates.",
    )
    report.add_argument(
        "--require-all-families", action="store_true",
        help="Refuse unless the boards cover every registered family.",
    )
    # The merged artifact goes to ``docs/figures/`` because ``results/`` is gitignored
    # and this is the file the docs cite and a test pins. It carries every per-cell AUROC
    # from every board job (its ``cells`` block), so the (large, intermediate) per-job
    # files under ``results/`` are not needed to audit it.
    add_report_inputs(report, board_required=True)
    report.set_defaults(func=run_report)

    ci = sub.add_parser(
        "ci",
        help="the published design end to end in one process (board, companions, "
             "transfer, report)",
    )
    add_report_inputs(ci, board_required=False)
    ci.add_argument(
        "--work-dir", type=Path, default=Path("results/sign_randomization/ci"),
        help="Where the intermediate board/transfer artifacts go (gitignored by default).",
    )
    smoke = ci.add_argument_group(
        "test-only options",
        "Each narrows the published design for a smoke run; any non-default value sets "
        "design.published_design = false in the artifact.",
    )
    smoke.add_argument("--families", nargs="+", default=CI_DEFAULTS["families"],
                       help="Board (and companion) families; default all registered.")
    smoke.add_argument("--only", nargs="+", default=CI_DEFAULTS["only"],
                       help="Board detectors; default all.")
    smoke.add_argument("--n-seeds", type=int, default=CI_DEFAULTS["n_seeds"])
    smoke.add_argument("--no-companions", dest="companions", action="store_false",
                       help="Skip the n5/n20 companion boards.")
    smoke.add_argument("--configs", nargs="+", default=CI_DEFAULTS["configs"],
                       choices=sorted(CONFIGS), help="Transfer configurations.")
    smoke.add_argument("--transfer-rows", nargs="+", default=CI_DEFAULTS["transfer_rows"],
                       help="Restrict the transfer rows (default: the design's per config).")
    smoke.add_argument("--n-trials", type=int, default=CI_DEFAULTS["n_trials"])
    smoke.add_argument("--n-seeds-train", type=int, default=CI_DEFAULTS["n_seeds_train"])
    smoke.add_argument("--n-seeds-test", type=int, default=CI_DEFAULTS["n_seeds_test"])
    smoke.add_argument("--train-families", nargs="+", default=CI_DEFAULTS["train_families"])
    smoke.add_argument("--test-families", nargs="+", default=CI_DEFAULTS["test_families"])
    ci.set_defaults(func=run_ci)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
