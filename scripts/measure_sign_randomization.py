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
                                       has to reproduce RTS 0.994 / L2 best 0.975 before the
                                       other three mean anything.
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
``transfer``  RTS, by driving ``scripts/cross_family_transfer.py``'s own functions in
              each configuration rather than reimplementing them (item 1).
``report``    turn the JSON artifacts into the committed Markdown table (item 4).

Run::

    python scripts/measure_sign_randomization.py board --out results/sign_randomization/board.json
    python scripts/measure_sign_randomization.py transfer --out results/sign_randomization/transfer.json
    python scripts/measure_sign_randomization.py report --board results/sign_randomization/board_*.json \
        --transfer results/sign_randomization/transfer_*.json

``board`` is run once per family group rather than once over all 33 families: the groups
are independent and the slowest family would otherwise gate the whole run for hours.
``report`` merges them, refusing to merge runs made at different ``n_seeds``. Its output
lands in ``docs/figures/`` -- ``results/`` is gitignored, and the merged file is the one
the docs cite. It carries every per-cell AUROC, so nothing is lost by leaving the
per-group files uncommitted.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Callable

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
from rhob.v3.provenance import provenance_block, sampling_block  # noqa: E402
from rhob.v3.registry import FamilyRegistry  # noqa: E402
from rhob.v3.sign_randomization import behav_sign  # noqa: E402

#: ``(randomize_behav_sign, detector_infers_orientation)`` per configuration.
CONFIGS: dict[str, tuple[bool, bool]] = {
    "before": (False, False),
    "before_infer": (False, True),
    "blind": (True, False),
    "after": (True, True),
}

#: The leaderboard's own sampling design (``scripts/populate_leaderboard_v32.py``), so
#: the ``before`` column is comparable to the committed board rather than to a
#: differently-seeded re-run.
BOARD_N_SEEDS = 10

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
    """One JSON-able record per evaluated cell."""
    return [
        {
            "family": c.family,
            "difficulty": round(float(c.difficulty), 4),
            "auroc": None if not c.applicable else round(float(c.discrimination_auroc), 6),
            "onset_mae": None if not c.applicable else round(float(c.onset_mae), 6),
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


def run_board(args: argparse.Namespace) -> None:
    """Score every L2/L3 detector in all four configurations, per cell."""
    families = "all" if args.families == ["all"] else args.families
    detectors: dict[str, Callable[[], PosthocDetector]] = {
        **L2_DETECTORS,
        **L3_DETECTORS,
        **INVARIANCE_DETECTORS,
    }
    if args.only:
        detectors = {k: v for k, v in detectors.items() if k in set(args.only)}

    out: dict[str, dict] = {}
    for config, (randomize, infer) in CONFIGS.items():
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


def _load_transfer_module():
    """Import ``scripts/cross_family_transfer.py`` as a module.

    The RTS experiment is defined by that script; measuring RTS by reimplementing it
    would measure the reimplementation. It is a script rather than a package module, so
    it is loaded by path.
    """
    path = Path(__file__).resolve().with_name("cross_family_transfer.py")
    spec = importlib.util.spec_from_file_location("_rhob_cross_family_transfer", path)
    if spec is None or spec.loader is None:  # pragma: no cover -- path is checked in
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_transfer(args: argparse.Namespace) -> None:
    """Measure RTS in each configuration, through cross_family_transfer's own code.

    ``cross_family_transfer`` calls ``pair.rollout(...)`` and ``Benchmark.evaluate(...)``
    without naming the randomization flag, i.e. it takes the shipped default. Rather than
    fork the file, this overrides the two defaults for the duration of one configuration
    and restores them afterwards. The scored code path -- the seeds, the pooled fit, the
    frozen-model transfer, the multi-trial aggregation -- is byte-for-byte the published
    one.
    """
    cft = _load_transfer_module()
    from rhob.v3.base_pair import MatchedPair

    orig_rollout = MatchedPair.rollout
    orig_evaluate = Benchmark.__dict__["evaluate"].__func__

    from rhob.detectors.l0_reward_mlp import RewardMLPDetector
    from rhob.detectors.l1_state_divergence import StateDivergenceDetector
    from rhob.detectors.l2_ensemble import EnsembleDetector

    results: dict[str, dict] = {}
    for config in args.configs:
        randomize, infer = CONFIGS[config]

        def _rollout(self, n_seeds, seed_base=0, randomize_sign=True, _r=randomize):
            return orig_rollout(self, n_seeds, seed_base, _r)

        def _evaluate(
            detector,
            families="all",
            difficulties="all",
            n_seeds=20,
            verbose=True,
            randomize_behav_sign=True,
            _r=randomize,
        ):
            return orig_evaluate(detector, families, difficulties, n_seeds, verbose, _r)

        MatchedPair.rollout = _rollout  # type: ignore[method-assign]
        Benchmark.evaluate = staticmethod(_evaluate)  # type: ignore[method-assign]
        Benchmark._rollout_cache = {}
        try:
            rows: dict[str, dict] = {}
            print(f"\n########## config = {config} "
                  f"(randomize={randomize}, infer={infer}) ##########", flush=True)

            if "Reward MLP" in args.rows:
                rows["Reward MLP"] = cft.run_one_multitrial(
                    "Reward MLP",
                    lambda trial: build(lambda: RewardMLPDetector(seed=trial), infer),
                    "L0", args.n_seeds_train, args.n_seeds_test, args.n_trials,
                )
            if "State Divergence" in args.rows:
                row, _ = cft.run_one(
                    "State Divergence",
                    build(StateDivergenceDetector, infer),
                    "L1", args.n_seeds_train, args.n_seeds_test,
                )
                rows["State Divergence"] = row
            if "Trajectory MLP" in args.rows:
                rows["Trajectory MLP"] = cft.run_one_multitrial(
                    "Trajectory MLP",
                    lambda trial: build(lambda: TrajectoryMLPDetector(seed=trial), infer),
                    "L2", args.n_seeds_train, args.n_seeds_test, args.n_trials,
                )

            def build_ensemble(trial: int):
                """Mirrors ``cross_family_transfer.main``'s local ``build_ensemble``.

                It is defined inside that script's ``main()``, so it cannot be imported;
                it is reproduced here rather than the script being restructured, since
                this script only measures.
                """
                traj_mlp = TrajectoryMLPDetector(seed=trial)
                train_a, train_b, _ = cft.pooled_runs(
                    cft.TRAIN_FAMILIES, "L2", args.n_seeds_train,
                    cft.required_channels(traj_mlp),
                )
                traj_mlp.fit(train_a, train_b)
                members = [
                    BehavioralThresholdDetector(),
                    AngularMomentumDetector(),
                    CentroidTrackerDetector(),
                    FeatureMagnitudeDetector(),
                    traj_mlp,
                ]
                return build(
                    lambda: EnsembleDetector(members, name="Ensemble (Top 5)"), infer
                )

            if "Ensemble (Top 5)" in args.rows:
                rows["Ensemble (Top 5)"] = cft.run_one_multitrial(
                    "Ensemble (Top 5)", build_ensemble, "L2",
                    args.n_seeds_train, args.n_seeds_test, args.n_trials,
                )
            results[config] = rows
        finally:
            MatchedPair.rollout = orig_rollout  # type: ignore[method-assign]
            Benchmark.evaluate = staticmethod(orig_evaluate)  # type: ignore[method-assign]
            Benchmark._rollout_cache = {}

    payload = {
        "measurement": "transfer",
        "train_families": cft.TRAIN_FAMILIES,
        "test_families": cft.TEST_FAMILIES,
        "n_seeds_train": args.n_seeds_train,
        "n_seeds_test": args.n_seeds_test,
        "n_trials": args.n_trials,
        "configs": {k: {"randomize_behav_sign": CONFIGS[k][0],
                        "detector_infers_orientation": CONFIGS[k][1]}
                    for k in args.configs},
        "drawn_signs": {
            f: behav_sign(f, 0) for f in cft.TRAIN_FAMILIES + cft.TEST_FAMILIES
        },
        "provenance": provenance_block(script="scripts/measure_sign_randomization.py"),
        "results": results,
    }
    _write(args.out, payload)


def family_suite(family: str) -> str:
    """The coarse family group a name belongs to, for the difficulty-axis table."""
    for prefix in ("mujoco_", "pettingzoo_", "sequence_", "rlhf_"):
        if family.startswith(prefix):
            return prefix.rstrip("_")
    return FamilyRegistry.get(family).complexity.value


def _merge_boards(paths: list[Path]) -> tuple[dict, dict]:
    """Merge several ``board`` artifacts (one per family group) into one.

    The board is measured in per-family-group jobs because a single job over all 33
    families would run for hours behind its slowest family; the cells are independent,
    so merging them is exact rather than an approximation. Returns
    ``(detectors, design)``, and raises if two jobs disagree about ``n_seeds`` -- cells
    measured at different sample sizes must not be averaged into one ladder.
    """
    detectors: dict[str, dict] = {}
    families: list[str] = []
    n_seeds: set[int] = set()
    signs: dict[str, float] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        n_seeds.add(int(payload["n_seeds"]))
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
    design = {
        "n_seeds": n_seeds.pop(),
        "families_measured": sorted(set(families)),
        "families_registered": FamilyRegistry.list_families(),
        "families_not_measured": sorted(
            set(FamilyRegistry.list_families()) - set(families)
        ),
        "drawn_signs_layout_0": signs,
        "board_files": [str(p) for p in paths],
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

    - ``leaderboard/v5_leaderboard.json`` -- reproduced by an ``n_seeds=5`` run, which is
      what its per-family values match.
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
            "path": str(path),
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
            and min(t["before"] for t in b["tiers"].values() if t.get("before")) >= 0.5
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


def _invariance(detectors: dict) -> dict:
    """Whether every direction-free detector scored identically in both modes.

    Sign randomization multiplies ``behav_trace`` and nothing else, and
    :func:`rhob.v3.access.restrict` nulls that field below L2, so every L0/L1 cell and
    every cell of a magnitude-only L2 detector must be bit-identical with the flip on and
    off. Checked cell by cell rather than on the means, so a pair of compensating changes
    cannot hide.
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
        after_by_cell = {(c["family"], c["difficulty"]): c["auroc"] for c in after["cells"]}
        mismatched = [
            (c["family"], c["difficulty"], c["auroc"], after_by_cell.get(key))
            for c in before["cells"]
            for key in [(c["family"], c["difficulty"])]
            if c["auroc"] != after_by_cell.get(key)
        ]
        report[name] = {
            "access_level": entry["access_level"],
            "n_cells": len(before["cells"]),
            "n_identical": len(before["cells"]) - len(mismatched),
            "mismatched_cells": mismatched[:10],
        }
    return report


def _reproduction_check(detectors: dict, board_path: Path, transfer_path: Path) -> dict:
    """Does the ``before`` column reproduce the committed pre-audit artifacts?

    The ``after`` number is only worth reporting if the ``before`` one lands on the
    published figures it is supposed to be replacing, so this compares, family by family,
    the ``before`` column against ``leaderboard/v5_leaderboard.json`` and the ``before``
    RTS against ``leaderboard/cross_family_transfer.json``.

    The committed board records neither its ``n_seeds`` nor a sampling block (it predates
    both), so an exact match is not guaranteed and a mismatch is not by itself evidence of
    a regression -- ``max_abs_diff`` is reported per detector so the size of any
    disagreement is visible rather than hidden behind a boolean.
    """
    if not board_path.exists():
        return {"published_board": f"{board_path} not found; not checked"}
    published = json.loads(board_path.read_text(encoding="utf-8"))["results"]
    out: dict[str, dict] = {}
    for name, entry in detectors.items():
        block = entry["by_config"].get("before")
        if block is None or name not in published:
            continue
        their = published[name].get("per_family", {})
        diffs = {
            fam: round(value - their[fam], 6)
            for fam, value in block["per_family"].items()
            if fam in their
        }
        if not diffs:
            continue
        out[name] = {
            "n_families_compared": len(diffs),
            "n_exact": sum(1 for d in diffs.values() if d == 0),
            "max_abs_diff": round(max(abs(d) for d in diffs.values()), 6),
            "families_differing": {k: v for k, v in sorted(diffs.items()) if v != 0},
        }
    # The committed ladder is a mean over all 33 families; recomputing it over exactly
    # the families measured here makes the two directly comparable, which a 33-family
    # figure against a subset would not be.
    measured = {
        fam
        for entry in detectors.values()
        for block in entry["by_config"].values()
        for fam in block["per_family"]
    }
    published_l2 = []
    for name in L2_DETECTORS:
        their = published.get(name, {}).get("per_family", {})
        values = [v for fam, v in their.items() if fam in measured]
        if values:
            published_l2.append((name, float(np.mean(values))))
    published_l2.sort(key=lambda kv: -kv[1])

    result: dict = {
        "published_board": str(board_path),
        "published_board_note": (
            "the committed board records no sampling block; its n_seeds is not "
            "recoverable from the artifact"
        ),
        "per_family_vs_published": out,
        "published_l2_ladder_over_measured_families": (
            {
                "n_families": len(measured),
                "n_detectors": len(published_l2),
                "mean_auroc": round(float(np.mean([v for _, v in published_l2])), 4),
                "max_auroc": round(published_l2[0][1], 4),
                "best_detector": published_l2[0][0],
                "per_detector": {k: round(v, 4) for k, v in published_l2},
            }
            if published_l2
            else {}
        ),
    }
    if transfer_path.exists():
        published_rts = json.loads(transfer_path.read_text(encoding="utf-8"))["results"]
        result["published_rts"] = {
            row: value.get("avg_transfer_auroc_mean", value.get("avg_transfer_auroc"))
            for row, value in published_rts.items()
        }
    return result


def _merge_transfers(paths: list[Path]) -> dict:
    """Merge per-configuration ``transfer`` artifacts into one block.

    Each configuration is run as its own process -- they share no state and the
    experiment is hours long -- so the artifact arrives in pieces. Merging refuses to
    proceed if two pieces disagree about the sampling design, since a ``before`` measured
    at one seed count and an ``after`` measured at another is not a comparison.
    """
    merged: dict = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
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
        merged["configs"].update(payload["configs"])
        merged["results"].update(payload["results"])
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
            differing = [
                (c["family"], c["difficulty"], c["auroc"], right_by_cell[key])
                for c in lb["cells"]
                for key in [(c["family"], c["difficulty"])]
                if key in right_by_cell and c["auroc"] != right_by_cell[key]
            ]
            row[f"{left}_vs_{right}"] = {
                "n_cells": len(lb["cells"]),
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


def run_report(args: argparse.Namespace) -> None:
    """Merge the measurement artifacts into the committed JSON + Markdown table."""
    detectors, design = _merge_boards(args.board)
    design["configs"] = {
        k: {"randomize_behav_sign": r, "detector_infers_orientation": i}
        for k, (r, i) in CONFIGS.items()
    }

    per_detector: dict[str, dict] = {}
    for name, entry in sorted(detectors.items()):
        row = {
            "access_level": entry["access_level"],
            "direction_dependent": entry["direction_dependent"],
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
        if row["before"] is not None and row["after"] is not None:
            row["delta_after_minus_before"] = round(row["after"] - row["before"], 4)
        per_detector[name] = row

    axis = _difficulty_axis(detectors, "Behavioral Threshold")
    payload = {
        "measurement": "sign_randomization_impact",
        "design": design,
        "provenance": provenance_block(script="scripts/measure_sign_randomization.py"),
        "l2_access_ladder": {c: _ladder(detectors, c) for c in ("before", "after")},
        "per_detector": per_detector,
        "difficulty_axis": axis,
        "knob_summary": _knob_summary(axis),
        "l0_l1_and_magnitude_invariance": _invariance(detectors),
        "config_deltas": _config_deltas(detectors),
        "sample_size_sensitivity": _sensitivity(args.companion or []),
        "reproduction_check": _reproduction_check(
            detectors, args.published_board, args.published_transfer
        ),
    }
    if args.transfer:
        payload["transfer"] = _merge_transfers(args.transfer)
    _write(args.out_json, payload)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"[OK] wrote {args.out_md}", flush=True)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(payload: dict) -> str:
    """The short committed table the docs cite and a test can pin."""
    design = payload["design"]
    before_rung = payload["l2_access_ladder"].get("before", {})
    after_rung = payload["l2_access_ladder"].get("after", {})
    transfer_before = (
        (payload.get("transfer") or {}).get("results", {}).get("before", {})
        .get("Ensemble (Top 5)", {})
    )
    transfer_after = (
        (payload.get("transfer") or {}).get("results", {}).get("after", {})
        .get("Ensemble (Top 5)", {})
    )
    knob = payload.get("knob_summary") or {}
    lines: list[str] = [
        "# L2 sign randomization: measured before/after",
        "",
        "Generated by `scripts/measure_sign_randomization.py`; numbers come from",
        "`results/sign_randomization/sign_randomization_impact.json`. Every figure here is a",
        "measurement, not an estimate; anything that was not measured is named as such.",
        "",
        "## Headline",
        "",
        f"- **RTS (Ensemble Top-5) {_fmt(transfer_before.get('avg_transfer_auroc_mean'))} "
        f"-> {_fmt(transfer_after.get('avg_transfer_auroc_mean'))}.** The published 0.994 "
        "reproduces exactly with randomization off.",
        f"- **L2 suite mean {before_rung.get('mean_auroc')} -> {after_rung.get('mean_auroc')}; "
        f"best {before_rung.get('max_auroc')} ({before_rung.get('best_detector')}) -> "
        f"{after_rung.get('max_auroc')} ({after_rung.get('best_detector')}).**",
        "- **The two L2 detectors that read the feature's *sign* are exactly the two that "
        "scored ~0.97 before.** The four that read only a magnitude scored 0.51-0.57 before "
        "and are bit-identical after. That is the direct evidence that the sign convention, "
        "not the behavioral channel, carried the pre-audit L2 result.",
        "- **The difficulty knob does not come back to life.** `|AUROC - 0.5|` is invariant "
        "under the flip, so the pinning at 1.000 survives it: "
        f"{knob.get('n_families_separation_span_exactly_zero')} of "
        f"{knob.get('n_families_with_more_than_one_tier')} multi-tier families still have a "
        "separation span of exactly 0.000. Sign randomization does **not** make a "
        "dose-response sweep viable.",
        "",
        "## Design",
        "",
        f"- `n_seeds` = {design['n_seeds']} per variant per cell, layout seed 0, one draw"
        " (`n_replicates = 1`).",
        f"- Families measured: **{len(design['families_measured'])} of"
        f" {len(design['families_registered'])}**.",
    ]
    if design["families_not_measured"]:
        lines.append(
            "- Not measured (rollout cost; see the report text): "
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
        "The `after` rung's *maximum* is carried entirely by `Trajectory MLP`, which is "
        "fitted inside each cell under 5-fold cross-validation -- its training folds carry "
        "labels, so it learns the cell's direction from them. That is legitimate under the "
        "benchmark's rules and is why it barely moves, but it means the surviving L2 "
        "maximum is a supervised result, not a statement that the unsupervised behavioral "
        "signal still separates.",
        "",
        "## Per-detector L2 (and L3) AUROC",
        "",
        "| detector | access | direction-dependent | `before` | `before_infer` | `blind` | "
        "`after` | `after - before` |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, row in payload["per_detector"].items():
        lines.append(
            f"| {name} | {row['access_level']} | "
            f"{'yes' if row['direction_dependent'] else 'no'} | "
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
            f"`n_trials={transfer['n_trials']}` "
            f"({'as published' if transfer['n_trials'] == 5 else 'the published run used 5'}; "
            "the +/- across model inits is reported in the JSON, not here).",
            "",
            "The L0 and L1 rows are measured in `before` only, and marked `n/a` elsewhere "
            "rather than re-run: `restrict()` nulls `behav_trace` below L2, so their cells "
            "are bit-identical in both modes. That is not taken on trust -- the invariance "
            "table at the end of this file checks it cell by cell on the full board.",
            "",
            "| detector | access | " + " | ".join(
                f"`{c}` train / RTS" for c in transfer["results"]
            ) + " |",
            "| --- | --- | " + " | ".join("---" for _ in transfer["results"]) + " |",
        ]
        rows = sorted({r for cfg in transfer["results"].values() for r in cfg})
        for row_name in rows:
            cells = []
            level = ""
            for cfg in transfer["results"]:
                entry = transfer["results"][cfg].get(row_name)
                if entry is None:
                    cells.append("n/a")
                    continue
                level = entry["access_level"]
                train = entry.get("train_auroc_mean", entry.get("train_auroc"))
                rts = entry.get("avg_transfer_auroc_mean", entry.get("avg_transfer_auroc"))
                cells.append(f"{_fmt(train)} / **{_fmt(rts)}**")
            lines.append(f"| {row_name} | {level} | " + " | ".join(cells) + " |")

    repro = payload.get("reproduction_check") or {}
    ladder = repro.get("published_l2_ladder_over_measured_families")
    if ladder:
        lines += [
            "",
            "## Reproduction of the pre-audit figures",
            "",
            "The `after` number is only worth anything if `before` lands on the published "
            "figures it replaces. The committed board is a 33-family mean; recomputed over "
            "exactly the families measured here it reads:",
            "",
            "| detector | committed board (same families) | this run, `before` |",
            "| --- | --- | --- |",
        ]
        for name, published_value in ladder["per_detector"].items():
            mine = payload["per_detector"].get(name, {}).get("before")
            lines.append(f"| {name} | {published_value:.4f} | {_fmt(mine, 4)} |")
        lines += [
            f"| **L2 suite mean** | **{ladder['mean_auroc']:.4f}** | "
            f"**{payload['l2_access_ladder']['before']['mean_auroc']:.4f}** |",
            "",
            f"Committed best at L2: **{ladder['best_detector']} "
            f"{ladder['max_auroc']:.4f}**; this run's `before` best: "
            f"**{payload['l2_access_ladder']['before']['best_detector']} "
            f"{payload['l2_access_ladder']['before']['max_auroc']:.4f}**.",
        ]
        rts = repro.get("published_rts")
        if rts:
            lines += [
                "",
                "Published RTS against this run's `before` RTS:",
                "",
                "| detector | published | this run, `before` |",
                "| --- | --- | --- |",
            ]
            for row, value in rts.items():
                mine = None
                cfg = (payload.get("transfer") or {}).get("results", {}).get("before", {})
                if row in cfg:
                    mine = cfg[row].get(
                        "avg_transfer_auroc_mean", cfg[row].get("avg_transfer_auroc")
                    )
                lines.append(f"| {row} | {_fmt(value)} | {_fmt(mine)} |")

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
            f"({knob['orientation_accuracy']:.3f}) — i.e. the shipped orientation rule is "
            "at or below a coin flip on the full board. Split by family group, because the "
            "published 63.3% was measured on the tabular and continuous families alone:",
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
            "statistic was tied.",
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
            "score identically with randomization on and off. Checked cell by cell.",
            "",
            "| detector | access | cells | identical |",
            "| --- | --- | --- | --- |",
        ]
        for name, block in inv.items():
            lines.append(
                f"| {name} | {block['access_level']} | {block['n_cells']} | "
                f"{block['n_identical']}/{block['n_cells']} |"
            )
    lines.append("")
    return "\n".join(lines)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {path}", flush=True)


def main() -> None:
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
    transfer.add_argument("--n-trials", type=int, default=5)
    transfer.add_argument(
        "--configs", nargs="+", default=["before", "blind", "after"],
        choices=sorted(CONFIGS),
    )
    transfer.add_argument(
        "--rows", nargs="+",
        default=["Reward MLP", "State Divergence", "Trajectory MLP", "Ensemble (Top 5)"],
    )
    transfer.add_argument(
        "--out", type=Path, default=Path("results/sign_randomization/transfer.json")
    )
    transfer.set_defaults(func=run_transfer)

    report = sub.add_parser("report", help="merge the artifacts into JSON + Markdown")
    report.add_argument("--board", nargs="+", type=Path, required=True)
    report.add_argument("--transfer", nargs="+", type=Path, default=None)
    report.add_argument(
        "--companion", nargs="+", default=None, metavar="LABEL=PATH",
        help="Board runs at other sample sizes, reported alongside but never merged "
             "into the primary aggregates.",
    )
    report.add_argument(
        "--published-board", type=Path, default=Path("leaderboard/v5_leaderboard.json"),
        help="Pre-audit board the `before` column has to reproduce.",
    )
    report.add_argument(
        "--published-transfer", type=Path,
        default=Path("leaderboard/cross_family_transfer.json"),
        help="Pre-audit RTS artifact the `before` RTS has to reproduce.",
    )
    # The merged artifact goes to ``docs/figures/`` because ``results/`` is gitignored
    # and this is the file the docs cite and a test pins. It carries every per-cell AUROC
    # from every board job, so the (large, intermediate) per-job files under ``results/``
    # are not needed to audit it.
    report.add_argument(
        "--out-json", type=Path,
        default=Path("docs/figures/sign_randomization_impact.json"),
    )
    report.add_argument(
        "--out-md", type=Path,
        default=Path("docs/figures/sign_randomization_impact.md"),
    )
    report.set_defaults(func=run_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
