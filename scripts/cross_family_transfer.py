#!/usr/bin/env python
"""Cross-family generalization experiment: train on 6 families, test on 8 held out.

This is the real experiment behind the paper's Cross-Family Transfer Analysis
(Section 6.3, Table ``tab:transfer``). Earlier drafts of that table used
illustrative placeholder numbers; this script produces the real ones from the
actual RHOB v3/v5 environments, families, and detectors.

TEST_FAMILIES was expanded from 3 to 8 families (v1.4): the 5 families added in
v1.4 (reward_channel_tampering, sensor_calibration_tampering, monitored_sandbagging,
eval_probe_sandbagging, rlhf_reward_model_overopt) introduce 3 mechanism types
(REWARD_TAMPERING, DECEPTIVE_ALIGNMENT, RM_OVEROPTIMIZATION) that appear in neither
TRAIN_FAMILIES nor the original 3 test families -- testing transfer to them is a
strictly more meaningful "mechanistically unseen" evaluation than the original split,
which only exercised 2 unseen mechanism types on top of what training already covers.
TRAIN_FAMILIES is unchanged, so "train AUROC" numbers stay comparable to prior results.

Methodology:
  1. "Train" AUROC: in-distribution performance of each detector on the 6 train
     families, measured with the same 5-fold stratified cross-validation used
     throughout the paper (``Benchmark.evaluate``), cell by cell and out of fold. That
     includes the Top-5 ensemble: :class:`FoldFittedEnsemble` gives it a ``fit`` that
     refits its Trajectory MLP member on each training fold. (Until 2026-11-27 the
     ensemble had no ``fit``, so ``Benchmark.evaluate`` scored it directly, with a
     Trajectory MLP fitted beforehand on exactly the runs it was then scored on: its
     "train AUROC" was in-sample for that member.)
  2. Each detector is then fit ONCE on the pooled runs of the train families that it
     can be scored on at all (:func:`pooled_runs`) to produce a single frozen,
     deployable model -- standard practice: cross-validation is for honest scoring,
     the shipped model uses all available training data.
  3. That frozen model is evaluated -- with no further fitting -- on the 8 held-out
     families (goal misgeneralization, physics exploitation, distributional shift,
     plus the 5 v1.4 families spanning 3 mechanism types unseen during training),
     which it has never seen in any form (:func:`transfer_eval`).
  4. A Top-5 L2 ensemble (4 stateless behavioral detectors + a Trajectory MLP) is
     evaluated the same way, to measure whether combining detectors recovers some of
     the transfer gap. Its frozen Trajectory MLP is fitted on the same pooled runs as
     the Trajectory MLP row's, with the same seed, so the two rows share that model.

Not applicable is never imputed
-------------------------------
A cell is ``null`` (not applicable) rather than scored at the detector's no-signal
constant in three cases, and each is recorded with its reason:

- the family does not emit the channel the detector reads, or its runs are shorter
  than the detector's declared ``min_episodes``. This is
  :func:`rhob.v3.benchmark.not_applicable_reason`, the rule ``Benchmark.evaluate``
  applies, applied here too: to the pooled fit (such a family contributes no training
  rows, so it is listed in ``train_families_excluded_from_pooled_fit``) and to every
  held-out cell. Two of the eight test families (monitored_sandbagging,
  eval_probe_sandbagging) and four of the six train families ship
  ``state_counts=None``, and continuous_camping's 40-episode runs are below the
  50-episode windows of Reward MLP, State Divergence and the ensemble.
- the frozen model cannot consume the family's version of the channel: ``classify``
  raises on it (a state-visitation histogram dimensioned by another family's
  discretization, for instance). Such a cell is not a measurement of transfer either,
  so it is null, and ``per_family_incompatible_runs`` counts the runs it raised on.
  Until 2026-11-27 these runs were scored at 0.5: State Divergence's frozen 49-bin
  gridworld centroids raised on every run of all six held-out families that emit
  ``state_counts``, so its published transfer row was 1200 imputed scores.
- the pooled fit failed or left the detector untrained, in which case nothing frozen
  exists to transfer (``pooled_fit_error``).

A held-out family scored on some of its difficulty cells but not all is averaged over
the scored ones, and says so: ``per_family_cells_scored`` and
``per_family_cells_not_applicable`` count its cells, and
``test_partial_not_applicable_reasons`` gives the reasons for the ones left out.

continuous_camping was never really training data for Reward MLP or State Divergence:
both fits already dropped runs shorter than their windows, so the frozen models are the
same as before these rules, and what changed is that the artifact now lists the family
as excluded instead of implying it was used.

Output: ``leaderboard/cross_family_transfer_v5.json`` by default, with a ``configs``
block naming the configuration it was measured in (sign randomization on, detectors
orienting themselves), so :mod:`rhob.v3.leaderboard.transfer` reads it as the shipped
configuration. ``leaderboard/cross_family_transfer.json`` (2026-07-09) is the
historical record -- pre-sign-randomization and pre-dating every rule above -- and this
script refuses to overwrite it.

Run: python scripts/cross_family_transfer.py [--n-seeds-train N] [--n-seeds-test N]
         [--n-trials N] [--out PATH] [--only ROW ...]

Run it with torch on one thread (``OMP_NUM_THREADS=1``; CI's numerics pins set it). The
networks are small, and with torch's default of one thread per core they can spend most
of their time synchronizing threads when the machine is busy. Under the pins the
published design took 5m23s on a 4-core box; unpinned on the same box, one trial's
Reward MLP train AUROC took anywhere from 10 s to 324 s, and the Trajectory MLP's 489 s.
The rollout cache (:func:`_rollout`) is not what makes the difference: it stops trials
and rows from re-simulating the pooled and held-out runs, while the model fits -- most of
the time -- are untouched by it, and the train AUROC's rollouts were already cached by
``Benchmark.evaluate``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

# Run the checkout this file lives in, not whatever the editable install points at.
# In a git worktree those differ, and the failure is silent: the script imports a
# same-named module from the OTHER tree and publishes results attributed to this one.
# Enforced for every script by tests/test_scripts_run_their_own_checkout.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rhob.v3.families  # noqa: F401,E402  -- importing registers every family
from rhob.v3.access import restrict  # noqa: E402
from rhob.v3.benchmark import (  # noqa: E402
    Benchmark,
    RolloutCacheKey,
    not_applicable_reason,
    offer_population,
    required_channels,
)
from rhob.v3.provenance import provenance_block, sampling_block  # noqa: E402
from rhob.v3.registry import FamilyRegistry  # noqa: E402

from rhob.detectors.l0_reward_mlp import RewardMLPDetector  # noqa: E402
from rhob.detectors.l1_state_divergence import StateDivergenceDetector  # noqa: E402
from rhob.detectors.l2_trajectory_mlp import TrajectoryMLPDetector  # noqa: E402
from rhob.detectors.l2_behavioral_threshold import BehavioralThresholdDetector  # noqa: E402
from rhob.detectors.l2_angular_momentum import AngularMomentumDetector  # noqa: E402
from rhob.detectors.l2_centroid_tracker import CentroidTrackerDetector  # noqa: E402
from rhob.detectors.l2_feature_magnitude import FeatureMagnitudeDetector  # noqa: E402
from rhob.detectors.l2_ensemble import EnsembleDetector  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]

TRAIN_FAMILIES = [
    "gridworld_camping",
    "continuous_camping",
    "proxy_correlation_gaming",
    "shortcut_exploitation",
    "novelty_farming",
    "orbit_chirality",
]
TEST_FAMILIES = [
    "goal_misgeneralization",
    "physics_exploitation",
    "distributional_shift",
    "reward_channel_tampering",
    "sensor_calibration_tampering",
    "monitored_sandbagging",
    "eval_probe_sandbagging",
    "rlhf_reward_model_overopt",
]

TEST_SEED_BASE = 50_000  # disjoint from any training rollout seeds

# Mirrors MatchedPair.rollout (src/rhob/v3/base_pair.py): the legitimate variant is
# rolled out at ``seed_base + 1000 + s``, the hacking variant at ``seed_base + s``.
LEGIT_SEED_OFFSET = 1000

# Both pooled_runs() and transfer_eval() call ``fam.generate_pair_at(d)`` with
# BaseFamily's default ``seed=0``, and Benchmark.evaluate runs at its default
# ``layout_seed=0``, so the whole experiment rests on a single environment layout per
# (family, difficulty). Recorded in the sampling block rather than left implicit.
LAYOUT_SEEDS = [0]

#: Where the artifact goes unless ``--out`` says otherwise.
DEFAULT_OUT = Path("leaderboard/cross_family_transfer_v5.json")
#: The 2026-07-09 artifact. Kept unchanged as the record of what was published before
#: sign randomization and the not-applicable rules; this script never writes it.
HISTORICAL_ARTIFACT = Path("leaderboard/cross_family_transfer.json")

ENSEMBLE_NAME = "Ensemble (Top 5)"
#: The rows the experiment reports, in the order it runs them.
ROWS: tuple[str, ...] = ("Reward MLP", "State Divergence", "Trajectory MLP", ENSEMBLE_NAME)
#: Rows with a seeded neural network, repeated over ``n_trials`` model initialisations.
#: State Divergence is deterministic given its runs and is fitted once.
MULTI_TRIAL_ROWS = frozenset({"Reward MLP", "Trajectory MLP", ENSEMBLE_NAME})
ROW_LEVELS: dict[str, str] = {
    "Reward MLP": "L0",
    "State Divergence": "L1",
    "Trajectory MLP": "L2",
    ENSEMBLE_NAME: "L2",
}

#: The sampling design the published artifact is measured at. ``published_design`` in
#: the artifact is true only when a run used exactly this, all four rows and the two
#: family lists above, so a reduced smoke run cannot be mistaken for the result.
PUBLISHED_DESIGN: dict[str, int] = {"n_seeds_train": 10, "n_seeds_test": 20, "n_trials": 5}

#: The configuration ``main`` measures in, written into the artifact's ``configs`` block.
#: True by construction: every rollout here passes ``randomize_sign=True`` explicitly,
#: and no detector's population hook is disabled.
SHIPPED_CONFIG = {"shipped": {"randomize_behav_sign": True, "detector_infers_orientation": True}}


# --------------------------------------------------------------------------- rollouts
def _rollout(pair, n_seeds: int, seed_base: int, randomize_sign: bool):
    """``pair.rollout(...)``, through ``Benchmark._rollout_cache`` under its own key.

    Rollouts are a deterministic function of the key, so this changes no value: it only
    stops every trial of every row from re-simulating the same runs. The train families'
    pooled runs are exactly the cells ``Benchmark.evaluate`` rolled out for the train
    AUROC (same pair, layout 0, ``seed_base`` 0), so they come from the same entries.
    """
    key = RolloutCacheKey(
        family=pair.family,
        difficulty=pair.difficulty,
        n_seeds=n_seeds,
        randomize_behav_sign=randomize_sign,
        layout_seed=int(getattr(pair, "layout_seed", 0)),
        seed_base=seed_base,
    )
    cached = Benchmark._rollout_cache.get(key)
    if cached is None:
        cached = pair.rollout(n_seeds, seed_base=seed_base, randomize_sign=randomize_sign)
        Benchmark._rollout_cache[key] = cached
    return cached


def pooled_runs(
    families: Sequence[str],
    level: str,
    n_seeds: int,
    gate=None,
    *,
    randomize_sign: bool = True,
):
    """Roll out every default-difficulty pair for ``families``, restricted to ``level``.

    A family is left out of the pool when :func:`~rhob.v3.benchmark.not_applicable_reason`
    says ``gate`` (the detector about to be fitted) cannot be scored on its runs: a run
    with no state-visitation histogram, or shorter than the detector's windows,
    contributes nothing to a fit except a silently-dropped row, and counting it as
    training data overstates how much data the frozen model saw. ``gate=None`` pools
    every family.

    Returns ``(runs_a, runs_b, skipped_families, skipped_reasons)``, the last a
    ``{family: reason}`` dict.
    """
    runs_a_all, runs_b_all = [], []
    skipped: list[str] = []
    reasons: dict[str, str] = {}
    for fam_name in families:
        fam = FamilyRegistry.get(fam_name)
        fam_a, fam_b = [], []
        for d in fam.default_difficulties():
            pair = fam.generate_pair_at(d)
            runs_a, runs_b, _ = _rollout(pair, n_seeds, 0, randomize_sign)
            fam_a.extend(restrict(r, level) for r in runs_a)
            fam_b.extend(restrict(r, level) for r in runs_b)
        reason = None if gate is None else not_applicable_reason(gate, fam_a + fam_b, level)
        if reason is not None:
            skipped.append(fam_name)
            reasons[fam_name] = reason
            continue
        runs_a_all.extend(fam_a)
        runs_b_all.extend(fam_b)
    return runs_a_all, runs_b_all, skipped, reasons


class TransferEval:
    """What :func:`transfer_eval` measured on the held-out families.

    A plain class rather than a dataclass: this script is also loaded by path
    (``scripts/measure_sign_randomization.py``, the tests), outside ``sys.modules``,
    where ``dataclasses`` cannot resolve postponed annotations.

    Attributes:
        per_family: mean AUROC over each family's scored cells; ``None`` = not applicable.
        not_applicable: families whose every cell was not applicable, in order evaluated.
        reasons: ``{family: reason}`` for each not-applicable family.
        incompatible_runs: ``{family: runs}`` -- how many ``classify`` calls raised.
        cells_scored: ``{family: cells}`` the family's mean in ``per_family`` is over.
        cells_not_applicable: ``{family: cells}`` left out of it.
        partial_reasons: ``{family: reason}`` for each family that was scored on some of
            its cells but not all, so its mean is over fewer difficulty tiers than the
            others' -- recorded, since nothing else in the row would show it.
    """

    def __init__(
        self,
        per_family: dict[str, Optional[float]],
        not_applicable: Optional[list[str]] = None,
        reasons: Optional[dict[str, str]] = None,
        incompatible_runs: Optional[dict[str, int]] = None,
        cells_scored: Optional[dict[str, int]] = None,
        cells_not_applicable: Optional[dict[str, int]] = None,
        partial_reasons: Optional[dict[str, str]] = None,
    ):
        self.per_family = per_family
        self.not_applicable = not_applicable if not_applicable is not None else []
        self.reasons = reasons if reasons is not None else {}
        self.incompatible_runs = incompatible_runs if incompatible_runs is not None else {}
        self.cells_scored = cells_scored if cells_scored is not None else {}
        self.cells_not_applicable = (
            cells_not_applicable if cells_not_applicable is not None else {}
        )
        self.partial_reasons = partial_reasons if partial_reasons is not None else {}

    @property
    def average(self) -> Optional[float]:
        """Mean over the scored families; ``None`` if none was scored."""
        scored = [v for v in self.per_family.values() if v is not None]
        return float(np.mean(scored)) if scored else None


def _channel_name(detector, level: str) -> str:
    channels = required_channels(detector)
    return "/".join(channels) if channels else f"{level} input"


def transfer_eval(
    detector,
    families: Sequence[str],
    level: str,
    n_seeds: int,
    *,
    randomize_sign: bool = True,
) -> TransferEval:
    """Score a frozen, already-fitted detector on held-out families.

    A cell is not scored -- and a family whose every cell is not scored is ``None``,
    excluded from the average -- in two cases:

    - :func:`~rhob.v3.benchmark.not_applicable_reason`: the family does not emit the
      channel this detector reads, or its runs are shorter than the detector's
      ``min_episodes``. Scoring it would report the detector's fallback constant.
    - the frozen model cannot consume the family's version of the channel: ``classify``
      raises ``ValueError``/``RuntimeError`` on a run. Some L1/L2 features (e.g. raw
      state-visitation histograms) are dimensioned by a family's own state-space
      discretization and are not representationally compatible with another family's.
      That is a genuine transfer failure, but not a measurement of one: there is no
      score to rank. The whole cell is left out rather than its consumable runs scored,
      since scoring only the runs the model happened to accept would select on the
      failure, and the number of runs that raised is reported per family.

    Neither case is imputed at 0.5 (until 2026-11-27 the second one was). Every family's
    scored and not-applicable cell counts are recorded, and a family scored on only some
    of its cells keeps its reasons in ``partial_reasons``: its mean is over fewer
    difficulty tiers, which the per-family number alone would not say.
    """
    out = TransferEval(per_family={})
    for fam_name in families:
        fam = FamilyRegistry.get(fam_name)
        cell_aurocs: list[float] = []
        cell_reasons: list[str] = []
        n_incompatible = 0
        for d in fam.default_difficulties():
            pair = fam.generate_pair_at(d)
            runs_a, runs_b, _ = _rollout(pair, n_seeds, TEST_SEED_BASE, randomize_sign)
            restricted_a = [restrict(r, level) for r in runs_a]
            restricted_b = [restrict(r, level) for r in runs_b]
            cell_runs = restricted_a + restricted_b
            reason = not_applicable_reason(detector, cell_runs, level)
            if reason is not None:
                cell_reasons.append(reason)
                continue
            # Same unlabeled-population hook Benchmark._evaluate_cell offers, so a
            # detector that orients itself from the cell can do so on a held-out family
            # too. Without it the transfer number would penalize an L2 detector for a
            # coin flip (the family's drawn behavioral sign) rather than for failing to
            # generalize -- and the frozen model's own training families give it no way
            # to know this one's orientation.
            offer_population(detector, cell_runs)
            labels = np.array([1] * len(restricted_a) + [0] * len(restricted_b))
            scores: list[float] = []
            raised: list[BaseException] = []
            for r in cell_runs:
                try:
                    scores.append(detector.classify(r))
                except (ValueError, RuntimeError) as exc:
                    raised.append(exc)
            if raised:
                n_incompatible += len(raised)
                first = raised[0]
                cell_reasons.append(
                    f"the frozen model cannot consume this family's "
                    f"{_channel_name(detector, level)}: classify raised on "
                    f"{len(raised)} of {len(cell_runs)} runs "
                    f"({type(first).__name__}: {first}); not scored"
                )
                continue
            if len(set(labels)) > 1:
                cell_aurocs.append(float(roc_auc_score(labels, scores)))
            else:
                cell_reasons.append("the cell has runs of one variant only; not scored")
        if n_incompatible:
            out.incompatible_runs[fam_name] = n_incompatible
        out.cells_scored[fam_name] = len(cell_aurocs)
        out.cells_not_applicable[fam_name] = len(cell_reasons)
        if cell_aurocs:
            out.per_family[fam_name] = float(np.mean(cell_aurocs))
            if cell_reasons:
                out.partial_reasons[fam_name] = "; ".join(dict.fromkeys(cell_reasons))
        else:
            out.per_family[fam_name] = None
            out.not_applicable.append(fam_name)
            out.reasons[fam_name] = "; ".join(dict.fromkeys(cell_reasons)) or "no cell"
    if out.incompatible_runs:
        print(
            f"  [note] classify() raised on {sum(out.incompatible_runs.values())} runs "
            f"(representational mismatch); those cells are not applicable, not 0.5: "
            f"{', '.join(f'{k} ({v})' for k, v in out.incompatible_runs.items())}",
            flush=True,
        )
    if out.not_applicable:
        print(
            f"  [note] not applicable and excluded from the transfer average: "
            f"{', '.join(out.not_applicable)}",
            flush=True,
        )
    if out.partial_reasons:
        print(
            "  [note] scored on some cells only: "
            + ", ".join(
                f"{f} ({out.cells_scored[f]} of "
                f"{out.cells_scored[f] + out.cells_not_applicable[f]})"
                for f in out.partial_reasons
            ),
            flush=True,
        )
    return out


# --------------------------------------------------------------------------- detectors
def _stateless_l2_members() -> list:
    """The four Top-5 members that carry no fitted state."""
    return [
        BehavioralThresholdDetector(),
        AngularMomentumDetector(),
        CentroidTrackerDetector(),
        FeatureMagnitudeDetector(),
    ]


class FoldFittedEnsemble(EnsembleDetector):
    """The Top-5 L2 ensemble, with a ``fit`` that refits its one fitted member.

    :class:`~rhob.detectors.l2_ensemble.EnsembleDetector` has no ``fit``, so
    ``Benchmark.evaluate`` scores it directly, without cross-validation. Handed a
    Trajectory MLP fitted on the train families, its train AUROC was therefore in-sample
    for that member. This subclass trains a fresh ``TrajectoryMLPDetector(seed=trial)``
    on whatever runs ``fit`` is given and swaps it in, so ``Benchmark.evaluate``'s
    5-fold cross-validation refits it on every training fold and scores it out of fold,
    like every other row; the pooled fit before the transfer step goes through the same
    ``fit``. The four other members carry no fitted state -- only the orientation the
    population hook sets on them per cell, which ``fit`` leaves alone.

    Script-local on purpose: it changes how this experiment scores the ensemble, not
    what :class:`EnsembleDetector` is.
    """

    def __init__(self, trial: int = 0):
        self.trial = trial
        super().__init__(
            _stateless_l2_members() + [TrajectoryMLPDetector(seed=trial)], name=ENSEMBLE_NAME
        )

    @property
    def pooled_fit_gate(self):
        """The member :func:`pooled_runs` gates the pooled fit on: the one being fitted.

        The ensemble itself declares the largest member minimum (Feature Magnitude's 50
        episodes), which would keep continuous_camping out of the Trajectory MLP's
        training data although that member needs 2. Gating on the member trains it on
        exactly the pool the Trajectory MLP row's frozen model sees.
        """
        return self.members[-1]

    @property
    def is_trained(self) -> bool:
        return bool(getattr(self.members[-1], "is_trained", True))

    def fit(self, runs_a, runs_b) -> None:
        mlp = TrajectoryMLPDetector(seed=self.trial)
        mlp.fit(runs_a, runs_b)
        self.members = [*self.members[:-1], mlp]


def build_ensemble(trial: int) -> FoldFittedEnsemble:
    """The Top-5 ensemble for model-init ``trial``, unfitted (see :class:`FoldFittedEnsemble`)."""
    return FoldFittedEnsemble(trial)


#: ``trial -> detector`` for each row. ``trial`` seeds the neural networks.
ROW_FACTORIES: dict[str, Callable[[int], Any]] = {
    "Reward MLP": lambda trial: RewardMLPDetector(seed=trial),
    "State Divergence": lambda trial: StateDivergenceDetector(),
    "Trajectory MLP": lambda trial: TrajectoryMLPDetector(seed=trial),
    ENSEMBLE_NAME: build_ensemble,
}


# --------------------------------------------------------------------------- one row
def _round_or_none(value: float | None, digits: int = 3) -> float | None:
    """Round for the artifact; ``None`` (not applicable) and NaN both become ``null``."""
    if value is None or value != value:
        return None
    return round(float(value), digits)


def _family_reasons(cells) -> dict[str, str]:
    """``{family: reason}`` over not-applicable cells, reasons de-duplicated in order."""
    by_family: dict[str, list[str]] = {}
    for c in cells:
        if c.na_reason:
            by_family.setdefault(c.family, [])
            if c.na_reason not in by_family[c.family]:
                by_family[c.family].append(c.na_reason)
    return {k: "; ".join(v) for k, v in by_family.items()}


def run_one(
    name: str,
    detector,
    level: str,
    n_seeds_train: int,
    n_seeds_test: int,
    *,
    randomize_sign: bool = True,
    train_families: Sequence[str] = TRAIN_FAMILIES,
    test_families: Sequence[str] = TEST_FAMILIES,
) -> tuple[dict, Any]:
    """Train AUROC (CV), pooled fit, frozen transfer, for one detector instance."""
    print(f"=== {name} ({level}) ===", flush=True)
    t0 = time.time()
    train_families = list(train_families)
    test_families = list(test_families)

    train_bench = Benchmark.evaluate(
        detector,
        families=train_families,
        difficulties="all",
        n_seeds=n_seeds_train,
        verbose=False,
        randomize_behav_sign=randomize_sign,
    )
    train_auroc = train_bench.overall_auroc
    na_train = train_bench.na_families
    note = f"  ({len(na_train)} train families N/A: {', '.join(na_train)})" if na_train else ""
    print(
        f"  in-distribution (5-fold CV) train AUROC: {train_auroc:.3f} "
        f"over {len(train_bench.scored_cells)} measured cells{note}  [{time.time()-t0:.0f}s]",
        flush=True,
    )

    gate = getattr(detector, "pooled_fit_gate", detector)
    train_a, train_b, skipped_train, skipped_reasons = pooled_runs(
        train_families, level, n_seeds_train, gate, randomize_sign=randomize_sign
    )
    if skipped_train:
        print(
            "  [note] excluded from the pooled fit: "
            + "; ".join(f"{k} ({v})" for k, v in skipped_reasons.items()),
            flush=True,
        )

    pooled_fit_error: Optional[str] = None
    if hasattr(detector, "fit"):
        if not train_a or not train_b:
            pooled_fit_error = "no train family passed the not-applicable rule; nothing to fit"
        else:
            try:
                detector.fit(train_a, train_b)
            except (ValueError, RuntimeError) as e:
                pooled_fit_error = f"{type(e).__name__}: {e}"
            else:
                if getattr(detector, "is_trained", True) is False:
                    pooled_fit_error = "fit left the detector untrained"

    if pooled_fit_error is None:
        ev = transfer_eval(
            detector, test_families, level, n_seeds_test, randomize_sign=randomize_sign
        )
    else:
        # Nothing frozen exists to transfer, and an untrained detector returns its
        # fallback constant: every held-out family is not applicable, not 0.5.
        print(f"  [warn] pooled fit failed ({pooled_fit_error}); no transfer step", flush=True)
        reason = f"pooled fit failed ({pooled_fit_error}); no frozen model to transfer"
        ev = TransferEval(
            per_family={f: None for f in test_families},
            not_applicable=list(test_families),
            reasons={f: reason for f in test_families},
            cells_scored={f: 0 for f in test_families},
            cells_not_applicable={
                f: len(FamilyRegistry.get(f).default_difficulties()) for f in test_families
            },
        )

    avg_transfer = ev.average
    gap = (
        (train_auroc - avg_transfer) / train_auroc * 100
        if avg_transfer is not None and train_auroc == train_auroc and train_auroc
        else None
    )

    shown = {k: ("N/A" if v is None else round(v, 3)) for k, v in ev.per_family.items()}
    print(f"  transfer per-family: {shown}", flush=True)
    avg_str = "N/A" if avg_transfer is None else f"{avg_transfer:.3f}"
    gap_str = "N/A" if gap is None else f"{gap:.1f}%"
    print(
        f"  avg transfer AUROC: {avg_str}   gap: {gap_str}   [{time.time()-t0:.0f}s total]",
        flush=True,
    )

    return {
        "access_level": level,
        "train_auroc": _round_or_none(train_auroc),
        "train_cells_measured": len(train_bench.scored_cells),
        "train_cells_not_applicable": len(train_bench.na_cells),
        "train_families_not_applicable": na_train,
        "train_not_applicable_reasons": _family_reasons(train_bench.na_cells),
        "train_families_excluded_from_pooled_fit": skipped_train,
        "train_excluded_from_pooled_fit_reasons": skipped_reasons,
        "pooled_fit_error": pooled_fit_error,
        "per_family_transfer": {k: _round_or_none(v) for k, v in ev.per_family.items()},
        "test_families_not_applicable": ev.not_applicable,
        "test_not_applicable_reasons": ev.reasons,
        "per_family_cells_scored": ev.cells_scored,
        "per_family_cells_not_applicable": ev.cells_not_applicable,
        "test_partial_not_applicable_reasons": ev.partial_reasons,
        "per_family_incompatible_runs": ev.incompatible_runs,
        "avg_transfer_auroc": _round_or_none(avg_transfer),
        "generalization_gap_pct": _round_or_none(gap, 1),
    }, detector


def run_one_multitrial(
    name: str,
    build_detector: Callable[[int], Any],
    level: str,
    n_seeds_train: int,
    n_seeds_test: int,
    n_trials: int,
    *,
    randomize_sign: bool = True,
    train_families: Sequence[str] = TRAIN_FAMILIES,
    test_families: Sequence[str] = TEST_FAMILIES,
) -> dict:
    """Repeat ``run_one`` across ``n_trials`` independently-seeded model instances
    and report mean/std, not a single draw.

    The neural-net detectors here (Reward MLP, Trajectory MLP) seed torch inside
    ``fit`` from their ``seed`` argument, which is the trial index -- so trial ``k`` is
    reproducible, and the trials differ only in weight initialization and minibatch
    order. An earlier, unseeded version of this experiment reported a single run's
    transfer AUROC as *the* result; repeating the exact same fit procedure 10 times on
    the Trajectory MLP produced held-out AUROCs on distributional_shift ranging from
    0.00 to 1.00 (genuinely bimodal, not noise around a stable mean). A single-run
    number for a detector with that much seed-sensitivity is not a reproducible
    measurement and should not be reported as one.

    A family is not applicable in the aggregate only if it was in every trial (channel
    presence and run length are properties of the family, not of the seed, so in
    practice it is all or none); a family scored in some trials but not others is
    averaged over the trials that scored it and counted in
    ``per_family_trials_not_applicable``. Cell counts are aggregated conservatively:
    ``per_family_cells_scored`` is the fewest cells any trial scored a family on and
    ``per_family_cells_not_applicable`` the most it left out, like
    ``per_family_incompatible_runs`` (the most runs any trial raised on).
    """
    train_aurocs: list[Optional[float]] = []
    transfer_aurocs: list[Optional[float]] = []
    gaps: list[Optional[float]] = []
    per_family_trials: dict[str, list[float]] = {f: [] for f in test_families}
    na_counts: dict[str, int] = {f: 0 for f in test_families}
    reasons: dict[str, str] = {}
    partial_reasons: dict[str, str] = {}
    cells_scored: dict[str, int] = {}
    cells_na: dict[str, int] = {}
    incompatible: dict[str, int] = {}
    fit_errors: list[str] = []
    last_result: dict = {}

    for trial in range(n_trials):
        detector = build_detector(trial)
        result, _ = run_one(
            name,
            detector,
            level,
            n_seeds_train,
            n_seeds_test,
            randomize_sign=randomize_sign,
            train_families=train_families,
            test_families=test_families,
        )
        last_result = result
        train_aurocs.append(result["train_auroc"])
        transfer_aurocs.append(result["avg_transfer_auroc"])
        gaps.append(result["generalization_gap_pct"])
        for fam, v in result["per_family_transfer"].items():
            if v is None:
                na_counts[fam] = na_counts.get(fam, 0) + 1
            else:
                per_family_trials.setdefault(fam, []).append(v)
        for fam, why in result["test_not_applicable_reasons"].items():
            reasons.setdefault(fam, why)
        for fam, why in result["test_partial_not_applicable_reasons"].items():
            partial_reasons.setdefault(fam, why)
        for fam, n in result["per_family_cells_scored"].items():
            cells_scored[fam] = min(cells_scored.get(fam, n), n)
        for fam, n in result["per_family_cells_not_applicable"].items():
            cells_na[fam] = max(cells_na.get(fam, 0), n)
        for fam, n in result["per_family_incompatible_runs"].items():
            incompatible[fam] = max(incompatible.get(fam, 0), n)
        if result["pooled_fit_error"]:
            fit_errors.append(f"trial {trial}: {result['pooled_fit_error']}")

    def _agg(values: Sequence[Optional[float]], fn, digits: int = 3) -> float | None:
        """Aggregate the trials that produced a number; ``None`` if none did."""
        present = [v for v in values if v is not None]
        return round(float(fn(present)), digits) if present else None

    def _show(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.3f}"

    per_family_mean = {k: _agg(v, np.mean) for k, v in per_family_trials.items()}
    per_family_std = {k: _agg(v, np.std) for k, v in per_family_trials.items()}
    not_applicable = [f for f in per_family_mean if per_family_mean[f] is None]
    partial = {f: n for f, n in na_counts.items() if 0 < n < n_trials}

    print(
        f"  [{name}] across {n_trials} seeded trials: "
        f"train={_show(_agg(train_aurocs, np.mean))}+/-{_show(_agg(train_aurocs, np.std))}  "
        f"transfer={_show(_agg(transfer_aurocs, np.mean))}"
        f"+/-{_show(_agg(transfer_aurocs, np.std))}",
        flush=True,
    )

    return {
        "access_level": level,
        "n_trials": n_trials,
        "train_auroc_mean": _agg(train_aurocs, np.mean),
        "train_auroc_std": _agg(train_aurocs, np.std),
        "train_cells_measured": last_result.get("train_cells_measured"),
        "train_cells_not_applicable": last_result.get("train_cells_not_applicable"),
        "train_families_not_applicable": last_result.get("train_families_not_applicable", []),
        "train_not_applicable_reasons": last_result.get("train_not_applicable_reasons", {}),
        "train_families_excluded_from_pooled_fit": last_result.get(
            "train_families_excluded_from_pooled_fit", []
        ),
        "train_excluded_from_pooled_fit_reasons": last_result.get(
            "train_excluded_from_pooled_fit_reasons", {}
        ),
        "pooled_fit_error": "; ".join(fit_errors) or None,
        "per_family_transfer_mean": per_family_mean,
        "per_family_transfer_std": per_family_std,
        "test_families_not_applicable": not_applicable,
        "test_not_applicable_reasons": {f: reasons[f] for f in not_applicable if f in reasons},
        "per_family_trials_not_applicable": partial,
        "per_family_cells_scored": cells_scored,
        "per_family_cells_not_applicable": cells_na,
        "test_partial_not_applicable_reasons": partial_reasons,
        "per_family_incompatible_runs": incompatible,
        "avg_transfer_auroc_mean": _agg(transfer_aurocs, np.mean),
        "avg_transfer_auroc_std": _agg(transfer_aurocs, np.std),
        "generalization_gap_pct_mean": _agg(gaps, np.mean, digits=1),
        "all_trial_train_aurocs": [_round_or_none(v) for v in train_aurocs],
        "all_trial_transfer_aurocs": [_round_or_none(v) for v in transfer_aurocs],
    }


def run_rows(
    rows: Sequence[str],
    *,
    n_seeds_train: int,
    n_seeds_test: int,
    n_trials: int,
    randomize_sign: bool = True,
    prepare: Optional[Callable[[Any], Any]] = None,
    train_families: Sequence[str] = TRAIN_FAMILIES,
    test_families: Sequence[str] = TEST_FAMILIES,
    timings: Optional[dict[str, float]] = None,
) -> dict[str, dict]:
    """Measure each of ``rows`` (names from :data:`ROWS`), in :data:`ROWS` order.

    The one entry point both this script's ``main`` and
    ``scripts/measure_sign_randomization.py transfer`` call, so the two measure RTS
    through the same code. ``prepare`` is applied to every freshly built detector
    before it is scored -- the sign-randomization script uses it to switch off the
    population hook for its "detector assumes the old orientation" configurations.
    """
    unknown = [r for r in rows if r not in ROW_FACTORIES]
    if unknown:
        raise ValueError(f"unknown transfer rows {unknown}; choose from {list(ROWS)}")
    wanted = set(rows)
    results: dict[str, dict] = {}
    for row in ROWS:
        if row not in wanted:
            continue
        factory = ROW_FACTORIES[row]

        def build(trial: int, _factory=factory):
            detector = _factory(trial)
            return prepare(detector) if prepare is not None else detector

        t0 = time.time()
        common = dict(
            randomize_sign=randomize_sign,
            train_families=train_families,
            test_families=test_families,
        )
        if row in MULTI_TRIAL_ROWS:
            results[row] = run_one_multitrial(
                row, build, ROW_LEVELS[row], n_seeds_train, n_seeds_test, n_trials, **common
            )
        else:
            results[row], _ = run_one(
                row, build(0), ROW_LEVELS[row], n_seeds_train, n_seeds_test, **common
            )
        if timings is not None:
            timings[row] = round(time.time() - t0, 1)
    return results


# --------------------------------------------------------------------------- artifact
#: What a value in the artifact means. Mirrors ``leaderboard/v5_leaderboard.json``'s.
CELL_SEMANTICS: dict[str, str] = {
    "per_family_null": (
        "not applicable -- no measurement exists: the family does not emit the RunData "
        "channel this detector's access level is defined over, or its horizon is "
        "shorter than the detector's min_episodes (rhob.v3.benchmark."
        "not_applicable_reason), or the frozen model could not consume the family's "
        "version of the channel (classify raised; per_family_incompatible_runs counts "
        "the runs), or the pooled fit failed (pooled_fit_error). "
        "test_not_applicable_reasons says which. Excluded from the transfer average; "
        "never imputed as 0.5."
    ),
    "per_family_number": (
        "measured mean AUROC of the frozen model over that held-out family's scored "
        "difficulty cells; for a multi-trial row, the mean over model-init trials."
    ),
    "per_family_cells": (
        "per_family_cells_scored / per_family_cells_not_applicable: how many of a "
        "held-out family's difficulty cells its number is over and how many were not "
        "applicable (for a multi-trial row, the fewest scored and the most left out in "
        "any trial). A family scored on some cells but not all keeps the reasons for the "
        "rest in test_partial_not_applicable_reasons."
    ),
    "train_auroc": (
        "5-fold stratified cross-validated AUROC per train cell (Benchmark.evaluate), "
        "averaged over the scored cells -- out of fold for every row, the ensemble "
        "included (its Trajectory MLP member is refitted on each training fold)."
    ),
    "pooled_fit": (
        "the frozen model is fitted once on the train families that pass the "
        "not-applicable rule for the detector being fitted (for the ensemble, its "
        "Trajectory MLP member); the others are listed in "
        "train_families_excluded_from_pooled_fit."
    ),
    "null_rts": (
        "avg_transfer_auroc(_mean) and generalization_gap_pct(_mean) are null when no "
        "held-out family was scored."
    ),
}


def published_design(args: argparse.Namespace, rows: Sequence[str]) -> bool:
    """Whether this run is the published design rather than a reduced smoke run."""
    return (
        set(rows) == set(ROWS)
        and list(args.train_families) == TRAIN_FAMILIES
        and list(args.test_families) == TEST_FAMILIES
        and all(getattr(args, k) == v for k, v in PUBLISHED_DESIGN.items())
    )


def build_payload(
    args: argparse.Namespace,
    rows: Sequence[str],
    results: dict[str, dict],
    timings: dict[str, float],
) -> dict:
    """The artifact ``main`` writes."""
    return {
        "train_families": list(args.train_families),
        "test_families": list(args.test_families),
        "n_seeds_train": args.n_seeds_train,
        "n_seeds_test": args.n_seeds_test,
        "n_trials": args.n_trials,
        "published_design": published_design(args, rows),
        "configs": SHIPPED_CONFIG,
        "provenance": provenance_block(
            script="scripts/cross_family_transfer.py", wall_clock_s=timings
        ),
        # Train and test draw from disjoint seed bases (TEST_SEED_BASE), so the
        # held-out evaluation never reuses a rollout the model was fitted on.
        # ``n_replicates=1``: the *environment* draw is a single one, shared by
        # every trial and every detector. ``model_init_trials`` is the separate
        # axis this experiment does replicate -- independently-seeded neural-net
        # weight initializations (see run_one_multitrial). Replicating model init
        # says nothing about sampling error in the environment draw.
        "sampling": sampling_block(
            n_seeds=args.n_seeds_train,
            n_layouts=len(LAYOUT_SEEDS),
            layout_seeds=LAYOUT_SEEDS,
            rollout_seeds_hacking=list(range(args.n_seeds_train)),
            rollout_seeds_legit=[LEGIT_SEED_OFFSET + s for s in range(args.n_seeds_train)],
            n_replicates=1,
            model_init_trials=args.n_trials,
            test_seed_base=TEST_SEED_BASE,
            n_seeds_test=args.n_seeds_test,
            rollout_seeds_hacking_test=[TEST_SEED_BASE + s for s in range(args.n_seeds_test)],
            rollout_seeds_legit_test=[
                TEST_SEED_BASE + LEGIT_SEED_OFFSET + s for s in range(args.n_seeds_test)
            ],
        ),
        "cell_semantics": CELL_SEMANTICS,
        "results": results,
    }


def _is_historical(path: Path) -> bool:
    """Whether ``path`` names the 2026-07-09 artifact, from the cwd or the repo root."""
    target = path.resolve()
    return target in {HISTORICAL_ARTIFACT.resolve(), (_REPO / HISTORICAL_ARTIFACT).resolve()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-seeds-train", type=int, default=PUBLISHED_DESIGN["n_seeds_train"])
    parser.add_argument("--n-seeds-test", type=int, default=PUBLISHED_DESIGN["n_seeds_test"])
    parser.add_argument(
        "--n-trials", type=int, default=PUBLISHED_DESIGN["n_trials"],
        help="Independent model-init trials for the neural-net rows (Reward MLP, "
             "Trajectory MLP, Ensemble); trial k seeds them with k. A single trial is "
             "not a reproducible measurement of a seed-sensitive fit -- see "
             "run_one_multitrial. State Divergence is deterministic and fitted once.",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"Output JSON path (default {DEFAULT_OUT}). Point it elsewhere for smoke "
             f"runs. {HISTORICAL_ARTIFACT} is the historical record and is refused.",
    )
    parser.add_argument(
        "--only", nargs="+", default=None, choices=list(ROWS),
        help="Run only these detector rows (smoke testing). Default: all four.",
    )
    parser.add_argument(
        "--train-families", nargs="+", default=list(TRAIN_FAMILIES), metavar="FAMILY",
        help="Smoke testing only: train on these families instead. The artifact then "
             "records published_design=false.",
    )
    parser.add_argument(
        "--test-families", nargs="+", default=list(TEST_FAMILIES), metavar="FAMILY",
        help="Smoke testing only: evaluate on these held-out families instead.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n_trials < 1:
        parser.error("--n-trials must be at least 1")
    if _is_historical(args.out):
        parser.error(
            f"refusing to overwrite {HISTORICAL_ARTIFACT}: it is the 2026-07-09 record of "
            f"what was published before sign randomization and is kept unchanged. "
            f"Write to {DEFAULT_OUT} (the default) or a scratch path."
        )
    registered = set(FamilyRegistry.list_families())
    unknown = sorted(set(args.train_families + args.test_families) - registered)
    if unknown:
        parser.error(f"unknown families: {', '.join(unknown)}")
    overlap = sorted(set(args.train_families) & set(args.test_families))
    if overlap:
        parser.error(f"a held-out family cannot also be a train family: {', '.join(overlap)}")

    rows = list(args.only) if args.only else list(ROWS)
    timings: dict[str, float] = {}
    results = run_rows(
        rows,
        n_seeds_train=args.n_seeds_train,
        n_seeds_test=args.n_seeds_test,
        n_trials=args.n_trials,
        randomize_sign=True,
        train_families=args.train_families,
        test_families=args.test_families,
        timings=timings,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(args, rows, results, timings)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n[OK] Saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
