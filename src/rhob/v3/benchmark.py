"""Benchmark: the single evaluation entry point for RHOB v3.

``Benchmark.evaluate(detector, families, difficulties, n_seeds)`` rolls out every
(family, difficulty) pair, scores the detector under its declared access level, and
returns a :class:`BenchmarkResults` with a printable summary. Detectors are the
post-hoc kind (``classify(run) -> float``, ``detect_onset(run) -> int``); every
existing RHOB detector already satisfies this interface.

Detectors that must be *fit* on labeled data before they can classify anything
(``StateDivergenceDetector``, ``RewardMLPDetector``, ``TrajectoryMLPDetector``, and
others exposing a ``.fit(runs_a, runs_b)`` method) are automatically evaluated with
5-fold stratified cross-validation and out-of-fold scoring -- the same rigor used by
the CR1 evaluation pipeline (``scripts/evaluate_detectors.py``) -- so a detector is
never scored on runs it was fit on.

A cell whose family does not emit the channel the detector reads is scored **N/A**
(NaN), not 0.5 -- see :data:`_ACCESS_LEVEL_CHANNEL` and :func:`missing_channels`.

Every cell's L2 behavioral feature arrives in a per-family orientation the benchmark
draws and withholds (:mod:`rhob.v3.sign_randomization`), so a detector cannot read the
label off ``behav_trace``'s sign. What a detector *is* offered instead is the cell's
runs, pooled and unlabeled, through :func:`offer_population`; recovering the direction
from that is the detector's job, not the harness's.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from rhob.detectors.posthoc import PosthocDetector, RunData
from rhob.v3.access import restrict
from rhob.v3.registry import FamilyRegistry

# Post-hoc detectors are the v3 detector interface.
Detector = PosthocDetector

# The :class:`~rhob.detectors.posthoc.RunData` channel a detector at each access level
# is *defined over*. An access level is not a preference -- it names the signal the
# detector reads. Hand an L1 (state-visitation) detector a run whose ``state_counts``
# is None and it has no input at all: every L1 detector in this repo returns its 0.5
# no-signal default in exactly that case (``l1_bimodal_occupancy.py`` classify(),
# ``l1_state_divergence.py`` classify() via ``_steady_hist`` returning None, and the
# same guard in the other six). Averaging that constant 0.5 into a mean AUROC reports
# a hardcoded fallback as if it were a measurement.
#
# This mattered, and at scale: 25 of the 33 registered families ship
# ``state_counts=None`` (every MuJoCo, PettingZoo, sequence-generation and shared-RLHF
# family, plus 6 others), covering 88 of the 123 leaderboard cells. The published L1
# access-level number was therefore a hardcoded 0.5 on 71% of its cells, diluted with
# 35 cells of real measurement. Recomputing the committed v5 board over the measured
# cells only: the L1 detector-suite mean moves 0.520 -> 0.570, and State Divergence --
# the best L1 detector -- moves 0.621 -> 0.927, i.e. the fabricated cells were hiding a
# strong result, not propping up a weak one. Cells whose required channel is absent are
# therefore scored NaN and excluded from every mean, and reported as ``not_applicable``
# downstream.
#
# A detector may override the default for its level by exposing a ``required_channels``
# attribute (a tuple of ``RunData`` field names). That hook exists for the L3 ceiling
# detectors, which do not share one input channel: the true-reward oracle reads
# ``true_rewards`` while the perfect-feature oracle reads ``behav_trace``.
_ACCESS_LEVEL_CHANNEL: dict[str, tuple[str, ...]] = {
    "L0": ("proxy_rewards",),
    "L1": ("state_counts",),
    "L2": ("behav_trace",),
    "L3": ("true_rewards",),
}


def required_channels(detector: PosthocDetector) -> tuple[str, ...]:
    """The ``RunData`` fields ``detector`` needs before a cell is scoreable."""
    declared = getattr(detector, "required_channels", None)
    if declared is not None:
        return tuple(declared)
    return _ACCESS_LEVEL_CHANNEL.get(detector.access_level, ())


def missing_channels(runs: list[RunData], required: tuple[str, ...]) -> list[str]:
    """Which of ``required`` are absent from *any* run in ``runs``.

    A channel counts as absent when it is ``None`` or an empty array -- both are how
    a family (or :func:`~rhob.v3.access.restrict`) signals "this signal does not
    exist here", and neither carries information a detector could score.
    """
    missing = []
    for channel in required:
        for run in runs:
            value = getattr(run, channel, None)
            if value is None or np.size(value) == 0:
                missing.append(channel)
                break
    return missing


@dataclass
class CellResult:
    """One (family, difficulty) evaluation cell.

    ``na_reason`` is None for a scored cell. When it is set, the family did not emit
    the channel this detector reads, ``discrimination_auroc``/``onset_mae`` are NaN,
    and the cell must be excluded from aggregates rather than counted as chance.
    """

    family: str
    mechanism: str
    difficulty: float
    discrimination_auroc: float
    onset_mae: float
    n_seeds: int
    na_reason: Optional[str] = None

    @property
    def applicable(self) -> bool:
        """True when the detector's input channel was present and the cell was scored."""
        return self.na_reason is None


@dataclass
class BenchmarkResults:
    """The full result of a benchmark run."""

    detector_name: str
    access_level: str
    cells: list[CellResult] = field(default_factory=list)

    @property
    def scored_cells(self) -> list[CellResult]:
        """Cells that were actually measured (channel present, AUROC not NaN)."""
        return [c for c in self.cells if c.applicable and not np.isnan(c.discrimination_auroc)]

    @property
    def na_cells(self) -> list[CellResult]:
        """Cells skipped because the family does not emit this detector's input channel."""
        return [c for c in self.cells if not c.applicable]

    @property
    def na_families(self) -> list[str]:
        """Families for which every cell was not applicable."""
        na = {c.family for c in self.na_cells}
        scored = {c.family for c in self.scored_cells}
        return sorted(na - scored)

    @property
    def overall_auroc(self) -> float:
        """Mean AUROC over the *measured* cells; NaN if nothing was measurable.

        Not-applicable cells are excluded, not imputed: a detector whose input channel
        a family never emits has produced no evidence about that family, and averaging
        in a 0.5 placeholder would pull the mean toward chance by construction.
        """
        vals = [c.discrimination_auroc for c in self.scored_cells]
        return float(np.mean(vals)) if vals else float("nan")

    def summary(self) -> str:
        """Return (and print) a human-readable AUROC table."""
        lines = [
            f"RHOB v3 results for {self.detector_name} (access {self.access_level})",
            "=" * 68,
            f"{'family':<24}{'mechanism':<14}{'diff':>6}{'AUROC':>9}{'onsetMAE':>10}",
            "-" * 68,
        ]
        for c in sorted(self.cells, key=lambda c: (c.family, -c.difficulty)):
            auroc_str = "      N/A" if not c.applicable else f"{c.discrimination_auroc:>9.3f}"
            mae_str = "       N/A" if not c.applicable else f"{c.onset_mae:>10.3f}"
            lines.append(
                f"{c.family:<24}{c.mechanism:<14}{c.difficulty:>6.2f}{auroc_str}{mae_str}"
            )
        lines.append("-" * 68)
        lines.append(f"{'OVERALL mean AUROC':<58}{self.overall_auroc:>10.3f}")
        n_na = len(self.na_cells)
        if n_na:
            reasons = sorted({c.na_reason for c in self.na_cells if c.na_reason})
            lines.append(
                f"{'  (over ' + str(len(self.scored_cells)) + ' measured cells; ' + str(n_na) + ' N/A, excluded)':<68}"
            )
            for reason in reasons:
                lines.append(f"  N/A reason: {reason}")
        text = "\n".join(lines)
        print(text)
        return text


def _stratified_folds(n_a: int, n_b: int, k: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """k (train_idx, test_idx) splits over [0, n_a+n_b), stratified by class.

    Indices [0, n_a) are class A (hacking), [n_a, n_a+n_b) are class B (legitimate).
    """
    rng = np.random.default_rng(seed)
    idx_a = rng.permutation(n_a)
    idx_b = n_a + rng.permutation(n_b)
    folds_a = np.array_split(idx_a, k)
    folds_b = np.array_split(idx_b, k)
    all_idx = np.arange(n_a + n_b)
    splits = []
    for i in range(k):
        test_idx = np.concatenate([folds_a[i], folds_b[i]])
        train_idx = np.setdiff1d(all_idx, test_idx)
        splits.append((train_idx, test_idx))
    return splits


#: Seed for the permutation that hides the label-implied ordering of a cell's runs
#: before they are shown to :meth:`~rhob.detectors.posthoc.PosthocDetector.observe_cell`.
#:
#: The cell is assembled as ``[all hacking runs] + [all legitimate runs]``, so passing
#: it unshuffled would leak the labels through the index -- the population hook is meant
#: to be unlabeled, and "the first half are the positives" is a label. Fixed rather than
#: drawn, because a detector that reorders or re-weights by position must produce the
#: same leaderboard cell on every run (``REPRODUCIBILITY.md``).
#:
#: This removes the *incidental* leak, not a determined one: the permutation is a
#: deterministic function of a constant in this file, so a detector could reconstruct it
#: the same way it could import ``true_rewards``. That is the standing rule, not a hole
#: this constant can close -- reading the label by any route disqualifies a detector.
_POPULATION_SHUFFLE_SEED = 20_260_803


def offer_population(detector: PosthocDetector, runs: list[RunData]) -> None:
    """Show ``detector`` the cell's runs, unlabeled and in a label-free order.

    Public because any evaluation path that scores a cell outside
    :func:`_evaluate_cell` must offer the same thing -- ``scripts/cross_family_transfer.py``
    scores frozen models on held-out families by hand, and a detector denied the
    population there would be penalized for the family's drawn sign rather than for
    failing to generalize. A detector with no ``observe_cell`` is left untouched.
    """
    observe = getattr(detector, "observe_cell", None)
    if observe is None:
        return
    order = np.random.default_rng(_POPULATION_SHUFFLE_SEED).permutation(len(runs))
    observe([runs[i] for i in order])


def _onset_mae(preds: list[int], onsets_true: list[int], n_episodes: int) -> float:
    errors = []
    for pred, true_onset in zip(preds, onsets_true):
        if pred < 0 or true_onset < 0:
            errors.append(1.0)
        else:
            errors.append(abs(pred - true_onset) / n_episodes)
    return float(np.mean(errors)) if errors else float("nan")


def _evaluate_cell(
    detector: PosthocDetector,
    runs_a: list[RunData],
    runs_b: list[RunData],
    onsets_a: list[int],
    level: str,
    n_episodes: int,
) -> tuple[float, float, Optional[str]]:
    """Score one (family, difficulty) cell: ``(discrimination_auroc, onset_mae, na_reason)``.

    Detectors exposing ``.fit`` are evaluated with 5-fold (or fewer, if the sample is
    small) stratified cross-validation and out-of-fold scoring; others are scored
    directly, since they carry no fittable state to leak across folds.

    If the family does not emit the channel this detector reads, the cell is *not
    scored at all*: it returns ``(nan, nan, reason)``. Running the detector anyway
    would produce its no-signal constant (0.5 for every L1 detector here), which is
    a property of the fallback branch, not of the family.
    """
    restricted_a = [restrict(r, level) for r in runs_a]
    restricted_b = [restrict(r, level) for r in runs_b]
    n_a, n_b = len(restricted_a), len(restricted_b)
    labels = np.array([1] * n_a + [0] * n_b)
    all_runs = restricted_a + restricted_b

    required = required_channels(detector)
    missing = missing_channels(all_runs, required)
    if missing:
        reason = (
            f"family does not provide {'/'.join(missing)}, which this {level} "
            f"detector requires; not scored"
        )
        return float("nan"), float("nan"), reason

    # Unlabeled population, before any scoring. This is what makes a detector able to
    # orient itself under sign randomization without the harness handing it a direction
    # (see PosthocDetector.observe_cell). It runs before the deepcopy below, so every
    # cross-validation fold inherits the same cell-level view -- which is sound because
    # nothing in it is a label.
    offer_population(detector, all_runs)

    if not hasattr(detector, "fit"):
        scores = [detector.classify(r) for r in all_runs]
        onset_preds = [detector.detect_onset(r) for r in restricted_a]
        auroc = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
        return auroc, _onset_mae(onset_preds, onsets_a, n_episodes), None

    k = min(5, n_a, n_b)
    if k < 2:
        # Too few seeds for cross-validation: fit once on everything (optimistic,
        # but the only option) rather than fail outright.
        fitted = copy.deepcopy(detector)
        fitted.fit(restricted_a, restricted_b)
        scores = [fitted.classify(r) for r in all_runs]
        onset_preds = [fitted.detect_onset(r) for r in restricted_a]
        auroc = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
        return auroc, _onset_mae(onset_preds, onsets_a, n_episodes), None

    oof_scores = np.full(n_a + n_b, np.nan)
    onset_by_index: dict[int, int] = {}
    for train_idx, test_idx in _stratified_folds(n_a, n_b, k):
        fold_detector = copy.deepcopy(detector)
        train_a = [all_runs[i] for i in train_idx if labels[i] == 1]
        train_b = [all_runs[i] for i in train_idx if labels[i] == 0]
        fold_detector.fit(train_a, train_b)
        for i in test_idx:
            oof_scores[i] = fold_detector.classify(all_runs[i])
            if i < n_a:
                onset_by_index[i] = fold_detector.detect_onset(all_runs[i])

    auroc = float(roc_auc_score(labels, oof_scores)) if len(set(labels)) > 1 else float("nan")
    onset_preds = [onset_by_index.get(i, -1) for i in range(n_a)]
    return auroc, _onset_mae(onset_preds, onsets_a, n_episodes), None


class RolloutCacheKey(NamedTuple):
    """Everything that changes a cell's rollout arrays, named rather than positional.

    A plain tuple invites two mistakes that this type makes impossible. Adding a field
    silently breaks any reader indexing from the end -- when ``layout_seed`` and
    ``seed_base`` were appended in 2026-08, a test asserting ``key[-1]`` was the
    randomize flag started reading ``seed_base`` and comparing ``0 == False``, which is
    true in Python and would have passed had the values lined up differently. And a
    field omitted here is a field the cache ignores, which is how a replication study
    ends up serving one draw to every replicate.
    """

    family: str
    difficulty: float
    n_seeds: int
    randomize_behav_sign: bool
    layout_seed: int
    seed_base: int


class Benchmark:
    """Main evaluation entry point."""

    # Rollout data is a deterministic function of the full key below, so evaluating N
    # detectors over the same cells re-uses one simulation instead of repeating it N
    # times -- for PettingZoo's real multi-agent physics rollouts that difference made a
    # 30-detector leaderboard run go from an estimated multiple days to hours.
    # restrict() (called downstream in _evaluate_cell) returns a copy rather than
    # mutating its input, so sharing these RunData objects across detectors is safe.
    #
    # EVERY input that changes the cached arrays must appear in the key, or the cache
    # silently serves one draw's data to a caller that asked for another:
    #
    #   randomize_behav_sign -- a cell rolled out un-randomized and then served to a
    #     randomized evaluation would reinstate the sign convention this benchmark now
    #     withholds.
    #   layout_seed, seed_base -- these are what make a *replicate* a replicate. They
    #     were absent until 2026-08, when the leaderboard was still a single draw and
    #     both were pinned at 0, so nothing could observe the omission. The moment
    #     replication was attempted it became load-bearing: R replicates run in one
    #     process would all hit the layout-0/seed-0 entry, return bit-identical AUROCs,
    #     and yield a zero-width confidence interval that looks like a rigorous result
    #     and measures nothing. tests/test_v3/test_replication_draws_differ.py fails if
    #     either field is dropped from this key.
    _rollout_cache: dict[
        "RolloutCacheKey", tuple[list[RunData], list[RunData], list[int]]
    ] = {}

    @staticmethod
    def evaluate(
        detector: PosthocDetector,
        families: str | list[str] = "all",
        difficulties: str | list[float] = "all",
        n_seeds: int = 20,
        verbose: bool = True,
        randomize_behav_sign: bool = True,
        layout_seed: int = 0,
        seed_base: int = 0,
    ) -> BenchmarkResults:
        """Evaluate a detector across the benchmark suite.

        Args:
            detector: A post-hoc detector (``classify``/``detect_onset``/``access_level``).
            families: ``"all"`` or a family name / list of names.
            difficulties: ``"all"`` or a target-L2 / list of target-L2 values.
            n_seeds: Runs per variant per pair.
            verbose: Print per-cell progress.
            randomize_behav_sign: Randomize the orientation of ``behav_trace`` per
                ``(family, layout_seed)`` before the detector sees it, so an L2
                detector cannot read the label off the feature's sign
                (:mod:`rhob.v3.sign_randomization`). Default ``True``, and any L2 number
                reported as a RHOB result must be measured with it on. Set ``False``
                only to reproduce the pre-audit convention for a before/after
                comparison -- an L2 result obtained with it off is a measurement of
                ``CONTRIBUTING.md``, not of the detector. (The leaderboard artifacts
                committed before the 2026-08 audit predate this flag and were produced
                under the old convention; they have not been regenerated.)
            layout_seed: Environment layout every cell is generated at. Also selects
                each family's behavioral orientation under sign randomization.
            seed_base: Offset of the per-run seed sequence drawn from each pair.

        ``(layout_seed, seed_base)`` together identify one *draw* of the benchmark. The
        defaults ``(0, 0)`` reproduce the historical single-draw suite; a published
        AUROC from a single draw carries that draw's sampling error (SE 0.19 at chance
        for ``n_seeds=5``) and should not be reported without an interval. Vary the pair
        to obtain independent replicates -- see ``scripts/replicate_leaderboard.py``.
        """
        level = detector.access_level
        name = getattr(detector, "name", type(detector).__name__)
        results = BenchmarkResults(detector_name=name, access_level=level)

        pairs = FamilyRegistry.generate_suite(families, difficulties, layout_seed=layout_seed)
        for pair in pairs:
            cache_key = RolloutCacheKey(
                family=pair.family,
                difficulty=pair.difficulty,
                n_seeds=n_seeds,
                randomize_behav_sign=randomize_behav_sign,
                layout_seed=layout_seed,
                seed_base=seed_base,
            )
            cached = Benchmark._rollout_cache.get(cache_key)
            if cached is None:
                cached = pair.rollout(
                    n_seeds, seed_base=seed_base, randomize_sign=randomize_behav_sign
                )
                Benchmark._rollout_cache[cache_key] = cached
            runs_a, runs_b, onsets_a = cached
            auroc, mae, na_reason = _evaluate_cell(
                detector, runs_a, runs_b, onsets_a, level, pair.n_episodes
            )
            results.cells.append(
                CellResult(
                    family=pair.family,
                    mechanism=pair.mechanism.value,
                    difficulty=pair.difficulty,
                    discrimination_auroc=auroc,
                    onset_mae=mae,
                    n_seeds=n_seeds,
                    na_reason=na_reason,
                )
            )
            if verbose:
                shown = "N/A" if na_reason else f"{auroc:.3f}"
                suffix = f"  ({na_reason})" if na_reason else ""
                print(f"  {pair.family} @ L2*={pair.difficulty:.2f}: AUROC={shown}{suffix}")
        return results

    @staticmethod
    def list_families() -> list[dict]:
        """Registered families with their taxonomy metadata."""
        out = []
        for n in FamilyRegistry.list_families():
            fam = FamilyRegistry.get(n)
            lo, hi = fam.difficulty_range()
            out.append(
                {
                    "family": fam.name,
                    "mechanism": fam.mechanism.value,
                    "complexity": fam.complexity.value,
                    "difficulty_range": (lo, hi),
                }
            )
        return out
