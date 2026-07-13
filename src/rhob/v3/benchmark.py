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
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score

from rhob.detectors.posthoc import PosthocDetector, RunData
from rhob.v3.access import restrict
from rhob.v3.registry import FamilyRegistry

# Post-hoc detectors are the v3 detector interface.
Detector = PosthocDetector


@dataclass
class CellResult:
    """One (family, difficulty) evaluation cell."""

    family: str
    mechanism: str
    difficulty: float
    discrimination_auroc: float
    onset_mae: float
    n_seeds: int


@dataclass
class BenchmarkResults:
    """The full result of a benchmark run."""

    detector_name: str
    access_level: str
    cells: list[CellResult] = field(default_factory=list)

    @property
    def overall_auroc(self) -> float:
        vals = [c.discrimination_auroc for c in self.cells if not np.isnan(c.discrimination_auroc)]
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
            lines.append(
                f"{c.family:<24}{c.mechanism:<14}{c.difficulty:>6.2f}"
                f"{c.discrimination_auroc:>9.3f}{c.onset_mae:>10.3f}"
            )
        lines.append("-" * 68)
        lines.append(f"{'OVERALL mean AUROC':<58}{self.overall_auroc:>10.3f}")
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
) -> tuple[float, float]:
    """Score one (family, difficulty) cell: (discrimination_auroc, onset_mae).

    Detectors exposing ``.fit`` are evaluated with 5-fold (or fewer, if the sample is
    small) stratified cross-validation and out-of-fold scoring; others are scored
    directly, since they carry no fittable state to leak across folds.
    """
    restricted_a = [restrict(r, level) for r in runs_a]
    restricted_b = [restrict(r, level) for r in runs_b]
    n_a, n_b = len(restricted_a), len(restricted_b)
    labels = np.array([1] * n_a + [0] * n_b)
    all_runs = restricted_a + restricted_b

    if not hasattr(detector, "fit"):
        scores = [detector.classify(r) for r in all_runs]
        onset_preds = [detector.detect_onset(r) for r in restricted_a]
        auroc = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
        return auroc, _onset_mae(onset_preds, onsets_a, n_episodes)

    k = min(5, n_a, n_b)
    if k < 2:
        # Too few seeds for cross-validation: fit once on everything (optimistic,
        # but the only option) rather than fail outright.
        fitted = copy.deepcopy(detector)
        fitted.fit(restricted_a, restricted_b)
        scores = [fitted.classify(r) for r in all_runs]
        onset_preds = [fitted.detect_onset(r) for r in restricted_a]
        auroc = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
        return auroc, _onset_mae(onset_preds, onsets_a, n_episodes)

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
    return auroc, _onset_mae(onset_preds, onsets_a, n_episodes)


class Benchmark:
    """Main evaluation entry point."""

    # Rollout data depends only on (family, difficulty, n_seeds) -- each family's
    # generate_pair(difficulty) is called with the same default seed=0, and
    # MatchedPair.rollout(n_seeds) always draws the same deterministic seed sequence
    # from it, so it is 100% reproducible across calls regardless of which detector
    # is being scored. Without this cache, evaluating N detectors over the same
    # families/difficulties re-simulates every environment from scratch N times --
    # for PettingZoo's real multi-agent physics rollouts this made a 30-detector
    # leaderboard run balloon from hours to an estimated multiple days. restrict()
    # (called downstream in _evaluate_cell) returns a copy rather than mutating its
    # input, so sharing these RunData objects across detectors is safe.
    _rollout_cache: dict[tuple[str, float, int], tuple[list[RunData], list[RunData], list[int]]] = {}

    @staticmethod
    def evaluate(
        detector: PosthocDetector,
        families: str | list[str] = "all",
        difficulties: str | list[float] = "all",
        n_seeds: int = 20,
        verbose: bool = True,
    ) -> BenchmarkResults:
        """Evaluate a detector across the benchmark suite.

        Args:
            detector: A post-hoc detector (``classify``/``detect_onset``/``access_level``).
            families: ``"all"`` or a family name / list of names.
            difficulties: ``"all"`` or a target-L2 / list of target-L2 values.
            n_seeds: Runs per variant per pair.
            verbose: Print per-cell progress.
        """
        level = detector.access_level
        name = getattr(detector, "name", type(detector).__name__)
        results = BenchmarkResults(detector_name=name, access_level=level)

        pairs = FamilyRegistry.generate_suite(families, difficulties)
        for pair in pairs:
            cache_key = (pair.family, pair.difficulty, n_seeds)
            cached = Benchmark._rollout_cache.get(cache_key)
            if cached is None:
                cached = pair.rollout(n_seeds)
                Benchmark._rollout_cache[cache_key] = cached
            runs_a, runs_b, onsets_a = cached
            auroc, mae = _evaluate_cell(detector, runs_a, runs_b, onsets_a, level, pair.n_episodes)
            results.cells.append(
                CellResult(
                    family=pair.family,
                    mechanism=pair.mechanism.value,
                    difficulty=pair.difficulty,
                    discrimination_auroc=auroc,
                    onset_mae=mae,
                    n_seeds=n_seeds,
                )
            )
            if verbose:
                print(f"  {pair.family} @ L2*={pair.difficulty:.2f}: AUROC={auroc:.3f}")
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
