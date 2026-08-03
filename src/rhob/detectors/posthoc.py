"""Post-hoc detectors for offline evaluation on completed runs.

These detectors consume a full run's recorded per-episode signals and produce:
- A discrimination score (hacking probability) for binary classification
- An onset episode prediction
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RunData:
    """A completed run's per-episode signals, access-level-appropriate.

    ``behav_trace`` is the pair's anti-symmetric L2 feature: ``mean(row) -
    mean(col)`` for gridworld, ``mean_x - centre`` for continuous. The two variants
    take opposite signs along it -- the same convention the admission gate itself uses
    to certify L2.

    Which of the two signs is the hacking one is **not** something a detector may
    assume. The benchmark multiplies both variants of a pair by a per-``(family,
    layout_seed)`` orientation before any detector sees the data, and does not disclose
    it (:mod:`rhob.v3.sign_randomization`). Under the pre-2026-08 fixed convention the
    sign of this field *was* the label, which made an L2 AUROC of 1.000 a restatement
    of ``CONTRIBUTING.md`` rather than a measurement. A detector that needs the
    direction has to infer it -- see :meth:`PosthocDetector.observe_cell`.

    ``true_rewards`` is oracle-only: the ground truth used to build onset
    labels and score the benchmark, never a legitimate detector input. No
    detector's ``classify``/``detect_onset`` may read it -- a detector that did
    would trivially "solve" hacking detection by reading the answer.
    """

    proxy_rewards: np.ndarray  # [n_episodes] proxy reward per episode (L0+)
    true_rewards: np.ndarray  # [n_episodes] oracle only -- not a detector input
    state_counts: Optional[np.ndarray]  # [n_episodes, n_bins] state-visitation histogram (L1+)
    behav_trace: Optional[np.ndarray]  # [n_episodes] anti-symmetric behavioral feature (L2)


class PosthocDetector(ABC):
    """Base class for post-hoc detectors."""

    @property
    @abstractmethod
    def access_level(self) -> str:
        """'L0', 'L1', or 'L2'."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name."""
        raise NotImplementedError

    def observe_cell(self, runs: list[RunData]) -> None:
        """Optionally look at one evaluation cell's runs, **unlabeled**, before scoring.

        The benchmark calls this once per (family, difficulty) cell with every run in
        it -- both variants pooled, shuffled into a label-free order, already restricted
        to the detector's access level -- and then calls :meth:`classify` on those same
        runs. It is the transductive, unsupervised information a post-hoc detector
        genuinely has: "here is a batch of runs, some fraction of which are hacking".

        It carries no labels, and it is not :meth:`fit`. ``fit(runs_a, runs_b)`` (the
        cross-validated supervised hook) is given the answer and is scored out-of-fold
        because of it; this is not, so it needs no fold discipline.

        It exists because sign randomization (:mod:`rhob.v3.sign_randomization`) makes
        the behavioral feature's *direction* unknowable from any single run, while
        leaving it inferable in principle from a population. Without this hook the only
        place direction could be recovered is the evaluation layer, and recovering it
        there -- e.g. reporting ``max(auroc, 1 - auroc)`` -- would hand every detector
        the answer back for free and measure nothing. Orienting is the detector's job,
        so the detector is what gets the population.

        Default: no-op. A detector whose score does not depend on a direction (every
        magnitude-based one) needs nothing here.
        """

    @abstractmethod
    def classify(self, run: RunData) -> float:
        """Score a run for hacking likelihood. Higher = more likely hacking."""
        raise NotImplementedError

    @abstractmethod
    def detect_onset(self, run: RunData) -> int:
        """Predict onset episode. Return -1 if no onset detected."""
        raise NotImplementedError
