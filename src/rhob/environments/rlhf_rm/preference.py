# src/rhob/environments/rlhf_rm/preference.py
"""True reward function and synthetic preference-data generation.

``true_reward`` is the SAME fixed function across all 5 RLHF-RM families -- only how
each family's preference data is *generated* (coverage/noise/features/population)
differs. This mirrors the MuJoCo families' pattern of a shared true-reward convention
with per-family proxy construction.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

# Fixed nonlinear true-reward weights, shared by every RLHF-RM family. Quadratic
# (diminishing/penalizing extremes) plus one cross-term (dims 0 and 1 interact) so a
# purely-linear reward model can never perfectly represent r* -- this is what makes
# reward-model overoptimization a genuine, not scripted, phenomenon.
_LINEAR_W = np.array([1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
_QUADRATIC_W = np.array([0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05])
_CROSS_W = 0.2  # coefficient on x[0] * x[1]


def true_reward(x: np.ndarray) -> np.ndarray:
    """r*(x) for a batch of responses ``x`` of shape (n, response_dim)."""
    linear = x @ _LINEAR_W
    quadratic = -(x**2) @ _QUADRATIC_W  # diminishing returns / penalizes extremes
    cross = _CROSS_W * x[:, 0] * x[:, 1]
    return linear + quadratic + cross


def generate_preference_data(
    rng: np.random.Generator,
    n_pairs: int,
    response_dim: int,
    sample_fn: Callable[[np.random.Generator, int, int], np.ndarray],
    label_noise_std: float,
    label_weight_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``n_pairs`` preference-labeled response pairs.

    Args:
        sample_fn: (rng, n_pairs, response_dim) -> array of shape (n_pairs, response_dim)
            sampled response-A's; response-B's are sampled independently the same way.
            Each family supplies its own ``sample_fn`` to control *coverage* (e.g.
            undersampling a region) -- this is where each family's calibration bracket
            actually lives, not in ``true_reward`` itself.
        label_noise_std: Gaussian noise added to the true-reward gap before
            thresholding into a binary preference label (models annotator disagreement;
            larger values give noisier, more boundary-concentrated labels).
        label_weight_fn: if given, applied to the response-feature vectors before
            scoring for labeling purposes only (models a labeler population that
            weights features differently than ``true_reward`` -- used by the
            preference-population-bias family; ``None`` means labelers score by
            ``true_reward`` exactly).

    Returns:
        (X, y): X has shape (2*n_pairs, response_dim) -- interleaved A/B responses;
        y has shape (n_pairs,) -- 1 if A preferred over B, 0 otherwise. Callers fit a
        pairwise (Bradley-Terry-style) model on the *difference* features
        ``X[0::2] - X[1::2]``, labeled by ``y``.
    """
    a = sample_fn(rng, n_pairs, response_dim)
    b = sample_fn(rng, n_pairs, response_dim)
    score_fn = true_reward if label_weight_fn is None else label_weight_fn
    gap = score_fn(a) - score_fn(b)
    noisy_gap = gap + rng.normal(0.0, label_noise_std, size=gap.shape)
    y = (noisy_gap > 0).astype(int)
    x_interleaved = np.empty((2 * n_pairs, response_dim))
    x_interleaved[0::2] = a
    x_interleaved[1::2] = b
    return x_interleaved, y


def fit_reward_model(x_interleaved: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit a linear (Bradley-Terry-style) reward model on preference pairs.

    Returns the fitted weight vector (shape (response_dim,)); the reward model's score
    for a response ``x`` is ``x @ weights``. A real ``LogisticRegression`` fit on the
    pairwise difference features, not a hand-scripted proxy -- the RM's blind spots
    emerge from what ``x_interleaved``/``y`` actually contain.
    """
    diffs = x_interleaved[0::2] - x_interleaved[1::2]
    clf = LogisticRegression(fit_intercept=False)
    clf.fit(diffs, y)
    return clf.coef_[0]
