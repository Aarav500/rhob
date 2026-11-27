"""Bayesian Online Changepoint Detection (Adams & MacKay, 2007) -- Bayesian baseline (L0)."""

from __future__ import annotations

import numpy as np

from rhob.detectors.posthoc import PosthocDetector, RunData

#: Fewest rewards the recursion is run on. Below this ``classify`` returns 0.5 and
#: ``detect_onset`` returns -1 for every run.
MIN_EPISODES = 5

#: Run-length hypotheses kept per step: the standard practical truncation that bounds the
#: recursion's cost. It binds only on runs longer than this many episodes.
MAX_RUN_LENGTHS = 500


class BOCPDDetector(PosthocDetector):
    """Classical Bayesian Online Changepoint Detection on the proxy-reward stream.

    Maintains a posterior over "run length since last changepoint" using a
    Normal-Gamma conjugate model for the reward-generating process (Adams &
    MacKay, 2007, arXiv:0710.3742) and a constant hazard rate. This is a
    general-purpose Bayesian time-series changepoint method, not designed for
    RHOB or matched-proxy hacking specifically -- included for a fair,
    citable Bayesian-methods comparison point.

    Score is the posterior probability that a changepoint occurred somewhere in the run,
    ``1 - P(run length == full length)``: one minus the mass the run-length posterior
    keeps, after the last reward, on the segment that began at the first one. Onset is
    the first step at which that probability exceeds 0.5, i.e. the first step after
    which "no changepoint yet" is the less likely account of the rewards seen so far.

    Until 2026-11-27 the code scored ``max_t P(r_t = 0 | x_0..x_t)`` instead, the
    probability that a changepoint happened *at* step ``t``. With a constant hazard that
    quantity is the hazard itself at every step (the changepoint and growth terms share
    the same predictive, which cancels on normalisation), so every run scored 0.004, the
    onset rule could never fire, and the published AUROCs ranked rounding noise.
    """

    def __init__(self, hazard: float = 1.0 / 250.0):
        if not 0.0 < hazard < 1.0:
            raise ValueError(f"hazard must lie strictly between 0 and 1, got {hazard!r}")
        self.hazard = hazard

    @property
    def access_level(self) -> str:
        return "L0"

    @property
    def name(self) -> str:
        return "Bayesian Online Changepoint Detection"

    @property
    def min_episodes(self) -> int:
        """Five rewards, the length the recursion is run from.

        Below that ``_changed_probabilities`` returns None and ``classify`` returns 0.5
        for every run. The bound does not depend on ``hazard``.
        """
        return MIN_EPISODES

    def classify(self, run: RunData) -> float:
        changed = self._changed_probabilities(run.proxy_rewards)
        if changed is None or len(changed) == 0:
            return 0.5
        return float(changed[-1])

    def detect_onset(self, run: RunData) -> int:
        changed = self._changed_probabilities(run.proxy_rewards)
        if changed is None:
            return -1
        crossed = np.where(changed > 0.5)[0]
        return int(crossed[0]) if len(crossed) else -1

    def _changed_probabilities(self, rewards: np.ndarray) -> np.ndarray | None:
        """``P(a changepoint occurred in x_0..x_t | x_0..x_t)`` for every step ``t``.

        One minus :meth:`_log_no_change_probabilities`, taken through ``expm1`` so a
        posterior close to "no change" keeps its precision.
        """
        log_no_change = self._log_no_change_probabilities(rewards)
        if log_no_change is None:
            return None
        return -np.expm1(log_no_change)

    def _log_no_change_probabilities(self, rewards: np.ndarray) -> np.ndarray | None:
        """``log P(no changepoint in x_0..x_t | x_0..x_t)`` for every step ``t``.

        Standard BOCPD recursion with a Normal-Inverse-Gamma predictive. After step ``t``
        hypothesis ``j`` is the segment that has absorbed the last ``j`` rewards; index 0
        is the segment that has just begun and has seen none. The growth chain's tail,
        the last hypothesis, is therefore the segment that began at the first reward, and
        its posterior mass is the probability that no changepoint has happened yet.

        Truncation keeps the ``MAX_RUN_LENGTHS`` most probable hypotheses in order, so
        the tail survives a truncation exactly when the last index kept is the last index
        there was. Once it is dropped nothing can restore it (every survivor is a younger
        segment), so ``full_alive`` stays False and the probability is 0 (log -inf) from
        that step on, rather than the mass of whichever hypothesis is now last.
        """
        if rewards is None or len(rewards) < self.min_episodes:
            return None
        x = rewards.astype(np.float64)
        n = len(x)
        log_hazard, log_survive = np.log(self.hazard), np.log1p(-self.hazard)

        # Normal-Inverse-Gamma prior hyperparameters (weakly informative).
        mu0, kappa0, alpha0, beta0 = float(x[0]), 1.0, 1.0, 1.0

        mu = np.array([mu0])
        kappa = np.array([kappa0])
        alpha = np.array([alpha0])
        beta = np.array([beta0])
        log_run_length_probs = np.array([0.0])  # log P(run length = 0) at t=0

        log_no_change = np.full(n, -np.inf)
        full_alive = True
        for t in range(n):
            xt = x[t]
            # Predictive: Student-t, approximated via Gaussian with scale for stability.
            pred_var = beta * (kappa + 1) / (alpha * kappa)
            pred_var = np.maximum(pred_var, 1e-8)
            log_pred = -0.5 * np.log(2 * np.pi * pred_var) - 0.5 * (xt - mu) ** 2 / pred_var

            log_growth = log_run_length_probs + log_pred + log_survive
            log_cp = self._logsumexp(log_run_length_probs + log_pred + log_hazard)

            new_log_probs = np.concatenate([[log_cp], log_growth])
            new_log_probs -= self._logsumexp(new_log_probs)

            # Bayesian update (prepend prior for the new run-length-0 hypothesis).
            new_mu = np.concatenate([[mu0], (kappa * mu + xt) / (kappa + 1)])
            new_kappa = np.concatenate([[kappa0], kappa + 1])
            new_alpha = np.concatenate([[alpha0], alpha + 0.5])
            new_beta = np.concatenate(
                [[beta0], beta + (kappa * (xt - mu) ** 2) / (2 * (kappa + 1))]
            )

            mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta
            log_run_length_probs = new_log_probs

            # Truncate history to bound cost (standard practical BOCPD trick).
            if len(mu) > MAX_RUN_LENGTHS:
                keep = np.argsort(log_run_length_probs)[-MAX_RUN_LENGTHS:]
                keep.sort()
                if keep[-1] != len(mu) - 1:
                    full_alive = False
                mu, kappa, alpha, beta = mu[keep], kappa[keep], alpha[keep], beta[keep]
                log_run_length_probs = log_run_length_probs[keep]
                log_run_length_probs -= self._logsumexp(log_run_length_probs)

            if full_alive:
                log_no_change[t] = log_run_length_probs[-1]

        return log_no_change

    @staticmethod
    def _logsumexp(log_values: np.ndarray) -> float:
        m = np.max(log_values)
        if not np.isfinite(m):
            return m
        return float(m + np.log(np.sum(np.exp(log_values - m))))
