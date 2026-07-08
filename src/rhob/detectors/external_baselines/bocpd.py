"""Bayesian Online Changepoint Detection (Adams & MacKay, 2007) -- Bayesian baseline (L0)."""

from __future__ import annotations

import numpy as np

from rhob.detectors.posthoc import PosthocDetector, RunData


class BOCPDDetector(PosthocDetector):
    """Classical Bayesian Online Changepoint Detection on the proxy-reward stream.

    Maintains a posterior over "run length since last changepoint" using a
    Normal-Gamma conjugate model for the reward-generating process (Adams &
    MacKay, 2007, arXiv:0710.3742) and a constant hazard rate. This is a
    general-purpose Bayesian time-series changepoint method, not designed for
    RHOB or matched-proxy hacking specifically -- included for a fair,
    citable Bayesian-methods comparison point.

    Score is the posterior probability of a changepoint having occurred by
    the end of the run (1 - P(run length == full length)). Onset is the
    first step where the changepoint probability (run length resets to ~0)
    exceeds 0.5.
    """

    def __init__(self, hazard: float = 1.0 / 250.0):
        self.hazard = hazard

    @property
    def access_level(self) -> str:
        return "L0"

    @property
    def name(self) -> str:
        return "Bayesian Online Changepoint Detection"

    def classify(self, run: RunData) -> float:
        cp_probs = self._changepoint_probabilities(run.proxy_rewards)
        if cp_probs is None or len(cp_probs) == 0:
            return 0.5
        return float(np.max(cp_probs))

    def detect_onset(self, run: RunData) -> int:
        cp_probs = self._changepoint_probabilities(run.proxy_rewards)
        if cp_probs is None:
            return -1
        crossed = np.where(cp_probs > 0.5)[0]
        return int(crossed[0]) if len(crossed) else -1

    def _changepoint_probabilities(self, rewards: np.ndarray) -> np.ndarray | None:
        """Standard BOCPD recursion with a Normal-Inverse-Gamma predictive."""
        if rewards is None or len(rewards) < 5:
            return None
        x = rewards.astype(np.float64)
        n = len(x)

        # Normal-Inverse-Gamma prior hyperparameters (weakly informative).
        mu0, kappa0, alpha0, beta0 = float(x[0]), 1.0, 1.0, 1.0

        mu = np.array([mu0])
        kappa = np.array([kappa0])
        alpha = np.array([alpha0])
        beta = np.array([beta0])
        log_run_length_probs = np.array([0.0])  # log P(run length = 0) at t=0

        cp_probs = np.zeros(n)
        for t in range(n):
            xt = x[t]
            # Predictive: Student-t, approximated via Gaussian with scale for stability.
            pred_var = beta * (kappa + 1) / (alpha * kappa)
            pred_var = np.maximum(pred_var, 1e-8)
            log_pred = -0.5 * np.log(2 * np.pi * pred_var) - 0.5 * (xt - mu) ** 2 / pred_var

            log_growth = log_run_length_probs + log_pred + np.log(1 - self.hazard)
            log_cp = self._logsumexp(log_run_length_probs + log_pred + np.log(self.hazard))

            new_log_probs = np.concatenate([[log_cp], log_growth])
            new_log_probs -= self._logsumexp(new_log_probs)

            cp_probs[t] = float(np.exp(new_log_probs[0])) if t > 0 else 0.0

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
            if len(mu) > 500:
                keep = np.argsort(log_run_length_probs)[-500:]
                keep.sort()
                mu, kappa, alpha, beta = mu[keep], kappa[keep], alpha[keep], beta[keep]
                log_run_length_probs = log_run_length_probs[keep]
                log_run_length_probs -= self._logsumexp(log_run_length_probs)

        return cp_probs

    @staticmethod
    def _logsumexp(log_values: np.ndarray) -> float:
        m = np.max(log_values)
        if not np.isfinite(m):
            return m
        return float(m + np.log(np.sum(np.exp(log_values - m))))
