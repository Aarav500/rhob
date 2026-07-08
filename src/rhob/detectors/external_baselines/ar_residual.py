"""Autoregressive residual detector -- classical sequence-model baseline (L2)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from rhob.detectors.posthoc import PosthocDetector, RunData


class ARResidualDetector(PosthocDetector):
    """Fits a linear AR(p) model of the legitimate variant's behavioral trace,
    then scores runs by their forecasting-residual magnitude.

    This is the classical "sequence model" baseline for change/anomaly
    detection: model the dynamics of the normal process, flag large
    prediction residuals. The AR model is fit by ordinary least squares on
    ``runs_b`` (legitimate) only; ``runs_a`` (hacking) is never used for
    fitting -- only for evaluation, consistent with the other baselines here.
    """

    def __init__(self, order: int = 5, window: int = 100):
        self.order = order
        self.window = window
        self.coeffs: Optional[np.ndarray] = None
        self.residual_scale: float = 1.0
        self.is_trained = False

    @property
    def access_level(self) -> str:
        return "L2"

    @property
    def name(self) -> str:
        return "AR(p) Residual"

    def fit(self, runs_a: list[RunData], runs_b: list[RunData]) -> None:
        X_rows, y_vals = [], []
        for run in runs_b:
            trace = run.behav_trace
            if trace is None or len(trace) <= self.order:
                continue
            for t in range(self.order, len(trace)):
                X_rows.append(trace[t - self.order : t])
                y_vals.append(trace[t])
        if not X_rows:
            return
        X = np.array(X_rows)
        y = np.array(y_vals)
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        self.coeffs = coeffs
        preds = X_aug @ coeffs
        residuals = y - preds
        self.residual_scale = float(np.std(residuals) + 1e-6)
        self.is_trained = True

    def classify(self, run: RunData) -> float:
        residuals = self._residuals(run)
        if residuals is None or len(residuals) == 0:
            return 0.5
        w = self.window
        late = residuals[-w:] if len(residuals) >= w else residuals
        z = float(np.mean(np.abs(late)) / self.residual_scale)
        return float(1.0 / (1.0 + np.exp(-(z - 2.0))))

    def detect_onset(self, run: RunData) -> int:
        residuals = self._residuals(run)
        if residuals is None or len(residuals) < 10:
            return -1
        z = np.abs(residuals) / self.residual_scale
        threshold = 3.0
        run_len = 0
        for t, val in enumerate(z):
            if val > threshold:
                run_len += 1
                if run_len >= 3:
                    return t - 2 + self.order
            else:
                run_len = 0
        return -1

    def _residuals(self, run: RunData) -> Optional[np.ndarray]:
        if not self.is_trained or self.coeffs is None:
            return None
        trace = run.behav_trace
        if trace is None or len(trace) <= self.order:
            return None
        rows = [trace[t - self.order : t] for t in range(self.order, len(trace))]
        X = np.array(rows)
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        preds = X_aug @ self.coeffs
        y = trace[self.order :]
        return y - preds
