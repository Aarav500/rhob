"""External baseline detectors drawn from the broader change-point / anomaly-detection /
representation-learning / Bayesian-inference literature (not designed for RHOB).

These exist so RHOB compares detection *approaches* fairly, rather than only showcasing
in-house L0-L3 baselines. Each wraps a well-known, citable method:

- ``page_hinkley``      : classical sequential change-point detection (Page, 1954; Hinkley, 1971)
- ``isolation_forest``  : unsupervised anomaly detection (Liu et al., 2008)
- ``ar_residual``       : autoregressive sequence-model residual (classical time-series baseline)
- ``pca_reconstruction``: representation learning via linear reconstruction error
- ``bocpd``             : Bayesian Online Changepoint Detection (Adams & MacKay, 2007)
"""

from rhob.detectors.external_baselines.ar_residual import ARResidualDetector
from rhob.detectors.external_baselines.bocpd import BOCPDDetector
from rhob.detectors.external_baselines.isolation_forest import IsolationForestDetector
from rhob.detectors.external_baselines.page_hinkley import PageHinkleyDetector
from rhob.detectors.external_baselines.pca_reconstruction import PCAReconstructionDetector

__all__ = [
    "PageHinkleyDetector",
    "IsolationForestDetector",
    "ARResidualDetector",
    "PCAReconstructionDetector",
    "BOCPDDetector",
]
