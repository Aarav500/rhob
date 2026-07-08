"""Built-in baseline detectors."""

from rhob.detectors.baselines.cusum import CUSUMDetector
from rhob.detectors.baselines.random_detector import RandomDetector

__all__ = ["CUSUMDetector", "RandomDetector"]
