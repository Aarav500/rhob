"""RHOB v3: the community-scale benchmark layer.

A clean, post-hoc evaluation API layered over the validated v1 matched-proxy code,
kept in its own namespace so it does not collide with the frozen Milestone-1
top-level exports (``rhob.Detector``/``rhob.evaluate`` refer to the streaming API).

Usage::

    from rhob.v3 import Benchmark, Detector

    class MyDetector(Detector):
        access_level = "L2"
        name = "my_detector"
        def classify(self, run): return float(run.behav_trace[-1])
        def detect_onset(self, run): return len(run.proxy_rewards) // 2

    results = Benchmark.evaluate(MyDetector(), families="all", n_seeds=5)
    results.summary()
"""

from __future__ import annotations

from rhob.detectors.posthoc import PosthocDetector as Detector
from rhob.detectors.posthoc import RunData
from rhob.v3.base_family import BaseFamily
from rhob.v3.base_pair import MatchedPair
from rhob.v3.benchmark import Benchmark, BenchmarkResults, CellResult
from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import DifficultyTier, EnvironmentComplexity, HackingMechanism

# Importing the families package registers every built-in family.
from rhob.v3 import families  # noqa: E402,F401

__all__ = [
    "Benchmark",
    "BenchmarkResults",
    "CellResult",
    "Detector",
    "RunData",
    "BaseFamily",
    "MatchedPair",
    "FamilyRegistry",
    "HackingMechanism",
    "EnvironmentComplexity",
    "DifficultyTier",
]
