"""Detector abstraction and baseline implementations.

This module contains:
- base.py: streaming detector interface (online evaluation during training)
- posthoc.py: post-hoc detector interface (offline evaluation on completed runs)
- l0_* : reward-only detectors
- l1_* : state-visitation detectors
- l2_* : full-trajectory detectors
- l3_* : oracle ceiling detectors (never a practical access level; for bounding results)
"""

from rhob.detectors.base import AbstractDetector, OverheadEstimate
from rhob.detectors.external_baselines import (
    ARResidualDetector,
    BOCPDDetector,
    IsolationForestDetector,
    PageHinkleyDetector,
    PCAReconstructionDetector,
)
from rhob.detectors.l0_reward_autocorr import RewardAutocorrelationDetector
from rhob.detectors.l0_reward_cusum import RewardCUSUMDetector
from rhob.detectors.l0_reward_kde import RewardKDEDetector
from rhob.detectors.l0_reward_mlp import RewardMLPDetector
from rhob.detectors.l0_reward_peak import RewardPeakDetector
from rhob.detectors.l0_reward_skewness import RewardSkewnessDetector
from rhob.detectors.l0_reward_threshold import RewardThresholdDetector
from rhob.detectors.l0_reward_trend import RewardTrendDetector
from rhob.detectors.l0_reward_variance_ratio import RewardVarianceRatioDetector
from rhob.detectors.l0_spectral import SpectralRewardDetector
from rhob.detectors.l0_variance_window import VarianceWindowDetector
from rhob.detectors.l0_max_plateau import MaxPlateauDetector
from rhob.detectors.l0_gradient_reversal import GradientReversalDetector
from rhob.detectors.l1_centroid_drift import CentroidDriftDetector
from rhob.detectors.l1_coverage_rate import StateCoverageRateDetector
from rhob.detectors.l1_entropy_trend import VisitationEntropyTrendDetector
from rhob.detectors.l1_occupancy_polarization import OccupancyPolarizationDetector
from rhob.detectors.l1_state_divergence import StateDivergenceDetector
from rhob.detectors.l1_state_frequency import StateFrequencyAnomalyDetector
from rhob.detectors.l1_bimodal_occupancy import BimodalOccupancyDetector
from rhob.detectors.l1_transition_entropy import TransitionEntropyDetector
from rhob.detectors.l2_angular_momentum import AngularMomentumDetector
from rhob.detectors.l2_behavioral_threshold import BehavioralThresholdDetector
from rhob.detectors.l2_centroid_tracker import CentroidTrackerDetector
from rhob.detectors.l2_ensemble import EnsembleDetector
from rhob.detectors.l2_trajectory_mlp import TrajectoryMLPDetector
from rhob.detectors.l2_feature_magnitude import FeatureMagnitudeDetector
from rhob.detectors.l2_feature_consistency import FeatureConsistencyDetector
from rhob.detectors.l2_reward_feature_correlation import RewardFeatureCorrelationDetector
from rhob.detectors.l3_perfect_feature import PerfectFeatureOracleDetector
from rhob.detectors.l3_true_reward_oracle import TrueRewardOracleDetector
from rhob.detectors.posthoc import PosthocDetector, RunData

__all__ = [
    "AbstractDetector",
    "OverheadEstimate",
    "PosthocDetector",
    "RunData",
    # L0 (13 total)
    "RewardThresholdDetector",
    "RewardCUSUMDetector",
    "RewardMLPDetector",
    "RewardVarianceRatioDetector",
    "RewardKDEDetector",
    "SpectralRewardDetector",
    "RewardPeakDetector",
    "RewardAutocorrelationDetector",
    "RewardSkewnessDetector",
    "RewardTrendDetector",
    "VarianceWindowDetector",
    "MaxPlateauDetector",
    "GradientReversalDetector",
    # L1 (8 total)
    "StateDivergenceDetector",
    "VisitationEntropyTrendDetector",
    "StateCoverageRateDetector",
    "StateFrequencyAnomalyDetector",
    "CentroidDriftDetector",
    "OccupancyPolarizationDetector",
    "BimodalOccupancyDetector",
    "TransitionEntropyDetector",
    # L2 (8 total)
    "BehavioralThresholdDetector",
    "TrajectoryMLPDetector",
    "EnsembleDetector",
    "AngularMomentumDetector",
    "CentroidTrackerDetector",
    "FeatureMagnitudeDetector",
    "FeatureConsistencyDetector",
    "RewardFeatureCorrelationDetector",
    # L3 (oracle ceiling)
    "TrueRewardOracleDetector",
    "PerfectFeatureOracleDetector",
    # External baselines (classical/non-RHOB-specific methods, for fair comparison)
    "PageHinkleyDetector",
    "IsolationForestDetector",
    "ARResidualDetector",
    "PCAReconstructionDetector",
    "BOCPDDetector",
]
