"""Registered environment families. Importing this package registers them all."""

from rhob.v3.families import (  # noqa: F401
    continuous_camping,
    distributional_shift,
    eval_probe_sandbagging,
    goal_misgeneralization,
    gridworld_camping,
    monitored_sandbagging,
    novelty_farming,
    orbit_chirality,
    physics_exploitation,
    proxy_correlation_gaming,
    reward_channel_tampering,
    rlhf_reward_model_overopt,
    sensor_calibration_tampering,
    shortcut_exploitation,
)

__all__ = [
    "gridworld_camping",
    "continuous_camping",
    "proxy_correlation_gaming",
    "shortcut_exploitation",
    "novelty_farming",
    "orbit_chirality",
    "goal_misgeneralization",
    "physics_exploitation",
    "distributional_shift",
    "reward_channel_tampering",
    "sensor_calibration_tampering",
    "monitored_sandbagging",
    "eval_probe_sandbagging",
    "rlhf_reward_model_overopt",
]
