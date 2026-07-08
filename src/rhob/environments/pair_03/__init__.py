"""Pair 03 -- medium-difficulty continuous pair (d=0.5, sigma_a=0.5)."""

from rhob.environments.pair_03.config import CONFIG, L2_MIN, PAIR_ID
from rhob.environments.pair_03.env_hacking import generate_hacking_run
from rhob.environments.pair_03.env_legitimate import generate_legitimate_run
from rhob.environments.pair_03.onset_label import label_onset

__all__ = [
    "CONFIG",
    "L2_MIN",
    "PAIR_ID",
    "generate_hacking_run",
    "generate_legitimate_run",
    "label_onset",
]
