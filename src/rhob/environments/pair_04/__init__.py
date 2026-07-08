"""pair_04 continuous pair."""

from rhob.environments.pair_04.config import CONFIG, L2_MIN, PAIR_ID
from rhob.environments.pair_04.env_hacking import generate_hacking_run
from rhob.environments.pair_04.env_legitimate import generate_legitimate_run
from rhob.environments.pair_04.onset_label import label_onset

__all__ = [
    "CONFIG",
    "L2_MIN",
    "PAIR_ID",
    "generate_hacking_run",
    "generate_legitimate_run",
    "label_onset",
]
