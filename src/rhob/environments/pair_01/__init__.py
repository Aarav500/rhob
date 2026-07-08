"""CR1 environment Pair 01 -- the first matched-difficulty pair.

Two transpose-isomorphic gridworld variants with a temporal onset:

* :func:`make_hacking_env` -- Environment A: the agent switches to a misaligned
  exploit tile (true reward collapses).
* :func:`make_legitimate_env` -- Environment B: the agent switches to an aligned
  improvement tile (true reward rises).

By construction the proxy-reward dynamics match (L0 at chance) while the
behavioural signature separates (L2 high). Admission is verified by
:func:`rhob.evaluation.admission_gate.compute_admission`.
"""

from rhob.environments.pair_01.config import DEFAULT, Pair01Config
from rhob.environments.pair_01.env_hacking import make_hacking_env
from rhob.environments.pair_01.env_legitimate import make_legitimate_env
from rhob.environments.pair_01.onset_label import label_onset
from rhob.environments.pair_01.rollout import RunLog, epsilon_schedule, generate_run

__all__ = [
    "Pair01Config",
    "DEFAULT",
    "make_hacking_env",
    "make_legitimate_env",
    "label_onset",
    "RunLog",
    "generate_run",
    "epsilon_schedule",
]
