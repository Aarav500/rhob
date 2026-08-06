| Access level | Detectors | Mean AUROC | SD across detectors | Best AUROC | Best detector |
|---|---|---|---|---|---|
| L0 | 13 | 0.4866 | 0.0250 | 0.5311 | Reward CUSUM |
| L1 | 8 | 0.5200 | 0.0387 | 0.6210 | State Divergence |
| L2 | 7 | 0.7213 | 0.2052 | 0.9750 | Behavioral Threshold |
| L3 | 1 | 0.9830 | 0.0000 | 0.9830 | True Reward Oracle |

SD is the spread across the detectors at a level, not a confidence interval: the detectors share one evaluation draw, so these are not independent replicates.

L0 is computed over 27 families. 6 are held out because the admission ledger marks their proxy DEGENERATE -- it is constant, so an L0 detector reads 0.5 on them by construction and not by matching: **distributional_shift**, **monitored_sandbagging**, **orbit_chirality**, **physics_exploitation**, **rlhf_reward_model_overopt**, **shortcut_exploitation**. Putting them back would read 0.4895 instead of 0.4866. They are held out of L0 only; the other levels do not read the proxy reward and score these families like any other.

Excluded from the L3 aggregate: **Perfect Feature Oracle** is a relabelled duplicate of **Behavioral Threshold** (identical scores on every family) and is reported only as a cross-check, never as an independent measurement at L3.
