| Access level | Detectors | Mean AUROC | SD across detectors | Best AUROC | Best detector |
|---|---|---|---|---|---|
| L0 | 13 | 0.4835 | 0.0328 | 0.5474 | Reward CUSUM |
| L1 | 8 | 0.5789 | 0.1543 | 0.9820 | State Divergence |
| L2 | 7 | 0.5771 | 0.1644 | 0.9440 | Trajectory MLP |
| L3 | 1 | 0.9830 | 0.0000 | 0.9830 | True Reward Oracle |

SD is the spread across the detectors at a level, not a confidence interval: the detectors share one evaluation draw, so these are not independent replicates.

L0 is computed over 27 families (Gradient Reversal, Max Plateau, Reward CUSUM and Reward MLP over 20 of them; Reward Autocorrelation, Reward Peak, Reward Skewness, Reward Trend and Variance Window over 8 of them: the other families run fewer episodes than the detector declares it needs, and it is not applicable there). 6 are held out because the admission ledger marks their proxy DEGENERATE -- constant, nearly constant, or identical in every run, so an L0 detector's AUROC on them is the tie convention (0.5) or rounding error, not evidence of matching: **distributional_shift**, **monitored_sandbagging**, **orbit_chirality**, **physics_exploitation**, **rlhf_reward_model_overopt**, **shortcut_exploitation**. Putting them back would read 0.4871 instead of 0.4835. They are held out of L0 only; the other levels do not read the proxy reward and score these families like any other. Cells the ledger measures NOT ADMITTED are not held out and stay in the mean; whether they should is an open question (ROADMAP.md).

Excluded from the L3 aggregate: **Perfect Feature Oracle** is a relabelled duplicate of **Behavioral Threshold** (identical scores on every family) and is reported only as a cross-check, never as an independent measurement at L3.
