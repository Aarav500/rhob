| Access level | Detectors | Mean AUROC | SD across detectors | Best AUROC | Best detector |
|---|---|---|---|---|---|
| L0 | 13 | 0.4876 | 0.0258 | 0.5351 | Reward CUSUM |
| L1 | 8 | 0.5701 | 0.1366 | 0.9270 | State Divergence |
| L2 | 7 | 0.5716 | 0.1647 | 0.9430 | Trajectory MLP |
| L3 | 1 | 0.9830 | 0.0000 | 0.9830 | True Reward Oracle |

SD is the spread across the detectors at a level, not a confidence interval: the detectors share one evaluation draw, so these are not independent replicates.

L0 is computed over 27 families (Reward Skewness over 8 of them: it is not applicable on the other 19). 6 are held out because the admission ledger marks their proxy DEGENERATE -- constant, nearly constant, or identical in every run, so an L0 detector's AUROC on them is the tie convention (0.5) or rounding error, not evidence of matching: **distributional_shift**, **monitored_sandbagging**, **orbit_chirality**, **physics_exploitation**, **rlhf_reward_model_overopt**, **shortcut_exploitation**. Putting them back would read 0.4898 instead of 0.4876. They are held out of L0 only; the other levels do not read the proxy reward and score these families like any other. Cells the ledger measures NOT ADMITTED are not held out and stay in the mean; whether they should is an open question (ROADMAP.md).

Eight L0 detectors declare no minimum horizon and return a constant score on a run shorter than their windows (Reward Autocorrelation, Reward Peak, Reward Trend, Variance Window below 100 episodes; Gradient Reversal below 60 episodes; Max Plateau, Reward CUSUM, Reward MLP below 50 episodes). On a family that short the constant ties on every pair and is recorded as 0.5. That 0.5 is a fallback, not a measurement, and it is counted in the L0 figures above. Only Reward Skewness declares its horizon; it is null below it instead.

Excluded from the L3 aggregate: **Perfect Feature Oracle** is a relabelled duplicate of **Behavioral Threshold** (identical scores on every family) and is reported only as a cross-check, never as an independent measurement at L3.
