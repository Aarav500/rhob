# Cross-platform check: the same draw on two dependency stacks

Replicate 0 (`layout_seed=0`, `seed_base=0`) was run twice: once on Windows against the
pinned `requirements-lock.txt`, once on Amazon Linux where that lock does not install
(it was generated on win32) so pip resolved the declared version *ranges* instead.
Same code, same seeds, same draw -- only the library build differs.

**23 of 30 detectors agree to within rounding (<=0.001). 7 differ, maximum |difference| 0.0140** (Behavioral Threshold and its registered duplicate
Perfect Feature Oracle, which tie at exactly 0.500 -> 0.514).

## How large is that, honestly

The comparison must be **like for like, per detector**. Behavioral Threshold moves 0.0140
across platforms and has a draw-to-draw standard deviation of 0.0686
across the 20 replicates, so for that detector platform variation is about **4.9x**
smaller than sampling variation -- not an order of magnitude.

An earlier version of this file claimed "roughly an order of magnitude". It reached that
figure by setting the 0.0140 from Behavioral Threshold against a 0.0713 standard deviation
belonging to *Occupancy Polarization*, a different detector whose own cross-platform
difference is only 0.0010. Comparing one detector's platform sensitivity to another
detector's sampling noise flatters the result, and the ratio it produces is not about any
single measurement. The number is corrected here; the conclusion it supported is unchanged.

Against the *range* across draws (0.407-0.646 for Behavioral Threshold, width 0.239) the
platform difference is ~17x smaller, but a range is not a standard deviation and the two
should not be swapped to reach a rounder multiplier.

## What this means

RHOB's published AUROCs are reproducible to about **+-0.015 across dependency stacks**, not
bit-identical. That moves no conclusion in the replication -- the rung orderings and the
identity of the best detector at L1, L2 and L3 are all among the 23 invariant detectors --
but a reader reproducing a single number should expect the third decimal to move.

**All 20 committed replicates are the Amazon Linux range-resolved runs.** The ledger, the
committed board and the sign-randomization artifacts are the pinned Windows stack. That is a
third measurement basis in this repository, alongside seed count and draw count.
*(2026-11-19: the admission ledger and the single-draw board have since been regenerated on
CI under the hashed Linux lock, `requirements/ci-admission-linux-py312.txt`, Python 3.12.16
(`d4b700d`, `52656cb`); the sign-randomization artifacts are still the Windows stack.)*
*(2026-11-26: since `f83a152` every detector on the board whose score was a constant below
some run length declares that length, and for 19 of them that length is above some family's run
length. Their rows in all 20 replicates were re-scored on CI under the hashed Linux lock
(run 45747922473), a second platform check of these draws: of the 23,480 cells those 19
detectors had scored, 23,066 (98.2%) came back with the same AUROC. Of the 414 that did
not, 319 are on `sequence_keyword_stuffing`, whose code changed after the draws were made
(`e7bb3f3`, 2026-09-19), 89 on three DEGENERATE families, and 6 elsewhere. Of the 89, 83 are
proxy-reading detectors (Gradient Reversal, Reward-Feature Correlation) on `orbit_chirality`
and `rlhf_reward_model_overopt`, whose proxy is constant to about 1e-16, where a tie breaks
differently across the two stacks; the other 6 are L1 detectors, which read `state_counts`
and never the proxy, so a proxy tie does not explain them (Bimodal Occupancy on
`distributional_shift`, whose proxy is exactly constant, five times; Centroid Drift on
`rlhf_reward_model_overopt` once). The merge
(run 45776113470, `827ae57`, `--keep-old-values`) therefore kept every value the draws
were scored with and applied only the cells below a declared length, which changed from a
fallback 0.500 to not applicable. The table below predates that and the 2026-09-03 re-score of
Reward Skewness (`253c7ec`): for those 19 detectors, State Divergence's 0.927 among them,
and for Reward Skewness (0.508 here, 0.516 since), it is not what `replicate_000.json`
holds now; the other 10 rows are. Both stacks returned the same constant on the cells that changed, so
none of the differences below comes from them.)*

| Detector | Windows (locked) | Linux (ranges) | abs diff |
|---|---|---|---|
| Perfect Feature Oracle | 0.500 | 0.514 | 0.0140 |
| Behavioral Threshold | 0.500 | 0.514 | 0.0140 |
| Reward Variance Ratio | 0.493 | 0.481 | 0.0120 |
| Reward-Feature Correlation | 0.570 | 0.559 | 0.0110 |
| Reward KDE | 0.504 | 0.496 | 0.0080 |
| Reward Threshold | 0.451 | 0.455 | 0.0040 |
| Gradient Reversal | 0.473 | 0.477 | 0.0040 |
| Occupancy Polarization | 0.494 | 0.495 | 0.0010 |
| Angular Momentum | 0.542 | 0.543 | 0.0010 |
| Visitation Entropy Trend | 0.496 | 0.496 | 0.0000 |
| Variance Window | 0.498 | 0.498 | 0.0000 |
| True Reward Oracle | 0.983 | 0.983 | 0.0000 |
| Transition Entropy | 0.529 | 0.529 | 0.0000 |
| Trajectory MLP | 0.939 | 0.939 | 0.0000 |
| State Frequency Anomaly | 0.534 | 0.534 | 0.0000 |
| State Divergence | 0.927 | 0.927 | 0.0000 |
| State Coverage Rate | 0.549 | 0.549 | 0.0000 |
| Spectral Reward | 0.480 | 0.480 | 0.0000 |
| Reward Trend | 0.487 | 0.487 | 0.0000 |
| Reward Skewness | 0.508 | 0.508 | 0.0000 |
| Reward Peak | 0.502 | 0.502 | 0.0000 |
| Reward MLP | 0.484 | 0.484 | 0.0000 |
| Reward CUSUM | 0.518 | 0.518 | 0.0000 |
| Reward Autocorrelation | 0.490 | 0.490 | 0.0000 |
| Max Plateau | 0.490 | 0.490 | 0.0000 |
| Feature Magnitude | 0.559 | 0.559 | 0.0000 |
| Feature Consistency | 0.516 | 0.516 | 0.0000 |
| Centroid Tracker | 0.356 | 0.356 | 0.0000 |
| Centroid Drift | 0.489 | 0.489 | 0.0000 |
| Bimodal Occupancy | 0.542 | 0.542 | 0.0000 |

Regenerate: run `scripts/replicate_leaderboard.py --replicate-id 0` on both platforms
and compare `results/replication/replicate_000.json`.
