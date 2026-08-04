# Cross-platform check: the same draw on two dependency stacks

Replicate 0 (`layout_seed=0`, `seed_base=0`) was run twice: once on Windows against the
pinned `requirements-lock.txt`, once on Amazon Linux where that lock does not install
(it was generated on win32) so pip resolved the declared version *ranges* instead.
Same code, same seeds, same draw -- only the library build differs.

**23 of 30 detectors agree to within rounding (<=0.001). 7 differ, maximum |difference| 0.0140.**

This is small but not zero, and it means RHOB's published AUROCs are reproducible to
about +-0.015 across dependency stacks, not bit-identical. For comparison, the
draw-to-draw standard deviation of a single detector reaches 0.071 and its range across
20 draws reaches 0.24 -- so platform variation is roughly an order of magnitude smaller
than the sampling variation the replication study exists to quantify, and does not
affect any conclusion drawn from the intervals. It does mean a reader reproducing a
single number should expect the third decimal to move.

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
