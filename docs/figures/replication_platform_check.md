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

*(2026-11-27: the sampling variation above is variation over rollout seeds at one layout.
Until `827aec9`, `FamilyRegistry.generate_suite` dropped its `layout_seed`, so all 20
draws, made on 2026-08-04, were built and oriented at layout 0, whatever layout seed they
record. Replicate 0 is the layout-0 draw by design, so the comparison below is
unaffected.)* *(2026-11-27, later: those 20 draws have been replaced by a re-draw in
which each draw runs at its own layout and behavioural orientation (CI run 46011019246,
`b93c351`). The 0.0686, 4.9x, 0.407-0.646 and ~17x above are the 2026-08-04 draws'
figures. Over the re-draw Behavioral Threshold's draw-to-draw standard deviation is 0.0923
and its range 0.354-0.670; the two-stack comparison below was not repeated against it.)*

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
*(2026-11-27: all 20 committed replicates are now the CI re-draw (run 46011019246 at
`54cd63c`, `b93c351`), drawn under the hashed Linux lock and the six numerics pins on 12
Intel and 8 AMD runners, with the board drawn in the same run on an AMD runner, so the
board and the replicates now share one basis; the ledger is on the same lock without the
pins. `scripts/check_redraw.py` requires replicate 0 to equal the board cell for cell, so
a re-draw is itself a platform check of one draw on two runners. This one passed it: an
Intel runner's replicate 0 and an AMD runner's board agree on every cell, all 30
detectors on all 33 families. The first re-draw (run 45995235365, at
`d85a60c`) did not: 16 `continuous_camping` cells differed, because that family's DQN
camper trained to different weights on Intel and AMD runners under the pins. It loads
frozen weights since `ddc8ab5` (CHANGELOG.md). The table below compares the 2026-08-04
replicate 0 on two stacks. The 10 rows the note above calls current are current for that
file as last written in `827ae57`, not for the re-drawn `replicate_000.json`.)*
*(2026-11-27, later: the sign-randomization artifacts, which the 2026-11-19 note leaves on
the Windows stack, were regenerated on CI too (run 46011028034 at `54cd63c`, `0ff0d42`),
under the hashed Linux lock and the six numerics pins on an AMD runner, so they now share
the board's and the replicates' basis; the measurement they replaced, of 2026-08-03,
recorded Windows, Python 3.13.14 and a dirty tree. Its cross-family transfer is a check of
one design on two vendors' runners: its trial `k` seeds the networks with `k`, as
`scripts/cross_family_transfer.py` does, so in the shipped configuration its 3 trials are
the first 3 of `leaderboard/cross_family_transfer_v5.json` (run 46011034429, on an Intel
runner), and Trajectory MLP's and the ensemble's train and transfer AUROCs agree on all
three, to the three decimals both files record. Reward MLP is scored there only with
randomization off, which does not reach an L0 detector's input (the flip multiplies only
`behav_trace`, which `restrict` withholds below L2): its train AUROCs agree on all three
trials and its transfer AUROC on two, trial 2 reading 0.483 there and 0.473 in the v5
file, a difference neither artifact explains.)*

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
and compare `results/replication/replicate_000.json`. *(2026-11-27: the replicate 0 this
file compares is the 2026-08-04 one as first drawn,
`git show e0ba8df:results/replication/replicate_000.json` (last written, re-scored, in
`827ae57`, where 20 of its rows differ as the 2026-11-26 note says); the committed file is
now the re-draw's.)*
