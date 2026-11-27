# RHOB Threat Model and External Validity

This document states what RHOB measures, what it does not measure, and which
conclusions its construction can and cannot support. It exists because the
benchmark's central number — an AUROC separating a "hacking" variant from a
"legitimate" one — is easy to read as a claim about detecting reward hacking in
deployed systems, and it is not that claim.

Read this before citing an RHOB number.

---

## 1. What RHOB is

RHOB is a **construction method and a certification harness**, and secondarily a
leaderboard.

The method is the **matched pair**. A family mints two variants of the same task:

- a *hacking* variant, whose behavior raises a proxy reward while a hidden true
  reward falls, and
- a *legitimate* variant, whose behavior raises the same proxy reward while the
  true reward also rises,

calibrated so that the **proxy reward is statistically indistinguishable between
them** while the **true reward diverges**. The true reward is oracle-only: it is
exposed to L3 detectors and to the admission gate, and is withheld from every
detector at L0/L1/L2 by `rhob.v3.access.restrict`.

The certification harness is the **admission gate**
(`src/rhob/v3/admission_gate.py`), which tests six properties per (family,
difficulty) cell and publishes the whole grid — passes, failures *and* cells where a
property could not be measured at all — to `admission/ADMISSION_LEDGER.md`. That third
category is not a formality; see §4.

The point of the pairing is to make a specific question answerable in isolation:
*if the proxy reward genuinely cannot distinguish the two cases, what other
observable can?* RHOB answers that by construction rather than by hoping a
natural dataset happens to contain the contrast.

## 2. What RHOB is NOT

**RHOB is not a measurement of how often reward hacking occurs.** Base rates in
RHOB are 50% by construction: every family emits equal numbers of hacking and
legitimate runs. Nothing in the benchmark speaks to prevalence, and an AUROC
carries no information about performance at a realistic base rate. A detector at
AUROC 0.98 here may still be useless at a 1-in-10,000 base rate.

**RHOB is not an online or real-time detection benchmark.** Every detector is
*post hoc*: `PosthocDetector.classify` receives a completed run and returns one
score. The `detect_onset` method localizes onset *within an already-finished
run*, and inside the benchmark its output is no longer scored (§6). Nothing is
evaluated under a streaming constraint, a latency budget, or a decision cost.

**RHOB is not adversarial.** No family is designed against a specific detector,
and no hacking variant adapts to being watched. A detector that scores well here
has not been shown to be robust to an optimizer that knows the detector exists.
The two sandbagging families (`monitored_sandbagging`, `eval_probe_sandbagging`)
condition on a *scripted* monitoring signal; they model an agent that was written
to behave differently under observation, not one that learned to.

**RHOB is not an LLM benchmark.** The families labelled RLHF (19–23) and
sequence-generation (29–33) are synthetic analogues: feature-vector "responses"
scored by a small fitted linear reward model, and token sequences over a 24-token
vocabulary scored against a fixed hidden Markov grammar. No language model is
trained or evaluated anywhere in RHOB. Results on those families are evidence
about the *mechanism* they instantiate, not about language models.

**RHOB is not a claim that L0 monitoring fails in general.** See §4.

## 3. The scripted-policy limitation (the big one)

**In 26 of 33 families, both variants are hand-written policies.** The "hacking"
behavior is a script the family author wrote to game the proxy; the "legitimate"
behavior is a second script. Neither is discovered by an optimizer. The MuJoCo,
PettingZoo and sequence-generation families all follow this pattern, with a scripted
exploit routine.

This section previously said 28, and used `gridworld_camping` as its example of a
scripted family. It is not one. Its two variants come from `make_hacking_env` and
`make_legitimate_env`, two authored *environments*, but the agent in both is a
`TabularQLearner` that learns from the proxy reward
(`src/rhob/evaluation/eval_rollout.py`), so the camping is found by the learner. Two
things there are still authored: when the exploit tile becomes available (episode 250)
and a 20-episode forced exploration burst after it
(`src/rhob/environments/pair_01/config.py`), so the config places its onset. With both
set to 0 the learner still finds the tile on 40 of 40 seeds, and the onset label
dissolves: SD 76.5 episodes, bimodal (`results/emergent_pilot/README.md`, c6bb6f6).
`continuous_camping` contains a pre-trained DQN (`rhob.agents.dqn.train_camper`), but
that network was trained on a generic navigate-and-hold task and then frozen (its weights
are committed, `rhob/environments/continuous/frozen_camper.py`, since 2026-11-27), and the
switch to camping is triggered at an authored `activation_episode`, so nothing discovers
the hack there. Seven families contain a learner; in six of them (`gridworld_camping`
and the five below) an optimizer finds the exploit.

The five RLHF-RM families (19–23) are the main partial exception and are worth stating
precisely, because "scripted" is not quite right for them: each runs **real
policy-gradient ascent** against a **genuinely fitted** preference reward model,
with a KL penalty to a reference policy (`src/rhob/environments/rlhf_rm/rollout.py`).
The overoptimization dynamics there are produced by an optimizer, not typed in.
What is still designed rather than discovered is the *flaw* — sparse preference
coverage, biased labels, a truncated feature set, a slack KL coefficient — and
the fact that one arm was set up to exploit it. So: emergent *exploitation*, a
constructed *vulnerability*, and an assigned *label*.

### What the scripted construction does let you conclude

- **That the information is there.** If an L2 detector separates the variants on a
  cell where the proxy reward is certified indistinguishable, then the behavioral
  trace carries information that the proxy reward does not. The ledger certifies that
  on 42 of the 123 cells (§4). That is a statement about the *observation channels*,
  and it survives the policies being scripted, because it is a statement about what a
  signal contains, not about where the signal came from. On the single-draw board the
  best L2 detector is Trajectory MLP at 0.944, and it is fitted to each cell's labels.
  Earlier versions of this bullet quoted Behavioral Threshold at 0.975; that figure
  depended on the fixed sign convention, and with the sign randomized it reads 0.512.
  Measured on one set of rollouts with the sign fixed and then randomized (10 seeds per
  variant, `docs/figures/sign_randomization_impact.md`), it reads 0.981 and 0.479.
- **That a detector is not merely a change detector.** The matched-proxy
  construction plus the `true_reward_diverges` criterion means a detector cannot
  win by noticing that *something changed* — both variants change.
- **Negative results, strongly.** If a detector fails on a family where the
  discriminating feature was placed there deliberately and is known to be
  present, that failure is informative. RHOB's negative results are its most
  transferable ones.

### What it does NOT let you conclude

- **Nothing about whether real learned agents produce these signatures.** The
  behavioral trace exists because a family author computed it. A trained policy
  that games the same proxy may leave a completely different, subtler, or no
  low-dimensional signature. RHOB cannot tell you which.
- **Nothing about detection difficulty in the wild.** Difficulty in RHOB is a
  *dial* (`difficulty_range`, tiers 0.60–0.98) calibrated to hit a target L2
  separability. It is the author's chosen separability, not an empirical
  distribution of how hard real reward hacking is to see.
- **Nothing about the frequency, diversity, or coverage of real hacking
  mechanisms.** The 33 families cover 9 mechanism labels chosen from the
  literature and from the authors' judgement. There is no argument anywhere in
  this repo that they are representative, exhaustive, or independently sampled.
- **Little about generalization to genuinely novel mechanisms.** The
  cross-family transfer score (RTS) trains on 6 families and tests on 8 held-out
  ones, but all 14 were written by the same authors under the same construction
  recipe, with the same `behav_trace` convention (anti-symmetric, ~0 pre-onset,
  opposite signs post-onset). Transfer across families that share an authored
  convention is a weaker claim than transfer to mechanisms nobody designed.

**In one sentence:** RHOB establishes what is *detectable in principle* from each
observation channel under a controlled proxy match; it does not establish what is
*detectable in practice* from agents that were never told to leave a trace.

## 4. The L0 result is a negative control, not a finding

RHOB's L0 (reward-only) detectors sit at chance: mean AUROC 0.4835 across 13
detectors, best 0.5474 (Reward CUSUM) — over the 27 families whose proxy the gate can
actually read, family-weighted. Not every detector is averaged over all 27: nine are not
applicable on the families too short for their windows, so Reward Peak, Reward
Autocorrelation, Reward Skewness, Reward Trend and Variance Window are averaged over 8
of them and Reward CUSUM, Max Plateau, Reward MLP and Gradient Reversal over 20; the
best figure, 0.5474, is over 20. The six the ledger marks DEGENERATE are held out; with
them in, family-weighted as well, it reads 0.4871. See §4's last bullet. Those figures
come from the single-draw board as re-drawn on 2026-11-27 under the six numerics pins
(`b93c351`, from code at `54cd63c`, CI run 46011019246; as re-scored on 2026-11-26,
`03d978e` from code at `f83a152`, CI run 45747902322, it read 0.4834 and 0.4877, with the
same best), on code in which every scored detector that needs a minimum horizon declares
it. The first such fix was Reward Skewness's
(b796f9c on this branch, first committed as 584ba13, 2026-09-02): Skewness needs 100
episodes and is recorded N/A below that, on 19 of those 27 families. Before the fix it
returned a constant 0.5 there. The 20-draw board has been re-scored with those cells not
applicable (253c7ec, first committed as 9da6b21), and again on 2026-11-26 for the 19
detectors whose new declaration some families fall below (CI runs 45747922473 and
45776113470, keeping every value the draws were scored with). On 2026-11-27 all 20 draws
were drawn again, whole, by the same run as the single draw, at the same commit and under
the same pins, each at its own layout seed, 0 to 19: the single draw on an AMD runner, the
20 on 12 Intel and 8 AMD runners. `scripts/check_redraw.py` gated the set, and replicate 0
equals the single-draw board cell for cell. The 2026-08-04 draws they replace resampled
rollout seeds only, at layout 0 (§5). A first attempt at the re-draw (CI run 45995235365)
was refused because
replicate 0 and the board differed on 16 `continuous_camping` cells: the family's camper
trained to different weights on Intel and AMD runners, and it now loads frozen ones (§3).
On the re-drawn set the maximum over unsupervised L0 detectors, taken in each draw and
averaged over the 20, is 0.570 [0.560, 0.579] (0.568 [0.555, 0.580] on the 2026-08-04
draws as re-scored). That maximum is selection-biased upward: no single L0 detector's
20-draw overall (over all its cells, degenerate families included) reaches 0.508 (Reward
Skewness 0.507 [0.496, 0.519]; on the 2026-08-04 draws Spectral Reward led at 0.506
[0.495, 0.517]; `leaderboard/v5_replicated.json`). The re-draw changed the chance
control on the 20 draws: 12 of the 13 L0 detectors' 95% intervals contain 0.500, and
Reward CUSUM's, 0.486 [0.475, 0.497], lies wholly below it, so on these draws it
separates the variants weakly, in the direction that would not flag hacking. These
per-detector figures are each detector's overall AUROC as `leaderboard/v5_replicated.json`
records it and `ReplicatedLeaderboard.control_check` reads it: cell-weighted over every
cell the detector measured, the six DEGENERATE families' cells included. The 20-draw
suite mean and maxima here, like the single-draw 0.4835 and 0.5474 above, are
family-weighted and hold those families out. On the 2026-08-04 draws all 13 intervals
contained 0.500. The suite mean over L0 detectors, 0.498 [0.492, 0.504], contains 0.500
(on the 2026-08-04 draws it was 0.495 [0.490, 0.500], whose upper end, 0.500015,
contained it by 0.000015); the level's best-detector maximum over all 13, 0.575 [0.565,
0.586], excludes it, by the selection just described.

Until 2026-11-26 Skewness was the only L0 detector that declared its horizon. Eight
others did not, and below their windows they fell back to a constant score, which reads
AUROC 0.5 and which both boards recorded as a measured value: Reward Peak, Reward
Autocorrelation, Reward Trend and Variance Window on all 19 families of 40 or 60
episodes, and Max Plateau, Reward MLP, Gradient Reversal and Reward CUSUM on the 7 of
40. Those were 104 of the 332 per-family values behind the 2026-11-19 board's L0 mean of
0.4876, and 7 of the 27 behind its best, Reward CUSUM's 0.5351. Each now declares
`min_episodes` (`f83a152`; `tests/test_detectors/test_min_episodes.py` holds every
declaration to the detector's code): the first four need 100 episodes, Reward CUSUM 51,
Max Plateau and Reward MLP 50, and Gradient Reversal 60. Both boards have been re-scored
with those cells not applicable and were re-drawn on 2026-11-27 under the same
declarations, so the 0.4835 above rests on the remaining 228 per-family values, every one
a measurement. The admission gate does not use those eight
detectors.

**This is not a discovery that reward-only monitoring fails.** It is a
manipulation check. The families are *built* so that the proxy reward carries no
signal, and the admission gate *checks*, cell by cell, whether they succeeded; it
admits 42 of the 123 cells (below). An L0 detector
scoring 0.5 is what a construction that worked looks like; it is close to a tautology, and
earlier versions of this README presented it as an empirical finding, which was
wrong.

The honest reading is directional:

- **L0 at chance is necessary, not sufficient.** A construction that held puts L0 at
  chance, but L0 at chance does not show that it held; the ledger does that, cell by
  cell, and admits 42 of the 123 cells. On 26 more `proxy_matched` passes but
  `proxy_distribution_matched` fails: on 19 a shape detector's AUROC is outside
  [0.40, 0.60], and on the other 7 every one is inside and only an interval reaches past
  the band. On 16 of the 24 that fail `proxy_matched` the mean L0 AUROC is inside the band
  but its interval is not; in 15 of those 16 the interval excludes 0.5. All of those cells
  are in the L0 mean, and an L0 mean at chance cannot tell them from an admitted cell.
- **L0 above chance ⇒ the construction leaked.** The family is broken, and every
  other number on it is suspect. This has happened: `distributional_shift` once
  let an L0 detector reach 0.89 AUROC through a proxy leak (see
  REPRODUCIBILITY.md).

A check that cannot fail is not a check, so the failures are published. The
admission ledger (`admission/ADMISSION_LEDGER.md`, regenerate with
`python scripts/admission_ledger.py`) records every criterion's outcome *and* its
numbers for every cell in scope, failures included. That is what makes the negative
control falsifiable rather than decorative.

### There is a third outcome, and it does not support the control

A criterion returns `PASS` (measured, and met), `FAIL` (measured, and not met), or
`DEGENERATE` — **not measurable**. The third case exists because the proxy criteria
are equivalence tests on an AUROC, and an AUROC over a *constant* proxy is 0.5 by the
tie convention rather than by matching: the bootstrap interval collapses to a point
and the test certifies against any margin. A gate that cannot distinguish "carefully
matched" from "carries no information" has the same defect — a check nobody can fail —
that the equivalence test was introduced to remove. The gate therefore checks both
that the statistic could have come out differently and that the proxy carries signal
above numerical dust (a scale-free relative SD above 1e-4); either failing makes the
cell degenerate. `true_reward_diverges` already returned DEGENERATE when the design
supplied too few runs to resample. Since 584ba13 it does the same for arms tied by
arithmetic, which it used to report as FAIL, "measured and found not to diverge".

**As of the 2026-11-19 run, which covers all 33 families, of 123 cells: 42 ADMITTED, 30 DEGENERATE, 51 NOT ADMITTED.**
The 30 degenerate cells are every difficulty tier of six families:
`distributional_shift`, `orbit_chirality` and `rlhf_reward_model_overopt`, whose proxy
rewards are constant by construction, and `monitored_sandbagging`,
`shortcut_exploitation` and `physics_exploitation`, whose proxies vary but are the same,
or nearly, in every run, so the L0 statistic orders next to nothing.

The 51 NOT ADMITTED cells are measured failures: 24 fail `proxy_matched`, 40 fail
`proxy_distribution_matched` (14 fail both), and one, `proxy_correlation_gaming` @ 0.60,
fails `behavioral_separated`. 46 of them are in the 23 families the ledger covered for
the first time on 2026-11-19: all 13 MuJoCo cells, 14 of the 15 PettingZoo cells, 9 of
the 15 RLHF cells and 8 of the 15 sequence cells in those families, and the other two
are `continuous_camping` @ 0.90 and `proxy_correlation_gaming` @ 0.60. The ten families
covered before keep every verdict. The ledger's four `continuous_camping` cells were
certified without the numerics pins and before the family loaded frozen camper weights
(`ddc8ab5`, 2026-11-27), each with the camper its runner trained, so their figures, the
@ 0.90 verdict and the onset SDs in §6 among them, depend on which CPU vendor ran them;
they have not been certified with the frozen weights. A `FAIL` on an equivalence test
means matching within the margin was not established, which covers two cases. In 8 of the 24
`proxy_matched` failures the mean L0 AUROC is outside the band [0.40, 0.60]. In the
other 16 the point estimate is inside the band and the 90% interval is not, so
equivalence was not established at this design rather than refuted. That is not the same
as consistent with 0.5: in 15 of the 16 the interval excludes 0.5, and only
`mujoco_sensor_decoupling` @ 0.90 has one that includes it.

For citation purposes this means:

- **The matched-proxy property is established on the 42 ADMITTED cells of 123** — not
  on all 123, and not on the "30 of 35 admitted" an earlier version of this document
  reported before the degeneracy guard existed. A 43rd cell, `proxy_correlation_gaming`
  @ 0.60, passes both proxy criteria and fails only `behavioral_separated`, so it is NOT
  ADMITTED although neither proxy criterion fails there.
- **Degenerate cells are excluded from the L0-at-chance negative control.** On a
  constant proxy, L0 detectors sit at chance for a reason that has nothing to do with
  matching; counting those cells would inflate the control with results that could not
  have come out otherwise.
- **NOT ADMITTED cells are not excluded.** The 51 cells the ledger measured and did not
  admit stay in the L0 mean; on 50 of them the gate measured the proxy and could not
  certify it matched. Only the six DEGENERATE families are held out. Whether the NOT
  ADMITTED cells should stay in is an open question for the author
  ([ROADMAP.md](../ROADMAP.md)).
- **Those families are still valid hacking families.** They separate at L2 and their
  true rewards diverge: `behavioral_separated` and `true_reward_diverges` pass on all
  30 cells. What is unavailable is the certification, not the mechanism.
- **The L0 rung is now reported both ways.** With all **six** degenerate families
  withheld the L0 suite mean is **0.4835 over 27 families**; with them in it is 0.4871
  (both family-weighted). Nine of the 13 detectors are averaged over fewer of the 27
  (five over 8, four over 20: the families shorter than the horizon each declares; see
  this section's opening). With no exclusion and weighted by cell, the same board reads
  0.4872. Computed from the unrounded means, the exclusion is worth 0.0036 (0.4870635 −
  0.4834738 = 0.0035897) and the re-weighting 0.0001 (0.4871538 − 0.4870635 =
  0.0000903); the rounded figures differ by the same amounts. On the board before the
  2026-11-27 re-draw the three figures were 0.4834, 0.4877 and 0.4881 (an exclusion worth
  0.0043 and a re-weighting worth 0.0004), before the 2026-11-26 re-score they were
  0.4876, 0.4898 and 0.4903, and before the
  2026-11-19 regeneration they were 0.4866, 0.4895 and 0.4898. That last 0.4898 is the
  cell-weighted figure, which releases before the exclusion published; the 2026-11-19
  board's family-weighted 0.4898 is a different statistic that rounds to the same
  value. The exclusion applies to L0 and only L0 — degeneracy is a
  property of the proxy reward, the one channel L0 reads, so L1/L2/L3 keep those
  families as ordinary benchmark items. **The ledger, not either mean, is where the
  control is adjudicated.**

**Scope:** since its 2026-11-19 regeneration on CI (`d4b700d`, from code at `a2aa435`,
CI run 44447220658) the ledger covers all 33
families and every scored tier, 123 cells; before that it covered 10. Covered is not
certified-and-passing: 42 of the 123 cells are ADMITTED. Note also that
what runs in CI for a family is a *reduced-power smoke screen* at equivalence margin
±0.256, not certification at ±0.10; 22 of 33 families run even that, and 11 do not —
most of those have hand-written per-family tests instead, which is not the same as the
gate's six criteria. And no test anywhere in `tests/` calls
`AdmissionGate.certify_all_tiers` on a *registered* family: the two tests that exercise
it use synthetic fixtures, so `scripts/admission_ledger.py` is the only caller that
puts a real family through full certification.

## 5. Statistical limits you should not read past

- **The leaderboard is one draw.** Every cell in `leaderboard/v5_leaderboard.json`
  comes from a single unreplicated 5-seeds-vs-5-seeds comparison, at one layout
  seed (`layout_seeds: [0]`). The standard error of a single-cell AUROC near 0.5
  is **0.19** (`sqrt((n+m+1)/(12nm))` at n=m=5, the Mann–Whitney null). That board
  carries no confidence intervals; the 20-draw replication in
  `leaderboard/v5_replicated.json` does, for aggregates. Since its 2026-11-27 re-draw
  (`b93c351`) those 20 draws are at layout seeds 0 to 19 (its `draws` block; the file
  records `replicates.layouts_resampled` as true), so its intervals include layout
  variation, as far as a layout seed varies anything: every family's behavioral-sign
  orientation is drawn from it; of the families' own generators, 12 ignore it, 19 use it
  only as a rollout-seed offset, and only `proxy_correlation_gaming` and
  `rlhf_reward_model_overopt` build a different layout from it. Until then the committed
  draws, the 2026-08-04 set, resampled rollout seeds only, at layout 0. Do not read the
  third decimal of a single-draw cell, and do not treat a gap of 0.02 between two
  detectors as a result unless an interval says so.
- **All 30 detectors share that one draw.** They are scored on the identical
  rolled-out runs, so their cell values are correlated and their differences are
  not independent measurements.
- **The certified distribution is not the scored distribution.** The admission
  gate averages over 12 independent layout seeds; the leaderboard scores layout
  seed 0 only. A family certified as matched *on average across layouts* may
  still be unmatched on the specific layout the leaderboard used. Since 2026-11-27 the
  20-draw board's draws do vary the layout seed, 0 to 19, so its intervals partly cover
  layout variation where the single draw does not, with the qualification in the first
  bullet: a different layout only in `proxy_correlation_gaming` and
  `rlhf_reward_model_overopt`, a different orientation in every family. The gate's 12
  layout seeds are drawn by its own generator and are not those 20, so the certified and
  the scored distributions are still not the same.
- **The "±" in any access-level table is between-detector spread**, not
  measurement uncertainty. It describes how heterogeneous the detector suite at
  that level happens to be, and shrinks or grows as detectors are added.
- **Mean-per-level is a statistic about the suite, not the benchmark.** A level's
  mean is dragged down by however many weak detectors were written for it. Read
  the max alongside it: at L2 the mean is 0.5771 over 7 detectors while the max
  is 0.9440.

## 6. Known structural caveats in the current artifacts

Stated here so they are not discovered mid-citation:

- **L1 is thin.** 25 of 33 families emit no `state_counts`, so an L1 detector has
  nothing to read on them; the board records those 88 cells as `null`. Of the 8 families
  that do emit it, `continuous_camping` runs 40 episodes, below every L1 detector's
  declared `min_episodes` (50 to 101), so its 4 cells are `null` too, with the horizon as
  the reason. Every L1 figure is therefore over 7 families (31 of 123 cells). That is not
  comparable to the denominators used at L0 (27 families, the six DEGENERATE ones held
  out, and 20 or 8 for the nine detectors some of those families are too short for) and
  L2 (33 families; 26 for Feature Magnitude and Reward-Feature Correlation, 14 for
  Feature Consistency). See the README's L1 table.
- **Short-horizon cells are not applicable, not 0.5, since 2026-11-26.** Until then
  eight L0 detectors (Reward Peak, Reward Autocorrelation, Reward Trend and Variance
  Window on the 19 families of 40 or 60 episodes; Max Plateau, Reward MLP, Gradient
  Reversal and Reward CUSUM on the 7 of 40), three L2 detectors (Feature Consistency on
  the 19; Feature Magnitude and Reward-Feature Correlation on the 7) and every L1
  detector on `continuous_camping` returned their fallback 0.5 on families too short for
  their windows, and both boards published those as measured: 104 L0, 33 L2 and 8 L1
  per-family values. Every one of those detectors now declares `min_episodes`
  (`f83a152`), and both boards were re-scored with those cells `null` (§4). A figure
  quoted from either board before that re-score has the constants in it.
- **L3 contains one detector.** "Perfect Feature Oracle" is a relabelled
  subclass of the L2 behavioral baseline that reads only `behav_trace`; it is
  retained as a cross-check and excluded from every access-level aggregate. The
  oracle gap above L2 on the single-draw board is 0.9830 − 0.9440 = **0.0390 AUROC**,
  where the best L2 detector is the label-fitted Trajectory MLP (0.0400 on the board
  before the 2026-11-27 re-draw); before sign
  randomization it was 0.9830 − 0.9750 = 0.0080. Over 20 draws, taking the best
  detector at each level, it is +0.032 [+0.025, +0.039], positive in 19 of 20
  (`leaderboard/v5_replicated.json`; +0.032 [+0.027, +0.037], positive in 20 of 20, on
  the 2026-08-04 draws): small, but not none.
- **Some shipped families have known failing tiers.** Family 30
  (`sequence_format_camping`) ships with a documented, unresolved `proxy_matched`
  failure at some difficulty tiers (CHANGELOG 1.8.0), and is in the leaderboard. The
  ledger confirms it: NOT ADMITTED at 0.90 and 0.80, where the mean L0 AUROC is 0.8023
  [0.7746, 0.8300] and 0.7924 [0.7457, 0.8390] against the band [0.40, 0.60], and
  ADMITTED at 0.70. Family 29 (`sequence_keyword_stuffing`) shipped with one too, but
  its calibration lever resolved to 0.0 at every tier; since 86acb92 (2026-09-19) it
  passes the reduced-power smoke screen at all three scored tiers (its diagnostic
  records ADMITTED at the smoke margin), which is not certification
  (`results/family_diagnostics/sequence_keyword_stuffing_gate.json`). The ledger
  admits it at 0.80 and 0.70 and not at 0.90, where the mean L0 AUROC is 0.4077, 90% CI
  [0.3677, 0.4477]: the point estimate is inside the band and the interval is not.
- **Some shipped families cannot be certified as built.** A proxy an L0 detector cannot
  read makes the matching criteria unmeasurable rather than satisfied. Six registered
  families are in that state — `distributional_shift`, `monitored_sandbagging`,
  `orbit_chirality`, `physics_exploitation`, `rlhf_reward_model_overopt`,
  `shortcut_exploitation` — covering **30 of the 123 (family, difficulty) cells**, all
  recorded DEGENERATE. The ledger now covers every family and finds no others, so at
  the shipped design the set is complete. Fixing this needs a family redesign, not more
  compute.
- **Admission checking is not enforced repository-wide.** 22 of 33 families have a
  reduced-power smoke screen in CI and 11 have none; 10 never reach `AdmissionGate` in
  any test at all; all 33 are in the ledger, which is a separate run, not a test; and
  `certify_all_tiers` is never called on a registered family by any test. A family's
  presence in the leaderboard implies none of these.
- **The onset label is authored, so nothing inside the benchmark scores it.** In 118 of
  the 123 admission-ledger cells, every cell of the 31 families whose onset label is the
  configured schedule, `onset_std` is exactly 0.0 (`admission/admission_ledger.json`),
  because the family writes the onset down. The other five are `continuous_camping` (four
  tiers, SD 0.23–2.05 episodes against a ceiling of 4.0, certified with a runner-trained
  camper, not the frozen one; see §4) and `gridworld_camping` (SD 9.87 against 50). The
  config places their onsets too, but the label is read off what the
  trained agent does after the activation episode. `onset_localizable` passes those five
  as well, so it fails no cell. Two things follow. The leaderboard's internal onset column
  could not fail, and 584ba13 removed it: `CellResult` no longer carries it, and onset
  error is still computed only for `external/hvta`, where the label was not written by
  us. And `rhob.audit` coded `onset_localizable` as degenerate (e9a1ace) when the ledger
  had 50 cells, all with `onset_std` exactly 0.0. The protocol (`docs/SURVEY_PROTOCOL.md`)
  codes a criterion degenerate only when its statistic took one value across the whole
  shipped design, and over the full ledger's 123 cells `onset_std` takes 6 distinct
  values, so since 2026-11-19 the self-coding (`audit/SELF_CODING.md`) records it
  `not_applicable`. The ledger's coverage changed, not the criterion: on the 31 families
  whose onset label is the configured schedule the statistic is still constant and
  passes every cell by construction. The Pair 01 pilot in §3 shows that the authoring is
  what pins it.
- **Five external baseline detectors ship but are on no board.** The classical
  baselines under `src/rhob/detectors/external_baselines/` do not appear in any
  committed leaderboard artifact; they are scored only on HVTA (`external/hvta/`).
  Since 2026-11-27 each declares its `min_episodes` (Page-Hinkley Test 3, Isolation
  Forest 2, AR(p) Residual `order + 1`, PCA Reconstruction `steady_window`, Bayesian
  Online Changepoint Detection 5), held to its code by
  `tests/test_detectors/test_min_episodes.py`. Until that date Bayesian Online
  Changepoint Detection was degenerate: it scored `max_t P(r_t = 0)`, which a constant
  hazard pins at the hazard (0.004) on every run, so its published HVTA AUROCs ranked
  rounding noise. Its rows on HVTA's primary Sudoku draw and on the other three
  channels were re-scored with the fix; its rows on the three replicate Sudoku draws
  were not, and `external/hvta/RESULTS.md` marks them pre-fix.

## 7. If you are using RHOB

Appropriate uses:

- Sanity-checking that a detector reads the channel it claims to read.
- Ruling detectors *out*: a method that cannot separate a deliberately-planted
  signal will not find a subtle one.
- Reusing the **method** — matched-pair construction plus an equivalence-tested
  admission gate — to build evaluation environments with certified, falsifiable
  properties in your own domain. This is the part of RHOB most likely to
  transfer.

Inappropriate uses:

- Citing an RHOB AUROC as an estimate of real-world detection performance.
- Ranking two detectors whose scores differ by less than ~0.2 on the current
  single-draw leaderboard.
- Citing the access-level ladder as evidence that more access monotonically buys
  more detection power. On the single-draw board the best detector does not rise at
  every rung (0.5474, 0.9820, 0.9440, 0.9830: up at L1, down at L2, up at L3), and the
  four are not over one denominator: the L0 figure (Reward CUSUM) is family-weighted
  over the 20 of 27 families long enough for it, the L1 figure is over the 31 cells of
  the 7 families that emit `state_counts` and run long enough for it (0.9840
  family-weighted), and the L2 and L3 figures are over all 123 cells, so neither the
  L1-to-L2 fall nor the L1 maximum's nearness to the L3 oracle (0.9820 against 0.9830,
  on a quarter of the cells) is like for like. The L1 and L2 maxima are also both
  label-fitted detectors (State Divergence, Trajectory MLP), 0.038 apart. Over 20 draws,
  as re-drawn on 2026-11-27 at layout seeds 0 to 19:
  across all detectors L2−L1 is −0.052 [−0.058, −0.046], which separates, with L1 above
  L2 in all 20 draws (under the mean over detectors it is +0.026 [+0.010, +0.044], the
  other way); the L1 maximum, 0.998 [0.996, 1.000], is above the L3 oracle's 0.979
  [0.976, 0.981], over 31 cells of 7 families against 123 of 33; L3−L2 is +0.032
  [+0.025, +0.039]; and without the five label-fitted detectors L1−L0 is −0.032
  [−0.055, −0.009], which separates against L1: the best unsupervised L0 detector is
  above the best unsupervised L1 one, and only 4 of the 20 draws favour L1 (under the
  mean over detectors it is −0.005 [−0.024, +0.014], which does not separate, so this
  step depends on the statistic; `leaderboard/v5_replicated.json`). On the 2026-08-04
  draws, at layout 0 only, those read −0.055 [−0.061, −0.050], +0.012 [+0.001, +0.023],
  an oracle of 0.975 [0.971, 0.978], +0.032 [+0.027, +0.037], and −0.029 [−0.054, −0.005]
  with 5 of 20 draws favouring L1 against −0.002 [−0.022, +0.017]: the same verdicts.
- Citing L0-at-chance as evidence that reward-only monitoring is insufficient in
  deployed systems. It is a check on RHOB's proxy matching, consistent with it having
  worked, and the ledger is what certifies it, on 42 of the 123 cells.
- Citing an admitted-cell count without its degenerate count. "42 of 123 admitted",
  "30 of 123 not measurable" and "51 of 123 measured and not admitted" are one result,
  and quoting the first alone reproduces the error the ledger's tri-state exists to
  prevent.

---

*Corrections to this document are as welcome as corrections to the code. If a
claim here is stronger than the evidence supports, that is a bug — open an issue.*
