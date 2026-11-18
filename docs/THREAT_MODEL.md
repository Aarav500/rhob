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
that network was trained on a generic navigate-and-hold task and then frozen, and the
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

- **That the information is there.** If an L2 detector separates the variants at
  AUROC 0.975 while the proxy reward is certified indistinguishable, then the
  behavioral trace carries information that the proxy reward does not. That is a
  statement about the *observation channels*, and it survives the policies being
  scripted, because it is a statement about what a signal contains, not about
  where the signal came from.
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

RHOB's L0 (reward-only) detectors sit at chance: mean AUROC 0.4866 across 13
detectors, best 0.5311 — over the 27 families whose proxy the gate can actually read.
The six the ledger marks DEGENERATE are held out; with them in it reads 0.4895. See
§4's last bullet. Those figures come from the single-draw board, which predates the
Reward Skewness fix (584ba13, 2026-09-02) and has not been regenerated: Skewness needs
100 episodes and returned a constant 0.5 below that, which on that board is 19 of
those 27 families. The 20-draw board has been re-scored with those cells not applicable
(9da6b21), and there the maximum over unsupervised L0 detectors, taken in each draw and averaged over the
20, is 0.551 [0.541, 0.562]. That maximum is selection-biased upward: no single L0
detector's 20-draw mean reaches 0.507 (Spectral Reward 0.506 [0.495, 0.517];
`leaderboard/v5_replicated.json`).

**This is not a discovery that reward-only monitoring fails.** It is a
manipulation check. The families are *built* so that the proxy reward carries no
signal, and the admission gate *certifies* that they succeeded. An L0 detector
scoring 0.5 confirms the construction worked; it is close to a tautology, and
earlier versions of this README presented it as an empirical finding, which was
wrong.

The honest reading is directional:

- **L0 at chance ⇒ the construction held.** Good; the rest of the benchmark is
  interpretable.
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

**As of the current run, of 50 cells in scope: 15 ADMITTED, 30 DEGENERATE, 5 NOT ADMITTED.**
The 30 degenerate cells are every difficulty tier of six families:
`distributional_shift`, `orbit_chirality` and `rlhf_reward_model_overopt`, whose proxy
rewards are constant by construction, and `monitored_sandbagging`,
`shortcut_exploitation` and `physics_exploitation`, whose proxies vary but are the same
in every run, so the L0 statistic orders next to nothing.

For citation purposes this means:

- **The matched-proxy property is established on 15 of the 50 in-scope cells** — not on
  all 50, and not on the "30 of 35 admitted" an earlier version of this document
  reported before the degeneracy guard existed.
- **Degenerate cells are excluded from the L0-at-chance negative control.** On a
  constant proxy, L0 detectors sit at chance for a reason that has nothing to do with
  matching; counting those cells would inflate the control with results that could not
  have come out otherwise.
- **Those families are still valid hacking families.** They separate at L2 and their
  true rewards diverge. What is unavailable is the certification, not the mechanism.
- **The L0 rung is now reported both ways.** With all **six** degenerate families
  withheld the L0 suite mean is **0.4866 over 27 families**; with them in it is 0.4895
  (both family-weighted). Releases before this one published 0.4898, the same statistic
  with no exclusion and weighted by cell — 0.0029 of the difference is the exclusion,
  0.0003 the re-weighting. The exclusion applies to L0 and only L0 — degeneracy is a
  property of the proxy reward, the one channel L0 reads, so L1/L2/L3 keep those
  families as ordinary benchmark items. **The ledger, not either mean, is where the
  control is adjudicated.**

**Scope limit:** the ledger currently covers 10 of 33 families. The remaining 23
are **uncertified**, which is not the same as certified-and-passing. Any L0
number on an uncertified family is uninterpretable in the above sense. Note also that
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
  `leaderboard/v5_replicated.json` does, for aggregates. Do not read the third decimal
  of a single-draw cell, and do not treat a gap of 0.02 between two detectors as a
  result unless an interval says so.
- **All 30 detectors share that one draw.** They are scored on the identical
  rolled-out runs, so their cell values are correlated and their differences are
  not independent measurements.
- **The certified distribution is not the scored distribution.** The admission
  gate averages over 12 independent layout seeds; the leaderboard scores layout
  seed 0 only. A family certified as matched *on average across layouts* may
  still be unmatched on the specific layout the leaderboard used.
- **The "±" in any access-level table is between-detector spread**, not
  measurement uncertainty. It describes how heterogeneous the detector suite at
  that level happens to be, and shrinks or grows as detectors are added.
- **Mean-per-level is a statistic about the suite, not the benchmark.** A level's
  mean is dragged down by however many weak detectors were written for it. Read
  the max alongside it: at L2 the mean is 0.7213 over 7 detectors while the max
  is 0.9750.

## 6. Known structural caveats in the current artifacts

Stated here so they are not discovered mid-citation:

- **L1 is thin.** 25 of 33 families emit no `state_counts`, so every L1 detector
  returns its hardcoded 0.5 fallback on them. L1 figures are only meaningful over
  the 8 families that emit the channel (35 of 123 cells) and are not comparable
  to the 33-family denominators used at L0 and L2. See the README's L1 table.
- **L3 contains one detector.** "Perfect Feature Oracle" is a relabelled
  subclass of the L2 behavioral baseline that reads only `behav_trace`; it is
  retained as a cross-check and excluded from every access-level aggregate. The
  oracle gap above L2 on the single-draw board is 0.9830 − 0.9750 = **0.0080 AUROC**.
  Over 20 draws, taking the best detector at each level, it is +0.032 [+0.027, +0.037],
  positive in 20 of 20 (`leaderboard/v5_replicated.json`): small, but not none.
- **Some shipped families have known failing tiers.** Family 30
  (`sequence_format_camping`) ships with a documented, unresolved `proxy_matched`
  failure at some difficulty tiers (CHANGELOG 1.8.0), and is in the leaderboard.
  Family 29 (`sequence_keyword_stuffing`) shipped with one too, but its calibration
  lever resolved to 0.0 at every tier; since 86acb92 (2026-09-19) it passes the
  reduced-power smoke screen at all three scored tiers (its diagnostic records ADMITTED
  at the smoke margin), which is not certification
  (`results/family_diagnostics/sequence_keyword_stuffing_gate.json`).
- **Some shipped families cannot be certified as built.** A proxy an L0 detector cannot
  read makes the matching criteria unmeasurable rather than satisfied. Six registered
  families are in that state — `distributional_shift`, `monitored_sandbagging`,
  `orbit_chirality`, `physics_exploitation`, `rlhf_reward_model_overopt`,
  `shortcut_exploitation` — covering **30 of the 123 (family, difficulty) cells**, all
  of them inside the ledger's scope and all recorded DEGENERATE. The 23 uncertified
  families have not been swept at the certification design, so 30 is a lower bound.
  Fixing this needs a family redesign, not more compute.
- **Admission checking is not enforced repository-wide.** 22 of 33 families have a
  reduced-power smoke screen in CI and 11 have none; 10 never reach `AdmissionGate` in
  any test at all; 10 are in the ledger; and `certify_all_tiers` is never called on a
  registered family by any test. A family's presence in the leaderboard implies none of
  these.
- **The onset label is authored, so nothing inside the benchmark scores it.** Every one
  of the 50 admission-ledger cells has `onset_std` exactly 0.0
  (`admission/admission_ledger.json`), because the family writes the onset down. Two
  things follow. The leaderboard's internal onset column could not fail, and 584ba13
  removed it: `CellResult` no longer carries it, and onset error is still computed only
  for `external/hvta`, where the label was not written by us. And `rhob.audit` codes
  `onset_localizable` as degenerate (e9a1ace), because a statistic that is constant on
  the shipped design passes every cell by construction. The Pair 01 pilot in §3 shows
  that the authoring is what pins it.
- **Five external baseline detectors ship but are unscored.** The classical
  baselines under `src/rhob/detectors/external_baselines/` are implemented but do
  not appear in any committed leaderboard artifact.

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
  more detection power. On the single-draw board it does between L1 and L2 and not
  between L2 and L3. Over 20 draws neither holds: across all detectors L2−L1 is +0.001
  [−0.004, +0.006] and L3−L2 is +0.032 [+0.027, +0.037]; without the five label-fitted
  detectors L1−L0 is −0.017 [−0.039, +0.002] (`leaderboard/v5_replicated.json`).
- Citing L0-at-chance as evidence that reward-only monitoring is insufficient in
  deployed systems. It is evidence that RHOB's proxy matching worked.
- Citing an admitted-cell count without its degenerate count. "15 of 50 admitted"
  and "30 of 50 not measurable" are one result, and quoting the first alone
  reproduces the error the ledger's tri-state exists to prevent.

---

*Corrections to this document are as welcome as corrections to the code. If a
claim here is stronger than the evidence supports, that is a bug — open an issue.*
