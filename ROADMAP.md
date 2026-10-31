# RHOB roadmap

## What changed

This file used to aim RHOB at becoming "the shared, default benchmark for reward-hacking
detection — the way ImageNet became the default benchmark for image classification", with
the RHOB Transfer Score (RTS) as its headline number and a twice-yearly "RHOB Challenge" to
give the field a reason to keep coming back. That plan was withdrawn on 2026-09-02
(`838a8d7`), and its reasons were restated more exactly on 2026-10-31. The audit behind
those reasons became the paper this repository is now the case study for.

- **The headline RTS did not measure transfer.** The top-5 ensemble's 0.994 on held-out
  families fell to 0.508, chance, once each family's behavioural feature stopped carrying
  the label in its sign. That is one run of three trials, and in it the label-fitted
  Trajectory MLP kept 0.706 ([docs/figures/sign_randomization_impact.json](docs/figures/sign_randomization_impact.json)),
  so what survives is a label-fitted detector's transfer, not the headline. Even before
  that, the held-out families were written by one author under one construction recipe, so
  the most RTS could show was transfer within a single authored convention.
- **RHOB cannot be the default measure of detection.** In 27 of its 33 families the hack
  itself is authored, and in all 33 so is the flaw it exploits. A benchmark built that way
  can say what a signal contains. It cannot say how well a detector catches hacking an
  optimizer found, which is the question the ImageNet framing promised to answer.
- **The onset column the acronym refers to could not fail.** Every one of the 50
  admission-ledger cells has `onset_std` exactly 0.0, because those onsets are authored, so
  a constant prediction equal to the label scores a perfect error of 0.000. Two things
  follow. `584ba13` removed the internal onset column from the leaderboard: `CellResult` no
  longer carries it, and it is still computed only for `external/hvta`, where the label was
  not written by us. And `rhob.audit` codes `onset_localizable` as degenerate (`e9a1ace`).

The withdrawn text is in git history rather than here: last at `06df9f3` on the branch
that carried this rewrite, and at `9352529`, the last version both lines of history share,
on main's. A plan whose headline metric is retracted should not sit
beside one that is current.

## The goal now

What the repository is: a 33-family matched-pair benchmark with an admission gate, a
20-draw replicated leaderboard, and the audit record of the checks in it that could not
have failed. The benchmark is the evidence. The paper is the claim.

What it is for: making the failure class findable in other people's benchmarks. The
contribution is the class — checks that cannot return a negative verdict under the design
that ships — and the diagnostics that find it. RHOB is the case study, and the main evidence
that the diagnostics find real defects. The paper counts seven instances in it, three of
them introduced by the fix to an earlier one. Two more were found in the released artifact
after the paper (`584ba13`): Reward Skewness returned a constant 0.5 below its 100-episode
horizon, on 19 families, which padded the L0 negative control, and `true_reward_diverges`
reported FAIL on arms tied by arithmetic, with no tie guard; it now returns DEGENERATE.

## Retired

- The RHOB Transfer Score as a headline metric, and any leaderboard ranked on it.
- The challenge cadence and the workshop shared task built around it.
  [docs/WORKSHOP_PITCH.md](docs/WORKSHOP_PITCH.md) is kept as a withdrawn record.
- The internal onset column, for the reason above.
- Any claim that RHOB detects reward hacking. The replicated result is that the three
  sub-oracle rungs sit within 0.09 of chance for unsupervised detectors: max AUROC L0 0.551
  [0.541, 0.562], L1 0.534 [0.515, 0.554] and L2 0.587 [0.576, 0.600], against 0.975
  [0.971, 0.978] for the L3 oracle, over 20 draws
  ([leaderboard/v5_replicated.json](leaderboard/v5_replicated.json)).
- The old v1.5, v1.6 and v2.0 milestones — RTS for every detector, a multi-agent
  extension, 25 or more families. Each would enlarge a benchmark whose central construction
  is now the thing in question.

## Milestones

- **`cannotfail` 0.1 — done (2026-10-31).** D1 and D2 as a standalone package in
  [`tools/cannotfail`](tools/cannotfail), installable without RHOB and held to RHOB's
  admission gate by [`tests/test_cannotfail_parity.py`](tests/test_cannotfail_parity.py).
- **One convention for every published number — open; the 20-draw L0 re-score done
  (2026-09-03).** A cell a detector could not have scored is N/A, never a 0.5. The L1 row was fixed in the
  2026-08 correction. The L0 Reward Skewness row had the same defect from a different cause
  (its windows need 100 episodes; 19 families run 40 or 60) and was re-scored on all 20
  draws in `9da6b21`: those 19 families are now N/A, leaving 0.504 [0.494, 0.514] over 64
  cells on 14 families. Five of the 14 are still at exactly 0.500, from a near-constant
  proxy rather than the horizon. That is disclosed and not fixed. The single-draw board
  (`leaderboard/v5_leaderboard.json`) has not been regenerated and still carries the padded
  Skewness cells on 19 of its 27 L0 families.
- **Extend the admission ledger to the short-horizon families — open.** The nightly smoke
  screen has been re-measured with Reward Skewness declaring its horizon: 19 short-horizon
  cells admit and 14 fail on a real late-window density leak (`9da6b21`; the 14 are marked
  `xfail(strict=True)` per tier in `tests/test_v3/test_family_*.py`). The certification
  ledger still covers only its 10 families, whose horizons run 100 to 200 episodes.
  Extending it moves the 30-of-50 calibration counts below and the parity test that
  asserts them, in the same change.
- **One maintainer outside the author's network runs it on their own score vectors — by
  2026-12-15.** A run of the diagnostics on a benchmark the author did not build.
- **The paper at a venue.** A submission to TMLR, after the paper's claims are made to
  match the repository's. The arXiv route has produced no endorsement; a journal needs
  none.
- **The survey: the port, then axis A.** First the port: the paper's 77 preconditions as
  rows, reconciled against its Appendix F by `rhob.audit.frame.reconcile`, coded twice for
  claim shape per [docs/SURVEY_PROTOCOL.md](docs/SURVEY_PROTOCOL.md). Authors are contacted
  per [docs/AUTHOR_OUTREACH.md](docs/AUTHOR_OUTREACH.md) before any table naming them is
  published. Then axis A: the enumeration's search patterns select against
  equivalence-test vocabulary, and the pre-registered remediation (revised patterns, the
  three unaudited members, the power re-run) addresses it. It is the one item that changes
  what the paper is evidence *about*.
- **HVTB's onset-timestamp patch — verified, not submitted.** The patch for the 89
  hack-verifiable Terminal Bench tasks is verified in Docker on 89 of 89
  ([`external/hvtb/verify_all_results.json`](external/hvtb/verify_all_results.json),
  `ea898e4`) and has not been submitted upstream (`b7a0395`, 2026-09-21). It records
  wall-clock time, not a step index.
- **After 2026-12-15: test the no-learner objection on someone else's learner.** Wong,
  Engels and Nanda's RL reward-hacking environment is open source, with a run that has a
  loophole and a baseline that does not. The question is whether hacking that an optimizer
  found is invisible to the proxy, as RHOB's construction assumes. The outcomes are fixed
  before any data exists, and one of them is stated now: **if the proxy separates the
  hacking and legitimate runs, that is the headline**, and it means RHOB's matched-proxy
  regime may be an artifact of scripting. The in-repo precedent is the Pair 01 pilot
  ([results/emergent_pilot/](results/emergent_pilot/README.md), `c6bb6f6`). The shipped
  config of Pair 01, the environment behind `gridworld_camping`, authors both when its
  exploit tile becomes available (episode 250) and a 20-episode forced exploration burst,
  so the config places its onset. With both set to 0 the tabular Q-learner still finds the
  tile on 40 of 40 seeds, and the onset label dissolves: mechanistic onset SD 76.5
  episodes, bimodal, and firing on all 40 legitimate twins. That shows authoring is what
  pins onset. It does not test the proxy, which the pilot keeps matched by construction;
  the outside probe asks that separate question.
- **D3 in `cannotfail`** — only once a second benchmark's harness shows what a channel
  digest would need to be told.

## Not planned

Learned families as certified benchmark members, a multi-agent hacking class as a
contribution, or new learned-agent trace collection for RHOB's own families before the
TMLR submission. Each was designed and argued against; the arguments are in the paper's
limitations section. The ledger extension above reruns the scripted families through the
gate and is not that. The learner probe above is scheduled after 2026-12-15, and it is a test of the
construction, not a new benchmark member.

## What RHOB itself is for now

- **A fixture for the diagnostics.** Its admission ledger is the calibration set. Fed the
  ledger's measured resolution and informativeness, `cannotfail` must mark the same 30
  `proxy_matched` cells degenerate as the gate does, and it must call `onset_localizable`'s
  statistic fixed, for the reason under What changed. Recomputing the 30 from score vectors
  would need the runs regenerated, since the ledger stores aggregates.
- **A worked example of a check built so it can fail.** The matched-pair construction and
  the admission gate. Its nightly smoke screen first failed twelve families (2026-08-17,
  recorded in `ba26a4a`). 33 of its 36 DEGENERATE cells turned out to be Reward Skewness
  tied below its horizon (`584ba13`; the docstring of
  [`tests/test_v3/test_gate_honest_verdicts.py`](tests/test_v3/test_gate_honest_verdicts.py)),
  the harness describing itself rather than the families. The re-measurement is under
  Milestones.
- **Not a leaderboard to rank detectors on**, for the reasons under What changed.

## Discoverability: the Inspect Evals Register

HVTB — the Terminal Bench 2.1 hack-rate eval, whose code has been at
[`inspect_eval/`](inspect_eval/) on main since `b083e55` (2026-09-19) — is listed in the
[Inspect Evals Register](https://ukgovernmentbeis.github.io/inspect_evals/evals/hvtb/),
maintained by Generality Labs with contributions from the UK AI Security Institute, Arcadia
Impact and the Vector Institute. The register lists this repository's `inspect-eval`
branch, whose root is that directory, pinned at `a2c7341`.

That is a distribution channel, not a milestone. What is listed measures agents' hack
rates; it is neither the diagnostics nor RHOB's detector suite, the register publishes no
numbers for it, and its checks are structural — the Contributor Guide's scoping, validity
and scoring standards are "not mandatory for registry acceptance." It does not count as an
outside user. What it plausibly buys is that someone searching for reward-hacking evals in
Inspect finds this repository without having been told about it, and nothing here
substitutes for asking people directly.

## What would change or accelerate this

- **One outside maintainer running the diagnostics on their own benchmark.** Every run so
  far is the author's own, on a benchmark the author built. The first run by someone who
  can contest the result is qualitatively different. This replaces, on purpose, the bar
  earlier versions of this file set, an external detector submission to RHOB.
- **A co-author or named collaborator from an established lab**, which changes the
  project's provenance from one researcher's repository to something others have checked.
- **A per-step label for HVTB.** HVTB is an external corpus in which hacking is decidable
  by construction and caused by the agent rather than scheduled by the authors, which RHOB
  is not. Its label is per trial. A step-indexed onset from real agents would let a
  detector's timing be scored against hacking nobody authored; that is future work, and a
  different paper.
