# Terminal Wrench and HVTB: a feasibility check, and a finding that cuts against us

Not a port. A survey of two external corpora, done before committing to either, plus one
result that weakens a claim this project has been building on.

## The corpora

| | HVTB | Terminal Wrench |
|---|---|---|
| source | `arXiv:2608.22103` (Roth, Bercovich, Efroni) | `arXiv:2604.17596` (Bercovich et al.) |
| release | Harbor Hub, `hv-terminal-bench-2-1` | `github.com/few-sh/terminal-wrench`, public, 1.2 GB |
| environments | 89 | 331 (957 task-model entries) |
| trajectories | 2,225 | 6,289: 3,632 hack, 2,657 non-hack, 2,352 baseline |
| models | frontier (unspecified on the page) | Claude Opus 4.6, Gemini 3.1 Pro, GPT-5.4 |
| hack classes | 2 (hidden solution, read tests) | 11 |

Both are real agents choosing to hack, which is the gap every HVTA number carries: our
policies are scripted and hack on a schedule we chose.

## Feasibility: L0 is blocked, L2 is not

Terminal Wrench's layout is explicit. Per trial:

    trial/agent/trajectory.json
    trial/agent/episode-N/{prompt.txt,response.txt}   <- step structure exists
    trial/verifier/reward.txt                         <- ONE reward, at trial level

Step structure is present and rich. A per-step numeric score is not: `reward.txt` sits
under `verifier/` for the whole trial, and Terminal Bench runs its tests after the agent
submits. HVTB is the same shape -- inotify watchers set flags the verifier reads when the
agent finishes.

So RHOB's L0 reward-only detectors, which need a per-step proxy series, cannot run on
either corpus as released. L2 behavioural detection can: `episode-N/` gives a
step-indexed action sequence. Terminal Wrench also ships `stripped_trajectories/` with
agent reasoning blanked and actions intact, which is an access-level split someone else
has already run.

### Onset: resolved for Terminal Wrench, and the answer is no

Checked directly against `tasks/regex-chess/gemini-3.1-pro/hack_trajectories/v5`:

    trial/agent/episode-0 .. episode-8     9 episodes, real per-step structure
    trial/agent/trajectory.json
    trial/verifier/reward.txt              2 BYTES -- one scalar for the whole trial
    trial/result.json                      started_at / finished_at, trial-level only
    metadata.json                          classification, reward, episode_count

`metadata.json` labels the TRAJECTORY, not the step: `classification`,
`judged_serious_exploit`, `brief_exploit_summary`, `episode_count: 9`. Nothing anywhere
records which episode the exploit happened in. So onset error cannot be measured on this
corpus at all, and L0 cannot run because the only reward is 2 bytes at trial level.

### And the labels are LLM-judged, which matters more

`"classification_source": "stored_judgment"`, with `judged_serious_exploit`,
`judged_legitimate_solve` and `judged_rewarding_nonhack` alongside it. The README says
these come from scoring with `gpt-5.4` against a judge prompt.

Scoring RHOB's detectors against those labels would mean measuring a detector against an
LLM judge's opinion. That is the exact unreliability the hack-verifiable methodology was
built to escape, and it would reintroduce a circularity as bad as the one the HVTA port
was meant to break -- just with someone else's judge instead of our own oracle.

Terminal Wrench is therefore excellent for what it was built for, monitorability studies,
and wrong for ours. It gives real agents and step structure, and neither of the two
things RHOB needs: a per-step proxy and a structurally-decided onset.

### Which leaves HVTB's watcher as the only candidate

HVTB detects structurally, by inotify, with no judge. Its author has since added a
watcher on test-file edits, which fires at a moment rather than setting an end-of-run
flag. Whether that moment survives into the released traces as a step index or a
timestamp is the open question, and it is now the ONLY route to an external corpus that
labels onset rather than the episode. There is no fallback if the answer is no.

## The finding: we may have invented the augment class

The HVTA work concluded that RHOB's `true_reward_diverges` criterion recognises hacking
that SUBSTITUTES for competence and is blind to hacking that AUGMENTS a competent agent.
Two of HVTA's four channels are the augment kind, so the gate would reject them.

The prediction was that Terminal Wrench's `metric-spoofing` and `mutable-input-tampering`
categories would be augment-type. Checked against the worked examples: they are not.

* `mongodb-sales-aggregation-engine` (metric-spoofing) -- the agent built a pytest plugin
  fabricating passing TestReport objects, and never wrote any MongoDB code. Competence
  zero, proxy perfect.
* `logistic-regression-divergence` (metric-spoofing) -- patched `predict()` to replay
  captured training labels and faked convergence. The model never trains.
* `train-fraud-detection-model` (constraint-loophole) -- injected coefficients and an
  intercept so the model predicts "not fraud" always, riding 97% class imbalance.

All substitutes. Working through the other categories by their own descriptions, every
one avoids doing the real work. The augment class appears nowhere in 331 environments and
6,289 trajectories, nor in HVTB's 89.

**It appears only in the two HVTA channels we built ourselves.** Two readings, and the
second is the uncomfortable one:

1. The class is real but structural to multi-agent settings -- it needs a counterparty to
   corrupt or outguess, and single-agent coding benchmarks cannot produce one.
2. The class is an artifact of our own construction. We wrote those two scripted
   opponents, so we may have manufactured a category rather than found one.

Reading 1 is defensible and is what was said to the HVTA authors. But nothing measured so
far distinguishes it from reading 2, and the difference matters: "the gate is blind to a
real class of hacking" is a much stronger claim than "the gate is blind to a class we
constructed." Until an external multi-agent corpus shows an augment-type hack, the paper
should make the weaker claim.

## What would settle it

An external corpus with two or more interacting agents and decidable hack detection.
Neither HVTB nor Terminal Wrench is one. Absent that, the honest framing is that our gate
recognises one class, that a second class is constructible, and that whether the second
occurs in the wild is open.
