# Survey protocol: coding benchmark preconditions for falsifiability

**Status: draft, not yet executed. No external benchmark has been coded under it.**

This document fixes the decision rules *before* any benchmark is coded, so that the
result is a measurement rather than a reading. It exists because the survey the paper
describes currently has no machine-readable form in this repository — no benchmark
list, no coding sheet, no runnable diagnostic — and a survey that names other people's
work and cannot be re-run is the defect this project is about, pointed outward.

The thesis this survey is meant to test is the repository's own:

> A criterion is not a check until it is accompanied by a demonstration that it can
> fail — a power curve, or an adversarial probe that passes it — and a negative
> control is decorative unless the failures are published alongside the passes.

---

## 1. What gets coded

The unit is a **precondition**: a criterion a benchmark states that a task, pair,
instance, or split must satisfy before its results are reported. Not an evaluation
metric, and not a result.

A precondition is in scope if it is stated in the benchmark's paper, README, or code
in a form specific enough to check. Aspirational language in a limitations section is
not a precondition.

One row per (benchmark, criterion). The row records five judgements and two
provenance fields; the schema is `rhob.audit.criterion.CodedCriterion` and the
on-disk form is JSON (`audit/*.json`).

## 2. The judgement that decides everything: claim shape

Every finding in this survey depends on one call, so it is the one with the most
explicit rule and the only one requiring two independent coders.

**A criterion is `equivalence`-shaped if passing it requires evidence that two
quantities are the same, or that one quantity is near a fixed value.** The test for
this is the *direction noise pushes the verdict*:

- If more noise, fewer samples, or a weaker measurement makes the criterion **easier**
  to pass, it is `equivalence`-shaped. Absence of evidence is being read as evidence
  of absence.
- If more noise makes it **harder** to pass, it is `difference`- or
  `threshold`-shaped, and a difference test is the correct tool.

This rule is what keeps the survey honest about difference tests. Four of RHOB's own
six criteria are difference tests and should be: as `README.md` puts it, "ties push
the statistic *toward* the null and therefore toward failing — the safe direction."
A survey that counted those as defects would be wrong about its own author.

Worked calls:

| Criterion | Shape | Why |
|---|---|---|
| "the two variants earn indistinguishable proxy reward" | `equivalence` | Noisier measurement → easier to pass |
| "the true rewards differ (CI excludes 0)" | `difference` | Noisier measurement → harder to pass |
| "mean AUROC ≥ 0.60" | `threshold` | Noisier measurement → harder to pass |
| "the contamination rate is below 1%" | `equivalence` | A weaker detector finds less contamination and passes more easily |
| "no task appears in the training set" | `equivalence` | A weaker search finds fewer matches and passes more easily |

The last two matter: **decontamination and leakage checks are the most common
equivalence-shaped preconditions in ML benchmarking**, and they are the place this
survey is most likely to find something real. A survey that only looked at explicit
statistical criteria would miss them.

## 3. The three diagnostics

Implemented in `rhob.audit.diagnostics`, applied in order, first hit wins.

**Note on provenance.** The paper names three diagnostics; this repository has never
listed them. The three below are reconstructed from the defects the RHOB audit
actually found. If the paper's three differ, these get renamed to match the paper —
not the reverse.

### 3.1 Direction — `INVERTED`

An `equivalence`-shaped claim tested with a difference test, a point threshold, or
nothing at all. The pass condition is "no difference was found", so the criterion
cannot return a negative against a design with no power.

*RHOB's own instance:* `proxy_matched` was `abs(mean_auroc - 0.5) < 0.10`. At four
seeds per side the standard error is 0.0625, so a family whose true L0 AUROC was
0.611 — clearly leaking — passed **43.5%** of the time.

*Remedy offered:* an equivalence test at the margin the criterion already implies.
The published claim does not change; the sample size does.

### 3.2 Degeneracy — `DEGENERACY_UNGUARDED`

An equivalence test with nothing checking that its statistic could have taken another
value. Where every compared value ties, the interval collapses to a point and the
criterion certifies against any margin.

*RHOB's own instance:* an AUROC over a constant proxy is exactly 0.5 by the
half-credit-per-tie convention. The bootstrap SE is 0, the interval is
`[0.5000, 0.5000]`, and it fits inside any margin. **15 of the first 35 cells** passed
this way, all of them families whose proxy is constant by construction.

*Remedy offered:* a third outcome for "not measurable", and hold those cells out of
any control that assumes the criterion was tested.

### 3.3 Demonstrated failure — `UNDEMONSTRATED`

Nothing published shows the criterion returning a negative: no power curve, no
adversarial probe it rejects, no rejection reported alongside the passes.

**This is the mildest verdict and is expected to be the majority one.** It is a
statement about what was published, not about whether the check works. A benchmark
may reject things routinely and simply never report it. The remedy is usually a
reporting change.

A table that presents `UNDEMONSTRATED` as equivalent to "this check cannot fail" is
making this project's own error and must not be published.

## 4. The denominator

The headline statistic is **inverted criteria over equivalence-shaped criteria**.

It is *not* "criteria supported by an equivalence test over all preconditions". That
denominator mixes in criteria for which a difference test is correct, and it fails on
its own author: post-audit RHOB scores 2 of 6 under it and looks mostly broken, while
scoring 0 of 2 inverted — sound on both criteria where the distinction bites.

`rhob.audit.diagnostics.SurveyResult.inverted_rate()` returns the defensible pair.
There is deliberately no function returning the other one.

If no equivalence-shaped criterion is found in a benchmark, the result is reported as
**"no equivalence-shaped preconditions found"**, never as a rate of zero.

## 5. Coding procedure

1. **Pre-register the benchmark list before coding any of it.** Selection criteria
   stated in advance, list frozen, and the frozen list published with the results —
   including benchmarks that were coded and produced no findings. A survey that only
   publishes the benchmarks where it found something has selected on its outcome.
2. **Two independent coders per benchmark**, blind to each other, for `claim_shape`
   at minimum. Disagreements are recorded in the row's `notes`, not resolved silently.
   Report the disagreement rate.
3. **Every row carries a source** precise enough for a third party to check: file and
   line, or paper section. `CodedCriterion.is_admissible()` refuses rows without one,
   and unscored rows are published with their reason rather than dropped.
4. **Code RHOB first and publish the result**, including where it fails. The
   worked example is `audit/rhob_self_coding.json`.

## 6. Author contact, before publication

Not a courtesy step — a correctness step. The authors of a benchmark are the people
most able to tell you the coding is wrong, and the most likely to know about a power
analysis or a rejection that exists but was never published. The survey is better for
having asked.

The sequence:

1. Send each benchmark's authors **their own rows only**, with sources, the exact
   diagnostic applied, and the remedy. Not the comparative table, and not other
   benchmarks' rows.
2. State plainly that RHOB failed the same diagnostics and link the self-coding.
3. Ask two questions: *is this coding wrong?* and *is there a power analysis or a
   published rejection we missed?*
4. Give a stated window — **four weeks** is the working default — and honour it.
5. Publish corrections and author responses **in the table**, next to the row, not in
   an appendix.
6. A benchmark whose every row comes back `SOUND` or `NOT_APPLICABLE` is named in the
   frozen list and carries no finding. It does not get an email and does not appear in
   a findings table.

## 7. What this survey cannot conclude

- **`UNDEMONSTRATED` is not "the check cannot fail".** It is "nothing published shows
  it failing". Most rows will be this.
- **It says nothing about whether the benchmark's results are wrong.** An
  unfalsifiable precondition means a number is unsupported, not that it is false.
- **Coding is a judgement**, concentrated in §2. The inter-coder disagreement rate is
  part of the result and gets published with it.
- **The frozen list is a convenience sample** of benchmarks that are public,
  documented, and in scope. It is not representative of the field and no rate computed
  from it should be described as a rate in the field.

## 8. Open question blocking execution

The paper reports a survey of **19** published benchmarks and, per `README.md`, states
that the failure class **does not** generalize — recorded in §5 as a claim withdrawn.
A separate figure has been quoted internally as "zero of 77 preconditions across 15
published benchmarks".

These are not reconcilable from anything in this repository: the counts differ (19 vs
15), and it is unclear whether the second figure is a surviving descriptive statistic
from §5 or part of what was withdrawn. **No table naming any benchmark may be
published until that is settled**, and settled in the direction the paper's own §5
records, not the direction that makes the better headline.
