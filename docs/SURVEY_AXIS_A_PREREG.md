# Axis A, the power re-run: pre-registration

**Status (2026-11-16): written, not yet registered; no row filled.** It awaits two things,
in order: the sealed record's digest in §13, then a deposit of this file whose date a third
party sets (§13). The 15 rows await the author's §5 row-level coding sheet and a second
human coder. The computation this plan fixes (§5) has not been run on them. The paper
already reports a D2 power result on 20 of the survey's designs
([What is already known](#what-is-already-known)).

This file was written on 2026-11-16, before any row of the axis-A sheet was filled. It
counts as registered once a deposit of it exists whose date a third party sets: an OSF
registration or a Zenodo deposit (a GitHub release archived by Zenodo counts), recorded in
[§13](#13-records-and-amendments). The registered text is this file as deposited. A date in
git is not such a timestamp. Whoever commits sets a commit's date, and whoever tags sets an
annotated tag's; a pushed tag can be deleted and pushed again on another commit; and this
repository's history has been rewritten before. A tag may point at the registered commit,
but it is not the registration. Every change to this file after the deposit is an
amendment (§13). The registered analysis stays the primary one, and no amendment can create
an outcome that strengthens the paper.

**What this is.** The analysis plan for one step of the paper's axis-A remediation: the D2
power re-run on the claims the paper counts as sub-class A. It fixes the question, the
fields, the effect sizes, the computation, the decision rules, and what each outcome does
to the paper's wording, before the computation it fixes has been run on them. The other
two steps of the remediation are out of scope (§11).

**What this is not.** A blind registration. The author coded these 15 claims for the paper,
so their designs are not unknown to the author, and the author has seen the paper's own
power result on 20 of the survey's designs (below). What is fixed in advance is the
analysis, before the computation it fixes has been run on them. It should be weighed as
that.

## What is already known

The paper already reports a D2 power result on the survey's designs. Its §5:

> Of the 20 designs recoverable well enough to power, **0** met the `CANNOT_FAIL` bar; what
> the sample supports is *underpowered,* 15 of 20, 0.750, CI [0.500,0.958]. We keep
> "cannot fail" for the class and the case study, where it is demonstrated rather than
> inferred ([the paper's] §6), and do not claim it as an external prevalence

Appendix F repeats it: "among the 20 designs recoverable well enough to power, 15 are
underpowered (0.750, CI [0.500,0.958]) while **0** meet `CANNOT_FAIL`."

The abstract names sub-class A by its power:

> the degenerate-statistic class is rare wherever measurable ... while the no-power class
> is not (19.5% of 77 enumerated preconditions, *none* of which uses an equivalence test)

That puts "no power" on all 15 claims, which is what this re-run tests (§7.2).

The paper states one D2 criterion, and only for the `CANNOT_FAIL` bar: an MDE, the minimum
detectable effect at power 0.80, beyond the statistic's attainable range. Its checklist
(App. G):

> if the minimum detectable effect at power 0.80 exceeds the statistic's attainable range,
> no dataset could have failed the check. Our own 5-vs-5 cell returns mde = 0.5365 against
> max attainable = 0.5: verdict `CANNOT_FAIL`.

It states neither the criterion nor the effect by which those 15 of the 20 were judged
underpowered. Its §4.2 describes D2 as a half-width compared with the margin, then the
rejection rate against the size of the violation. The author holds the paper's records, and
writes that criterion into the sealed record before registration (§13).

**The App. G screen does not establish its own bar.** The bar is that no dataset could
have failed the check, which is rule 3's meaning here (§6). The screen tests something
else. Its MDE uses the null SD for the alternative as well, (z₀.₉₇₅ + z₀.₈₀)·0.1915 = 0.5365
at 5 vs 5, so it is a normal-approximation screen, not a power bound. At 5 vs 5 it says
`CANNOT_FAIL`, yet by the same approximation the test rejects once the AUROC is 0.375 from
chance, inside the attainable 0.5. An exact two-sided test rejects every complete
separation there (p = 0.0079). The computation pinned here (§5) gives power 0.833 at AUROC
0.95 and 1.0 at AUROC 1.0; its power first reaches 0.80 at an AUROC of 0.944, 0.444 from
chance, inside the range. So App. G's sentence, and its `CANNOT_FAIL` verdict on its own
5-vs-5 cell, are wrong, and the paper needs correcting there
([ROADMAP.md](../ROADMAP.md), "The paper at a venue"). The §5 count of 0 is not weakened by
this: every design rule 3 calls "cannot reject" also has a screen MDE above 0.5 (§7.2).

**Why this registration's grid differs.**

- **One fixed violation.** The paper's stated bar is an MDE per design, and how it judged
  "underpowered" is not stated. Here every claim is asked the same question: its power at a
  true AUROC of 0.7, a violation of 0.2 from chance, against the same 0.8 (§4.1). 0.6 and
  0.8 are reported and decide nothing.
- **Rule 3 asks the bar's question directly**, through the exact test's minimum p, and only
  where the source shows an exact test and states its level and sidedness (§6). At 5 vs 5,
  the paper's own cell, the minimum p is 0.0079 and the power at AUROC 0.7 is 0.156 (§5).
  Rule 4 calls that underpowered, not cannot reject.
- **A different set.** The paper's 20 are the survey's designs recoverable well enough to
  power. The paper does not say which they are, or that all are sub-class A. This
  re-run takes the 15 sub-class A claims, coded twice, with fixed rules for a silent or
  conflicting source (§3), and arithmetic pinned to one file by its digest (§5).

**How many of the 15 are among the 20 cannot be determined from committed material.** The
paper gives the result as counts and names none of the 20 designs.
[`audit/survey_s1_frame.json`](../audit/survey_s1_frame.json) transcribes Appendix F's
per-benchmark totals, which say neither which preconditions were powered nor which 4
benchmarks hold the 15. The row-level coding sheet is not in this repository
([SURVEY_PROTOCOL.md](SURVEY_PROTOCOL.md) §8).

The author ran that calculation for the paper and has seen its results, so for any of the
15 among the 20, a power verdict was known before these rules were written. Before
registration the author writes a sealed record (§13): which of the 15 are among the 20 and
the paper's verdict on each, the criterion behind "underpowered", and the author's own
expected counts under these rules. Only its SHA-256 goes in this file. The record is
private until every group's outreach window has closed, like the rows (§10), and is then
published with them.

---

## 1. The question

The paper's surviving prevalence claim is sub-class A, an equivalence asserted from a
non-significant difference test: 15 of the 77 enumerated preconditions (19.5%, App. F), in
4 of 15 benchmarks (26.7%, §5) (the 15 and the 4 are transcribed in
[`audit/survey_s1_frame.json`](../audit/survey_s1_frame.json)).

Reading non-significance as equivalence is the wrong inference at any design. Whether the
check could also not fail depends on its power. So, for each of the 15:

> At the design it shipped, how often would its difference test have rejected if the claim
> were violated by a true AUROC of 0.7?

0.7 is not the smallest violation that matters. RHOB's own gate is built to reject 0.6
(§4.1). It is a larger violation, so that "underpowered" at 0.7 is the stronger statement.

A test that would have rejected at least 80% of the time, at AUROC 0.7, was a check that
could fail, read with the wrong inference (rule 5, §6). A test that would have rejected
less often was a check that could not reliably fail: underpowered, the paper's word, here
judged at one fixed effect (rule 4).
Only a test that could not have rejected on any data is called a check that cannot fail
here (rule 3): an exact test, at a level and sidedness the source states, whose minimum p
is above that level. The paper keeps "cannot fail" for the class and the case study, and
does not claim it as an external prevalence. Nothing here claims more.

## 2. The unit

One claim: a row of the paper's §5 coding sheet whose support is coded "difference test".
The 15 are taken as the paper counts them, and the sheet holds those rows and no others.
The sealed record fixes which rows they are (§13). The script checks the count against the
paper (§7.1), and the rows against the sealed record.

Where one claim rests on several difference tests (one per task, say, reported as "no task
differed"), the row codes the most powerful: the test with the highest power at AUROC 0.7,
computed as §5 at that test's own level and sidedness. A test whose power §5 does not
compute (§6 rule 2) ranks below every test whose power it does. Among tests tied on power,
the row codes the one with the smaller exact rank floor (§5, step 3), computed for every
test, mean-difference tests included, then the one the source reports first, in the order
`source_location` cites: the earliest page, then the earliest position on the page; for
code, the file, then the line. A rank test and a mean-difference test on the same arms tie
on both, so the source's order decides. `other_tests` records every other test of the
claim (§3). The script refuses a row whose `other_tests` holds a test more powerful than
the coded one. A claim resting on several tests cannot reject at any data only if every
one of its tests cannot (§6 rule 3).

Coding one test is not neutral, and which way it leans depends on where the violation is:

- **Shared across the tasks.** The claim fails if any one test rejects, so its chance to
  fail is 1 − ∏(1 − p_i), p_i being test i's power, at least the coded test's power. Coding
  one test understates the claim's chance to fail. That favours the paper.
- **Confined to one task j, not known in advance.** The other tests still reject at their
  size α_i, or never, for a test that cannot reject at any data (size 0). So the claim's
  chance to fail is 1 − (1 − p_j)∏_{i≠j}(1 − α_i), with that size in place of α_i. That is
  at least p_j, and it can be above or below the coded test's power. Coding one test can
  overstate it or understate it.

Sensitivity analysis (d) (§7.3) takes the shared case, and can only withdraw the outcome
that strengthens the paper. Its figure is at least the confined one whichever test holds
the violation, because each p_i in (d) is at least that test's size in the confined
formula: α_i, or 0 for a test that cannot reject at any data. So wherever the confined case
would put a row at 0.8 or above, (d) does too. Where the confined figure is below the coded
test's power, the row could only move toward "could not reliably fail", and no sensitivity
analysis here may strengthen the paper.

## 3. What is coded per claim

The sheet is a UTF-8 CSV in the format of
[`audit/axis_a_sheet_TEMPLATE.csv`](../audit/axis_a_sheet_TEMPLATE.csv), whose one row is
synthetic. The first four columns identify the claim and come from the sealed record. The
two coders fill the twelve columns from `sub_class_a` to `other_tests` independently (§9).

| Column | What it records | Values |
|---|---|---|
| `claim_id` | A label for the row. It is private, like the row | text, unique in the sheet |
| `benchmark` | The benchmark, as the frame names it | a frame member's name |
| `criterion` | The claim as the source states it, verbatim. A long claim is quoted in part, with an ellipsis, never paraphrased | text |
| `source_location` | Where a third party can check it: file and line, or paper section, table or page | text |
| `sub_class_a` | The coders' reading of the source: is this an equivalence asserted from a non-significant difference test? | `yes`, `no` |
| `test_used` | The difference test the claim rests on; for a claim resting on several, the one §2 codes | one of the values in §4.2 |
| `n_arm_a`, `n_arm_b` | The number of units in each arm *as the test treated them*: the n it was computed on. Their order does not matter (§5) | whole numbers of at least 1, or both blank |
| `unit_of_replication` | What one of those units is: an item, prompt, task, seed, model or run | text |
| `alpha` | The per-test level the source compared the test's p-value against | a number from 1e-10 to 0.5 |
| `sidedness` | Whether the test was two-sided or one-sided | `two_sided`, `one_sided` |
| `defaults_applied` | Which of the rules below for a silent or conflicting source decided a value, for any test the claim rests on | any of `alpha`, `sidedness`, `test`, `conflict`, separated by `;`; blank when none |
| `exact_test` | Whether the source shows that every rank test the claim rests on is exact: an exact or permutation method named, or a library default that is exact at that size, with scores shown to have no ties | `yes`, `no`, `unknown` |
| `scores` | What the measured scores are | `continuous`, `ordinal`, `binary`, `unknown` |
| `clustered` | Whether the units are grouped (several items per prompt, task, seed or model) in a way the test ignored | `yes`, `no`, `unknown` |
| `other_tests` | Every other difference test the claim rests on (§2), one entry each, `test_used:n_arm_a:n_arm_b:alpha:sidedness`, each value as its own column allows, both n blank when not stated | entries separated by `;`; blank when the claim rests on one test |
| `coder_1`, `coder_2` | The two human coders, by initials. `coder_2` is not the author | two different values |
| `disagreed_fields` | The coded columns the two coders' own sheets hold differently (§9) | names separated by `;`; blank when none |
| `notes` | How each disagreement was resolved; which source each value under a rule below came from; the software and method the source names; anything else a reader needs to check the row | text |

The row's `defaults_applied` and `exact_test` describe the claim as a whole, every test it
rests on: `alpha` is listed if any test's alpha was not stated, and so on for each value,
and `exact_test` is `yes` only if every rank test is shown exact. The script does not check
a listed default against any one test's value. A listed `alpha` or `sidedness` keeps rule 3
from the row (§6), and a listed `sidedness` brings it into analysis (f) (§7.3).

Rules for a silent or conflicting source, fixed now so that no coder chooses them with the
result in view. Where a rule sets a value that favours one side, it says which:

- **alpha not stated:** 0.05, with `alpha` in `defaults_applied`. It leans either way:
  toward the paper if the source meant 0.10, toward the benchmark if it meant 0.01.
- **alpha after a correction for several tests:** the per-test level the source compared
  the unadjusted p-value against, α/m for Bonferroni over m tests. For a step-down or
  false-discovery-rate procedure, which has no fixed per-test level, the nominal α, which
  leans toward the benchmark. `notes` names the procedure. Where the source states no α,
  the default 0.05 is corrected the same way (0.05/m for Bonferroni), with `alpha` in
  `defaults_applied`.
- **Sidedness not stated:** `two_sided`, with `sidedness` in `defaults_applied`. It leans
  one way only, toward the paper: it doubles the minimum p and halves the level. At 0.05 it
  can move 3 vs 3, 3 vs 4, 2 vs 5 to 2 vs 7 and 1 vs 19 to 1 vs 38 into "cannot reject",
  and 25 to 31 a side into "underpowered". So rule 3 never applies to such a row (§6), and
  analysis (f) takes it as one-sided (§7.3).
- **The test not named in the text:** coded from the code only where the code shows the
  call that produced the reported p-value. Otherwise `other`, which is not computed, with
  `test` in `defaults_applied`.
- **n not stated**, and not recoverable from the source's tables, code or released data:
  both n columns blank, and the row is not computed. A coder does not estimate n. A total
  n is not recoverable unless its split is stated or fixed by construction, and one arm's
  n alone is not either.
- **Sources that conflict** (the text, the code and the released data give different n,
  alpha, sidedness or test): each value is coded from what produced the reported result.
  Where that cannot be settled, the value that gives the higher power: the larger n, the
  larger alpha, one-sided over two-sided. That leans toward the benchmark. `conflict` goes
  in `defaults_applied`, and `notes` says which source each value came from.
- **Exactness not shown:** `unknown`, treated as `no`, so rule 3 does not apply (§6). It
  can only shrink j, the count the paper may call "cannot fail".
- **Scores not determinable:** `unknown`, which analysis (e) treats as `binary`, so it can
  only withdraw the outcome that strengthens the paper.
- **Clustering not determinable from the source:** `unknown`, which §8 treats as `yes`. It
  matters only in analysis (b), which can only withdraw the outcome that strengthens the
  paper.
- **A one-sided test:** computed as §5 steps 1 and 2, as a two-sided test at 2α in the
  larger of the two arm orders. That order may be the direction the test does not look,
  which errs toward "adequately powered".

The aggregate counts the rows resting on each of `alpha`, `sidedness`, `test` and
`conflict`.

## 4. Effect sizes

### 4.1 The grid

The primary effect is a **true AUROC of 0.7**. The secondaries are **0.6 and 0.8**. The
power threshold is **0.8**.

AUROC here is the probability that a unit from one arm scores above a unit from the other,
ties counting half. It is the quantity a Mann-Whitney test tests, and it is scale-free, so
one grid applies to claims about different measured quantities. A true AUROC of 0.7 means
that, with no ties, 70% of cross-arm pairs are ordered one way.

Why these values:

- **One grid for all 15, not a margin per claim.** The claims measure different things,
  and a "smallest violation that matters" chosen claim by claim would be chosen by the
  author with the design in view. A common grid removes that choice.
- **0.7 as primary, because it is more generous to the benchmarks than 0.6, RHOB's own
  margin.** It is a larger violation than one RHOB's own case treats as a defect: before the
  audit, RHOB's `proxy_matched` check passed a family whose true L0 AUROC was 0.611, clearly
  leaking, 43.5% of the time ([SURVEY_PROTOCOL.md](SURVEY_PROTOCOL.md) §3.1). In this
  calculation power rises from AUROC 0.6 to 0.7 to 0.8 on every design checked (1 to 299
  units in one arm, 1 to 298 in the other in steps of 3, both orders), so a test that is
  underpowered at 0.7 is underpowered at 0.6 as well. "Underpowered" at 0.7 is therefore the
  stronger statement, and "adequately powered" at 0.7 the weaker one. It is not 0.8:
  adequate power against so large a violation says little about the smaller ones the claims
  are meant to rule out.
- **0.6, RHOB's own margin.** RHOB's admission gate certifies a match only inside
  0.5 ± 0.10 (`EQUIVALENCE_MARGIN` in `src/rhob/v3/admission_gate.py`), so 0.6 is the
  smallest violation the case study's own gate is built to reject. It is reported for
  every row and decides no verdict.
- **0.8, a large violation.** A test with power below 0.8 even here says almost nothing
  when it fails to reject. It is reported for every row and decides no verdict.
- **Power 0.8** is the conventional floor for calling a design adequately powered, and the
  threshold at which `cannotfail power difference` prints its warning.

The 0.6 and 0.8 counts appear only in the table of counts, both together, beside the 0.7
count. They never appear in a lead, a heading, the abstract or a README sentence, or in a
sentence that qualifies the outcome's wording (§7.2).

For readers used to standardized mean differences: on normal data with equal variances, a
true AUROC θ corresponds to d = √2·Φ⁻¹(θ), that is 0.36, 0.74 and 1.19 for 0.6, 0.7 and 0.8.
That is a translation only; nothing is computed from it.

Not claimed: that a violation smaller than 0.6 never matters. For some claims it may. The
grid sets what is asked of every test. It is not a statement about what matters.

### 4.2 Tests, and the mapping for those that are not rank-based

| `test_used` | Test | How it is computed |
|---|---|---|
| `mann_whitney` | Mann-Whitney U, or Wilcoxon rank-sum, two independent arms | directly, as §5 |
| `t_test` | Two-sample t-test, Student or Welch | the Mann-Whitney calculation as a stand-in, below |
| `bootstrap_mean_difference` | A bootstrap or normal-theory interval for a difference in means, read as "the interval includes 0" | as `t_test` |
| `paired_t`, `wilcoxon_signed_rank` | Paired tests | not computed |
| `chi_square`, `fisher_exact`, `ks` | Tests on counts, or on whole distributions | not computed |
| `other` | Anything else, named in `notes` | not computed |

The t-test stand-in, stated:

1. **What is computed.** The Mann-Whitney power at the t-test's n per arm, at the same
   AUROC. On normal data with equal variances the Mann-Whitney test's asymptotic efficiency
   relative to the t-test is 3/π ≈ 0.955, so the stand-in understates a t-test's power, by
   about what 5% fewer units would cost. On heavy-tailed or skewed data the Mann-Whitney test
   can be the more powerful, and the stand-in overstates. At small n neither direction is a
   bound. On tied or binary scores both the rank calculation and the stand-in understate
   power, toward the paper; analysis (e) covers binary scores (§7.3).
2. **The sensitivity analysis.** Each such row computed in (a) is also computed at
   ⌈n·π/3⌉ per arm, the n at which the Mann-Whitney calculation matches a t-test on normal
   data. The rank-floor check of item 3 is made at the stated n only. The aggregate reports
   how many verdicts change, and §7.3 says what follows if any does.
3. **No floor.** The exact rank floor does not bind a t-test. At 3 vs 3 the smallest exact
   Mann-Whitney p is 0.100, but a t-test on the same six values (4, 5, 6 against 1, 2, 3)
   returns p = 0.021 (SciPy 1.18.1). So a mean-difference row never gets "cannot reject at
   any data". Its minimum p is reported as not applicable, and where the rank floor is above
   alpha the stand-in does not apply and the row is not computed.

Every other test is **not computed**, because no alternative is specified for it. A
mapping added later is an amendment, and is reported as §13 says: beside the registered
analysis, able to withdraw the outcome that strengthens the paper, never to create it.

## 5. The computation

[`scripts/survey_axis_a_power.py`](../scripts/survey_axis_a_power.py) runs it. The
arithmetic is `cannotfail`'s `min_two_sided_p` and `difference_power`, from
`tools/cannotfail/src/cannotfail/power.py` as last changed in `88a3f10` (cannotfail 0.1.0).
Two changes before it matter here. `7e73dcc` lets `min_two_sided_p` compute at any design
size; before it, the function raised OverflowError from 515 a side, and the script stopped
on any such row. `88a3f10` returns 0.0 for `min_two_sided_p` by a log-gamma check from 541
a side, where the exact ratio rounds to 0.0 anyway, and makes `difference_power` return 1
at a true AUROC of exactly 0 or 1 wherever the exact test can reject. No value at an AUROC
strictly between 0 and 1 changes, so no figure in this plan moves.

The files this plan pins, each by its SHA-256 with LF line endings:

| File | SHA-256 |
|---|---|
| `tools/cannotfail/src/cannotfail/power.py` | `63a94676b51294cc9528b3c830d66e610b3f459893327edfa4675122a5429cd2` |
| `audit/survey_s1_frame.json` | `4952ac6e2cfab407f68517c091ad839bd055c379070d298725fcdd80b4543936` |
| `scripts/survey_axis_a_power.py` | `25d97fe6c2e9316b889ad0ea40eead5395904e137242b6d93cf1a7319f739946` |
| `audit/axis_a_sheet_TEMPLATE.csv` | `27881e0fade52d1fbdd59d44b37863e5006b1e6e811186098ea57ac8da7e89a2` |

The script prints the digest of the power.py and of the frame it read, and whether each
matches. A run on any other file is not this analysis, and on such a run the script
reports its outcome as none. The script's own digest and the template's are checked by the
repository's tests (`tests/test_audit/test_axis_a_power.py`). The deposit (§13) holds all
four files with this one. Where the script and §§6-7 differ, §§6-7 govern, and the
difference is reported. Neither function needs SciPy; the script needs NumPy only because
importing `cannotfail` does.

For each row with `sub_class_a` = `yes`:

1. **Level.** alpha for a two-sided test. For a one-sided test, 2·alpha passed as the
   two-sided level: a two-sided test at 2α has the same upper rejection threshold as a
   one-sided test at α. That adds the opposite tail's rejection probability, which errs
   toward "adequately powered", in the benchmark's favour. For a one-sided test at 0.05,
   wherever a rejection is attainable, it is at most 0.019 at the grid's three AUROCs when
   both arms have 3 or more units (checked from 3 to 199 a side). With an arm of 1 or 2 it
   can reach 0.028, but power at 0.7 stays below 0.27 there, so no row's own verdict
   depends on it; in analysis (d) it can, toward adequately powered, as the 0.019 above can.
2. **Orientation.** `difference_power` is not symmetric in its two arms: at AUROC 0.7,
   5 vs 40 gives 0.302 and 40 vs 5 gives 0.259. Every power here is the larger of the two
   orders, for a one-sided test too, though only one order is the direction it looks. So
   the order of the arms in the sheet does not matter. That errs toward
   "adequately powered", in the benchmark's favour.
3. **Minimum p.** `min_two_sided_p(n_a, n_b)`, halved for a one-sided test: the smallest
   p-value an exact Mann-Whitney test can return at this size, reached when every unit of
   one arm outranks every unit of the other. For the exact test, ties can only raise it.
   Other methods can go below it, and a library default can switch to one without the
   source saying so. SciPy's `mannwhitneyu` uses the exact test by default only when one
   sample has 8 or fewer units and there are no ties (`method='auto'`). With ties it
   switches to a normal approximation with continuity correction: on fully tied scores it
   returns p = 0.047 at 3 vs 3 (exact floor 0.100), 0.025 at 3 vs 4 (exact 0.057) and 0.041
   at 1 vs 6 (exact 0.286). Without ties, its asymptotic method without continuity
   correction returns 0.0495 at 3 vs 3 (all SciPy 1.18.1). So the floor says what an exact
   test could do, and rule 3 uses it only where the source shows an exact test (§6).
4. **Power.** `difference_power(n_a, n_b, θ, alpha=level)` for θ in 0.6, 0.7 and 0.8. It is
   a normal approximation: the exact null SD sets the threshold and the Hanley-McNeil
   variance describes the alternative. Its docstring says it is close at 20 or more a side
   and only indicative below 10; at 5 vs 5 and 0.7 it gives 0.156 where a simulated exact
   test rejects about 0.12 of the time. It assumes untied scores, and on tied or binary
   scores it understates the test's power. Where no data could reach alpha it returns 0
   exactly. Any row's figure is reproduced with `cannotfail power difference --pos n_a
   --neg n_b --true-auroc θ --alpha level`, and again with the arms swapped.
5. **Mean-difference rows**, as §4.2.
6. **Binary scores, for analysis (e).** The power of a two-sided two-proportion z-test of
   0.3 against 0.7 at the row's level: Φ((0.4 − z·s₀)/s₁) + Φ((−0.4 − z·s₀)/s₁), with
   z = Φ⁻¹(1 − level/2), s₀ = √(p̄(1 − p̄)(1/n_a + 1/n_b)), p̄ = (0.3·n_a + 0.7·n_b)/(n_a + n_b)
   and s₁ = √(0.21/n_a + 0.21/n_b). On 0/1 scores, ties counting half, AUROC = 0.5 + (p_hi −
   p_lo)/2, so 0.7 means a gap of 0.4. For equal arms, 0.3 against 0.7 is the pair with
   the largest variance and the lowest power of any binary design at AUROC 0.7 (checked
   from 2 to 199 a side). With unequal arms another pair can be lower: at 12 vs 108, 0.800
   at 0.3 against 0.7, and 0.794 with 0.242 in the 12-unit arm and 0.642 in the other. So
   (e) can call a row adequately powered that a less favourable binary design would leave
   underpowered; that can only withdraw the outcome that strengthens the paper. At 24 a
   side it is 0.812, against 0.678 by step 4. The script computes it; `cannotfail` does
   not.

The design arithmetic, for orientation (two-sided, alpha 0.05, equal arms):

| n per arm | min p | power at 0.6 | at 0.7 | at 0.8 |
|---|---|---|---|---|
| 3 | 0.100 | 0 | 0 | 0 |
| 4 | 0.029 | 0.069 | 0.127 | 0.230 |
| 5 | 0.0079 | 0.076 | 0.156 | 0.306 |
| 10 | 1.1e-05 | 0.111 | 0.310 | 0.655 |
| 20 | 1.5e-11 | 0.185 | 0.590 | 0.953 |
| 31 | 4.3e-18 | 0.267 | 0.796 | 0.997 |
| 32 | 1.1e-18 | 0.275 | 0.810 | 0.998 |

Power at AUROC 0.7 first reaches 0.8 at 32 a side (25 a side one-sided at 0.05). At 0.6 it
takes 130 a side, and at 0.8 it takes 14. Unequal arms of 3 and 4 have a minimum p of 0.057,
so they cannot reject at 0.05.

## 6. The decision rules, per claim

Applied in order; the first that matches decides.

1. `sub_class_a` = `no`: **not sub-class A on re-coding.** Out of the power counts, and
   reported first (§7.1).
2. A test with no specified computation, an n not stated, or a mean-difference row whose
   exact rank floor is above alpha (§4.2): **not computed**, with the reason.
3. For a rank-based row, a minimum p above alpha, where the source shows an exact test
   (`exact_test` = `yes`), states alpha and sidedness (neither in `defaults_applied`), and
   every other test of the claim is a rank test whose own minimum p is above its own alpha:
   **cannot reject at any data.** A minimum p equal to alpha can still reject. A rank row
   whose minimum p is above alpha but which fails one of these conditions goes on to rule
   4, where its power is 0, so it is underpowered. The aggregate counts such rows, and why.
4. Power at AUROC 0.7 below 0.8: **underpowered.**
5. Otherwise: **adequately powered.**

"Cannot reject at any data" and "underpowered" together are the claims that could not
reliably fail. Only the first is a check that cannot fail (§1). "Adequately powered" is not
"sound". The inference is still the wrong one, and the right test, an equivalence test at a
stated margin, was still not run. It says only that a non-significant result at that design
was informative against a violation of AUROC 0.7. Like an `ok` from `cannotfail`, it is not evidence that the arms are matched.

## 7. What each outcome means for the paper

### 7.1 Membership comes first

If fewer than 15 rows are confirmed (`sub_class_a` = `yes`), the sub-class A count is
wrong. "15 of the 77 enumerated preconditions, in 4 of 15 benchmarks" becomes the confirmed
count C and its benchmark count B, whatever the power result, and that is reported before
any power figure. **This outcome weakens the paper.** The script prints it above the power
counts, as "C of 15 confirmed, over B of 4 benchmarks", and leads its last line with it. A
row the coders first disagreed on counts as resolved, and the disagreement is published
(§9).

A sheet that does not reconcile with the paper's counts (15 rows, App. F; over 4
benchmarks, §5; every one an audited frame member with no more rows than App. F enumerates
for it) is not the paper's sub-class A. Its output is not the power re-run. The script says
so above its counts and reports its outcome as none.

The script reports an outcome only on the registered inputs: the pinned power.py and frame
(§5); the sealed record whose digest §13 records, with the sheet's identification columns
equal to its own; and the two coders' own sheets, three different files, which the sheet
must match (§9): `disagreed_fields` names exactly the columns they hold differently, and
every other coded column holds the value both hold (not checked with `--after-responses`,
§7.4). Otherwise its last line is none, with the reasons.

### 7.2 The power outcome

Over the C confirmed rows: K are computed; W of those could not reliably fail (cannot
reject, or underpowered), of which j cannot reject; A = K − W are adequately powered.
Where §7.1 applies, C and B are its counts; otherwise C = 15 and B = 4. Every outcome
reports W, j, A, K, C and B, the not-computed rows by reason, and the number of benchmarks
with at least one of the W, out of the B. It also reports W, j, A and K separately for
three groups, as the sealed record marks them: the claims among the paper's 20
(`underpowered` or `not_underpowered`), which were partly foreseen; those marked `unknown`,
which may have been; and those marked `no`, which were not. The outcome decides what the
paper may lead with.

The table is read top to bottom, and the first outcome that matches decides. The one that
weakens the paper comes first, so a low K cannot hide it.

| Outcome | When | What the sub-class A wording becomes |
|---|---|---|
| **Most adequately powered** | A > W, whatever K is | **This weakens the paper.** In A of the K claims whose test could be computed, by the registered computation, the test's power against a true AUROC of 0.7 was at least 0.8, so those checks could fail; how many of the A have clustered or unknown units (§8), and how many are mean-difference stand-ins (§4.2), is given beside it. They still rest on the wrong inference. The statement that sub-class A generalised must be rewritten: the inference recurred in C of 77, in B of 15 benchmarks, and of the K claims whose test could be computed, a check that could not reliably fail is evidenced in W. Nothing is said either way of the C − K not computed. That narrower count leads wherever the 15 is quoted now: the paper's abstract, §5 and App. F; the README's "Does the class occur elsewhere?"; `docs/site/index.html`'s "Does it occur elsewhere?"; `tools/cannotfail/README.md`; and [SURVEY_PROTOCOL.md](SURVEY_PROTOCOL.md) §8. The docstring of `power.py` also quotes it; that file is pinned (§5), so its docstring changes only after the re-run is published. |
| **Inconclusive** | A ≤ W and K ≤ C/2 | The re-run adds no power statement. The count stands as a count of the inference, C of 77 in B of 15 benchmarks, with the abstract reworded as below, and one added sentence: the power re-run could compute K of the C claims, and why the rest could not be. |
| **Most could not reliably fail** | W > A and K > C/2 | The paper may add that in W of the K claims whose test could be computed, the difference test could not reject on any data (j) or had power below 0.8 against a true AUROC of 0.7 (W − j). This strengthens the case that sub-class A's checks could not reliably fail. Any power statement is limited to W of the K computed: never to all C, and never to the claims not computed. "Cannot fail" is said only of the j, each at its shipped design, never as a rate. Rule 3 counts a row in j only where its source shows an exact test and states its level and sidedness (§6). |
| **Split** | W = A and K > C/2 | Both counts side by side, with no lead claim either way. The inference recurred in C of 77, in B of 15 benchmarks; of the K computed, W could not reliably fail and A could. The bold "generalises" (paper §5) and "**did**" (README) are rewritten as under Most adequately powered. |

Whatever the outcome:

- **The abstract.** Its "the no-power class is not [rare] (19.5% of 77 ...)" is reworded to
  name the inference rather than the power, for example "an equivalence asserted from a
  non-significant difference test recurs in 19.5% of 77 ...", with C for 15 where §7.1
  applies. Any power statement in it is limited to W of the K computed, as the table allows.
- **The paper's own D2 sentence.** At the same alpha and sidedness, rule 3 is narrower than
  the App. G screen: every design it calls "cannot reject" has a screen MDE above 0.5. So a
  "cannot reject" verdict on a claim among the paper's 20 designs contradicts the paper's
  "0 met the `CANNOT_FAIL` bar" (§5, App. F). Either the design was coded differently from
  the paper's (n, alpha or sidedness), or the paper's figure was wrong. The script counts
  such claims from the sealed record. If there is one, the paper's sentence is corrected in
  the same change, with the coding difference stated, and that correction is reported ahead
  of any wording that says "cannot fail" of the j. The script also counts the claims among
  the 20 whose verdict differs from the paper's own. That count is reported beside the
  sealed record's, and is not a correction: the paper's effect for "underpowered" may not
  be the one used here.
- **The secondaries** are quoted only as §4.1 says.

### 7.3 Robustness

The outcome is computed six times: (a) on every row; (b) with the rows whose `clustered`
is `yes` or `unknown` dropped; (c) with the mean-difference rows decided at the
efficiency-adjusted n of §4.2; (d) with each claim that rests on several tests (§2) decided
on its combined chance to fail; (e) with the rows whose `scores` are `binary` or `unknown`
decided on the two-proportion power of §5, step 6; (f) with the rows whose sidedness rests
on the default decided as one-sided (§5, step 1). The script computes all six. The paper
may lead with "most could not reliably fail" only if all six return it. If (a) returns it
and any of (b) to (f) does not, the outcome is reported as **not robust**, all six are
reported, and the wording follows Split.

In (c) to (f), a row not computed in (a) stays not computed, and a row moves only toward
adequately powered: its verdict from (a) is replaced only where the analysis finds a power
of at least 0.8. So each of (c) to (f) can only withdraw "most could not reliably fail". In
(c) the floor rule is applied at the original n. A row it leaves not computed would be
underpowered at the adjusted n, so leaving it out errs toward withdrawing "most could not
reliably fail". In (e), a row that cannot reject at any data keeps that verdict: for the
exact test, ties only raise the floor. In (f), a row whose `defaults_applied` lists
`sidedness` has its coded test taken as one-sided, even when the default applied only to
another of the claim's tests; that too can only withdraw the outcome that strengthens the
paper.

Analysis (d), stated:

1. It applies to the rows computed in (a) whose claim rests on several tests (`other_tests`
   not blank).
2. For each test i of the claim, the coded one included, p_i is its power at AUROC 0.7
   computed as §5 at its own level and sidedness, or its α_i if that is larger. Taking the
   larger keeps (d) at or above the confined case of §2 whatever the approximation does. In
   the larger of its two orders (§5, step 2), no design checked whose minimum p is at or
   below its level has power below that level (1 to 300 a side, and arms of 1 to 3 against
   up to 10⁶, at levels from 10⁻⁶ to 1), so in practice the power is used. p_i is α_i for a test whose power §5
   does not compute (§6 rule 2), and for a rank test whose minimum p is above its alpha
   where rule 3's conditions on the row do not hold, since such a test may still reject at
   about its size. p_i is 0 only for a rank test that cannot reject at any data under those
   conditions. The combined chance to fail is 1 − ∏(1 − p_i), at least the coded test's
   power.
3. A combined chance of at least 0.8 is **adequately powered**. Below it, the row keeps its
   verdict from (a).
4. It assumes the tests use disjoint units. Where they share some (a common control arm,
   say), it overstates the chance to fail, which errs toward withdrawing the strengthening
   outcome.

Every other outcome of (a) (most adequately powered, inconclusive, split), and the
membership correction, stand on (a) alone. A sensitivity analysis can withdraw an outcome
that strengthens the paper, never one that weakens it. (c) and (d) each run one side of a
two-sided uncertainty. Their other sides, heavy-tailed data and a violation confined to a
less powerful test, are not run, because they could only strengthen the paper. Ties on
ordinal scores are not corrected at all, and that favours the paper (§12).

### 7.4 Replies from groups

The registered outcome is the one computed at §14 step 5 on the reconciled sheet whose
salted digest §13 records from before that run. It is reported first, under its own label.

- Corrections from a group are applied to a copy of that sheet, never to it. The copy is
  run with the step-5 command, the copy in place of the sheet, plus `--after-responses`,
  and reported as a second analysis, "after author responses", listing each changed field
  by row and who proposed it. Where the author and a group disagree, the copy takes the
  group's reading, and both readings are printed beside the row.
- The copy keeps `claim_id`, `benchmark`, `criterion` and `source_location` exactly as the
  sealed record has them. A group's correction to one of them is recorded in the copy's
  `notes`, with who proposed it, and printed beside the row; none of the four enters the
  computation.
- No row is deleted. A row a group shows is not an equivalence asserted from a
  non-significant difference test is set to `sub_class_a` = `no` in the copy: a membership
  correction (§7.1). For axis A this replaces AUTHOR_OUTREACH.md's "The row comes out",
  which would leave a sheet that no longer reconciles.
- A power analysis a group sends is cited beside its row and does not change its verdict,
  which concerns the shipped design at the registered effect.
- The paper's wording follows the registered outcome. If the outcome after author
  responses weakens the paper, the weaker one leads, by the same rule as §7.3.

## 8. Clustering

`difference_power` assumes independent units. When a test treated grouped units as
independent (several items per prompt, task, seed or model), it was mis-sized: its real
false-positive rate is not alpha, and its real rejection rate under a violation is not the
figure computed here. The direction is not fixed, so the figure is neither an upper nor a
lower bound, as `cannotfail.power`'s module docstring also says. Clustered rows are computed
at the n the test used, flagged in the per-row table, counted in the aggregate, and dropped
in sensitivity analysis (b). A design-effect correction is not attempted: it needs the
intra-cluster correlation, which a source is not expected to report. Where one does, the row
records it in `notes`.

## 9. Two human coders

The twelve coded columns, `sub_class_a` to `other_tests`, are coded twice, by two people
working from the source independently and without seeing each other's sheet. One may be
the author. The other, `coder_2`, must be a person other than the author. An AI assistant
does not count as an independent coder, as the protocol's two-coder rule says
([SURVEY_PROTOCOL.md](SURVEY_PROTOCOL.md) §5, step 2): it asks for two independent
readings, and an assistant's reading is shaped by whoever prompts it.

Both coders are given the source and the four identification columns from the sealed
record, and nothing else from it. `criterion` is verbatim, never the author's paraphrase.
Each coder fills a copy of the template alone: the identification columns as given, both
coders' initials in `coder_1` and `coder_2`, the coded columns, and `disagreed_fields` left
blank. That is the coder's own sheet. Both own sheets are kept privately, under
`audit/private/` or outside the repository. Each coder runs `--check` on their own sheet,
which prints only `valid` or the errors and computes nothing, and corrects every error it
reports. The author, who holds the sealed record, also runs `--check --sealed-record` on
each, which counts the rows whose identification columns are not the record's, and the
record's claims the sheet does not hold, and prints nothing else. Then, before reconciliation starts, the salted SHA-256 of each is recorded in
§13, and both are published with the rows after every window has closed. If a recorded own
sheet later proves invalid, or its identification columns differ from the record's, its
coder corrects only what makes it so; §13 records the correction, dated, with both salted
digests, and both versions are published.

Disagreements are resolved by discussion once both sheets are complete, never by whichever
file is read second. Where discussion does not settle a field, `coder_2`'s value stands. A
value both own sheets hold is carried into the reconciled sheet unchanged. The reconciled
sheet records which columns the two own sheets hold differently (`disagreed_fields`) and how
each was resolved (`notes`). Columns are compared as the script reads them:
`defaults_applied` and `other_tests` as unordered lists, numbers as numbers, text ignoring
case and surrounding spaces, and the two arms in the order written. So two arms entered in
opposite orders are a difference, and `Item` against `item`, or `0.05` against `.05`, is
not. The script compares the two own sheets with the reconciled one, and reports no outcome
if `disagreed_fields` does not name exactly the columns they hold differently, or if
another coded column does not hold the value both hold. Before its digest is recorded, the
reconciled sheet passes `--check` against the sealed record and both own sheets, which
prints counts only (§14 step 4). The step-5 run, not `--check`, also reports the membership
count and the outcome on `coder_2`'s own sheet, beside the reconciled ones.
The per-column disagreement counts are printed with the aggregate, and published with it,
as the protocol requires.

The script checks that two different coder names are present, that the reconciled sheet
and the two own sheets are three different files, and that the reconciled sheet matches
the own sheets as above. It cannot check that both coders are people or that they coded
independently, and its output is not evidence that they did.

## 10. Privacy

Every row names a third-party benchmark. **Nothing about the 15 claims' power, aggregate
counts included, and nothing from the sealed record, is made public before every group's
window has closed** ([AUTHOR_OUTREACH.md](AUTHOR_OUTREACH.md); four weeks by default). Of the
private files, only salted digests are committed before then (§13). Until then:

- **The sheet, the coders' own sheets, the sealed record and any saved output live outside
  this repository, or under `audit/private/`**, which `.gitignore` excludes. The script
  refuses to read any of them from anywhere else in the checkout, the template included.
  The check covers paths in the script's own checkout only. The one sheet in the repository
  is the template, whose row is synthetic.
- **The per-row output names benchmarks** and stays with the two coders. The second coder
  holds every group's rows, as the protocol's two-coder rule requires, and is bound by this
  section. AUTHOR_OUTREACH.md's rule against sending the comparative table applies to
  everyone else. The per-row output does not go into an issue, a commit, a pull request, a
  chat or a public draft.
- **`--aggregate-only` prints counts only**: no benchmark, criterion, source, coder or claim
  id, and no per-benchmark breakdown. Like the per-row output, it stays with the two coders
  until every window has closed. Sharing it with anyone else before then is a §13 amendment
  that says with whom, by role, and why, and that person agrees not to pass it on. It is
  not made public before then, for two reasons. Nobody should first learn about their rows
  from a public count. And the paper publishes that the 15 claims fall in 4 benchmarks, so
  some counts can single benchmarks out: "4 of 4" benchmarks with a claim that could not
  reliably fail names no one and implicates all four.
- **Error output names no row content.** An invalid input is reported by line and column,
  with the allowed values, or as a whole file (one that is not UTF-8 text, or not a
  readable CSV), never with a cell's text.
- **Each group's email** carries its own rows with their power figures, and nothing about
  any other group's rows. `--benchmark NAME` prints one benchmark's rows and no counts; the
  criterion text and the source come from the sheet.
- **After every window has closed**, the counts, the rows, the coders' own sheets and the
  sealed record are published, with each group's response beside its rows, as
  AUTHOR_OUTREACH.md says and §7.4 adds for axis A.

## 11. Out of scope

- **The revised search patterns.** The enumeration's patterns select against
  equivalence-test vocabulary, and revising them is the other part of the axis-A
  remediation. A revised enumeration may find claims this re-run never sees. It gets its
  own registration.
- **The three frame members the paper did not audit**, marked `NOT_AUDITED` in
  `audit/survey_s1_frame.json`. The script refuses rows for them.
- The other 62 preconditions, and the claim-shape pass over all 77
  ([SURVEY_PROTOCOL.md](SURVEY_PROTOCOL.md) §8). That port is not a prerequisite for this
  re-run: the sealed record fixes which 15 rows it reads (§13).
- Re-running any benchmark's test on its data, or running an equivalence test for it. This
  computes what each shipped design could have detected. It measures nothing again.
- A multiplicity correction across the 15. Each verdict describes one design; none is a
  test.

## 12. What this cannot conclude

- Whether any benchmark's arms are in fact matched, or whether its results are wrong
  ([SURVEY_PROTOCOL.md](SURVEY_PROTOCOL.md) §7).
- That an adequately powered claim is sound (§6).
- That the paper's 15 of 20 is corroborated. The coder, the sources and, for the claims
  among the 20, the verdicts were known ([What this is not](#axis-a-the-power-re-run-pre-registration);
  §13). This is a re-computation under rules fixed in advance, not an independent
  replication. If the sealed record shows that the paper's "underpowered" used AUROC 0.7
  at power 0.8, rule 4 is close to the paper's own criterion, and agreement on those rows is
  expected, not found.
- A prevalence of checks that cannot fail. The paper declines that claim (§5), and nothing
  here supports more.
- Anything about the field. The frame is a convenience sample, and no count here is a rate
  in the field. The enumeration's selection (§11) is untouched.
- An exact power. The figures are a normal approximation, only indicative below 10 a side,
  for mean-difference tests a stand-in (§4.2), and for untied scores. On tied or discrete
  scores they understate power. Binary scores are covered, by analysis (e); on ordinal
  scores the understatement, which favours the paper, is not corrected.

## 13. Records and amendments

Until the deposit exists, this file is written, not registered. Each record below is made
before the step it names. Each record made after the deposit is timestamped by a third
party: a new version of the deposit, or an RFC 3161 or OpenTimestamps stamp of the digest. A
salted digest is the SHA-256 of the sealed record's salt, a newline, and the file's bytes:
`{ printf '%s\n' "$SALT"; cat FILE; } | sha256sum`. Every digest of a private file
committed before the windows close is salted, so no count or row can be recovered from it
by trial; the sealed record holds its own salt.

- **Sealed record: not yet made.** Owed before the deposit (§14 step 2). A private file,
  `audit/private/axis_a_sealed.txt`, in this format:

      axis-A sealed record
      salt: <64 lowercase hexadecimal digits, from secrets.token_hex(32)>
      paper_underpowered_bar: <one line: the effect or margin, fixed or per claim, power, alpha>
      expected: W=<w> j=<j>
      claims:
      claim_id,benchmark,criterion,source_location,paper_20
      <one CSV row for each of the 15 claims>

  `paper_underpowered_bar` is the criterion the paper's "underpowered" used: the effect or
  margin, whether it was one fixed value or each claim's own, and the power and alpha.
  `expected` is the author's own expectation of W and j over the 15 under these rules,
  written before any row is coded. `paper_20` is `underpowered` or `not_underpowered` for a
  claim among the paper's 20 designs, as the paper judged it; `no` for the rest; and
  `unknown` where the author's records cannot say. The four identification columns are the
  ones the sheet will hold: the rows of the author's §5 sheet, with `benchmark` spelled as
  the frame names it (§3). Its benchmarks must reconcile with the frame as the sheet's must
  (§7.1): every one an audited member, 4 of them, none with more of the 15 than Appendix F
  enumerates for it. The script refuses a record that fails, reporting counts and no name,
  so its step-5 outcome is none; `--check --sealed-record` alone checks the record before
  its digest is recorded (§14 step 2). Once made, it is recorded here as
  **Sealed record: SHA-256 `<digest>`**, the digest of the file's bytes, which the script
  reads from this line. The record is published, salt included, when every window has
  closed. Until then its counts stay out of this file: 15 claims over 4 benchmarks are few
  enough that a count can single benchmarks out (§10), and the possible counts are few
  enough that an unsalted digest of them could be inverted by trying each.
- **Deposit, the registration: not yet made.** After the sealed record's digest is in this
  file: a deposit, dated by a third party, of this file and the four files §5 pins, as of
  one commit. Record here the DOI or URL, the commit, and each deposited file's SHA-256.
  This record, made after the deposit, is not a change to the registered text. A tag on
  that commit (suggested name `axis-a-prereg-2026-11-16`) may point at it; its date is the
  author's and is not a timestamp. After that commit, PR #2's branch is not rebased,
  reworded or force-pushed, and main is brought in by a merge. PR #2 is merged with a merge
  commit, not squashed or rebased, so that the commit survives into the history of main.
- **The coders' own sheets: not yet made.** Before reconciliation (§14 step 4), once each
  passes `--check` (§9): the salted digest of each. A later correction to one (§9) is
  recorded here too, dated, with the old and the new salted digest.
- **The reconciled sheet: not yet made.** Before the first run at §14 step 5, once it
  passes the check of §14 step 4: its salted digest, by the formula above. The run prints
  it again, and it must equal this one. The outcome of step 5 on that sheet is the
  registered outcome (§7.4). If the step-5 run refuses it, or reports that it does not
  match the coders' own sheets, only what makes it so is corrected, and only from the own
  sheets: a name is added to or removed from `disagreed_fields`, a value both own sheets
  hold is restored, and an invalid entry takes the value both own sheets hold or, where
  they differ, `coder_2`'s (§9). Every other value stays as recorded. §13 records the
  correction, dated, with both salted digests; it is the one record made after step 5's
  first run. Both versions are published, and the outcome of step 5 on the corrected sheet
  is the registered outcome.

**Amendments.** Every change to this file after the deposit is an amendment, except the
records above. Each goes here, dated, with what had been seen when it was made and its
diff against the registered version. The registered analysis remains the primary analysis
and is always reported. An amendment to §§3-7, a new §4.2 mapping or another power.py
included, is reported beside it as a further analysis. That analysis can withdraw an
outcome that strengthens the paper, and leads if it weakens the paper. It never creates or
restores an outcome that strengthens the paper, whenever it was made.

Amendments: none as of 2026-11-16.

## 14. Procedure

1. Commit this file. No row is filled.
2. Write the sealed record (§13) privately, from the author's records of the paper's power
   calculation and the §5 sheet. Check it, correcting every error the check reports:

       python scripts/survey_axis_a_power.py --check --sealed-record audit/private/axis_a_sealed.txt

   Record its digest in §13 and commit. Make the deposit of
   that commit's files and record it in §13. PR #2 is merged when
   [RELEASING_CANNOTFAIL.md](RELEASING_CANNOTFAIL.md) says, with a merge commit (§13);
   coding need not wait for it.
3. Copy the template to `audit/private/` or outside the repository; nothing else in the
   checkout may hold a filled sheet. If you use `audit/private/`, also add the line
   `audit/private/` to `.git/info/exclude`: every working tree of the clone shares that
   file, so the sheets stay ignored when a commit from before this rule is checked out.
   Give both coders the source and the identification columns of the 15 claims from the
   sealed record.
4. Each coder fills their own sheet alone (§9) and checks it, correcting every error the
   check reports:

       python scripts/survey_axis_a_power.py audit/private/axis_a_coder_1.csv --check

   The author also checks each own sheet with `--sealed-record audit/private/axis_a_sealed.txt`
   added. Record the salted digest of each own sheet in §13. The two then reconcile into
   one sheet, recording `disagreed_fields` and `notes`. The author checks it against the
   record and both own sheets, and the two correct it until it is valid and every count is
   0. The check prints counts only, never a value or a verdict:

       python scripts/survey_axis_a_power.py audit/private/axis_a_sheet.csv --check \
           --sealed-record audit/private/axis_a_sealed.txt \
           --coder-sheets audit/private/axis_a_coder_1.csv audit/private/axis_a_coder_2.csv

   Then record the reconciled sheet's salted digest in §13, before step 5.
5. Run, privately:

       python scripts/survey_axis_a_power.py audit/private/axis_a_sheet.csv \
           --sealed-record audit/private/axis_a_sealed.txt \
           --coder-sheets audit/private/axis_a_coder_1.csv audit/private/axis_a_coder_2.csv

   Save any output under `audit/private/` or outside the repository. The run prints the
   reconciled sheet's salted digest, which must equal the one §13 records.
6. Send each group its own rows, from `--benchmark NAME` and the sheet, in the email
   [AUTHOR_OUTREACH.md](AUTHOR_OUTREACH.md) gives for a criterion coded INVERTED. For axis
   A, its question 2 ends "If so I'd like to cite it beside the row": the row stays (§7.4).
7. Apply the groups' corrections to a copy (§7.4), and run the step-5 command on the copy,
   in place of the sheet, with `--after-responses`.
8. After every window has closed, make the counts public with all six analyses (the same
   command with `--aggregate-only`), publish the rows, the coders' own sheets and the sealed
   record, with the responses beside the rows, and change the paper's wording as §7 says.
