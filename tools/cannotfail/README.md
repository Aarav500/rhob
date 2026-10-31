# cannotfail

A benchmark's headline usually rests on a claim about its own construction — *the two
arms are matched*, *this control is at chance* — and a check in the harness guards it.
`cannotfail` finds guards that **cannot return a negative verdict** under the design that
ships. Such a check is green in CI and looks exactly like one that was earned.

It implements two of the three diagnostics from *Checks That Cannot Fail* (Shah, 2026,
[doi:10.5281/zenodo.22005218](https://doi.org/10.5281/zenodo.22005218); the Zenodo record's
metadata currently shows an earlier title, *Detecting Reward Hacking Without an Oracle*,
but the PDF it hosts is this paper):

- **D1, resolution.** What range could the statistic have taken? Run it on your own
  score vectors. Needs only NumPy.
- **D2, power.** At the design you ship, could the precondition have failed? Needs SciPy
  for the equivalence half.

D3 (does a parameter reach anything the evaluation reads?) is not included: answering it
needs knowledge of your harness that a score file does not carry.

## Install

```bash
pip install "cannotfail @ git+https://github.com/Aarav500/rhob#subdirectory=tools/cannotfail"
pip install "cannotfail[power] @ git+https://github.com/Aarav500/rhob#subdirectory=tools/cannotfail"   # with D2
```

Python 3.10 or later. The package imports nothing from the benchmark it was built in.

## D1: run it on your score vectors

Give it a CSV or TSV with one row per scored item: the score, optionally its class, and
optionally the cell it belongs to (a dataset, a detector-by-task pair, whatever your
headline aggregates over).

```
$ cannotfail resolution scores.csv --score score --label is_hack --group cell
group              n  distinct     pos     neg  resolution      AUROC range  verdict
reward_only       48         1      24      24       0.000   [0.500, 0.500]  fixed (one distinct value)
mostly_tied       48         2      24      24       0.042   [0.479, 0.521]  pinned (AUROC confined within 0.5 +/- 0.0208)
behavioural       48        48      24      24       1.000   [0.000, 1.000]  ok

1 of 3 groups has at most one distinct value, so any rank statistic over it is fixed at 0.5.
1 of 3 groups is pinned: the AUROC cannot get further from 0.5 than 0.1, so its estimate cannot leave an equivalence band of that width. A test that also needs its interval inside the band can still fail, but only on the interval's width; a pass there says the scores tie.
```

That output is from [`examples/scores.csv`](examples/scores.csv). The idea behind it: in a
standard AUROC — the one `sklearn.metrics.roc_auc_score` computes — every cross-class pair
of equal scores contributes exactly 0.5, whatever the data. So if a fraction `1 − r` of the
pairs are equal, the AUROC is confined to `0.5 ± r/2`. That is an identity, not an
estimate. `resolution` is that `r`, over the finite scores; blank or non-finite scores are
left out and counted in the footer.

| Verdict | Meaning |
|---|---|
| `fixed` | A standard AUROC over these scores is 0.5 whatever happens: at most one distinct value, no unequal pair across the classes, or a class with no finite score. |
| `pinned` | A standard AUROC cannot get further from 0.5 than the margin (`--margin`, default 0.10). Its estimate can never leave an equivalence band of that width; a test that also needs its interval inside the band can still fail on the interval's width. |
| `dust` | The scores differ, but only below the tie tolerance described below — or, if you set `--dispersion-floor`, by less than that fraction of their own size. |
| `ok` | D1 found no reason the check cannot fail. That is not evidence it is sound. |

**The tie tolerance, and `dust`.** Rounding error in a scorer's own arithmetic can make
identical inputs score differently in the 11th significant digit, and a standard AUROC
will rank that noise. So the tool also counts ties the way the admission gate it was
lifted from does: `|a − b| ≤ max(1e-12, 1e-9 · max(|a|, |b|))`. When that tolerance would
confine the AUROC but exact equality does not, the verdict is `dust`, never `fixed`.
Whether those differences are real is a question about your scorer: at 12,345 the
tolerance is 1.2e-5, which is far above float64 rounding, so summed log-probabilities that
differ by 5e-6 come back `dust` here even when they separate the classes perfectly. Both
resolutions are in `--json`.

**What to do about a flagged cell.**

- `fixed`: find why the scorer returns one value there — a fallback or default, a score
  vector discarded before it reached the AUROC, a saturated output. Until it moves, report
  that cell as *not measured*, never as a result.
- `pinned`: the cell cannot support an equivalence claim at that margin, because a pass
  there is decided by the ties, not by the classes. Either the margin is wider than your
  scores can resolve, or the scores need more distinct values. Report it as not
  measurable rather than as matched.
- `dust`: decide whether those differences are signal or rounding, and say which in your
  write-up.

It also reads a `.json` list of objects or a `.jsonl` file, one object per scored item —
the shape leaderboard prediction files usually take — so a submission can be checked as it
was submitted:

```
$ cannotfail resolution predictions.json --score score --fail-on fixed
```

Without `--label` it counts distinct values only, which is enough to find a cell whose
scores are all one number. Use `--positive VALUE` when your labels are words: every row
then needs a label, and that value must occur. `--json` gives machine-readable output, and
`--fail-on fixed pinned` makes CI fail when either occurs. Lines starting with `#` are
skipped only above the header; Excel's UTF-8 files are fine.

The tie count is exact. It costs `O((n + m) log m)` plus one check per pair that falls
within a few rounding units of the tolerance boundary, so it handles score vectors far too
large for an `n × m` comparison.

## D2: could the precondition have failed?

The vocabulary, mapped to yours: an **item** is one scored thing (a run, a prompt, a
sample). A **block** — `--layouts` — is an independent unit your headline averages an AUROC
over: a seed, a dataset, an environment layout. `--per-side` is how many items of each
class are in one block. `--tests` is how many equivalence tests must all pass at once.

**An equivalence precondition**, certified by an equivalence test (TOST):

```
$ cannotfail power equivalence --per-side 4 --layouts 12
SE of the mean AUROC: 0.0625  (null SD 0.2165 over sqrt(12) layouts)
Confidence half-width: 0.1172  against a margin of 0.1
This design cannot certify equivalence at any observed mean: the half-width alone is at least the margin. A pass reported at this design is not a test result.

$ cannotfail power seeds --layouts 12 --tests 4
24
```

That is the benchmark the paper audits: its gate used 4 items a side, and needed 24.

**A precondition asserted from a non-significant difference test** ("no significant
difference, so the arms are matched") is the shape the paper's survey found in 15 of 77
published preconditions. Ask how often the test would have caught the smallest violation
that matters:

```
$ cannotfail power difference --pos 5 --neg 5 --true-auroc 0.7
Power of a two-sided Mann-Whitney test at alpha 0.05 to detect a true AUROC of 0.7 with 5 vs 5: 0.156
Below 0.8: a non-significant result at this design is weak evidence that the two classes are matched.
Fewer than 10 on a side: this is a normal approximation and only indicative.

$ cannotfail power difference --pos 3 --neg 3 --true-auroc 0.9
No data can make a two-sided Mann-Whitney test reject at alpha 0.05 with 3 vs 3: the smallest attainable p-value is 0.100. Power is 0, and a non-significant result here is no evidence of anything.
```

The first figure is a normal approximation. In 20,000 simulations of the exact test at
that design, it rejected 12.2% of the time, so read 0.156 as "about one in seven, perhaps
less". The second is exact.

## From Python

```python
import cannotfail as cf

r = cf.resolution(scores, labels)          # one vector
r.verdict, r.why, r.auroc_range, r.exact_resolution, r.tolerance_resolution

by_cell = cf.resolution_by_group(scores, cells, labels)
cf.summarize(by_cell).n_single_valued      # the paper's class-B count

cf.equivalence_power(n_per_side=24, n_layouts=12).power
cf.difference_power(n_pos=5, n_neg=5, true_auroc=0.7)
cf.min_two_sided_p(3, 3)                   # 0.1: no test at this size can reject at 0.05
```

## What it cannot tell you

- **That a check is sound.** `ok` means D1 found no reason the check cannot fail.
- **Noise from a small spread.** `dust` is a flag for you to judge, not a finding.
- **Real equivalence power.** It assumes no heterogeneity between blocks, so it is a
  ceiling: real blocks vary and real power is lower.
- **Real difference power for clustered items.** If several items share a prompt, task or
  seed, the real test is mis-sized in either direction and the figure is not a bound.
- **Anything about D3.** Whether a parameter is wired to anything the evaluation reads.

## Where it comes from, and how it is held to it

D1's tolerance rule and the equivalence half of D2 are lifted from the admission gate of
[RHOB](https://github.com/Aarav500/rhob), the benchmark the paper uses as its case study.
Because this package imports nothing from RHOB, it is a copy, and the RHOB repository's
`tests/test_cannotfail_parity.py` holds the copy to the original:

- The tolerance resolution is bit-identical to the gate's on adversarial inputs, with one
  documented exception. A pair of finite scores whose difference overflows float64
  (`|a − b|` above about 1.8e308) is tied in the gate, through the overflow, and ordered
  here.
- The design arithmetic (null SD, the t multiplier, required seeds) matches at every
  design point tested.
- The verdict cuts agree with the gate's on both sides of its resolution and
  informativeness boundaries.
- On RHOB's 50 committed admission-ledger cells, the verdict rule fed the gate's *measured*
  resolution and informativeness marks the same 30 degenerate. That checks the rule's
  direction, not its cut, because the ledger's values sit far from it (the boundary tests
  above pin the cut). It also flags `onset_localizable`, which the gate passes on every
  cell, as `fixed`: its statistic is 0.0 in all 50.

The difference-power function, the exact-equality resolution and the `dust` verdict are
new here, and are tested against brute force and simulation in this package's own tests.

## Cite

```bibtex
@misc{shah2026cannotfail,
  title={Checks That Cannot Fail: Unfalsifiable Construction Claims in Benchmarks, and Three Diagnostics That Find Them},
  author={Shah, Aarav},
  year={2026},
  howpublished={Zenodo},
  doi={10.5281/zenodo.22005218},
  url={https://doi.org/10.5281/zenodo.22005218}
}
```

MIT licence.
