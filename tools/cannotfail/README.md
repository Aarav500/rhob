# cannotfail

A benchmark's headline usually rests on a claim about its own construction — *the two
arms are matched*, *this control is at chance* — and a check in the harness guards it.
`cannotfail` finds guards that **cannot return a negative verdict** under the design that
ships. Such a check is green in CI and looks exactly like one that was earned.

It implements two of the three diagnostics from *Checks That Cannot Fail* (Shah, 2026,
[doi:10.5281/zenodo.22005218](https://doi.org/10.5281/zenodo.22005218)):

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
reward_only       48         1      24      24       0.000   [0.500, 0.500]  fixed
mostly_tied       48         2      24      24       0.042   [0.479, 0.521]  pinned
behavioural       48        48      24      24       1.000   [0.000, 1.000]  ok

1 of 3 groups have a score vector with a single distinct value: any rank statistic over them is fixed.
1 of 3 groups are pinned: their AUROC cannot leave 0.5 +/- 0.1, so an equivalence test at that margin cannot fail there.
```

That output is from [`examples/scores.csv`](examples/scores.csv). The idea behind it: every
tied cross-class pair contributes exactly 0.5 to an AUROC, whatever the data, so if a
fraction `1 - r` of pairs is tied, the AUROC is confined to `0.5 ± r/2`. That is an
identity, not an estimate.

| Verdict | Meaning |
|---|---|
| `fixed` | The statistic cannot move: one distinct value, no ordered pair, or an empty class. |
| `pinned` | The AUROC cannot leave `0.5 ± margin`, so an equivalence test at that margin passes whatever the data are. |
| `dust` | Only with `--dispersion-floor`: the scores differ, but by less than that fraction of their own size. |
| `ok` | D1 found no reason the check cannot fail. That is not evidence it is sound. |

Without `--label` it counts distinct values only, which is enough to find a cell whose
scores are all one number. Use `--positive VALUE` if your labels are not 0/1, `--json` for
machine-readable output, and `--fail-on fixed pinned` to make CI fail when either occurs.

Two scores count as tied when `|a − b| ≤ max(1e-12, 1e-9 · max(|a|, |b|))`, or when either
is NaN or infinite. The tolerance absorbs rounding error in a scorer's own arithmetic;
it is not meant to decide whether a weak signal is real. The count is exact and runs in
`O((n + m) log m)`, so it handles score vectors far too large for an `n × m` comparison.

## D2: could the precondition have failed?

**An equivalence precondition**, certified by an equivalence test (TOST):

```
$ cannotfail power equivalence --per-side 4 --layouts 12
SE of the mean AUROC: 0.0625  (null SD 0.2165 over sqrt(12))
Confidence half-width: 0.1172  against a margin of 0.1
This design cannot certify equivalence at any observed mean: the half-width alone is at least the margin. A pass reported at this design is not a test result.

$ cannotfail power seeds --layouts 12 --tests 4
24
```

That is the benchmark the paper audits: its gate used 4 runs a side, and needed 24.

**A precondition asserted from a non-significant difference test** ("no significant
difference, so the arms are matched") is the shape the paper's survey found in 15 of 77
published preconditions. Ask how often the test would have caught the smallest violation
that matters:

```
$ cannotfail power difference --pos 5 --neg 5 --true-auroc 0.7
Power of a two-sided Mann-Whitney test at alpha 0.05 to detect a true AUROC of 0.7 with 5 vs 5: 0.156
Below 0.8: a non-significant result at this design is weak evidence that the two classes are matched.
Fewer than 10 on a side: this is a normal approximation and only indicative.
```

## From Python

```python
import cannotfail as cf

r = cf.resolution(scores, labels)          # one vector
r.verdict, r.auroc_range, r.n_distinct

by_cell = cf.resolution_by_group(scores, cells, labels)
cf.summarize(by_cell).n_single_valued      # the paper's class-B count

cf.equivalence_power(n_per_side=24, n_layouts=12).power
cf.difference_power(n_pos=5, n_neg=5, true_auroc=0.7)
```

## What it cannot tell you

- **That a check is sound.** `ok` means D1 found no reason the check cannot fail.
- **Noise from a small spread.** `dispersion` is `SD / mean(|x|)`, so scores near 1000 that
  differ by 0.001 read the same as rounding noise around 1. That is why `dust` is off
  unless you set a floor you can defend for your scores.
- **Real power.** Both power figures assume no heterogeneity between layouts or clusters,
  so they are a ceiling. The difference power is a normal approximation, indicative below
  10 a side.
- **Anything about D3.** Whether a parameter is wired to anything the evaluation reads.

## Where it comes from, and how it is held to it

D1 and the equivalence half of D2 are lifted from the admission gate of
[RHOB](https://github.com/Aarav500/rhob), the benchmark the paper uses as its case study.
Because this package deliberately imports nothing from RHOB, it is a copy, and the RHOB
repository's `tests/test_cannotfail_parity.py` holds the copy to the original. Resolution
is bit-identical on adversarial inputs. The design arithmetic matches on every design point
tested. And fed the gate's own measurements, the verdicts reproduce the gate's outcome on
all 50 cells of RHOB's committed admission ledger, including the criterion that passes all
50 because its statistic is exactly 0.0 in every one.

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
