# cannotfail

[![cannotfail CI](https://github.com/Aarav500/rhob/actions/workflows/cannotfail.yml/badge.svg)](https://github.com/Aarav500/rhob/actions/workflows/cannotfail.yml)

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
pip install cannotfail              # D1
pip install "cannotfail[power]"     # with D2
pip install "cannotfail[parquet]"   # to read Parquet files (adds pyarrow)
pip install "cannotfail[inspect]"   # to read Inspect logs (adds inspect_ai, pandas, pyarrow)
```

Python 3.10 or later. The package imports nothing from the benchmark it was built in, and
the base install needs nothing but NumPy: the extras are imported only when they are used.

Or let [uv](https://docs.astral.sh/uv/) or [pipx](https://pipx.pypa.io/) fetch it into a
temporary environment and run it:

```bash
uvx cannotfail resolution scores.csv --score score
pipx run cannotfail resolution scores.csv --score score
uvx --from "cannotfail[power]" cannotfail power seeds --layouts 12 --tests 4   # D2 needs the extra
```

If pip or uv finds no release yet, or for the development version, install from the
repository:

```bash
pip install "cannotfail @ git+https://github.com/Aarav500/rhob#subdirectory=tools/cannotfail"
```

The badge above is the package's own CI. It runs the tests on Python 3.10, 3.13 and 3.14,
first with NumPy alone, which it checks by asserting that SciPy, pyarrow, pandas and
inspect_ai are absent, then adding SciPy, then pyarrow, then the `inspect` extra. It builds
the sdist and wheel and, from the wheel alone, runs each example on this page that shows its
output and compares the two. It also runs weekly, so a NumPy, SciPy, pyarrow, pandas or
inspect_ai release that breaks a test or an example turns it red with no change here. Green says the tool's tests pass and its examples print what this page shows. It
says nothing about your checks.

If you run it on your own benchmark, please tell the author, whatever it found, `ok`
included: [a short form](https://github.com/Aarav500/rhob/issues/new?template=cannotfail_run.yml),
or email aarav7.shah@gmail.com if the result should not be public.

## D1: run it on your score vectors

Give it a CSV or TSV with one row per scored item: the score, optionally its class, and
optionally the cell it belongs to (a dataset, a detector-by-task pair, whatever your
headline aggregates over). To try it on the example file first:

```bash
curl -O https://raw.githubusercontent.com/Aarav500/rhob/main/tools/cannotfail/examples/scores.csv
```

```
$ cannotfail resolution scores.csv --score score --label is_hack --group cell
group              n  distinct     pos     neg  resolution      AUROC range  verdict
reward_only       48         1      24      24       0.000   [0.500, 0.500]  fixed (one distinct value)
mostly_tied       48         2      24      24       0.042   [0.479, 0.521]  pinned (AUROC confined within 0.5 +/- 0.0208)
behavioural       48        48      24      24       1.000   [0.000, 1.000]  ok

1 of 3 groups has at most one distinct value, so any rank statistic over it is fixed at 0.5.
1 of 3 groups is pinned: the AUROC cannot get further from 0.5 than 0.1, so its estimate cannot leave an equivalence band of that width. A test that also needs its interval inside the band can still fail, but only on the interval's width; a pass there says the scores tie.
```

That output is from
[`examples/scores.csv`](https://github.com/Aarav500/rhob/blob/main/tools/cannotfail/examples/scores.csv).
The idea behind it: in a standard AUROC — the one `sklearn.metrics.roc_auc_score`
computes — every cross-class pair of equal scores contributes exactly 0.5, whatever the
data. So if a fraction `1 − r` of the pairs are equal, the AUROC is confined to
`0.5 ± r/2`. That is an identity, not an estimate. `resolution` is that `r`, over the
finite scores; blank or non-finite scores are left out and counted in the footer.

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

It also reads a `.json` list of objects, a `.jsonl` file, Parquet and Inspect logs, where a
nested field is named by a dotted path (`--score output.probs.1`), and `--group` can be given
more than once. The Recipes section below has one tested command for each.

Without `--label` it counts distinct values only, which is enough to find a cell whose
scores are all one number. Use `--positive VALUE` when your labels are words: every row
then needs a label, and that value must occur. `--json` gives machine-readable output, and
`--fail-on fixed pinned` makes CI fail when either occurs. Lines starting with `#` are
skipped only above the header; Excel's UTF-8 files are fine.

The tie count is exact. It costs `O((n + m) log m)` plus one check per pair that falls
within a few rounding units of the tolerance boundary, so it handles score vectors far too
large for an `n × m` comparison.

## Recipes: the files you already have

Each command below runs in [`examples/`](https://github.com/Aarav500/rhob/tree/main/tools/cannotfail/examples)
and prints exactly what is shown under it; the package's CI checks that it still does. The
data in every file there is made up. The Inspect log is real in the sense that inspect_ai
itself wrote it, with a mock model.

**Naming a column.** In CSV and TSV it is the header's name, taken literally, dots and all.
In JSON, JSONL and Parquet it is a dotted path: `scores.detector.value` is the field `value`
inside `detector` inside `scores`, and an integer indexes a list, so `output.probs.1` is the
second probability. A key that itself contains a dot is matched whole before the path is
split there. A path that resolves in some records and not others is blank in the others (a
blank score is left out and counted in the footer). One that resolves in none exits 2,
naming the path and the keys at the deepest level it reached. In an Inspect log a column is
one of `samples_df`'s, such as `score_includes`.

**Several groups.** Repeat `--group` to group by several columns:
`--group task --group model`. A comma never separates them, because names such as
`acc,none` contain commas. Each group's label is its values joined by ` | `, in the order
the flags were given, and the first column of the table is headed by the column names joined
the same way. With `--json` the groups are keyed by those labels, and each also carries its
values by column under `"group"`, with the columns listed once under `"group_by"`. If a value
itself contains ` | `, so that two groups would print alike, every label is written as a JSON
list instead. A single `--group` over scalar values prints the table it did before `--group`
could be repeated, except that in JSON and JSONL a dotted name now also reaches the nested
field in a record that has no key of that literal name. In JSON, JSONL and Parquet, a score,
label or group field that holds a list or an object is refused, with the path to index it
by. An Inspect log's `samples_df` gives such a field as JSON text, and a group is that text.

**The format** comes from the extension, in any letter case (`.json`; `.jsonl` or `.ndjson`;
`.parquet` or `.pq`; `.tsv` or `.tab`; and CSV otherwise). `--format` overrides it, and an Inspect log always needs `--format inspect`.

### lm-evaluation-harness `--log_samples`

The harness writes one JSONL file per task, `samples_<task>_<timestamp>.jsonl`, with a line
per document and answer filter and each per-sample metric as a top-level field.
[`examples/samples_toy_arith.jsonl`](https://github.com/Aarav500/rhob/blob/main/tools/cannotfail/examples/samples_toy_arith.jsonl)
is a synthetic file in that shape, as lm-evaluation-harness 0.4.13 builds it: twelve made-up
arithmetic questions on a `generate_until` task with two answer filters, `strict-match`,
which wants `#### <answer>`, and `flexible-extract`, which takes the last number. The
filtered answers and `exact_match` are computed as the harness's regex filter and
exact-match metric compute them. The made-up responses never write `#### <answer>`, so
`strict-match` extracts nothing:

```
$ cannotfail resolution samples_toy_arith.jsonl --score exact_match --group filter
group                   n  distinct  verdict
strict-match           12         1  fixed (one distinct value)
flexible-extract       12         2  ok

1 of 2 groups has at most one distinct value, so any rank statistic over it is fixed at 0.5.
No --label given, so only distinct values were counted; pass one for resolution.
```

Per-sample `exact_match`, like `acc`, is 0 or 1, and the file holds no class label, so D1
can only count distinct values here, and the one thing it can flag is `fixed`: a cell in
which every document scored the same. `strict-match` is 0.0 on all twelve because the filter
never matched, and no change in which answers were right could have moved it. `ok` means
only that both values occur. Fields of the dataset row are under `doc`, so
`--group doc.<field>` groups by one where the dataset has it.

### A leaderboard predictions JSON

[`examples/predictions.json`](https://github.com/Aarav500/rhob/blob/main/tools/cannotfail/examples/predictions.json)
is a synthetic submission: a list of objects, one per item, with the positive-class
probability second in `output.probs` and the class in `gold.is_hack`, for two tasks by two
models:

```
$ cannotfail resolution predictions.json --score output.probs.1 --label gold.is_hack --group task --group model
task | model          n  distinct     pos     neg  resolution      AUROC range  verdict
qa | model-a         12        12       6       6       1.000   [0.000, 1.000]  ok
qa | model-b         12         1       6       6       0.000   [0.500, 0.500]  fixed (one distinct value)
code | model-a       12        12       6       6       1.000   [0.000, 1.000]  ok
code | model-b       12         2       6       6       0.167   [0.417, 0.583]  pinned (AUROC confined within 0.5 +/- 0.0833)

1 of 4 groups has at most one distinct value, so any rank statistic over it is fixed at 0.5.
1 of 4 groups is pinned: the AUROC cannot get further from 0.5 than 0.1, so its estimate cannot leave an equivalence band of that width. A test that also needs its interval inside the band can still fail, but only on the interval's width; a pass there says the scores tie.
```

`model-b` returns 0.5 on every `qa` item, as a fallback that never moved would. On `code`
it leaves the fallback for one item, so 6 of its 36 cross-class pairs are ordered and its
AUROC cannot get further from 0.5 than 0.083, whatever the items are. In CI, add
`--fail-on fixed pinned`. If your file is one object holding the list, the error names the
key that holds it.

### Parquet

Needs `pip install "cannotfail[parquet]"`; without pyarrow, a `.parquet` file exits 2 with
that line.
[`examples/scores.parquet`](https://github.com/Aarav500/rhob/blob/main/tools/cannotfail/examples/scores.parquet)
holds the 144 rows of `scores.csv` with the score inside a struct column, `detector`, so it
prints the same table:

```
$ cannotfail resolution scores.parquet --score detector.score --label is_hack --group cell
group              n  distinct     pos     neg  resolution      AUROC range  verdict
reward_only       48         1      24      24       0.000   [0.500, 0.500]  fixed (one distinct value)
mostly_tied       48         2      24      24       0.042   [0.479, 0.521]  pinned (AUROC confined within 0.5 +/- 0.0208)
behavioural       48        48      24      24       1.000   [0.000, 1.000]  ok

1 of 3 groups has at most one distinct value, so any rank statistic over it is fixed at 0.5.
1 of 3 groups is pinned: the AUROC cannot get further from 0.5 than 0.1, so its estimate cannot leave an equivalence band of that width. A test that also needs its interval inside the band can still fail, but only on the interval's width; a pass there says the scores tie.
```

A struct is read as an object and a list column as a list, so a nested field has the path it
would have in JSON. Only the top-level columns the paths start in are read, unless one of
those names itself contains a dot, when the whole file is read and the rest dropped.
Not read: a partitioned dataset directory, as Spark or `pyarrow.parquet.write_to_dataset`
writes (point it at one file), and map columns (convert them to structs first).

### Inspect logs

Needs `pip install "cannotfail[inspect]"`, which adds inspect_ai, pandas and pyarrow.
inspect_ai 0.3.281 and later need Python 3.11, so on 3.10 pip installs 0.3.280. With
`--format inspect`, the file, or a directory of logs, is read by inspect_ai's own
`samples_df`, one row per sample and epoch. Its columns are named as `samples_df` names
them: `score_<scorer>`, `metadata_<field>`, `id` and `epoch`, and from the eval `task_name`
and `model`. `samples_df` reads a log only under a name Inspect itself would give it: a
lower-case `.eval`, or a `.json` whose name starts with the run's timestamp. A single file
is read whatever its name, by loading it with `read_eval_log` first. A directory holding a
renamed log exits 2 and names the file, rather than giving a verdict on the logs it could
read.
[`examples/hack_probe.eval`](https://github.com/Aarav500/rhob/blob/main/tools/cannotfail/examples/hack_probe.eval)
is a real log, written offline by inspect_ai 0.3.290 with its `mockllm/model` given a fixed
reply per sample, by
[`examples/hack_probe.py`](https://github.com/Aarav500/rhob/blob/main/tools/cannotfail/examples/hack_probe.py).
The probe states a probability, and on family `b` it says 0.5 every time:

```
$ cannotfail resolution hack_probe.eval --format inspect --score score_stated_probability --label metadata_is_hack --group metadata_family
group        n  distinct     pos     neg  resolution      AUROC range  verdict
a            4         4       2       2       1.000   [0.000, 1.000]  ok
b            4         1       2       2       0.000   [0.500, 0.500]  fixed (one distinct value)

1 of 2 groups has at most one distinct value, so any rank statistic over it is fixed at 0.5.
```

In the score column, Inspect's letter grades become what its own metrics average: `C` 1,
`I` 0, `P` 0.5 and `N` 0, and a boolean 1 or 0. Any other string that is not a number is an
error, where Inspect's `value_to_float` would quietly make it 0. Across a directory of logs:

```bash
cannotfail resolution logs/ --format inspect --score score_includes --group task_name --group model
```

For a column `samples_df` gives only with other arguments, export the frame yourself, with
`samples_df(...).to_csv("samples.csv")`, and run `cannotfail` on the CSV.

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

The first figure is a normal approximation. Simulating the exact test at that design, with
unit-variance normal scores in both classes and the positive class shifted by
√2·Φ⁻¹(0.7) ≈ 0.74 so that the true AUROC is 0.7, it rejects about 12% of the time, so read
0.156 as "about one in seven, perhaps less". The second is exact.

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
[`tests/test_cannotfail_parity.py`](https://github.com/Aarav500/rhob/blob/main/tests/test_cannotfail_parity.py)
holds the copy to the original:

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

The tie tolerance has a third copy, in the RHOB repository's Inspect eval:
`score_resolution` in
[`inspect_eval/hvtb_hack_detection/metrics.py`](https://github.com/Aarav500/rhob/blob/main/inspect_eval/hvtb_hack_detection/metrics.py),
with the same 1e-9 and 1e-12. It treats an overflowing finite pair as tied, as the gate does, and no test
holds it to either of the other two. A change to the gate's tolerance has to be applied to
all three.

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
