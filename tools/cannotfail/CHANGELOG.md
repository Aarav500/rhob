# Changelog

All notable changes to `cannotfail` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). A release is the tag
`cannotfail-vX.Y.Z` in the RHOB repository; how one is cut is in
[docs/RELEASING_CANNOTFAIL.md](https://github.com/Aarav500/rhob/blob/main/docs/RELEASING_CANNOTFAIL.md).
The RHOB benchmark keeps its own changelog, at the
[repository root](https://github.com/Aarav500/rhob/blob/main/CHANGELOG.md).

## [0.1.0] - Unreleased

The first release on PyPI. It implements D1 (resolution) and D2 (power) from *Checks That
Cannot Fail* (Shah, 2026, [doi:10.5281/zenodo.22005218](https://doi.org/10.5281/zenodo.22005218)).
D3 is not included: it needs knowledge of a harness that a score file does not carry.
Before this release the package was installable only with `git+https` from the RHOB
repository, where it has lived in `tools/cannotfail` since 2026-10-31.

### Added

- `cannotfail resolution` (D1), NumPy only. It reads a CSV, a TSV, a `.json` list of
  objects or a `.jsonl` file, and with an extra a Parquet file or an Inspect log, one row
  per scored item, with an optional class label and optional cells. For each cell it
  reports the distinct values and, given labels, the resolution `r` (the fraction of
  cross-class pairs that are strictly ordered), the AUROC
  range `0.5 ± r/2` that `r` allows, and a verdict: `fixed`, `pinned`, `dust` or `ok`.
  `ok` means D1 found no reason the check cannot fail; it is not evidence that the check
  is sound.
- Nested fields by dotted path, in JSON, JSONL and Parquet: `--score scores.detector.value`,
  where an integer segment indexes a list (`output.probs.1`) and a key that itself contains a
  dot is matched whole first. A path that resolves in some records and not others is blank
  in the others; one that resolves in none exits 2, naming the path and the keys at the
  deepest level it reached. CSV and TSV columns stay literal header names.
- Several grouping columns: `--group` can be repeated (`--group task --group model`), and is
  never split on commas, because metric names such as `acc,none` contain them. A group's
  label is its values joined by ` | `; with `--json` each group also lists its values by
  column under `"group"`, and the columns are listed once under `"group_by"`. Where a value
  contains ` | ` and two labels would coincide, every label is written as a JSON list. A
  single `--group` over scalar values prints as it did before the flag could be repeated,
  except that in JSON and JSONL a dotted name now also reaches the nested field in a record
  that has no key of that literal name.
- A score, label or group field in JSON, JSONL or Parquet that holds a list or an object is
  now refused (exit 2) with the path to index it by. A group field used to be grouped by its
  Python repr; a score field already exited 2.
- The extension is matched in any letter case, and `.ndjson` and `.pq` are recognised.
  Before, only lower-case `.json`, `.jsonl`, `.tsv` and `.tab` were recognised, and every
  other file, `.parquet` and `.eval` included, was read as comma-separated CSV.
- `--format csv|tsv|json|jsonl|parquet|inspect`, which overrides the extension. An `.eval`
  file without `--format inspect` exits 2 and says to pass it.
- Parquet, through the `parquet` extra (pyarrow 16 or later). A struct column is read by the
  dotted path it would have in JSON, and only the top-level columns the paths start in are
  read, unless one of those names contains a dot. Without pyarrow, a `.parquet` file exits 2 with the install line.
- Inspect logs, through the `inspect` extra (inspect_ai 0.3.179 or later, pandas 2.2.2,
  pyarrow 16) and `--format inspect`: a log or a directory of them, read by
  `inspect_ai.analysis.samples_df`, with the task and model columns. A single local log is
  read whatever its name; a local directory holding a log that `samples_df` would skip,
  under any name, exits 2 and names it. A URL is passed to `samples_df` as it is. In the
  score column
  Inspect's letter grades become the numbers its metrics average (`C` 1, `I` 0, `P` 0.5,
  `N` 0); any other string that is not a number is refused rather than made 0. Tested
  against `examples/hack_probe.eval`, a real log written offline by inspect_ai 0.3.290 with
  its mock model, and on Python 3.10 against inspect_ai 0.3.179 and 0.3.280.
- Recipes in the README for an lm-evaluation-harness `--log_samples` file, a leaderboard
  predictions JSON, a Parquet file and an Inspect log, each run on a file in `examples/`.
  The lm-evaluation-harness and leaderboard files are synthetic. Per-sample accuracy is 0 or
  1, so on such a file without a class label D1 can flag only `fixed`.
- Two resolutions per cell. The primary one counts ties by exact equality, which is what
  `sklearn.metrics.roc_auc_score` ranks. The other uses the tie tolerance of RHOB's
  admission gate, `|a − b| ≤ max(1e-12, 1e-9 · max(|a|, |b|))`. When only the tolerance
  would confine the AUROC, the verdict is `dust`, never `fixed`.
- Exact tie counting in `O((n + m) log m)`, plus one check per pair near the tolerance
  boundary, so score vectors too large for an `n × m` comparison can be checked.
- `cannotfail power equivalence` and `cannotfail power seeds` (D2): whether an equivalence
  test (TOST) on a mean AUROC over independent blocks can pass at a given design, how
  often, and how many items a side it needs. They assume no heterogeneity between blocks,
  so the power is a ceiling. They need SciPy, installed by the `power` extra.
- `cannotfail power difference` (D2), NumPy only: the power of a two-sided Mann-Whitney
  test to detect a given true AUROC, by normal approximation, and 0 where no data can make
  the exact test reject (3 against 3 at alpha 0.05). Any design size works. Where
  C(n + m, n) fits in a float, `min_two_sided_p`, the smallest p-value the exact test can
  return, is computed as before; past that, from 515 a side, it is the exact ratio rounded
  once, a subnormal float and 0.0 from 541 a side, which the logarithm settles without
  building the integer. Before 2026-11-16 it raised OverflowError there. At a true AUROC of
  exactly 0 or 1 the power is 1 wherever the exact test can reject at all; it used to
  follow the normal threshold, which at 1 vs N lies past 0.5 and gave 0.
- For CI: `--json` (strict JSON), `--fail-on fixed pinned dust` and
  `--fail-on-cannot-certify`. Exit status 0 on success, 1 when a named verdict occurred or
  the design cannot certify, 2 on bad input or a missing optional dependency, and 3 on an
  internal error.
- A Python API: `resolution`, `resolution_by_group`, `summarize`, `equivalence_power`,
  `required_seeds_per_layout`, `difference_power`, `min_two_sided_p` and the helpers they
  use.
- Packaging for PyPI: project URLs and classifiers, a README whose links are absolute, an
  sdist with the licence, README, changelog, tests and examples, and a wheel that holds the
  `cannotfail` package alone. The base install is NumPy alone, and the extras are imported
  only when used. The package has its own CI,
  [`cannotfail.yml`](https://github.com/Aarav500/rhob/blob/main/.github/workflows/cannotfail.yml),
  separate from the benchmark's, and is released by PyPI Trusted Publishing from
  [`cannotfail-release.yml`](https://github.com/Aarav500/rhob/blob/main/.github/workflows/cannotfail-release.yml).

D1's tolerance rule and the equivalence half of D2 are copies of RHOB's admission gate.
RHOB's
[`tests/test_cannotfail_parity.py`](https://github.com/Aarav500/rhob/blob/main/tests/test_cannotfail_parity.py)
holds them to it, with one documented exception: a finite pair whose difference overflows
float64 is tied in the gate and ordered here.

[0.1.0]: https://github.com/Aarav500/rhob/tree/cannotfail-v0.1.0/tools/cannotfail
