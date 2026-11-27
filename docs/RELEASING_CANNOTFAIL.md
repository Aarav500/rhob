# Releasing `cannotfail` to PyPI

[`tools/cannotfail`](../tools/cannotfail) is released to PyPI by
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/): PyPI is told once to
trust one workflow file in this repository, and that workflow proves who it is with a
short-lived OIDC token from GitHub. No API token is created or stored. The workflow is
[`.github/workflows/cannotfail-release.yml`](../.github/workflows/cannotfail-release.yml):

| Trigger | What it does |
|---|---|
| push of a tag `cannotfail-vX.Y.Z` | build, then TestPyPI, then PyPI |
| Actions → cannotfail-release → Run workflow | build, then TestPyPI. A dry run: it never reaches PyPI. |

The steps below need the owner's PyPI, TestPyPI and GitHub accounts. Nothing in this
repository can do them. As of 2026-11-15, `cannotfail` is on neither index:
`https://pypi.org/pypi/cannotfail/json` and `https://test.pypi.org/pypi/cannotfail/json`
both return 404, and so do `cannot-fail`, `cannot_fail`, `cannot.fail`, `cann0tfail` and
`cannotfai1`, spellings PyPI treats as the same name or as too similar to it.

## First release: 0.1.0, in this order

The release workflow refuses a tag whose commit is not on `main`, so nothing can be
uploaded before PR #2 is merged. And from the merge on, the READMEs and the homepage lead
with `pip install cannotfail`. They fall back to the `git+https` line when pip finds no
release, but until 0.1.0 is up, anyone can register the name. So steps 1 and 2 are done
before the merge, and steps 3 to 6 in one sitting.

### 1. Create the two GitHub environments

In the repository: Settings → Environments → New environment.

- `testpypi`. No rules are needed.
- `pypi`. Optionally, tick **Required reviewers** and add yourself. The PyPI job then
  waits for your approval in the Actions tab before it uploads anything. If you are the
  only reviewer, leave "Prevent self-review" unticked, or nobody can approve.
- Optionally, on both: Deployment branches and tags → **Selected branches and tags** →
  add a tag rule `cannotfail-v*`, and on `testpypi` also the branch `main`, so no other
  ref can deploy to them.

A workflow that names a missing environment creates it on first use, with no rules.
Creating them first is what lets you add the reviewer before anything runs.

### 2. Label run reports, and date the changelog

- Issues → Labels → **New label** → `cannotfail-run`. The "Report a cannotfail run" form
  applies it, and GitHub silently drops a label that does not exist, so without it run
  reports arrive unlabelled.
- [`tools/cannotfail/CHANGELOG.md`](../tools/cannotfail/CHANGELOG.md) says
  `## [0.1.0] - Unreleased`. On the PR #2 branch, change it to the day you will publish,
  for example `## [0.1.0] - 2026-11-16`. The release workflow refuses a tag whose
  changelog entry has no date; it does not check which date. Dating it inside the PR means
  the merge commit is the commit the dry run and the tag both build.

### 3. Add a pending trusted publisher on TestPyPI and on PyPI

TestPyPI is a separate site with separate accounts. Both need a verified email address,
and PyPI requires two-factor authentication on the account.

**TestPyPI.** Sign in at <https://test.pypi.org> (register first if needed), open
<https://test.pypi.org/manage/account/publishing/> (account menu → Publishing), choose
the **GitHub** tab under "Add a new pending publisher", and enter exactly:

| Field | Value |
|---|---|
| Project Name | `cannotfail` |
| Owner | `Aarav500` |
| Repository name | `rhob` |
| Workflow name | `cannotfail-release.yml` |
| Environment name | `testpypi` |

Then **Add**.

**PyPI.** The same at <https://pypi.org/manage/account/publishing/>, with one change:

| Field | Value |
|---|---|
| Project Name | `cannotfail` |
| Owner | `Aarav500` |
| Repository name | `rhob` |
| Workflow name | `cannotfail-release.yml` |
| Environment name | `pypi` |

The workflow name is the file name alone, not its path. The environment names must match
the `environment:` lines in the workflow. That is also a second lock on the dry run: its
upload job runs in the `testpypi` environment and PyPI trusts only `pypi`, so even an
upload to PyPI from that job by mistake would be refused.

**A pending publisher does not reserve the name.** PyPI's documentation: a pending
publisher "does **not** create a project or reserve a project's name **until** it is
actually used to publish. If you create a 'pending' publisher but another user registers
the project name before you actually publish to it, your 'pending' publisher will be
**invalidated**." The two indexes are separate namespaces, so the name can be lost on
one and not the other. Do steps 3 to 6 in one sitting: add the publishers, merge, dry-run and
tag, without a day in between.

### 4. Merge PR #2

As of 2026-11-15, `main` has no branch protection and no required checks, so moving the
`cannotfail` job out of `tests.yml` blocks nothing. Do not make the new
`test (Python 3.10)`, `test (Python 3.13)`, `test (Python 3.14)` or `package` checks
required. They run only when `tools/cannotfail/**` or their own workflow file changes,
and a required check that does not run blocks the merge. Wait for `cannotfail.yml` to pass
on `main`.

### 5. Dry run to TestPyPI

Actions → **cannotfail-release** → Run workflow → branch `main` → Run workflow. Or:

```bash
gh workflow run cannotfail-release.yml --ref main
```

Expect `build` and `publish-testpypi` to pass and `publish-pypi` to be skipped. `build`
checks that `pyproject.toml` and `src/cannotfail/__init__.py` agree on the version and
that CHANGELOG.md has an entry for it, builds the sdist and wheel, runs
`twine check --strict`, checks that the wheel holds the `cannotfail` package alone, and
runs the tool's tests against the wheel it is about to upload, installed with every extra
so that the Parquet and Inspect tests run rather than skip.

Then install from TestPyPI in a fresh venv and run the README's first example. Take
`cannotfail` alone from TestPyPI, with `--no-deps`, and its dependencies from PyPI: with
`--extra-index-url`, pip would take any package from whichever index has it, and anyone
can upload to TestPyPI under a name like `numpy`.

```bash
python3 -m venv /tmp/cannotfail-testpypi
/tmp/cannotfail-testpypi/bin/pip install "numpy>=1.24,<3" "scipy>=1.10,<2"
/tmp/cannotfail-testpypi/bin/pip install --no-deps --index-url https://test.pypi.org/simple/ cannotfail==0.1.0
/tmp/cannotfail-testpypi/bin/cannotfail --version            # cannotfail 0.1.0
cd tools/cannotfail/examples
/tmp/cannotfail-testpypi/bin/cannotfail resolution scores.csv --score score --label is_hack --group cell
/tmp/cannotfail-testpypi/bin/cannotfail power seeds --layouts 12 --tests 4      # 24
```

The resolution output must match the README's. Also open
<https://test.pypi.org/project/cannotfail/> and check that the README renders and that the
links in the sidebar work. TestPyPI's index can take a minute to show a new upload.

Running the dry run again for the same version does not fail: the TestPyPI upload is
told to skip files TestPyPI already has. That also means TestPyPI keeps the first files
it was given for a version: if the dry run shows something to fix, the fix will not
appear there under 0.1.0, though the tag run still uploads the fixed build to PyPI. To
rehearse the fix itself on TestPyPI, give it a new version number.

### 6. Push the tag

On `main`'s head, which should be the commit the dry run built:

```bash
git switch main && git pull
git tag -a cannotfail-v0.1.0 -m "cannotfail 0.1.0"
git push origin cannotfail-v0.1.0
```

The run builds once and uploads that one build to TestPyPI and then to PyPI. TestPyPI
already has 0.1.0 from the dry run, so that upload skips the existing files and the job
passes; PyPI gets this build. If you set a reviewer on `pypi`, approve the deployment in
the run's page.

Then, in a fresh venv:

```bash
pip install cannotfail==0.1.0
cannotfail --version                                    # cannotfail 0.1.0
uvx cannotfail@0.1.0 --version                          # the no-install route
```

Check <https://pypi.org/project/cannotfail/>. Once PyPI has it, the pending publisher
becomes an ordinary one on the project, and the project is yours. Nothing else has to change. The top-level
[README.md](../README.md), the package README and the homepage already lead with
`pip install cannotfail`, with the `git+https` line beside it for when pip finds no
release, so they read true before and after this step.

**If the tag run fails before an upload** (the version check, the build, the tests),
nothing was published. Fix the cause on `main`, then move the tag:

```bash
git tag -d cannotfail-v0.1.0
git push origin :refs/tags/cannotfail-v0.1.0
git tag -a cannotfail-v0.1.0 -m "cannotfail 0.1.0" && git push origin cannotfail-v0.1.0
```

**If an upload step fails** for a reason outside the repository (an index outage, an
expired approval), re-run the failed jobs from the run's page. The built files are kept
as the run's artifact, so the same files are uploaded. PyPI accepts the two files one at
a time, though, and the PyPI job does not skip existing files. If PyPI already lists one
of the two, a re-run fails on it. Download the run's artifact and upload the missing file
with `twine upload`, or re-run once with `skip-existing: true` set on the PyPI job.

**If something is wrong after PyPI has the files**, do not try to replace them. PyPI
never accepts a file name it has seen before, even after the release is deleted. Fix it
in a new version, 0.1.1, as below, and yank 0.1.0 on PyPI if it should not be installed.

## Later releases: 0.1.1 and on

1. In a pull request, with the change itself:
   - set the new version in both `tools/cannotfail/pyproject.toml` (`version`) and
     `tools/cannotfail/src/cannotfail/__init__.py` (`__version__`). The release workflow
     refuses a build where they differ;
   - add an entry at the top of `tools/cannotfail/CHANGELOG.md`, dated, such as
     `## [0.1.1] - 2026-12-01`, with `### Fixed`, `### Changed` or `### Added` beneath
     it, and a link line at the bottom:
     `[0.1.1]: https://github.com/Aarav500/rhob/compare/cannotfail-v0.1.0...cannotfail-v0.1.1`;
   - if any output the README shows has changed, copy the new output from the tool, not
     by hand. The `package` job in `cannotfail.yml` runs every README example that shows
     its output and fails on any difference.
2. Merge, and wait for `cannotfail.yml` to pass on `main`.
3. Optionally, run the dry run (step 5 above) for the new version.
4. Tag that commit `cannotfail-v0.1.1` and push the tag (step 6 above).

The versions follow [Semantic Versioning](https://semver.org/), as the changelog says. A
change to what a verdict means, to an exit status, or to the `--json` fields is not a
patch release: someone's CI reads them.

Neither the PyPI nor the TestPyPI setup is repeated. The publishers stay attached to the
project, and they name the workflow file, so renaming `cannotfail-release.yml` or either
environment stops publishing until the publishers on both sites are changed to match.

## Before 2027-02-18: the licence field

Every build prints two kinds of setuptools warning. `license = {text = "MIT"}` in
`tools/cannotfail/pyproject.toml` is a TOML-table form that setuptools 77 and later have
deprecated, and they say builds using it stop being supported after 2027-02-18; and
License classifiers are deprecated in favour of a licence expression. Both forms were
kept for 0.1.0. The fix, in one release:

- `license = "MIT"` and `license-files = ["LICENSE"]` in place of the table;
- delete the `License :: OSI Approved :: MIT License` classifier, since setuptools
  refuses a licence expression and a License classifier together;
- `requires = ["setuptools>=77"]` in `[build-system]`, since older setuptools refuse the
  string form.

Checked on 2026-11-15, with setuptools 84.2.1 and, for the last point, 76.1.0: each
behaves as described, and the changed project builds with `License-Expression: MIT` in
its metadata. If setuptools drops the table form sooner, the weekly run of
`cannotfail.yml` is what turns red.

2026-11-27: the repository's root `pyproject.toml` (the `rhob` package) has made this
change, with `license = "MIT"`, `license-files = ["LICENSE"]` and
`requires = ["setuptools>=77"]`, and it declares no License classifier.
`tools/cannotfail/pyproject.toml` has not: it still has the table and the classifier,
so all three parts above remain for its next release.

## What each workflow covers

- [`cannotfail.yml`](../.github/workflows/cannotfail.yml), the README badge. On pushes to
  `main` and pull requests that touch `tools/cannotfail/**` or the workflow, weekly on
  Mondays, and on demand. `test` runs the tool's tests on Python 3.10, 3.13 and 3.14, first
  with numpy only (asserting SciPy, pyarrow, pandas and inspect_ai are absent), then with
  SciPy, then pyarrow, then the `inspect` extra. `package` builds the sdist and wheel, runs
  `twine check --strict`, checks their contents, and installs the wheel alone into a fresh
  Python 3.10 venv. From there it runs the README's examples against the output the README
  prints, first with numpy only (the `resolution` and `power difference` examples, minus
  the Parquet and Inspect recipes; `power equivalence` and `power seeds` need SciPy) and
  then all of them with `[power,parquet,inspect]`, and then the tool's tests against that
  installed wheel. Since 2026-11-27 each job has a timeout (30 minutes for `test` and
  `package`), a newer push to the same pull request cancels the run in flight
  (scheduled, manual and `main` runs are never cancelled), and a `report` job, on
  scheduled runs only, opens or comments on the issue "Weekly cannotfail check is
  failing" when `test` or `package` does not pass and closes it when both pass again.
- [`cannotfail-release.yml`](../.github/workflows/cannotfail-release.yml), above. Since
  2026-11-27 its jobs time out: 20 minutes for `build`, 15 for each upload.
- [`tests.yml`](../.github/workflows/tests.yml) keeps
  [`tests/test_cannotfail_parity.py`](../tests/test_cannotfail_parity.py) in its pytest
  job. It imports `rhob`, to hold the tool's copy of the admission gate to the original,
  so it runs with RHOB installed and not in the tool's own CI. Its re-certification of
  four of the ledger's six DEGENERATE families is marked slow and runs in the nightly
  `admission-slow` job instead; the other two run on every push.
