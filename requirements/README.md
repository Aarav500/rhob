# requirements/

Pinned dependency sets for CI. There is one so far.

## ci-admission-linux-py312.txt

The hashed lock that the nightly `admission-slow` job in `.github/workflows/tests.yml`
installs. It is uv's resolution of `pip install -e ".[dev,continuous,mujoco,pettingzoo]"`:
every distribution that install pulls in, transitively, at one exact version, with the sha256
of every file PyPI holds for that version. Generated on 2026-11-16, it is resolved as PyPI
stood at 2026-11-15T05:00:20Z, the moment the 2026-11-15 verification run (43679157068)
began installing. That run passed every slow screen, and the lock is exactly the 81
distributions it installed, read from its job log. Among them are numpy
2.5.4, torch 2.15.0 from PyPI with the CUDA wheels it depends on, and mujoco 3.17.0 under
the `mujoco<4` bound. Releases since then, numpy 2.5.5 among them, wait for `drift.yml` and
a reviewed regeneration. rhob itself is not in it. CI installs it
afterwards with `pip install --no-deps -e .`. The one unpinned piece of that step is the
build backend (setuptools, wheel). pip fetches it into an isolated build environment to
build rhob's own editable wheel and then discards it, so it installs nothing into the
environment the tests use.

**Why.** The physics screen should change only when the code does. Installing unpinned, the
nightly met MuJoCo 4.0.0 the night it was published and went red on an unchanged commit
([post-mortem](../docs/postmortems/2026-11-mujoco-calibration-cache.md)). On the lock, the
same commit installs the same distributions every night. A new release reaches the nightly
only through a regeneration of this file, which is a diff someone reviews. What still floats
is the CPython 3.12 patch release, pip and the runner image. The calibration cache's physics
token is stable too, because the "Fingerprint the physics stack" step reads installed
versions and those no longer move. This file's versions give `029b0400d8d0fa6d`, the token
43679157068 computed.

**Scope.** CPython 3.12 on Linux x86_64. It is resolved for `x86_64-manylinux_2_28`, and the
script checks that `x86_64-manylinux_2_40`, the newest glibc tag uv can target, resolves
identically. `ubuntu-latest` was `ubuntu-26.04` in the 2026-11-07 run logs. The lock is not
for macOS, Windows, ARM or another Python, and it is not the environment of any committed
artifact. Each artifact that records one says so in its own provenance block.

**What was checked, 2026-11-16.** In CI, `workflow_dispatch` run 43836925199 of `tests.yml`
on the branch at `6ae7d7f` was the first to install this file. Its Install step passed:
pip's hash-checking install of all 81 lines, `pip install --no-deps -e .`, `pip check` and
the lock-satisfies-pyproject check. Its slow tier passed as 43679157068's did, 73 passed and
21 xfailed as marked. The warm step had left 23 families cold, 15 of them with cached
calibrations, and the runner was about half as fast (`mujoco_sensor_decoupling`'s calibration took 1662 s, against 804 s on 2026-11-15),
so the job took 153 of its 180 minutes, and its timeout is now 240. The rest was done in
scratch environments. Some of it was first run on a draft resolved at 2026-11-16T06:54:00Z,
which differed from this file only in numpy 2.5.5; the draft was dropped so the first lock
is a stack CI has already passed on.

- **It is the last green stack.** At this file's cutoff the script gives exactly the 81
  distributions that 43679157068 installed.
- **Same resolution as pip.** On the draft, pip 26.3 run unpinned with `--dry-run --report`
  on Python 3.12 picked the same 81 versions, plus rhob. A pip 26.3
  `install --dry-run --require-hashes -r` of the draft downloaded and hash-checked every
  wheel, torch's CUDA wheels included, and exited 0.
- **The check script catches what it should.** `uv pip sync --require-hashes` of this file
  made a Python 3.12 venv exactly these 81 distributions, hash-checking each file. There
  `scripts/check_ci_lock.py --installed` passes, and with imageio hidden from it, it fails
  and names `imageio>=2.14.1` from `gymnasium[mujoco]`, which `pip check` does not read.
  The static check fails on a scratch `pyproject.toml` with `mujoco>=3,<3.17`, with a
  requirement the lock lacks, or with one of the four extras renamed, and names it
  (`tests/test_ci_lock.py`).
- **The suites pass on it.** In that venv, with rhob added by `pip install --no-deps -e .`
  as CI does, a clean checkout of `9b9031d` gives 798 passed, 1 skipped and 23 xfailed in the
  fast suite. The skip is `test_dirty_file_paths_are_not_truncated`, which runs only in a
  dirty working tree. On the draft, the four MuJoCo families' slow smoke screens, run cold, gave 10
  passed and 12 xfailed as marked. On this file, the slow tier is 43679157068's result and
  43836925199's.
- **It reproduces past nightlies.** Resolving with a past cutoff reproduces that night's
  install. At `2026-11-06T12:03:42Z`, the moment the last green run (42214775392) began
  installing, the script gives exactly the 81 distributions that run installed. At
  `2026-11-07T11:01:10Z`, the first red run (42366363571), it matches that run's install
  except where `mujoco<4` applies. It gives mujoco 3.17.0 instead of 4.0.0, with no
  mujoco-core or mujoco-studio, and with absl-py, etils and zipp present.

## Regenerating: scripts/update_ci_lock.sh

    scripts/update_ci_lock.sh                       # resolve against PyPI as it is now
    scripts/update_ci_lock.sh 2026-11-15T05:00:20Z  # resolve as PyPI stood at that time
    scripts/update_ci_lock.sh --check               # does the committed file still reproduce?

It needs uv and network access to pypi.org. It runs `uv pip compile` with `--exclude-newer`,
which ignores every file uploaded after the cutoff, and it writes the cutoff, the uv version
and the source commit into the header. So a rerun with the recorded cutoff reproduces the
file, unless a pinned release has since been yanked or deleted, or a different uv resolves
differently. User and project uv configuration is ignored, and the index, resolution
strategy and platform are passed explicitly. It stops without writing anything if the 2.28
and 2.40 resolutions differ, because then the lock would not be what pip installs on the
runner.

When to regenerate:

- **After a dependency change in `pyproject.toml`.** `scripts/check_ci_lock.py` compares the
  two, and the fast suite runs it (`tests/test_ci_lock.py`), so a pull request that changes a
  requirement of the four extras without regenerating the lock fails its own `pytest` jobs on
  3.11 and 3.12, naming the requirement. That check reads only names and version bounds. The
  nightly's install step runs the script again with `--installed`, which also follows the
  extras a requirement asks for, such as `gymnasium[mujoco]`.
- **To adopt new releases.** Read the latest run of `.github/workflows/drift.yml` first. Its
  table is the diff that a regeneration would make, and its tests say whether that stack
  passes the fast suite and the MuJoCo screens. It does not run the PettingZoo, RLHF
  reward-model or sequence families' slow screens; those meet the new stack at step 3 below.

Before committing a regenerated lock:

1. Read `git diff requirements/`, since the pins that moved are the change under review. A
   move in any calibration-key distribution (numpy, scipy, mujoco, mujoco-core, gymnasium,
   torch, pettingzoo, mpe2, marked in the drift table) changes the cache token, so the next
   nightly recalibrates from cold, and that does not fit the warm step's 45-minute budget.
   The 2026-11-15 verification run, which did it, spent 37 minutes of that budget deriving 8
   families and then stopped rather than start one that could overrun, leaving 17 families
   cold, 11 of them with cached calibrations, for the slow screens to derive. Those screens
   passed in 31 minutes, and the job took 71. The next cold run, 43836925199, got a runner about half as
   fast and took 153.
2. A new MuJoCo major is a re-measurement of the MuJoCo families and the ledger, not a lock
   bump. See the comment on `mujoco<4` in `pyproject.toml`.
3. Run `tests.yml` by hand (workflow_dispatch) on the branch, so the nightly that sees the
   new stack first is not the one on `main`.

## Drift: .github/workflows/drift.yml

Runs weekly (Mondays, 05:43 UTC) and on demand. It installs the newest releases that
`pyproject.toml` allows on Python 3.12, prints a table of every version that differs from
this lock, then runs the fast suite and the slow smoke screens of the four MuJoCo families.
No calibration cache is restored, so every calibration is derived under the new stack. On
the draft stack (this file with numpy 2.5.5), those screens ran cold on a 4-core scratch
machine on 2026-11-16 in 21 minutes, with 10 passed and 12 xfailed as marked.

A failure shows on that workflow's own page, nowhere else, and does not change the nightly,
the badges or the lock. On a scheduled run that does not pass, a second job opens or comments
on one issue, "Weekly dependency drift check is failing", and closes it when a later scheduled
run passes. The job that runs the new releases has read-only access to the repository. Only
the issue job, which installs nothing, holds `issues: write`.

Cost per run is about 25 to 40 minutes on a standard runner, estimated rather than yet
observed. Install and the fast suite take about 2 and 4 minutes (the 2026-11-06 nightly).
The MuJoCo screens took 21 minutes on that machine. In two CI runs of the same full slow
tier the same tests ran about 1.55 times slower than there (2026-11-06) and at about the
same speed (2026-11-15), so the runner made the difference. Expect 20 to 35 minutes, and
up to about twice that on a slow runner (one on 2026-11-16 calibrated about half as fast).
That comes to two to three hours of runner time a month, up to about five on slow runners,
billed only if the repository is private. The header of `drift.yml` has the breakdown.

## Not to be confused with requirements-lock.txt

`requirements-lock.txt` at the repository root is a different and older file, and no CI job
installs it. It lists the versions of RHOB's direct dependencies found in one Windows
environment under Python 3.13 on 2026-07-29, with transitive dependencies unpinned and no
hashes, as its own header says. The admission ledger's provenance block records nine package
versions, and the eight besides rhob itself all match it. Nothing records the rest of that environment.
