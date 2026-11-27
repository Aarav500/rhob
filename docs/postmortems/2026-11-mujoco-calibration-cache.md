# Post-mortem: the calibration cache could not see a MuJoCo upgrade

Written 2026-11-16 from facts checked on 2026-11-15 and 2026-11-16 against the GitHub Actions
runs named below. The fix is `9cc96e2`, on the branch of [PR #2](https://github.com/Aarav500/rhob/pull/2);
until that merges, `main` runs the old key. The paper asks of a check *what observation would
have had to exist for this to report failure?* This is that question, asked of our own harness.

## Summary

From 2026-11-07 the scheduled `admission-slow` job in `tests.yml` failed every night on an
unchanged `main` commit. MuJoCo 4.0.0 had been published the afternoon before. CI installed
unpinned, and the cache that stores each family's calibration was keyed on repository source
alone, so `mujoco_joint_limit_gaming` ran under MuJoCo 4.0 on a hip amplitude calibrated
under 3.17. The cache did not cause the failure: a fresh calibration under 4.0 fails too.
What it did was make each nightly a mixture of two simulators, with nothing in its key able
to notice.

## Timeline (UTC)

| When | What |
|---|---|
| 2026-11-01 to 11-06 | `admission-slow` passes every night. |
| 2026-11-06 12:03 | Last green run, [42214775392](https://github.com/Aarav500/rhob/actions/runs/42214775392), installs mujoco 3.17.0. |
| 2026-11-06 15:39 | mujoco 4.0.0 is uploaded to PyPI, with the new mujoco-core 4.0.0 and mujoco-studio 4.0.0. |
| 2026-11-07 11:01 | First red run, [42366363571](https://github.com/Aarav500/rhob/actions/runs/42366363571), installs mujoco 4.0.0. |
| 2026-11-07 to 11-15 | Red every night on the same commit. Later runs install 4.0.1. |
| 2026-11-15 | Controlled experiment; fix `9cc96e2`; wording correction `b70d258`; verification run [43679157068](https://github.com/Aarav500/rhob/actions/runs/43679157068) on the branch. The scheduled nightly on `main` ([43711218047](https://github.com/Aarav500/rhob/actions/runs/43711218047)) is red again: until the merge, `main` has neither the `mujoco<4` pin nor the new key, so it installs MuJoCo 4.0.1 and serves it the 3.17 calibration. |
| 2026-11-16 | On the branch: provenance records the simulator versions; `admission-slow` moves to a hashed lock, with a weekly drift check; this write-up. |

## Evidence

**The two runs**, from their install logs:

| Package | 42214775392 (green) | 42366363571 (red) |
|---|---|---|
| mujoco | 3.17.0 | 4.0.0 |
| mujoco-core, mujoco-studio | not installed | 4.0.0 |
| absl-py, etils, zipp | installed | dropped |
| click | 8.5.0 | 8.5.1 |
| gymnasium, numpy, scipy, torch, pettingzoo, mpe2 | 1.4.0, 2.5.4, 1.18.1, 2.15.0, 1.27.0, 1.1.1 | identical |

Both restored the same tarball, `calib-Linux-py3.12-e993df02...`, and the warm step logged
`mujoco_joint_limit_gaming: already warm` (`scripts/warm_calibration_cache.py` prints that
when a family's calibrations return in under a second, that is, from cache).

**The failure.** At difficulty 0.9, `proxy_matched` failed with a TOST mean L0 AUROC of
0.7083, 90% CI [0.6001, 0.8166], against the smoke band [0.24, 0.76]. At 0.7 a strict xfail
passed (`XPASS(strict)`), which fails the run.

**The controlled experiment (2026-11-15).** Same machine, same command, `RHOB_CALIB_CACHE=0`
so every calibration was derived fresh, Python 3.12, numpy 2.5.4, scipy 1.18.1, gymnasium
1.4.0. Only MuJoCo differed. Each run took about 4.5 minutes.

| Tier | mujoco 3.17.0 | mujoco 4.0.0 |
|---|---|---|
| 0.9 | PASS | FAIL, 0.3333 [0.2264, 0.4403] |
| 0.8 | XFAIL, as documented | XFAIL |
| 0.7 | XFAIL, as documented | XPASS |

Under 3.17 the family behaves as its test file documents. Under 4.0 a fresh calibration also
fails at 0.9, on the other side of the band, so recalibrating does not rescue it. The
nightly's 0.7083 was neither simulator's result: it was 4.0 physics on a 3.17 amplitude.

## The mechanism

Before the fix both keys were computed from the repository tree. Each entry was keyed on
sha256 over the defining module's source, the function's qualname and its arguments; the CI
tarball on `hashFiles('src/rhob/v3/families/*.py', 'src/rhob/v3/calibration_cache.py')`. The
workflow comment above the restore step gave the reasoning: the calibrations are pure
functions of the family sources, so "a stale entry is unreachable rather than merely unlikely".

That was false twice over. A calibration also depends on the simulator that rolls it out,
which the job then installed after checkout, unpinned, and on the modules the family imports,
including `rhob.environments.calibration`, where `calibrate_scale` itself lives. Neither key
read those, and the entry key did not read the Python version or the machine either. So on
2026-11-07 nothing either key read had changed. The job computed the previous night's tarball
key, restored the tarball saved under 3.17, matched every entry, and served the 3.17 hip
amplitude to rollouts simulated by 4.0. For this key the paper's question had one answer, an
edit to a family module or to `calibration_cache.py`. A dependency upgrade was not an
observation it could receive, so for the change that mattered it could only report "current".

## The fix

In `9cc96e2`: the `mujoco` extra in `pyproject.toml` requires `mujoco>=3,<4`, which keeps the
nightly on the simulator the families were measured under. Every entry key also hashes the
installed versions of numpy, scipy, mujoco, mujoco-core, gymnasium, torch, pettingzoo and mpe2
(`absent` when not installed), the Python minor version, the machine, and the source of every
same-package module the defining module imports, followed transitively, imports inside
function bodies included. The CI tarball key adds the same version token and
`hashFiles('src/rhob/**/*.py')`, with a prefix fallback that is safe only because the entry
keys are now complete. Five regression tests in `tests/test_v3/test_calibration_cache.py` all
fail against the old module. The pin is a stopgap. The key is what makes the next upgrade a
cache miss instead of a silent mixture, pinned or not.

*(2026-11-27: the entry key changed in two more ways in `827aec9`. It now hashes each reachable
module's code, its syntax tree with docstrings removed, rather than its text, so a
comment, layout or docstring edit no longer invalidates an entry and any other edit
still does. Of the eleven commits that touched `src/` after `9cc96e2`, five had changed
every calibrated family's key by docstring or comment edits alone, each costing a cold
recalibration of about 43 minutes on CI; under the new fingerprint all eleven keep every
key. And the version token now also hashes the six numerics variables `regenerate.yml`
pins for every job (`OPENBLAS_CORETYPE`, `OPENBLAS_NUM_THREADS`, `MKL_CBWR`,
`MKL_NUM_THREADS`, `OMP_NUM_THREADS`, `ATEN_CPU_CAPABILITY`; `unset` when absent), since
the BLAS and ATen kernels they select can move the last bits of a calibration: pinned and
unpinned calibrations never share an entry, so a cache serves only a process started
under the same values it was warmed under. `admission-slow` sets none of them. The CI
tarball key is now defined once, in `.github/actions/calibration-cache`, as
`calib-v2-<os>-<image>-py3.12-np<pins>-<token>-<src>-<run id>-<attempt>`, and provenance
written since records the pins and the kernels they selected in a `numerics` block.)*

`b70d258` corrected the wording about what the cache did. The `workflow_dispatch` run
43679157068 on the branch succeeded. It started cold. The warm step spent 37 minutes of its
45-minute budget deriving 8 families and then stopped rather than start one that could
overrun, leaving 17 families cold, 11 of them with cached calibrations, for the slow smoke
screens, which derived them and passed in 31 minutes; the cache was saved. On 2026-11-16 `rhob.v3.provenance`
began recording the simulator versions; a test fails if the cache keys on one it does not
record.

Also on 2026-11-16, the branch changed `admission-slow` to stop installing unpinned. Like
the key fix, this reaches the nightly when PR #2 merges; until then the nightly on `main`
installs unpinned. On the branch it installs `requirements/ci-admission-linux-py312.txt`, a
hashed lock of all 81 distributions, resolved as PyPI stood when 43679157068 began
installing, so it is exactly the stack that run passed on. A new release reaches the
nightly only through `scripts/update_ci_lock.sh` and a reviewed diff.
`.github/workflows/drift.yml` installs the newest allowed releases each week, tabulates what
differs from the lock, and runs the fast suite and the MuJoCo screens on them. The per-push
`pytest` matrix still installs unpinned, on purpose. See
[requirements/README.md](../../requirements/README.md).

## What it was, and what it was not

- **Not the cause.** A fresh calibration under 4.0 fails as well. The failure came from a
  simulator the families were never measured under, reaching CI through the unpinned install.
- **What the cache did.** It made each nightly a mixture of two simulators, and nothing in
  its key could notice. It would have done the same, silently, to an upgrade that moved the
  numbers without failing a test. This one failed a test, with a number from neither simulator.
- **The shape of the paper's class, not an instance of it.** A validity check that could not
  return "stale" for the change that mattered has that shape. But a cache key is
  infrastructure, not a check guarding a claim about a benchmark's construction, so it does
  not add to the paper's count: seven instances in RHOB, plus two the README records since.
- **Not known to have moved a published number.** Every committed artifact written before
  2026-08-18, when `045b87f` added the disk cache, could not have read it. Two later sets
  could have, and neither records a simulator version: the 2026-09-03 Reward Skewness
  re-score in the replicates (and `leaderboard/v5_replicated.json`, built from it), which
  records numpy, scipy, scikit-learn, torch and rhob only, and `results/family_diagnostics/`
  (2026-09-19), which records none. Whether a stale entry reached them is open.
  *(2026-11-19: the single-draw board and the admission ledger have since been
  regenerated on CI, `52656cb` and `d4b700d`. Both restored the cache under the complete
  entry key, on the hashed lock, and both record their simulator versions: mujoco
  3.17.0.)*

## What is still open

- **Migrating to MuJoCo 4** means re-measuring the four MuJoCo families (`mujoco_camping`,
  `mujoco_goal_misgeneralization`, `mujoco_joint_limit_gaming`, `mujoco_sensor_decoupling`)
  and the ledger. The ledger's ten families import no simulator, but its provenance records
  mujoco 3.10.0, and that 4.0 leaves it unchanged should be measured, not inferred. Until then
  `mujoco<4` stays. *(2026-11-19: the regenerated ledger, `d4b700d`, covers all 33 families,
  the four MuJoCo families among them, and records mujoco 3.17.0 from the CI lock; all 13
  MuJoCo cells are NOT ADMITTED under it.)*
- **The replication does not say which simulator it ran.** The 20 committed replicates were
  written under the old list (numpy, scipy, scikit-learn, torch, rhob), so the MuJoCo they ran
  under cannot be recovered from them. Artifacts written from 2026-11-16 on record it.
- **The lock has been installed by CI on the branch only.** `workflow_dispatch` run
  [43836925199](https://github.com/Aarav500/rhob/actions/runs/43836925199) installed it on
  2026-11-16, hash-checking every file, and its slow tier passed as 43679157068's did (73
  passed, 21 xfailed). The nightly on `main` meets it only after the merge.
- **The drift check covers the MuJoCo families only.** The PettingZoo, RLHF reward-model and
  sequence families' slow screens meet a new release only at the dispatched run after a
  regeneration.
