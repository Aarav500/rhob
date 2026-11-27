"""Provenance stamping for every result artifact RHOB writes to disk.

Every JSON this repo publishes is a *measurement*, and a measurement that does not
record the code and inputs that produced it cannot be checked, reproduced, or
retracted with confidence. Before this module existed, ``leaderboard/*.json`` carried
nothing but a wall-clock timestamp and a version string: a reader holding a
leaderboard file had no way to tell which commit produced it, whether that commit's
working tree was clean, which library versions were installed, or how many seeds were
drawn. Two files with different numbers were indistinguishable from two files with the
same numbers measured differently.

:func:`provenance_block` returns the block to embed under a top-level ``"provenance"``
key. :func:`sampling_block` records the *draw*: how many seeds, which seeds, how many
independent replicates. The sampling block exists specifically so that a single
unreplicated draw is machine-readable as such (``"n_replicates": 1``) rather than
being invisible in an artifact that reports three-decimal AUROCs.

:func:`numerics_block`, embedded as ``provenance["numerics"]`` since 2026-11-27, records
what chose the floating-point kernels: the numerics pins CI sets, the CPU they ran on,
and the kernels the loaded libraries report they actually picked. The same code and
versions can still differ in the last bits between two machines, and when they do, this
block is what tells a reader whether the two runs were pinned, and on which CPUs.
"""

from __future__ import annotations

import importlib
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Any, Optional

from rhob.v3.calibration_cache import NUMERICS_ENV_VARS

# Packages whose versions can change a reported AUROC. numpy/scipy affect the
# environment rollouts and the JS-divergence/statistical primitives; scikit-learn
# owns ``roc_auc_score`` and the MLP detectors; torch is optional (only the
# neural-net detectors import it).
#
# The simulators were added on 2026-11-16. mujoco steps the four MuJoCo families,
# through gymnasium's environments; mujoco-core is new with MuJoCo 4.0 and is installed
# alongside it (None under 3.x). mpe2 supplies the scenarios of the five PettingZoo
# families and builds them on pettingzoo. Before this, only scripts/admission_ledger.py
# added them, to its own block, so no artifact committed before then other than the
# ledger, the 20-draw replication included, says which simulator produced it. The v5
# board and the admission ledger regenerated on 2026-11-19 carry the full list.
# Rendering-only dependencies (imageio, pygame-ce) are left out: no rollout renders.
#
# Every distribution the calibration cache keys on
# (``rhob.v3.calibration_cache.PHYSICS_DISTRIBUTIONS``) must appear here too, and
# tests/test_v3/test_na_channels_and_provenance.py fails if one does not. See
# docs/postmortems/2026-11-mujoco-calibration-cache.md for why both lists exist.
#
# A package that is not installed is recorded as None (JSON null), never omitted. That
# a run had no MuJoCo is a fact about the run, and a missing key would look the same as
# an artifact written before the key existed.
_TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "torch",
    "rhob",
    "mujoco",
    "mujoco-core",
    "gymnasium",
    "pettingzoo",
    "mpe2",
)

# The repo root, resolved from this file: src/rhob/v3/provenance.py -> ../../..
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _git(*args: str) -> Optional[str]:
    """Run a git command in the repo root; return stdout, or None on failure.

    Only the trailing newline is stripped, never leading whitespace: the first column
    of ``git status --porcelain`` is the index status and is a space for a file that is
    modified but unstaged, so a full ``.strip()`` silently eats the first character of
    the first path.

    Returns None rather than raising when git is unavailable or the tree is not a
    checkout (e.g. an sdist install, or a Docker image built by COPY): a missing SHA
    is honest provenance, a crashed experiment script is not.
    """
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.rstrip("\r\n")


def git_provenance() -> dict[str, Any]:
    """``{commit, short_commit, branch, dirty, dirty_files}`` for the repo root.

    ``dirty`` is True when ``git status --porcelain`` reports anything at all --
    including untracked files, which in this repo routinely means an uncommitted
    family or detector was on disk when the artifact was produced. A dirty artifact is
    not necessarily wrong, but it is not reproducible from the recorded SHA alone, and
    the reader is entitled to know that.
    """
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        # No git here, which is not exotic: the 2026-08 replication ran on a remote host
        # from a source tarball packed WITHOUT .git, so all twenty of its artifacts
        # recorded commit=null and no draw could be tied to a revision -- in a repository
        # whose seventh catalogued defect is scripts resolving to the wrong source tree.
        #
        # A deploy that strips .git must pass the revision explicitly. Where the answer
        # came from is recorded next to it, because "the operator asserted this" and "git
        # reported this" are different evidence and a reader must not have to guess which.
        declared = (os.environ.get("RHOB_SOURCE_COMMIT") or "").strip()
        if declared:
            return {
                "commit": declared,
                "short_commit": declared[:12],
                "branch": (os.environ.get("RHOB_SOURCE_BRANCH") or "").strip() or None,
                "dirty": None,
                "dirty_files": None,
                "source": "env:RHOB_SOURCE_COMMIT",
                "note": (
                    "No git repository at generation time; the revision was declared by "
                    "the caller and is unverified. dirty is unknown, not clean."
                ),
            }
        return {
            "commit": None,
            "short_commit": None,
            "branch": None,
            "dirty": None,
            "dirty_files": None,
            "source": None,
            "note": (
                "No git repository at generation time and RHOB_SOURCE_COMMIT was unset, "
                "so this artifact cannot be tied to a source revision."
            ),
        }
    commit = commit.strip()
    status = _git("status", "--porcelain")
    # Porcelain v1 lines are ``XY<space>PATH``: two status columns then the path.
    dirty_files = [line[3:] for line in status.splitlines() if len(line) > 3] if status else []
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    return {
        "commit": commit,
        "short_commit": commit[:12],
        "branch": branch.strip() if branch else None,
        "dirty": bool(dirty_files),
        "dirty_files": sorted(dirty_files),
        "source": "git",
    }


def package_versions() -> dict[str, Optional[str]]:
    """Installed versions of the packages that can move a reported number.

    Every name in :data:`_TRACKED_PACKAGES` is a key; a package that is not installed
    maps to None. Artifacts written before 2026-11-16 carry only the first five names
    (plus, in an admission ledger written then, the four simulators that
    ``scripts/admission_ledger.py`` added itself), so a reader must treat a missing
    name as "not recorded", which is different from None, "recorded as not installed".
    """
    versions: dict[str, Optional[str]] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = _pkg_version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


#: CPU flags that decide which kernels OpenBLAS, MKL and ATen dispatch to when nothing
#: pins them. Recorded as the subset the CPU has, not the whole flag list.
_KERNEL_ISA_FLAGS = (
    "sse4_2",
    "avx",
    "avx2",
    "fma",
    "avx512f",
    "avx512dq",
    "avx512bw",
    "avx512vl",
    "avx512_vnni",
    "avx512_bf16",
    "amx_tile",
)

#: Where :func:`cpu_info` reads the CPU from; a module attribute so a test can point it at
#: a fixture.
_CPUINFO_PATH = "/proc/cpuinfo"

# The fields of threadpoolctl's report that say which kernel a library picked. Its
# ``filepath`` is left out: it is a path on the generating machine, not a property of
# the numerics.
_BLAS_FIELDS = ("user_api", "internal_api", "prefix", "version", "architecture",
                "threading_layer", "num_threads")


def cpu_info() -> dict[str, Any]:
    """``{vendor, model, family, model_number, stepping, isa, logical_cpus, source}``.

    From ``/proc/cpuinfo`` (the first processor's entry) where it exists, so on every
    Linux runner; elsewhere only ``model`` is filled, from :func:`platform.processor`,
    and the rest are None. ``family``/``model_number``/``stepping`` are recorded as well
    as the marketing name because a cloud VM's name can be as vague as "Intel(R) Xeon(R)
    Processor @ 2.10GHz" while the numbers identify the microarchitecture.
    """
    info: dict[str, Any] = {
        "vendor": None,
        "model": None,
        "family": None,
        "model_number": None,
        "stepping": None,
        "isa": None,
        "logical_cpus": os.cpu_count(),
        "source": None,
    }
    fields: dict[str, str] = {}
    try:
        with open(_CPUINFO_PATH, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    if fields:
                        break  # end of the first processor's entry
                    continue
                key, sep, value = line.partition(":")
                if sep:
                    fields.setdefault(key.strip(), value.strip())
    except OSError:
        fields = {}
    if fields:
        flags = set(fields.get("flags", "").split())
        info.update(
            vendor=fields.get("vendor_id"),
            model=fields.get("model name"),
            family=fields.get("cpu family"),
            model_number=fields.get("model"),
            stepping=fields.get("stepping"),
            isa=[f for f in _KERNEL_ISA_FLAGS if f in flags] if flags else None,
            source="/proc/cpuinfo",
        )
    else:
        model = platform.processor() or None
        info.update(model=model, source="platform.processor" if model else None)
    return info


#: The line of ``torch.__config__.show()`` that names the ATen kernel level, present on
#: every torch 2.x: ``"  - CPU capability usage: AVX2"``.
_TORCH_CAPABILITY_LINE = re.compile(r"CPU capability usage:\s*(\S+)")


def _torch_loaded() -> bool:
    """Whether this process has imported torch. A ``None`` entry in ``sys.modules``
    blocks the import rather than recording one, so it counts as not loaded."""
    return sys.modules.get("torch") is not None


def _torch_cpu_capability() -> Optional[str]:
    """The ATen kernel level torch dispatches to, if torch is already imported.

    Never imports torch itself: stamping provenance must not load a library the run did
    not use, and importing torch would also fix its kernel choice in this process.

    Read from ``torch.backends.cpu.get_cpu_capability()``, which first shipped in torch
    2.1. ``pyproject.toml`` allows 2.0, so when it is missing the value is parsed from
    the ``CPU capability usage`` line of ``torch.__config__.show()``, which carries the
    same string (``"AVX2"`` under ``ATEN_CPU_CAPABILITY=avx2``) on every 2.x. None when
    neither is available.
    """
    if not _torch_loaded():
        return None
    try:
        backend = importlib.import_module("torch.backends.cpu")
        return str(backend.get_cpu_capability())
    except Exception:  # noqa: BLE001 -- provenance must never crash the run it describes
        pass
    try:
        summary = importlib.import_module("torch.__config__").show()
        match = _TORCH_CAPABILITY_LINE.search(summary)
    except Exception:  # noqa: BLE001 -- provenance must never crash the run it describes
        return None
    return match.group(1) if match else None


def _loaded_blas() -> Optional[list[dict[str, Any]]]:
    """The BLAS and OpenMP libraries loaded in this process, as threadpoolctl sees them.

    ``architecture`` is the kernel OpenBLAS actually selected (``"Haswell"`` under the
    CI pin, whatever the CPU), which is the evidence the pin took effect rather than
    that it was merely set. Only libraries already loaded are reported; None when
    threadpoolctl is unavailable.
    """
    try:
        from threadpoolctl import threadpool_info  # noqa: PLC0415
    except ImportError:
        return None
    try:
        libraries = threadpool_info()
    except Exception:  # noqa: BLE001 -- provenance must never crash the run it describes
        return None
    rows = [{field: lib.get(field) for field in _BLAS_FIELDS} for lib in libraries]
    return sorted(rows, key=lambda r: tuple(str(r[f]) for f in _BLAS_FIELDS))


def numerics_block() -> dict[str, Any]:
    """The ``"numerics"`` block: what selected this run's floating-point kernels.

    - ``env``: each of :data:`~rhob.v3.calibration_cache.NUMERICS_ENV_VARS` (the pins CI
      sets: ``OPENBLAS_CORETYPE``, ``MKL_CBWR``, ``ATEN_CPU_CAPABILITY`` and the three
      thread counts), or None when unset -- null, as for an uninstalled package, never
      omitted. The calibration cache hashes the same names, spelling unset ``"unset"``.
    - ``cpu``: :func:`cpu_info`.
    - ``torch_imported`` and ``torch_cpu_capability``: whether this process had loaded
      torch when the block was stamped, and if so the ATen kernel level it dispatches to
      (``"AVX2"`` under the pin, ``"AVX512"`` unpinned on a CPU that has it; read on
      every torch 2.x, see :func:`_torch_cpu_capability`). Null when torch was not
      imported; importing :mod:`rhob.v3` loads it with the detectors whenever it is
      installed, so in practice null means torch is absent.
    - ``blas``: :func:`_loaded_blas`.

    Two runs with equal ``env`` pins and equal ``blas`` architectures picked the same
    kernels; equal ``cpu`` is not required for that, and is not implied by it. A claim
    that results agree *across* CPU vendors needs two artifacts whose ``cpu.vendor``
    differ.
    """
    return {
        "env": {name: os.environ.get(name) for name in NUMERICS_ENV_VARS},
        "cpu": cpu_info(),
        "torch_imported": _torch_loaded(),
        "torch_cpu_capability": _torch_cpu_capability(),
        "blas": _loaded_blas(),
    }


def provenance_block(**extra: Any) -> dict[str, Any]:
    """The ``"provenance"`` block to embed in a result JSON.

    Stamp it once the run's work is done, as the scripts here do (each builds it while
    writing its artifact): ``numerics`` reports the libraries the process has loaded, so
    a block stamped before the first rollout could say nothing about torch.

    Args:
        **extra: Additional run-specific keys merged into the block (e.g. the
            command line, or a script identifier).
    """
    block: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git": git_provenance(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "numerics": numerics_block(),
        "argv": list(sys.argv),
    }
    block.update(extra)
    return block


def null_auroc_se(n: int, m: int) -> float:
    """The Mann-Whitney null standard error of an AUROC on ``n`` and ``m`` runs.

    ``sqrt((n + m + 1) / (12 n m))``: the spread of one cell's AUROC near 0.5 when the
    two variants are indistinguishable. 0.19 at n=m=5 (the leaderboard), 0.13 at
    n=m=10, 0.09 at n=m=20.
    """
    return math.sqrt((n + m + 1) / (12 * n * m))


def sampling_block(
    *,
    n_seeds: int,
    n_layouts: int,
    layout_seeds: list[int],
    rollout_seeds_hacking: list[int],
    rollout_seeds_legit: list[int],
    n_replicates: int = 1,
    shared_draw_across_detectors: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    """The ``"sampling"`` block: exactly which draw produced these numbers.

    ``n_replicates == 1`` means every number in the artifact comes from ONE draw of
    the rollout seeds, shared across every detector scored -- not an average over
    independent repetitions. At the leaderboard's ``n_seeds=5`` per variant, the
    standard error of a single-cell AUROC near 0.5 is 0.19 -- the Mann-Whitney null
    ``sqrt((n+m+1)/(12nm))`` at n=m=5 -- so a three-decimal cell value carries about
    one meaningful digit. (This constant read 0.16 until 2026-08, which was simply
    wrong and was stamped into every artifact generated before then; until 2026-11-27
    the note said 0.19 at n=m=5 whatever ``n_seeds`` was, so the transfer and
    sign-randomization artifacts, at 10 and 20 seeds, overstated it. The note now
    computes it with :func:`null_auroc_se`, and names the held-out cells' figure when an
    ``n_seeds_test`` differs.) This block is
    emitted so that fact is machine-readable from the artifact itself instead of
    having to be reconstructed from the script's source.

    Args:
        n_seeds: Rollout seeds per variant per cell.
        n_layouts: Distinct family-layout seeds (``generate_pair(difficulty, seed=...)``).
        layout_seeds: The layout seeds actually used.
        rollout_seeds_hacking: Seeds passed to the hacking-variant rollout closure.
        rollout_seeds_legit: Seeds passed to the legitimate-variant rollout closure.
        n_replicates: Independent repetitions of the whole draw. 1 = single draw.
        shared_draw_across_detectors: True when every detector in the artifact was
            scored on the identical rolled-out runs (the rollouts are deterministic in
            the seeds, so this holds whether they are cached or regenerated).
        **extra: Additional sampling keys (e.g. a separate test-set seed base).
    """
    se_txt = f"the null SE at n=m={n_seeds} is {null_auroc_se(n_seeds, n_seeds):.2f}"
    n_test = extra.get("n_seeds_test")
    if isinstance(n_test, int) and not isinstance(n_test, bool) and n_test != n_seeds:
        se_txt += f" ({null_auroc_se(n_test, n_test):.2f} at the held-out cells' n=m={n_test})"
    block: dict[str, Any] = {
        "n_seeds_per_variant": n_seeds,
        "n_layouts": n_layouts,
        "layout_seeds": list(layout_seeds),
        "rollout_seeds_hacking": list(rollout_seeds_hacking),
        "rollout_seeds_legit": list(rollout_seeds_legit),
        "n_replicates": n_replicates,
        "single_draw": n_replicates == 1,
        "shared_draw_across_detectors": True,
        "confidence_intervals": None,
        "note": (
            "Single unreplicated draw: all detectors are scored on the identical "
            "rolled-out runs, so their cell values are correlated and their "
            "differences are not independent measurements. No confidence intervals "
            "are computed for this artifact; see leaderboard/v5_replicated.json for "
            "the 20-draw replication that does. Do not read the third decimal of any "
            f"AUROC here as resolved: {se_txt}."
        )
        if n_replicates == 1
        else None,
    }
    block.update(extra)
    return block
