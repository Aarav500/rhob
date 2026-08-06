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
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Any, Optional

# Packages whose versions can change a reported AUROC. numpy/scipy affect the
# environment rollouts and the JS-divergence/statistical primitives; scikit-learn
# owns ``roc_auc_score`` and the MLP detectors; torch is optional (only the
# neural-net detectors import it) and is reported as null when absent.
_TRACKED_PACKAGES = ("numpy", "scipy", "scikit-learn", "torch", "rhob")

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
        return {
            "commit": None,
            "short_commit": None,
            "branch": None,
            "dirty": None,
            "dirty_files": None,
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
    }


def package_versions() -> dict[str, Optional[str]]:
    """Installed versions of the packages that can move a reported number."""
    versions: dict[str, Optional[str]] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = _pkg_version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


def provenance_block(**extra: Any) -> dict[str, Any]:
    """The ``"provenance"`` block to embed in a result JSON.

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
        "argv": list(sys.argv),
    }
    block.update(extra)
    return block


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
    wrong and was stamped into every artifact generated before then.) This block is
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
            "AUROC here as resolved: the null SE at n=m=5 is 0.19."
        )
        if n_replicates == 1
        else None,
    }
    block.update(extra)
    return block
