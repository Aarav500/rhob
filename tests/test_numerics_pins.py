"""The numerics pins CI sets are the ones the cache key and provenance record.

``rhob.v3.calibration_cache.NUMERICS_ENV_VARS`` names the environment variables that
select floating-point kernels. Three places must agree on that list, and nothing checks
them against each other at run time:

- ``.github/workflows/regenerate.yml`` sets them, workflow-wide, for every published
  artifact;
- the calibration-cache key folds them in (``environment_token``) and so does the
  composite action's tarball key (``.github/actions/calibration-cache/action.yml``);
- ``provenance.numerics.env`` records them.

A variable added to one and not the others would let a pinned and an unpinned run share
a cache entry, or publish an artifact whose provenance does not name a pin that was in
force. These tests read the files as text (PyYAML is not in the locked environment) and import
nothing heavier than the cache module.
"""

from __future__ import annotations

import re
from pathlib import Path

from rhob.v3.calibration_cache import NUMERICS_ENV_VARS

ROOT = Path(__file__).resolve().parents[1]
REGENERATE = ROOT / ".github" / "workflows" / "regenerate.yml"
CACHE_ACTION = ROOT / ".github" / "actions" / "calibration-cache" / "action.yml"

#: The shared contract: the values regenerate.yml pins. Changing one is a change to
#: every published artifact's arithmetic and belongs in the CHANGELOG.
CONTRACT = {
    "OPENBLAS_CORETYPE": "Haswell",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_CBWR": "COMPATIBLE",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "ATEN_CPU_CAPABILITY": "avx2",
}


def _top_level_env(text: str) -> dict[str, str]:
    """The workflow-level ``env:`` mapping: the unindented ``env:`` and its 2-space keys."""
    lines = text.splitlines()
    start = lines.index("env:")
    env: dict[str, str] = {}
    for line in lines[start + 1:]:
        if line and not line.startswith(" "):
            break
        m = re.match(r"^  ([A-Za-z_][A-Za-z0-9_]*):\s*(.*?)\s*$", line)
        if m:
            env[m.group(1)] = m.group(2).strip('"').strip("'")
    return env


def test_the_contract_names_exactly_the_cache_keys_variables():
    assert set(CONTRACT) == set(NUMERICS_ENV_VARS)


def test_regenerate_pins_every_variable_workflow_wide():
    env = _top_level_env(REGENERATE.read_text(encoding="utf-8"))
    assert {k: env.get(k) for k in NUMERICS_ENV_VARS} == CONTRACT


def test_no_job_or_step_in_regenerate_overrides_a_pin():
    """A job- or step-level ``env`` entry, or a shell assignment, would un-pin one artifact."""
    offenders = []
    for n, line in enumerate(REGENERATE.read_text(encoding="utf-8").splitlines(), 1):
        for var in NUMERICS_ENV_VARS:
            nested_key = re.match(rf"^\s{{3,}}{var}\s*:", line)
            assignment = re.search(rf"(?<![A-Za-z0-9_]){var}=", line)
            if nested_key or assignment:
                offenders.append(f"{REGENERATE.name}:{n}: {line.strip()}")
    assert offenders == []


def test_the_cache_actions_tarball_key_hashes_the_same_variables():
    text = CACHE_ACTION.read_text(encoding="utf-8")
    match = re.search(r"for v in (.*?); do", text, flags=re.S)
    assert match, "the action no longer lists the numerics variables in a for loop"
    listed = match.group(1).replace("\\", " ").split()
    assert tuple(listed) == NUMERICS_ENV_VARS


def _jobs(text: str) -> dict[str, list[str]]:
    """Each top-level job's lines, keyed by job id (text parsing; see the module docstring)."""
    lines = text.splitlines()
    jobs: dict[str, list[str]] = {}
    current = None
    for line in lines[lines.index("jobs:") + 1:]:
        if line and not line.startswith(" "):
            break
        m = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if m:
            current = jobs.setdefault(m.group(1), [])
        elif current is not None:
            current.append(line)
    return jobs


WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))


def test_every_job_of_every_workflow_has_a_timeout():
    """A hung job otherwise holds a runner, and a concurrency slot, for six hours."""
    missing = [
        f"{wf.name}:{job}"
        for wf in WORKFLOWS
        for job, body in _jobs(wf.read_text(encoding="utf-8")).items()
        if not any(re.match(r"^    timeout-minutes:", line) for line in body)
    ]
    assert WORKFLOWS and missing == []


def test_the_calibration_tarball_is_keyed_by_one_shared_action():
    """tests.yml and regenerate.yml must compute the key the same way, so they use the action."""
    for name in ("tests.yml", "regenerate.yml"):
        text = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert "uses: ./.github/actions/calibration-cache" in text, name
