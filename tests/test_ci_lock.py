"""requirements/ci-admission-linux-py312.txt must satisfy pyproject.toml.

The nightly ``admission-slow`` job installs that lock rather than pyproject.toml's ranges.
A pull request that tightens a bound or adds a dependency in one of the lock's four
extras, without rerunning scripts/update_ci_lock.sh, fails here, on the pull request,
instead of turning main's next nightly red. See requirements/README.md.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("tomllib")  # Python 3.11+; the lock itself is for 3.12

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_ci_lock.py"


def _load():
    spec = importlib.util.spec_from_file_location("_check_ci_lock", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = _load()


def test_the_committed_lock_satisfies_the_committed_pyproject():
    unmet = check.static_unmet()
    assert not unmet, (
        "requirements/ci-admission-linux-py312.txt no longer satisfies pyproject.toml; "
        "run scripts/update_ci_lock.sh (requirements/README.md):\n  " + "\n  ".join(unmet))


def test_every_extra_the_lock_is_built_for_is_checked():
    script = (REPO / "scripts" / "update_ci_lock.sh").read_text(encoding="utf-8")
    assert f"extras=({' '.join(check.EXTRAS)})" in script
    drift = (REPO / ".github" / "workflows" / "drift.yml").read_text(encoding="utf-8")
    assert f'pip install -e ".[{",".join(check.EXTRAS)}]"' in drift


def _pyproject(tmp_path, old, new):
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    path = tmp_path / "pyproject.toml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


def test_a_tightened_bound_is_caught(tmp_path):
    path = _pyproject(tmp_path, '"mujoco>=3,<4"', '"mujoco>=3,<3.17"')
    unmet = check.static_unmet(pyproject=path)
    assert len(unmet) == 1 and unmet[0].startswith("mujoco<3.17,>=3 ([mujoco]): the lock has ")


def test_a_new_requirement_is_caught(tmp_path):
    path = _pyproject(tmp_path, '"mpe2>=1.0,<2"', '"mpe2>=1.0,<2", "not-a-real-package-xyz>=1"')
    assert check.static_unmet(pyproject=path) == [
        "not-a-real-package-xyz>=1 ([pettingzoo]): not in the lock"]


def test_extras_outside_the_lock_and_other_platforms_are_not_checked(tmp_path):
    path = _pyproject(tmp_path, '"gradio>=4.0,<7"', '"gradio>=4.0,<7", "unlocked-thing>=1"')
    assert check.static_unmet(pyproject=path) == []  # `space` is not one of the four
    path = _pyproject(tmp_path, '"mpe2>=1.0,<2"',
                      '"mpe2>=1.0,<2", "winonly>=1; sys_platform == \'win32\'"')
    assert check.static_unmet(pyproject=path) == []  # markers are evaluated for the lock's target


def test_lock_pins_reads_every_pinned_distribution():
    text = (REPO / "requirements" / "ci-admission-linux-py312.txt").read_text(encoding="utf-8")
    pins = check.lock_pins()
    assert len(pins) == sum("==" in line and not line.startswith(("#", " ")) for line in text.splitlines())
    assert {"numpy", "mujoco", "torch"} <= set(pins) and "rhob" not in pins
