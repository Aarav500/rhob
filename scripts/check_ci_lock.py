#!/usr/bin/env python3
"""Does requirements/ci-admission-linux-py312.txt still satisfy pyproject.toml?

    python scripts/check_ci_lock.py              # read the lock file; installs nothing
    python scripts/check_ci_lock.py --installed  # read what was installed from it

The lock is what the nightly `admission-slow` job installs. If `pyproject.toml` changes
and the lock is not regenerated with `scripts/update_ci_lock.sh`, the nightly would test
a stack the project no longer declares, or fail on main the night after the merge.

- The default, static check reads the lock's pins and tests every requirement of the
  four extras the lock is built for against them: the name must be pinned, at a version
  the requirement allows. `tests/test_ci_lock.py` runs it in the fast suite, so the pull
  request that changes `pyproject.toml` is the one that fails. It cannot follow the extras
  a requirement asks for (`gymnasium[mujoco]`), because the lock does not record what an
  extra pulls in.
- `--installed` runs in `admission-slow` after the lock is installed. It reads rhob's
  installed requirements, and follows each extra a requirement asks for through that
  distribution's own metadata, so `gymnasium[mujoco]` needs gymnasium's `mujoco` extra
  to be satisfied too. `pip check` does neither: it reads base requirements only.

Markers are evaluated for the lock's target, CPython 3.12 on Linux x86_64
(requirements/README.md), not for the machine running the static check. Exit status is 0
when every requirement is met and 1 otherwise, with each unmet one listed.
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "requirements" / "ci-admission-linux-py312.txt"
PYPROJECT = ROOT / "pyproject.toml"
# Keep in step with `extras` in scripts/update_ci_lock.sh and drift.yml's install line
# (tests/test_ci_lock.py checks both).
EXTRAS = ("dev", "continuous", "mujoco", "pettingzoo")
TARGET = {
    "implementation_name": "cpython",
    "os_name": "posix",
    "platform_machine": "x86_64",
    "platform_python_implementation": "CPython",
    "platform_system": "Linux",
    "python_full_version": "3.12.0",
    "python_version": "3.12",
    "sys_platform": "linux",
}
_PIN = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s\\;]+)")


def lock_pins(lock: Path = LOCK) -> dict[str, str]:
    """``{canonical name: version}`` for every ``name==version`` line of the lock."""
    pins = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        m = _PIN.match(line)
        if m:
            pins[canonicalize_name(m.group(1))] = m.group(2)
    return pins


class UndefinedExtra(ValueError):
    """An extra the lock is built for, or one rhob asks of itself, that pyproject.toml lacks."""


def declared(pyproject: Path = PYPROJECT, extras: tuple[str, ...] = EXTRAS) -> list[tuple[Requirement, str]]:
    """Each requirement of the base install and of ``extras``, with where it is declared.

    Raises UndefinedExtra for an extra pyproject.toml does not define: read as empty, it
    would pass the check while checking nothing (an extra renamed without a regeneration).
    """
    import tomllib  # Python 3.11+; the lock is for 3.12

    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    name = canonicalize_name(project["name"])
    optional = project.get("optional-dependencies", {})

    def lines_of(extra):
        if extra not in optional:
            raise UndefinedExtra(f"pyproject.toml defines no extra {extra!r}; the lock is built for "
                                 f"{', '.join(extras)} (scripts/update_ci_lock.sh)")
        return optional[extra]

    out: list[tuple[Requirement, str]] = []
    todo = [("", project.get("dependencies", []))] + [(e, lines_of(e)) for e in extras]
    done = set()
    while todo:
        where, lines = todo.pop(0)
        if where in done:
            continue
        done.add(where)
        for line in lines:
            req = Requirement(line)
            if req.marker is not None and not req.marker.evaluate({**TARGET, "extra": ""}):
                continue
            if canonicalize_name(req.name) == name:  # rhob[...] inside its own extras
                todo += [(e, lines_of(e)) for e in sorted(req.extras)]
                continue
            out.append((req, f"[{where}]" if where else "dependencies"))
    return out


def static_unmet(pyproject: Path = PYPROJECT, lock: Path = LOCK) -> list[str]:
    pins = lock_pins(lock)
    unmet = []
    try:
        reqs = declared(pyproject)
    except UndefinedExtra as exc:
        return [str(exc)]
    for req, where in reqs:
        have = pins.get(canonicalize_name(req.name))
        if have is None:
            unmet.append(f"{req} ({where}): not in the lock")
        elif not req.specifier.contains(have, prereleases=True):
            unmet.append(f"{req} ({where}): the lock has {have}")
    return unmet


def installed_unmet(extras: tuple[str, ...] = EXTRAS) -> list[str]:
    provided = {canonicalize_name(e) for e in metadata.metadata("rhob").get_all("Provides-Extra") or []}
    unmet = [f"rhob as installed provides no extra {e!r}" for e in extras if canonicalize_name(e) not in provided]
    todo: list[tuple[Requirement, str]] = []
    for line in metadata.requires("rhob") or []:
        req = Requirement(line)
        if req.marker is None or any(req.marker.evaluate({"extra": e}) for e in ("", *extras)):
            todo.append((req, "rhob"))
    followed = set()
    while todo:
        req, via = todo.pop(0)
        try:
            have = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            unmet.append(f"{req} (from {via}): not installed")
            continue
        if not req.specifier.contains(have, prereleases=True):
            unmet.append(f"{req} (from {via}): installed {have}")
        for extra in sorted(req.extras):
            key = (canonicalize_name(req.name), extra)
            if key in followed:
                continue
            followed.add(key)
            for line in metadata.requires(req.name) or []:
                sub = Requirement(line)
                # Only what the extra adds; the base requirements are pip check's.
                if (sub.marker is not None and sub.marker.evaluate({"extra": extra})
                        and not sub.marker.evaluate({"extra": ""})):
                    todo.append((sub, f"{req.name}[{extra}]"))
    return unmet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--installed", action="store_true",
                        help="check the installed distributions instead of the lock file")
    args = parser.parse_args(argv)
    unmet = installed_unmet() if args.installed else static_unmet()
    what = "installed set" if args.installed else "lock"
    if unmet:
        print(f"The {what} no longer satisfies pyproject.toml; run scripts/update_ci_lock.sh "
              "(requirements/README.md):\n  " + "\n  ".join(unmet), file=sys.stderr)
        return 1
    print(f"The {what} satisfies every rhob requirement for {', '.join(EXTRAS)}"
          + (", extras followed" if args.installed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
