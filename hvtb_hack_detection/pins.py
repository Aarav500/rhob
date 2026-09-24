"""The exact HVTB task set and images this eval runs, and how to check a copy against them.

HVTB's 89 tasks are distributed through the Harbor registry, not a git host, and the
dataset states no licence, so this repository does not redistribute them. It pins them
instead: ``fixtures/hvtb_pins.json`` records the Harbor dataset digest, one content hash
per task directory, and one registry digest per prebuilt image. The loader refuses a task
directory whose content does not hash to its pin, and every sample runs its image by
digest rather than by a tag that can be re-pushed.

``task_digest`` reimplements Harbor's per-task content hash (harbor 0.23.0,
``harbor/publisher/packager.py``: ``Packager.collect_files`` and
``Packager.compute_content_hash``) with the standard library, so checking a download needs
neither the Harbor CLI nor Python 3.12. It covers only what that hash covers: the files a
Harbor task package is built from, with Harbor's default ignore patterns applied. It
refuses a task directory that carries its own ``.gitignore``, which Harbor would honour
and this does not; none of the 89 pinned tasks has one.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

_PINS_RESOURCE = "fixtures/hvtb_pins.json"

#: Top-level files and directories a Harbor task package is built from.
_TASK_FILES = ("task.toml", "instruction.md", "README.md", "trajectory.json")
_TASK_DIRS = ("environment", "tests", "solution", "steps")

#: Harbor's default ignore patterns, applied to the final path component. Files under a
#: ``__pycache__`` directory are ignored separately.
_IGNORED_NAMES = ("*.pyc", ".DS_Store", "*.swp", "*.swo", "*~")

_CHUNK_BYTES = 1 << 20


@lru_cache(maxsize=1)
def load_pins() -> dict[str, Any]:
    """The pinned dataset and image digests, read once from the package."""
    text = files("hvtb_hack_detection").joinpath(_PINS_RESOURCE).read_text(encoding="utf-8")
    pins: dict[str, Any] = json.loads(text)
    return pins


def pinned_download_command() -> str:
    """The Harbor command that fetches exactly the pinned task set."""
    pins = load_pins()
    return (
        f"uvx --from harbor=={pins['harbor_cli_version']} harbor download "
        f"{pins['harbor_dataset']}@{pins['harbor_dataset_digest']} -o <dir>"
    )


def _ignored(relative: str) -> bool:
    parts = relative.split("/")
    if "__pycache__" in parts[:-1]:
        return True
    return any(fnmatch.fnmatchcase(parts[-1], pattern) for pattern in _IGNORED_NAMES)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_digest(task_dir: Path) -> str:
    """Harbor's content hash of one task directory.

    Args:
        task_dir: A task directory as Harbor exports it (``<dir>/<dataset>/<task>``).

    Returns:
        The hex SHA-256 Harbor publishes for the task.

    Raises:
        ValueError: If the directory has a task-level ``.gitignore``.
    """
    task_dir = Path(task_dir).resolve()
    if (task_dir / ".gitignore").exists():
        raise ValueError(f"{task_dir}: a task-level .gitignore is not supported")
    found = [task_dir / name for name in _TASK_FILES if (task_dir / name).is_file()]
    for name in _TASK_DIRS:
        if (task_dir / name).is_dir():
            found.extend(p for p in (task_dir / name).rglob("*") if p.is_file())
    relatives = sorted({p.relative_to(task_dir).as_posix() for p in found})
    outer = hashlib.sha256()
    for relative in relatives:
        if not _ignored(relative):
            outer.update(f"{relative}\0{_file_sha256(task_dir / relative)}\n".encode())
    return outer.hexdigest()
