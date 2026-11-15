"""A disk layer under the families' in-process calibration memoization.

Every family calibrates a scalar (or a fitted reward-model vector) so its two variants
have a matched proxy. That calibration is a pure function of code, fixed seeds and the
installed numerical stack -- ``mujoco_camping._calibrate`` depends on ``wind_up_end``, the
code that rolls HalfCheetah out, and the MuJoCo that simulates it -- and it is expensive:
74s of HalfCheetah
rollouts for one ``mujoco_camping`` tier, against 229s for that tier's entire 96-rollout
screen.

``functools.lru_cache`` already shares it *within a process*. That was enough while the
smoke screen was one test per family looping over tiers. It stopped being enough when
the screen became one test per (family, difficulty) so the tiers could be distributed:
three tiers on three ``pytest-xdist`` workers means three processes each re-deriving the
same amplitude. Parallelising without this file would have bought wall clock by burning
strictly more CPU.

Correctness before speed
------------------------
A calibration cache that returns a stale value does not fail loudly -- it silently
changes the physics under every published number. So the key is deliberately
over-conservative. It hashes three things, none of them a hand-listed set of
dependencies:

- the **entire source of the defining module**, so editing a constant, the measure
  function, a bracket or the tolerance invalidates it (so does a docstring, which costs a
  recomputation nobody minds);
- the source of **every module of the same package it imports**, followed transitively
  through ``import`` statements anywhere in the file, function bodies included, so a
  change to the shared rollout helper a family calls is a change to the family's
  calibration;
- the **installed versions of the numerical stack** (:data:`PHYSICS_DISTRIBUTIONS`), the
  Python minor version and the machine architecture, because the same code under a new
  simulator is a different calibration.

The last two were added after the cache served a wrong number for a week. The key used
to hash the defining module alone, and CI installs unpinned dependencies. MuJoCo 4.0.0
was published on 2026-11-06, and from the next nightly ``mujoco_joint_limit_gaming`` was
rolled out under 4.0 with the hip amplitude calibrated under 3.17: its 0.9 tier stopped
admitting (L0 AUROC 0.708) and a strict xfail at 0.7 started passing, on an unchanged
commit. The key could not have changed, because nothing it read had. That is the failure
this repository's paper is about, in the cache that guards its own numbers.

Everything else fails safe. An unreadable, corrupt, or unparseable entry is a miss, not
an error. Writes go to a temp file and are atomically replaced, so concurrent xdist
workers cannot observe a half-written entry, and one file per key means they never
contend for the same one. ``RHOB_CALIB_CACHE=0`` disables the disk layer entirely and
is the first thing to try if a calibration is ever suspected.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

import numpy as np

#: Bump when the on-disk encoding changes in a way older entries cannot satisfy.
_FORMAT = 1

#: Distributions whose version is part of every key: anything a calibration's numbers can
#: depend on. Over-inclusive on purpose -- an upgrade that changes nothing costs one
#: recomputation; one that changes the physics and is missed costs a wrong number.
PHYSICS_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "mujoco",
    "mujoco-core",
    "gymnasium",
    "torch",
    "pettingzoo",
    "mpe2",
)

_DEFAULT_DIR = Path(__file__).resolve().parents[3] / ".rhob_cache" / "calibration"


def cache_dir() -> Path:
    """Where entries live. ``RHOB_CALIB_CACHE_DIR`` overrides the repo-local default."""
    override = (os.environ.get("RHOB_CALIB_CACHE_DIR") or "").strip()
    return Path(override) if override else _DEFAULT_DIR


def enabled() -> bool:
    """False when ``RHOB_CALIB_CACHE`` is set to a falsey value."""
    raw = (os.environ.get("RHOB_CALIB_CACHE") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def physics_versions() -> dict[str, str]:
    """Installed version of each of :data:`PHYSICS_DISTRIBUTIONS` ("absent" if not)."""
    out = {}
    for name in PHYSICS_DISTRIBUTIONS:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "absent"
    return out


@functools.lru_cache(maxsize=None)
def environment_token() -> str:
    """Hash of everything outside the source tree that a calibration's value depends on.

    Also what CI keys its cache tarball on, so a dependency upgrade starts a new tarball
    instead of restoring entries that would all miss anyway.
    """
    payload = json.dumps(
        {
            "versions": physics_versions(),
            "python": "%d.%d" % sys.version_info[:2],
            "machine": platform.machine(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _imported_names(module_name: str, source: str, is_package: bool) -> set[str]:
    """Every module name an ``import`` in ``source`` could load, relative ones resolved.

    Walks the whole tree, not just the top level: a calibration that imports its helper
    inside the function body depends on it all the same. ``from a import b`` yields both
    ``a`` and ``a.b``, since ``b`` may be a submodule; names that turn out not to be
    modules are dropped by the caller.
    """
    package = module_name if is_package else module_name.rpartition(".")[0]
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                anchor = ".".join(parts[: len(parts) - node.level + 1])
                base = f"{anchor}.{node.module}" if node.module else anchor
            else:
                base = node.module or ""
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def _source_of(name: str) -> tuple[str, bool] | None:
    """(source, is_package) of an importable pure-Python module, else None."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, AttributeError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    try:
        return Path(spec.origin).read_text(encoding="utf-8"), spec.submodule_search_locations is not None
    except OSError:
        return None


def _module_fingerprint(func: Callable) -> str:
    """SHA-256 over the source of ``func``'s module and of every module it reaches.

    The whole module rather than the function: a calibration reads module-level constants
    and helper functions, and listing them by hand is exactly where a missed dependency
    would turn into a silently stale value. For the same reason the hash follows the
    module's imports, transitively, within its own top-level package (``rhob`` for every
    shipped family). Third-party code is covered by version instead, in
    :func:`environment_token`. Falls back to a marker that never matches a real hash when
    the defining module's source is unavailable (a zipimport, an exec'd module), which
    makes such a function uncacheable rather than wrongly cached.
    """
    module = sys.modules.get(func.__module__)
    if module is None:
        return ""
    # Memoized per module *object*: a re-imported module is a new object and is hashed
    # afresh, while repeated calls from one imported module -- every calibration call
    # goes through here -- pay for the import walk once.
    return _fingerprint_of_module(func.__module__, module)


_FINGERPRINTS: dict[tuple[str, int], tuple[Any, str]] = {}


def _fingerprint_of_module(module_name: str, module: Any) -> str:
    cached = _FINGERPRINTS.get((module_name, id(module)))
    if cached is not None and cached[0] is module:
        return cached[1]
    value = _compute_fingerprint(module_name, module)
    _FINGERPRINTS[(module_name, id(module))] = (module, value)
    return value


def reachable_sources(module_name: str, module: Any) -> dict[str, str] | None:
    """Source of ``module`` and of every same-package module its imports reach.

    None when the module's own source is unavailable, or when the walk meets a file it
    cannot parse: either way the fingerprint cannot be trusted, so the caller must not
    cache.
    """
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):
        return None
    root = module_name.partition(".")[0]
    sources = {module_name: source}
    frontier = [(module_name, source, hasattr(module, "__path__"))]
    while frontier:
        name, text, is_package = frontier.pop()
        try:
            imported = _imported_names(name, text, is_package)
        except SyntaxError:
            return None
        for dep in sorted(imported):
            if dep in sources or (dep != root and not dep.startswith(root + ".")):
                continue
            found = _source_of(dep)
            if found is None:
                continue
            sources[dep] = found[0]
            frontier.append((dep, found[0], found[1]))
    return sources


def _compute_fingerprint(module_name: str, module: Any) -> str:
    sources = reachable_sources(module_name, module)
    if sources is None:
        return ""
    digest = hashlib.sha256()
    for name in sorted(sources):
        digest.update(name.encode("utf-8") + b"\0" + sources[name].encode("utf-8") + b"\0")
    return digest.hexdigest()


def _encode(value: Any) -> dict[str, Any]:
    """JSON-safe envelope that round-trips the types the calibrations return."""
    if isinstance(value, np.ndarray):
        return {"t": "ndarray", "dtype": str(value.dtype), "shape": list(value.shape),
                "v": value.ravel().tolist()}
    if isinstance(value, tuple):
        return {"t": "tuple", "v": [_encode(x) for x in value]}
    if isinstance(value, (bool, int, float, str)) or value is None:
        return {"t": "scalar", "v": value}
    if isinstance(value, np.generic):
        return {"t": "scalar", "v": value.item()}
    raise TypeError(f"calibration cache cannot encode {type(value).__name__}")


def _decode(blob: dict[str, Any]) -> Any:
    kind = blob["t"]
    if kind == "scalar":
        return blob["v"]
    if kind == "tuple":
        return tuple(_decode(x) for x in blob["v"])
    if kind == "ndarray":
        return np.asarray(blob["v"], dtype=np.dtype(blob["dtype"])).reshape(blob["shape"])
    raise ValueError(f"unknown cache entry kind {kind!r}")


def _key(func: Callable, fingerprint: str, args: tuple, kwargs: dict) -> str:
    payload = json.dumps(
        {
            "format": _FORMAT,
            "module": func.__module__,
            "qualname": func.__qualname__,
            "fingerprint": fingerprint,
            "environment": environment_token(),
            "args": [repr(a) for a in args],
            "kwargs": {k: repr(v) for k, v in sorted(kwargs.items())},
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read(path: Path) -> tuple[bool, Any]:
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
        return True, _decode(blob["value"])
    except (OSError, ValueError, KeyError, TypeError):
        # Unreadable or from an older encoding: a miss, never an error.
        return False, None


def _write(path: Path, func: Callable, args: tuple, value: Any) -> None:
    try:
        record = {
            "value": _encode(value),
            # Provenance for a human staring at a suspicious entry. Never read back.
            "produced_by": f"{func.__module__}.{func.__qualname__}",
            "args": [repr(a) for a in args],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(record, fh)
            os.replace(tmp, path)  # atomic; concurrent workers never see a partial file
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except (OSError, TypeError, ValueError):
        # A cache that cannot be written is a cache that is not used. Never fatal.
        pass


def disk_memo(func: Callable) -> Callable:
    """Memoize ``func`` in-process *and* on disk.

    A drop-in replacement for ``@functools.lru_cache(maxsize=None)`` on the families'
    calibration entry points. The in-process layer is kept underneath, so repeated calls
    within one process still never touch the filesystem.

    Exposes ``cache_clear()`` (clears only the in-process layer, matching lru_cache) and
    ``__wrapped__`` for tests that need the uncached function.
    """
    memo = functools.lru_cache(maxsize=None)(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not enabled():
            return memo(*args, **kwargs)
        fingerprint = _module_fingerprint(func)
        if not fingerprint:
            return memo(*args, **kwargs)
        path = cache_dir() / f"{_key(func, fingerprint, args, kwargs)}.json"
        hit, value = _read(path)
        if hit:
            return value
        value = memo(*args, **kwargs)
        _write(path, func, args, value)
        return value

    wrapper.cache_clear = memo.cache_clear  # type: ignore[attr-defined]
    wrapper.cache_info = memo.cache_info  # type: ignore[attr-defined]
    wrapper.__wrapped__ = func  # type: ignore[attr-defined]
    return wrapper
