"""A disk layer under the families' in-process calibration memoization.

Every family calibrates a scalar (or a fitted reward-model vector) so its two variants
have a matched proxy. That calibration is a pure function of code and fixed seeds --
``mujoco_camping._calibrate`` depends on ``wind_up_end`` and nothing else, the RLHF-RM
fits depend only on their own constants -- and it is expensive: 74s of HalfCheetah
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
over-conservative: it includes a hash of the **entire defining module's source**, not a
hand-listed set of dependencies. Editing a constant, the measure function, a bracket,
or the tolerance all invalidate it. So does editing a docstring in the same module,
which costs a recomputation nobody minds and removes the class of bug where someone
adds a dependency and forgets to declare it.

Everything else fails safe. An unreadable, corrupt, or unparseable entry is a miss, not
an error. Writes go to a temp file and are atomically replaced, so concurrent xdist
workers cannot observe a half-written entry, and one file per key means they never
contend for the same one. ``RHOB_CALIB_CACHE=0`` disables the disk layer entirely and
is the first thing to try if a calibration is ever suspected.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

#: Bump when the on-disk encoding changes in a way older entries cannot satisfy.
_FORMAT = 1

_DEFAULT_DIR = Path(__file__).resolve().parents[3] / ".rhob_cache" / "calibration"


def cache_dir() -> Path:
    """Where entries live. ``RHOB_CALIB_CACHE_DIR`` overrides the repo-local default."""
    override = (os.environ.get("RHOB_CALIB_CACHE_DIR") or "").strip()
    return Path(override) if override else _DEFAULT_DIR


def enabled() -> bool:
    """False when ``RHOB_CALIB_CACHE`` is set to a falsey value."""
    raw = (os.environ.get("RHOB_CALIB_CACHE") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _module_fingerprint(func: Callable) -> str:
    """SHA-256 of the source of the module that defines ``func``.

    The whole module rather than the function: a calibration reads module-level
    constants and helper functions, and listing them by hand is exactly where a missed
    dependency would turn into a silently stale value. Falls back to a marker that
    never matches a real hash when the source is unavailable (a zipimport, an exec'd
    module), which makes such a function uncacheable rather than wrongly cached.
    """
    module = sys.modules.get(func.__module__)
    try:
        source = inspect.getsource(module)  # type: ignore[arg-type]
    except (OSError, TypeError):
        return ""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


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
