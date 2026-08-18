"""The disk layer under the families' calibration memoization.

A calibration cache is the kind of optimization that does not fail loudly. If it ever
returns a stale value the physics silently changes under every published number, and
nothing goes red. So the properties worth pinning are not "is it fast" but: the cached
value equals the computed one, a change to the defining module invalidates it, and every
way it can go wrong degrades to a recomputation rather than to a wrong answer.
"""

from __future__ import annotations

import json
import sys
import textwrap

import numpy as np
import pytest

from rhob.v3 import calibration_cache as cc


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Every test gets its own empty cache directory."""
    monkeypatch.setenv("RHOB_CALIB_CACHE_DIR", str(tmp_path / "calib"))
    monkeypatch.delenv("RHOB_CALIB_CACHE", raising=False)
    return tmp_path / "calib"


# --------------------------------------------------------------------------- encoding


@pytest.mark.parametrize(
    "value",
    [
        0.20312, -1.0, 0, 7, True, False, None, "amplitude",
        (True, False, True),
        (0.1, 0.2),
        np.array([1.5, -2.5, 3.0]),
        np.zeros((2, 3)),
        np.array([1, 2, 3], dtype=np.int64),
    ],
    ids=repr,
)
def test_encoding_round_trips_every_type_a_calibration_returns(value):
    decoded = cc._decode(json.loads(json.dumps(cc._encode(value))))
    if isinstance(value, np.ndarray):
        assert isinstance(decoded, np.ndarray)
        assert decoded.dtype == value.dtype
        np.testing.assert_array_equal(decoded, value)
    else:
        assert decoded == value
        assert type(decoded) is type(value)


def test_encoding_refuses_what_it_cannot_represent():
    """Better a loud TypeError at write time than a silently mangled calibration."""
    with pytest.raises(TypeError):
        cc._encode({"a": 1})


# ------------------------------------------------------------------------ the memoizer


def _counting_fn():
    calls = {"n": 0}

    @cc.disk_memo
    def calibrate(x: int) -> float:
        calls["n"] += 1
        return x * 1.5

    return calibrate, calls


def test_value_is_identical_to_the_uncached_computation():
    calibrate, _ = _counting_fn()
    assert calibrate(4) == calibrate.__wrapped__(4)


def test_a_second_process_would_hit_the_disk_not_recompute():
    """Clearing the in-process layer stands in for the next xdist worker."""
    calibrate, calls = _counting_fn()
    first = calibrate(3)
    assert calls["n"] == 1

    calibrate.cache_clear()          # in-process memo gone; disk entry remains
    second = calibrate(3)
    assert second == first
    assert calls["n"] == 1, "recomputed despite a warm disk entry"


def test_distinct_arguments_do_not_collide():
    calibrate, calls = _counting_fn()
    assert calibrate(2) == 3.0
    assert calibrate(4) == 6.0
    assert calls["n"] == 2


def test_a_corrupt_entry_is_a_miss_not_an_error(_isolated_cache):
    calibrate, calls = _counting_fn()
    calibrate(5)
    entries = list(_isolated_cache.glob("*.json"))
    assert len(entries) == 1
    entries[0].write_text("{not json at all", encoding="utf-8")

    calibrate.cache_clear()
    assert calibrate(5) == 7.5          # recomputed, no exception
    assert calls["n"] == 2


def test_an_entry_from_an_older_encoding_is_a_miss(_isolated_cache):
    calibrate, calls = _counting_fn()
    calibrate(6)
    entry = next(_isolated_cache.glob("*.json"))
    entry.write_text(json.dumps({"value": {"t": "from-the-future", "v": 1}}), encoding="utf-8")

    calibrate.cache_clear()
    assert calibrate(6) == 9.0
    assert calls["n"] == 2


def test_writes_leave_no_temp_files_behind(_isolated_cache):
    """Entries are written to a temp file and atomically replaced, so a concurrent
    worker never reads a half-written entry."""
    calibrate, _ = _counting_fn()
    for x in range(5):
        calibrate(x)
    assert list(_isolated_cache.glob("*.tmp")) == []
    assert len(list(_isolated_cache.glob("*.json"))) == 5


def test_disabling_the_env_var_bypasses_the_disk_entirely(_isolated_cache, monkeypatch):
    monkeypatch.setenv("RHOB_CALIB_CACHE", "0")
    calibrate, calls = _counting_fn()
    calibrate(9)
    calibrate.cache_clear()
    calibrate(9)
    assert calls["n"] == 2, "in-process only; nothing should have been read back"
    assert not _isolated_cache.exists() or list(_isolated_cache.glob("*.json")) == []


def test_an_unwritable_cache_directory_does_not_break_calibration(tmp_path, monkeypatch):
    """A cache that cannot be written is a cache that is not used, never a crash.

    The directory is pointed at a path whose parent is a regular file, so ``mkdir``
    genuinely fails -- rather than monkeypatching ``_write`` to raise, which would only
    prove the monkeypatch works.
    """
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setenv("RHOB_CALIB_CACHE_DIR", str(blocker / "calib"))

    calibrate, calls = _counting_fn()
    assert calibrate(1) == 1.5
    calibrate.cache_clear()
    assert calibrate(1) == 1.5      # still correct, just uncached
    assert calls["n"] == 2


# ------------------------------------------------------------------------ invalidation


def test_the_key_changes_when_the_module_fingerprint_changes():
    calibrate, _ = _counting_fn()
    fn = calibrate.__wrapped__
    a = cc._key(fn, "fingerprint-before", (1,), {})
    b = cc._key(fn, "fingerprint-after", (1,), {})
    assert a != b


def test_editing_the_defining_module_changes_its_fingerprint(tmp_path, monkeypatch):
    """The property the whole design rests on: touch the source, lose the cache."""
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = tmp_path / "_calib_fingerprint_probe.py"

    mod.write_text(textwrap.dedent("""
        AMPLITUDE = 0.20
        def calibrate():
            return AMPLITUDE
    """), encoding="utf-8")
    sys.modules.pop("_calib_fingerprint_probe", None)
    import _calib_fingerprint_probe as probe  # noqa: PLC0415
    before = cc._module_fingerprint(probe.calibrate)

    # Change only a constant the function reads -- not the function body itself.
    mod.write_text(textwrap.dedent("""
        AMPLITUDE = 0.21
        def calibrate():
            return AMPLITUDE
    """), encoding="utf-8")
    sys.modules.pop("_calib_fingerprint_probe", None)
    import importlib  # noqa: PLC0415
    import linecache  # noqa: PLC0415
    linecache.clearcache()
    probe2 = importlib.import_module("_calib_fingerprint_probe")
    after = cc._module_fingerprint(probe2.calibrate)

    assert before and after
    assert before != after, (
        "a constant the calibration reads changed and the fingerprint did not -- "
        "this is the stale-value hazard the whole-module hash exists to close"
    )


# --------------------------------------------------------------- against a real family


@pytest.mark.slow
def test_a_real_family_calibration_matches_the_uncached_value(monkeypatch):
    """End to end on a shipped family: the cache must not move the number.

    Marked slow -- deliberately, and measured rather than assumed. It derives
    ``sequence_keyword_stuffing._legit_target_proxy`` twice, once through the cache and
    once with ``RHOB_CALIB_CACHE=0``, at 55s each: 113s added to a per-push suite that
    otherwise runs in about three and a half minutes. The synthetic tests above already
    pin every property of the mechanism, and the decorator is the same one the MuJoCo
    families use, so what this adds is the end-to-end reassurance rather than the
    coverage -- worth having nightly, not worth a third of the push job.
    """
    from rhob.v3.families import sequence_keyword_stuffing as fam  # noqa: PLC0415

    fam._legit_target_proxy.cache_clear()
    cached = fam._legit_target_proxy()

    monkeypatch.setenv("RHOB_CALIB_CACHE", "0")
    fam._legit_target_proxy.cache_clear()
    uncached = fam._legit_target_proxy()

    assert cached == uncached
