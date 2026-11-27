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


def _fresh_import(tmp_path, package: str, files: dict[str, str]):
    """Write a throwaway package and import its ``fam`` module from scratch."""
    import importlib  # noqa: PLC0415
    import linecache  # noqa: PLC0415

    pkg = tmp_path / package
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for name, body in files.items():
        (pkg / f"{name}.py").write_text(textwrap.dedent(body), encoding="utf-8")
    for mod in [m for m in sys.modules if m == package or m.startswith(package + ".")]:
        sys.modules.pop(mod)
    importlib.invalidate_caches()
    linecache.clearcache()
    return importlib.import_module(f"{package}.fam")


def test_editing_a_module_the_calibration_imports_changes_its_fingerprint(tmp_path, monkeypatch):
    """The helper a family calls is part of its calibration.

    Before 2026-11-15 the key hashed the defining module alone, so editing
    ``rhob.environments.calibration`` -- where ``calibrate_scale`` itself lives -- left
    every cached calibration in place. The import here sits inside the function body, the
    form a top-level-only scan would miss.
    """
    monkeypatch.syspath_prepend(str(tmp_path))
    fam = """
        def calibrate():
            from _calibprobe_pkg.helper import AMPLITUDE
            return AMPLITUDE
    """
    before = cc._module_fingerprint(
        _fresh_import(tmp_path, "_calibprobe_pkg", {"fam": fam, "helper": "AMPLITUDE = 0.20\n"}).calibrate
    )
    after = cc._module_fingerprint(
        _fresh_import(tmp_path, "_calibprobe_pkg", {"fam": fam, "helper": "AMPLITUDE = 0.21\n"}).calibrate
    )
    assert before and after
    assert before != after, "a helper the calibration imports changed and the fingerprint did not"


_PROBE_PACKAGE = "_calibprobe_doc_pkg"

_FAM = '''
"""A probe family."""

from _calibprobe_doc_pkg.helper import scale


def calibrate():
    """Bisect the amplitude."""
    return scale(0.20)
'''

_HELPER = '''
"""Shared helper."""


def scale(x):
    """Double it."""
    return 2 * x
'''


def _fingerprint_with(tmp_path, monkeypatch, fam: str = _FAM, helper: str = _HELPER) -> str:
    monkeypatch.syspath_prepend(str(tmp_path))
    module = _fresh_import(tmp_path, _PROBE_PACKAGE, {"fam": fam, "helper": helper})
    fingerprint = cc._module_fingerprint(module.calibrate)
    assert fingerprint
    return fingerprint


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda s: s.replace('"""A probe family."""', '"""Reworded."""'),
                     id="module-docstring"),
        pytest.param(lambda s: s.replace('"""Bisect the amplitude."""',
                                         '"""Bisect it.\n\n    A longer story.\n    """'),
                     id="function-docstring"),
        pytest.param(lambda s: s.replace("    return scale(0.20)",
                                         "    # the published amplitude\n    return scale(0.20)"),
                     id="comment"),
        pytest.param(lambda s: s.replace("    return scale(0.20)", "    return scale( 0.20 )")
                     + "\n\n\n", id="layout"),
    ],
)
def test_a_docstring_comment_or_layout_edit_keeps_the_fingerprint(tmp_path, monkeypatch, edit):
    """Five times in eleven days a docstring or comment edit, and nothing else, threw away
    every calibrated family's entry and cost a cold recalibration on CI. None of them
    could change a calibrated value, so none of them may change the key."""
    before = _fingerprint_with(tmp_path, monkeypatch)
    edited = edit(_FAM)
    assert edited != _FAM, "the edit did not apply; the test would prove nothing"
    assert _fingerprint_with(tmp_path, monkeypatch, fam=edited) == before


def test_a_docstring_edit_to_an_imported_helper_keeps_the_fingerprint(tmp_path, monkeypatch):
    before = _fingerprint_with(tmp_path, monkeypatch)
    helper = _HELPER.replace('"""Double it."""', '"""Multiply by two, as the paper says."""')
    helper = helper.replace('"""Shared helper."""', '"""Shared helper.\n\nNow documented."""')
    assert helper != _HELPER
    assert _fingerprint_with(tmp_path, monkeypatch, helper=helper) == before


@pytest.mark.parametrize(
    "edit",
    [
        # A docstring turned into code next to it: the docstring is still stripped, the
        # new statement is not.
        pytest.param(lambda s: s.replace('"""Bisect the amplitude."""',
                                         '"""Bisect the amplitude."""\n    X = 1'),
                     id="statement-added-after-docstring"),
        pytest.param(lambda s: s.replace("0.20", "0.21"), id="constant"),
        pytest.param(lambda s: s.replace("    return scale(0.20)",
                                         '    "not a docstring"\n    return scale(0.20)'),
                     id="bare-string-that-is-not-first"),
        # A no-op as code, and still a miss: ``[pass, return]`` is not the tree
        # ``[return]``. The normalization does not try to decide which edits are
        # semantically empty beyond docstrings; pinned so that stays a decision.
        pytest.param(lambda s: s.replace('"""Bisect the amplitude."""', "pass"),
                     id="docstring-replaced-by-a-statement"),
    ],
)
def test_any_other_edit_still_changes_the_fingerprint(tmp_path, monkeypatch, edit):
    """The normalization must strip docstrings and nothing else. A bare string that is
    not the first statement is kept on purpose -- an over-conservative miss, never a
    shared key between two programs that compute different things."""
    before = _fingerprint_with(tmp_path, monkeypatch)
    edited = edit(_FAM)
    assert edited != _FAM
    assert _fingerprint_with(tmp_path, monkeypatch, fam=edited) != before


def test_editing_a_helper_constant_still_changes_the_fingerprint(tmp_path, monkeypatch):
    before = _fingerprint_with(tmp_path, monkeypatch)
    assert _fingerprint_with(tmp_path, monkeypatch, helper=_HELPER.replace("2 * x", "3 * x")) != before


@pytest.mark.parametrize(
    "source, stripped_equivalent",
    [
        ('"""doc"""\nX = 1\n', "X = 1\n"),
        ("class C:\n    '''doc'''\n    y = 2\n", "class C:\n    y = 2\n"),
        ("def f():\n    'doc'\n    return 1\n", "def f():\n    return 1\n"),
        ("async def f():\n    '''doc'''\n    return 1\n", "async def f():\n    return 1\n"),
        # A body that was only a docstring executes as ``pass``.
        ("def f():\n    '''doc'''\n", "def f():\n    pass\n"),
        ("class C:\n    '''doc'''\n", "class C:\n    pass\n"),
        ('"""only a docstring"""\n', "pass\n"),
        ("X = 1  # a comment\n\n\n", "X = 1\n"),
    ],
)
def test_normalized_strips_exactly_the_docstrings(source, stripped_equivalent):
    assert cc._normalized(source) == cc._normalized(stripped_equivalent)


@pytest.mark.parametrize(
    "a, b",
    [
        # Not docstrings: a bytes literal, an f-string and a second string statement.
        ("def f():\n    b'doc'\n    return 1\n", "def f():\n    return 1\n"),
        ("def f():\n    f'doc'\n    return 1\n", "def f():\n    return 1\n"),
        ("def f():\n    return 1\n    'tail'\n", "def f():\n    return 1\n"),
        ("X = 'a'\n", "X = 'b'\n"),
    ],
)
def test_normalized_keeps_every_string_that_is_not_a_docstring(a, b):
    assert cc._normalized(a) != cc._normalized(b)


def test_a_docstring_edit_to_posthoc_keeps_every_calibrated_familys_key(monkeypatch):
    """The real case. ``rhob.detectors.posthoc`` defines ``RunData``, so every calibrated
    family reaches it, and three of the five full invalidations were edits to its
    docstrings alone (96d928c, f83a152, 67c73b4). Replayed here on the family whose
    calibration is the most expensive, through the real import walk."""
    import importlib  # noqa: PLC0415

    name = "rhob.v3.families.mujoco_joint_limit_gaming"
    module = importlib.import_module(name)
    real_source_of = cc._source_of

    def with_posthoc(transform):
        def source_of(dep):
            found = real_source_of(dep)
            if dep != "rhob.detectors.posthoc" or found is None:
                return found
            text, is_package = found
            new = transform(text)
            assert new != text, "the posthoc edit did not apply"
            return new, is_package

        return source_of

    before = cc._compute_fingerprint(name, module)
    assert "rhob.detectors.posthoc" in cc.reachable_sources(name, module)

    monkeypatch.setattr(cc, "_source_of", with_posthoc(
        lambda t: t.replace('"""A completed run', '"""A finished run', 1) + "\n# a comment\n"
    ))
    assert cc._compute_fingerprint(name, module) == before

    monkeypatch.setattr(cc, "_source_of", with_posthoc(lambda t: t + "\n_ADDED = 1\n"))
    assert cc._compute_fingerprint(name, module) != before


def test_normalization_is_memoized_across_families():
    """The modules every family reaches are parsed once per process, not once each."""
    import importlib  # noqa: PLC0415

    names = ["rhob.v3.families.mujoco_joint_limit_gaming", "rhob.v3.families.mujoco_camping"]
    cc._code_digest.cache_clear()
    for name in names:
        cc._compute_fingerprint(name, importlib.import_module(name))
    info = cc._code_digest.cache_info()
    assert info.hits > 0, "two families sharing base_pair et al. re-parsed every module"


# ------------------------------------------------------------------- numerics pins

#: What CI sets (the shared contract). The cache does not read these values; it hashes
#: whatever the process has, so the test only pins the names.
_CI_PINS = {
    "OPENBLAS_CORETYPE": "Haswell",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_CBWR": "COMPATIBLE",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "ATEN_CPU_CAPABILITY": "avx2",
}


@pytest.fixture
def _fresh_token():
    cc.environment_token.cache_clear()
    yield
    cc.environment_token.cache_clear()


def _token_under(monkeypatch, env: dict[str, str]) -> str:
    for name in cc.NUMERICS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    cc.environment_token.cache_clear()
    return cc.environment_token()


def test_the_key_hashes_exactly_the_pins_ci_sets():
    assert set(cc.NUMERICS_ENV_VARS) == set(_CI_PINS)


def test_an_unset_numerics_variable_is_hashed_as_unset(monkeypatch):
    for name in cc.NUMERICS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    assert cc.numerics_env() == {name: "unset" for name in cc.NUMERICS_ENV_VARS}
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert cc.numerics_env()["OMP_NUM_THREADS"] == "1"


@pytest.mark.usefixtures("_fresh_token")
@pytest.mark.parametrize("name", sorted(_CI_PINS))
def test_each_numerics_pin_is_part_of_the_environment_token(monkeypatch, name):
    """Pinned and unpinned calibrations can differ in the last bits, so they must never
    share an entry -- nor may a run that dropped just one of the pins."""
    unpinned = _token_under(monkeypatch, {})
    pinned = _token_under(monkeypatch, _CI_PINS)
    one_missing = _token_under(monkeypatch, {k: v for k, v in _CI_PINS.items() if k != name})
    only_this = _token_under(monkeypatch, {name: _CI_PINS[name]})
    assert len({unpinned, pinned, one_missing, only_this}) == 4


@pytest.mark.usefixtures("_fresh_token")
def test_pinning_the_numerics_turns_a_warm_entry_into_a_miss(monkeypatch):
    """End to end through the memoizer, as for a version change."""
    calibrate, calls = _counting_fn()
    _token_under(monkeypatch, {})
    calibrate(2)
    calibrate.cache_clear()
    calibrate(2)
    assert calls["n"] == 1, "precondition: the entry is warm"

    _token_under(monkeypatch, _CI_PINS)
    calibrate.cache_clear()
    calibrate(2)
    assert calls["n"] == 2, "a pinned run was served an unpinned calibration"


def test_the_key_changes_when_a_physics_library_changes_version(monkeypatch):
    """The failure that motivated this: MuJoCo 4.0.0 replaced 3.17.0 overnight, CI
    installed unpinned, and the cache served the 3.17 calibration under 4.0 physics."""
    calibrate, _ = _counting_fn()
    fn = calibrate.__wrapped__
    real = cc.physics_versions()

    def versions(mujoco):
        return lambda: {**real, "mujoco": mujoco}

    try:
        monkeypatch.setattr(cc, "physics_versions", versions("3.17.0"))
        cc.environment_token.cache_clear()
        old = cc._key(fn, "same-source", (1,), {})
        monkeypatch.setattr(cc, "physics_versions", versions("4.0.0"))
        cc.environment_token.cache_clear()
        new = cc._key(fn, "same-source", (1,), {})
    finally:
        cc.environment_token.cache_clear()
    assert old != new


def test_a_version_change_turns_a_warm_entry_into_a_miss(monkeypatch):
    """End to end through the memoizer, not just the key function."""
    calibrate, calls = _counting_fn()
    real = cc.physics_versions()
    try:
        calibrate(2)
        calibrate.cache_clear()
        calibrate(2)
        assert calls["n"] == 1, "precondition: the entry is warm"

        monkeypatch.setattr(cc, "physics_versions", lambda: {**real, "numpy": "99.0.0"})
        cc.environment_token.cache_clear()
        calibrate.cache_clear()
        calibrate(2)
        assert calls["n"] == 2, "a numpy upgrade was served the old entry"
    finally:
        cc.environment_token.cache_clear()


def test_every_physics_distribution_is_reported():
    versions = cc.physics_versions()
    assert set(versions) == set(cc.PHYSICS_DISTRIBUTIONS)
    assert versions["numpy"] != "absent"


def test_a_shipped_mujoco_family_hashes_the_helpers_it_calibrates_with():
    """Pinned on the family whose stale calibration turned the nightly red."""
    import importlib  # noqa: PLC0415

    name = "rhob.v3.families.mujoco_joint_limit_gaming"
    reached = cc.reachable_sources(name, importlib.import_module(name))
    assert reached is not None
    assert {"rhob.environments.mujoco.rollout", "rhob.environments.calibration"} <= set(reached)
    assert not [m for m in reached if not m.startswith("rhob.")], "only rhob's own source is hashed"


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
