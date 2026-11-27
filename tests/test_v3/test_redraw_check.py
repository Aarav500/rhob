"""The re-draw tooling: the board's cells sidecar, the replicate files, and the gate.

A re-draw replaces all 20 committed replicates at once (``regenerate.yml``'s ``redraw``
mode). It is only worth committing if the 20 files are one complete, uniform draw of one
clean commit, and replicate 0 -- the board's own draw ``(0, 0)`` -- reproduces the board
cell for cell. ``scripts/check_redraw.py`` is the gate; these tests show it accepts such
a set and refuses each way of not being one, including a single changed cell.

The second half runs the two scripts the long CI jobs run,
``scripts/v5_leaderboard_and_transfer.py --cells-out`` and
``scripts/replicate_leaderboard.py``, end to end on stub families that roll out in
milliseconds, and feeds what they write to the gate and to the aggregator. A crash in
either script after hours of rollouts costs every job of a re-draw; this is the cheap
place to find it.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

import rhob.v3.provenance as provenance_module
from rhob.detectors.posthoc import RunData
from rhob.v3.base_family import BaseFamily
from rhob.v3.base_pair import MatchedPair
from rhob.v3.benchmark import Benchmark
from rhob.v3.calibration_cache import NUMERICS_ENV_VARS
from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism

_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "scripts"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def check():
    return _load("_check_redraw_under_test", _SCRIPTS / "check_redraw.py")


@pytest.fixture(scope="module")
def board_script():
    return _load("_v5_board_under_test", _SCRIPTS / "v5_leaderboard_and_transfer.py")


@pytest.fixture(scope="module")
def replicate_script():
    return _load("_replicate_under_test", _SCRIPTS / "replicate_leaderboard.py")


@pytest.fixture(scope="module")
def aggregate_script():
    return _load("_aggregate_under_test", _SCRIPTS / "aggregate_replication.py")


# =============================================================== synthetic sets
_COMMIT = "c0ffee" + "0" * 34
_FAMILIES = ["fam_a", "fam_b"]
_DIFFICULTIES = (0.7, 0.8)
_N = 20


#: The CI pins, spelled out rather than read from the gate, so a change to either the gate or
#: regenerate.yml shows up here (see test_the_gate_requires_the_pins_regenerate_sets).
_PINS = {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_NUM_THREADS": "1", "MKL_CBWR": "COMPATIBLE",
         "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "ATEN_CPU_CAPABILITY": "avx2"}


def _pinned_numerics() -> dict:
    return {
        "env": dict(_PINS),
        "cpu": {"vendor": "AuthenticAMD", "model": "AMD EPYC 7763"},
        "torch_imported": True,
        "torch_cpu_capability": "AVX2",
        "blas": [{"internal_api": "openblas", "architecture": "Haswell", "num_threads": 1},
                 {"internal_api": "openmp", "architecture": None, "num_threads": 1}],
    }


def _provenance(generated: str, commit: str = _COMMIT) -> dict:
    return {
        "generated_utc": generated,
        "git": {"commit": commit, "short_commit": commit[:12], "branch": "b",
                "dirty": False, "dirty_files": [], "source": "git"},
        "python": "3.12.3",
        "python_implementation": "CPython",
        "platform": "Linux-6.8.0-x86_64-with-glibc2.39",
        "packages": {"numpy": "2.3.5", "torch": "2.9.1", "rhob": "1.8.0"},
        "numerics": _pinned_numerics(),
        "argv": [],
    }


def _cells(rid: int, k: int) -> list[dict]:
    """Detector ``k``'s cells on draw ``rid``; every third detector skips ``fam_b``."""
    out = []
    for f, fam in enumerate(_FAMILIES):
        for j, d in enumerate(_DIFFICULTIES):
            auroc = None if (fam == "fam_b" and k % 3 == 0) else round(
                0.4 + 0.01 * ((rid * 7 + k * 3 + f * 2 + j) % 50), 6
            )
            out.append({"family": fam, "difficulty": d, "auroc": auroc})
    return out


@pytest.fixture(scope="module")
def detector_names(check):
    return [d.name for d in check.DETECTORS]


@pytest.fixture
def good_set(check, detector_names):
    """A complete, uniform 20-draw set, its board and the board's cells sidecar."""
    board_prov = _provenance("2026-11-27T01:00:00Z")
    board = {
        "families": len(_FAMILIES),
        "families_evaluated": list(_FAMILIES),
        "provenance": board_prov,
        "sampling": {"n_seeds_per_variant": 5, "layout_seeds": [0]},
        "results": {
            name: {"access_level": "L0", "overall_auroc": 0.5, "per_family": {}}
            for name in detector_names
        },
    }
    board_cells = {
        "provenance": copy.deepcopy(board_prov),
        "cells": {name: _cells(0, k) for k, name in enumerate(detector_names)},
    }
    replicates = {}
    for rid in range(_N):
        layout_seed, seed_base = check.draw_for(rid)
        replicates[f"replicate_{rid:03d}.json"] = {
            "replicate_id": rid,
            "draw_schema": check.DRAW_SCHEMA,
            "layout_seed": layout_seed,
            "seed_base": seed_base,
            "n_seeds": 5,
            "families_evaluated": list(_FAMILIES),
            "detectors_evaluated": list(detector_names),
            "provenance": _provenance(f"2026-11-27T02:{rid:02d}:00Z"),
            "results": {
                name: {"access_level": "L0", "overall_auroc": 0.5,
                       "cells_detail": _cells(rid, k)}
                for k, name in enumerate(detector_names)
            },
        }
    return replicates, board, board_cells


def _failed(check, replicates, board, board_cells, **kwargs) -> dict[str, list[str]]:
    kwargs.setdefault("expect_commit", _COMMIT)
    kwargs.setdefault("registered_families", list(_FAMILIES))
    checks = check.run_checks(replicates, board, board_cells, **kwargs)
    return {c.name: c.problems for c in checks if not c.ok}


def test_a_complete_uniform_set_is_accepted(check, good_set):
    assert _failed(check, *good_set) == {}


def test_a_short_commit_prefix_is_accepted_but_not_a_short_stub(check, good_set):
    assert _failed(check, *good_set, expect_commit=_COMMIT[:12]) == {}
    assert "one commit" in _failed(check, *good_set, expect_commit=_COMMIT[:4])


def _rep(replicates, rid):
    return replicates[f"replicate_{rid:03d}.json"]


def _drop_replicate(r, b, c):
    del r["replicate_007.json"]


def _extra_replicate(r, b, c):
    extra = copy.deepcopy(_rep(r, 19))
    extra.update(replicate_id=20, layout_seed=20, seed_base=20_000_000)
    r["replicate_020.json"] = extra


def _misnumbered(r, b, c):
    _rep(r, 3)["replicate_id"] = 4


def _unparseable(r, b, c):
    r["replicate_011.json"] = "JSONDecodeError: Expecting value"


def _no_schema(r, b, c):
    _rep(r, 2).pop("draw_schema")


def _old_schema(r, b, c):
    _rep(r, 2)["draw_schema"] = 1


def _wrong_draw(r, b, c):
    _rep(r, 5)["layout_seed"] = 0


def _seeds_differ(r, b, c):
    _rep(r, 2)["n_seeds"] = 4


def _board_seeds_differ(r, b, c):
    b["sampling"]["n_seeds_per_variant"] = 10


def _detector_missing(r, b, c):
    rep = _rep(r, 9)
    rep["detectors_evaluated"] = [n for n in rep["detectors_evaluated"] if n != "Reward Peak"]
    rep["results"].pop("Reward Peak")


def _board_detector_missing(r, b, c):
    b["results"].pop("Reward Peak")
    c["cells"].pop("Reward Peak")


def _family_missing(r, b, c):
    _rep(r, 4)["families_evaluated"] = _FAMILIES[:1]


def _board_family_missing(r, b, c):
    b["families_evaluated"] = _FAMILIES[:1]


def _error_record(r, b, c):
    _rep(r, 6)["results"]["Reward Peak"] = {"access_level": "L0", "error": "boom"}


def _board_error(r, b, c):
    b["results"]["Reward Peak"] = {"access_level": "L0", "error": "boom"}


def _no_cells_detail(r, b, c):
    _rep(r, 6)["results"]["Reward Peak"].pop("cells_detail")


def _other_commit(r, b, c):
    _rep(r, 13)["provenance"]["git"]["commit"] = "d" * 40


def _no_commit(r, b, c):
    _rep(r, 13)["provenance"]["git"]["commit"] = None


def _dirty(r, b, c):
    _rep(r, 1)["provenance"]["git"]["dirty"] = True


def _dirty_unknown(r, b, c):
    _rep(r, 1)["provenance"]["git"]["dirty"] = None


def _board_dirty(r, b, c):
    b["provenance"]["git"]["dirty"] = True
    c["provenance"]["git"]["dirty"] = True


def _rescored(r, b, c):
    _rep(r, 8)["provenance"]["rescored"] = [{"detector": "Reward Peak", "commit": "abc"}]


def _packages_differ(r, b, c):
    _rep(r, 10)["provenance"]["packages"]["numpy"] = "2.3.4"


def _python_differs(r, b, c):
    _rep(r, 10)["provenance"]["python"] = "3.12.4"


def _numerics_differ(r, b, c):
    _rep(r, 11)["provenance"]["numerics"]["env"]["OPENBLAS_CORETYPE"] = None


def _kernel_differs(r, b, c):
    _rep(r, 11)["provenance"]["numerics"]["torch_cpu_capability"] = "AVX512"


def _everywhere(r, b, c):
    """Every provenance block of the set: the board's, the sidecar's, each replicate's."""
    return [b["provenance"], c["provenance"]] + [doc["provenance"] for doc in r.values()]


def _unpinned_everywhere(r, b, c):
    """Uniform, so "one environment" passes -- but no runner had the pins."""
    for prov in _everywhere(r, b, c):
        numerics = prov["numerics"]
        numerics["env"] = dict.fromkeys(_PINS)
        numerics["torch_cpu_capability"] = "AVX512"
        numerics["blas"][0]["architecture"] = "SkylakeX"


def _one_pin_missing_everywhere(r, b, c):
    for prov in _everywhere(r, b, c):
        prov["numerics"]["env"]["OMP_NUM_THREADS"] = None


def _pin_set_but_openblas_ignored_it(r, b, c):
    for prov in _everywhere(r, b, c):
        prov["numerics"]["blas"][0]["architecture"] = "Zen"


def _aten_ignored_the_pin(r, b, c):
    for prov in _everywhere(r, b, c):
        prov["numerics"]["torch_cpu_capability"] = "DEFAULT"


def _no_numerics_block(r, b, c):
    for prov in _everywhere(r, b, c):
        prov.pop("numerics")


def _board_detectors_reordered(r, b, c):
    b["results"] = dict(reversed(list(b["results"].items())))


def _sidecar_detectors_reordered(r, b, c):
    c["cells"] = dict(reversed(list(c["cells"].items())))


def _replicate_results_reordered(r, b, c):
    rep = _rep(r, 9)
    rep["results"] = dict(reversed(list(rep["results"].items())))


def _sidecar_from_another_run(r, b, c):
    c["provenance"]["generated_utc"] = "2026-11-26T00:00:00Z"


def _cell_missing(r, b, c):
    _rep(r, 12)["results"]["Reward Peak"]["cells_detail"].pop()


def _cell_added(r, b, c):
    _rep(r, 12)["results"]["Reward Peak"]["cells_detail"].append(
        {"family": "fam_a", "difficulty": 0.9, "auroc": 0.5}
    )


@pytest.mark.parametrize(
    ("mutate", "refused_by"),
    [
        (_drop_replicate, "complete ids"),
        (_extra_replicate, "complete ids"),
        (_misnumbered, ("complete ids", "draws")),
        (_unparseable, "complete ids"),
        (_no_schema, "draw schema"),
        (_old_schema, "draw schema"),
        (_wrong_draw, "draws"),
        (_seeds_differ, "seeds per variant"),
        (_board_seeds_differ, "seeds per variant"),
        (_detector_missing, "detectors"),
        (_board_detector_missing, "detectors"),
        (_family_missing, "families"),
        (_board_family_missing, "families"),
        (_error_record, "no errors"),
        (_board_error, "no errors"),
        (_no_cells_detail, "no errors"),
        (_other_commit, "one commit"),
        (_no_commit, "one commit"),
        (_dirty, "clean"),
        (_dirty_unknown, "clean"),
        (_board_dirty, "clean"),
        (_rescored, "not re-scored"),
        (_packages_differ, "one environment"),
        (_python_differs, "one environment"),
        (_numerics_differ, ("one environment", "numerics pins")),
        (_kernel_differs, ("one environment", "numerics pins")),
        (_unpinned_everywhere, "numerics pins"),
        (_one_pin_missing_everywhere, "numerics pins"),
        (_pin_set_but_openblas_ignored_it, "numerics pins"),
        (_aten_ignored_the_pin, "numerics pins"),
        (_no_numerics_block, ("one environment", "numerics pins")),
        (_board_detectors_reordered, "detectors"),
        (_sidecar_detectors_reordered, "detectors"),
        (_replicate_results_reordered, "detectors"),
        (_sidecar_from_another_run, "sidecar"),
        (_cell_missing, "cell grid"),
        (_cell_added, "cell grid"),
    ],
    ids=lambda v: v.__name__.strip("_") if callable(v) else "",
)
def test_each_failure_mode_is_refused(check, good_set, mutate, refused_by):
    """Refused, and by the check that names the fault -- not by a side effect of it."""
    replicates, board, board_cells = good_set
    mutate(replicates, board, board_cells)
    failed = _failed(check, replicates, board, board_cells)
    expected = {refused_by} if isinstance(refused_by, str) else set(refused_by)
    assert set(failed) == expected, f"expected {sorted(expected)}; failed: {sorted(failed)}"


def test_a_set_from_another_commit_is_refused(check, good_set):
    failed = _failed(check, *good_set, expect_commit="e" * 40)
    assert set(failed) == {"one commit"}


def test_platform_and_cpu_may_differ(check, good_set):
    """Hosted runners differ in kernel build and CPU; under the pins that is not a fault."""
    replicates, board, board_cells = good_set
    rep = _rep(replicates, 14)["provenance"]
    rep["platform"] = "Linux-6.11.0-x86_64-with-glibc2.39"
    rep["numerics"]["cpu"] = {"vendor": "GenuineIntel", "model": "Intel Xeon Platinum 8370C"}
    assert _failed(check, replicates, board, board_cells) == {}


def _top_level_env(text: str) -> dict[str, str]:
    """A workflow's unindented ``env:`` mapping (PyYAML is not in the locked environment)."""
    lines = text.splitlines()
    env: dict[str, str] = {}
    for line in lines[lines.index("env:") + 1:]:
        if line and not line.startswith(" "):
            break
        m = re.match(r"^  ([A-Za-z_][A-Za-z0-9_]*):\s*(.*?)\s*$", line)
        if m:
            env[m.group(1)] = m.group(2).strip('"').strip("'")
    return env


def test_the_gate_requires_the_pins_regenerate_sets(check):
    """The gate's pins, the cache's names and the workflow's values are one contract."""
    assert list(check.PINNED_NUMERICS_ENV) == list(NUMERICS_ENV_VARS)
    assert check.PINNED_NUMERICS_ENV == _PINS
    workflow = _top_level_env(
        (_REPO / ".github" / "workflows" / "regenerate.yml").read_text(encoding="utf-8")
    )
    assert {name: workflow.get(name) for name in NUMERICS_ENV_VARS} == _PINS


def test_a_set_that_lost_the_pins_is_one_line_per_fault(check, good_set):
    """21 runners with the same fault are reported once, with the files that have it."""
    replicates, board, board_cells = good_set
    _unpinned_everywhere(replicates, board, board_cells)
    problems = _failed(check, replicates, board, board_cells)["numerics pins"]
    assert len(problems) == 3, problems
    # The sidecar's provenance is held to the board's by "sidecar", so it is not re-listed.
    assert all(p.startswith("board, replicate_000.json, replicate_001.json, "
                            "replicate_002.json (+17): ") for p in problems)
    assert "OPENBLAS_CORETYPE=None (pin 'Haswell')" in problems[0]
    assert "ATen dispatched at 'AVX512'" in problems[1]
    assert "OpenBLAS selected SkylakeX" in problems[2]


def test_a_run_without_torch_needs_no_aten_level(check, good_set):
    """No ATen kernel ran, so there is no dispatch level to hold to the pin."""
    replicates, board, board_cells = good_set
    for prov in _everywhere(replicates, board, board_cells):
        prov["numerics"].update(torch_imported=False, torch_cpu_capability=None)
    assert _failed(check, replicates, board, board_cells) == {}


def test_a_pin_name_with_no_value_in_the_gate_is_refused(check, good_set, monkeypatch):
    """A variable added to NUMERICS_ENV_VARS must not pass the gate unchecked."""
    pins = {k: v for k, v in check.PINNED_NUMERICS_ENV.items() if k != "MKL_CBWR"}
    monkeypatch.setattr(check, "PINNED_NUMERICS_ENV", pins)
    failed = _failed(check, *good_set)
    assert set(failed) == {"numerics pins"}
    assert "give every pin its value" in failed["numerics pins"][0]


def test_one_changed_cell_of_replicate_0_is_refused_and_named(check, good_set):
    replicates, board, board_cells = good_set
    cell = _rep(replicates, 0)["results"]["Reward Trend"]["cells_detail"][1]
    cell["auroc"] = round(cell["auroc"] + 1e-6, 6)
    failed = _failed(check, replicates, board, board_cells)
    assert set(failed) == {"replicate 0 is the board"}
    assert any("Reward Trend / fam_a @ 0.8" in p for p in failed["replicate 0 is the board"])


def test_a_cell_that_became_not_applicable_is_a_mismatch(check, good_set):
    replicates, board, board_cells = good_set
    _rep(replicates, 0)["results"]["Reward Trend"]["cells_detail"][0]["auroc"] = None
    assert set(_failed(check, replicates, board, board_cells)) == {"replicate 0 is the board"}


def test_the_mismatch_allowance_is_a_count(check, good_set):
    replicates, board, board_cells = good_set
    cells = _rep(replicates, 0)["results"]["Reward Trend"]["cells_detail"]
    cells[0]["auroc"] = 0.123456
    assert _failed(check, replicates, board, board_cells, max_board_mismatch=1) == {}
    cells[1]["auroc"] = 0.123456
    failed = _failed(check, replicates, board, board_cells, max_board_mismatch=1)
    assert set(failed) == {"replicate 0 is the board"}


def test_only_replicate_0_is_compared_with_the_board(check, good_set):
    """Every other draw is a different draw; its cells are supposed to differ."""
    replicates, board, board_cells = good_set
    assert _rep(replicates, 1)["results"]["Reward Trend"]["cells_detail"] != (
        board_cells["cells"]["Reward Trend"]
    )
    assert _failed(check, replicates, board, board_cells) == {}


def _write_set(tmp_path: Path, replicates, board, board_cells) -> tuple[Path, Path, Path]:
    reps = tmp_path / "reps"
    reps.mkdir()
    for name, doc in replicates.items():
        (reps / name).write_text(json.dumps(doc))
    (tmp_path / "board.json").write_text(json.dumps(board))
    (tmp_path / "cells.json").write_text(json.dumps(board_cells))
    return reps, tmp_path / "board.json", tmp_path / "cells.json"


@pytest.fixture
def two_registered_families(monkeypatch):
    """The command line checks against the registry; register exactly ``_FAMILIES``."""
    monkeypatch.setattr(
        FamilyRegistry, "_families", {n: _StubFamily(n, 50) for n in _FAMILIES}
    )


def test_the_command_line_accepts_then_refuses_one_changed_cell(
    check, good_set, tmp_path, capsys, two_registered_families
):
    reps, board, cells = _write_set(tmp_path, *good_set)
    argv = ["--from-dir", str(reps), "--board", str(board), "--board-cells", str(cells),
            "--expect-commit", _COMMIT]
    assert check.main(argv) == 0
    assert "ACCEPTED: all" in capsys.readouterr().out

    path = reps / "replicate_000.json"
    doc = json.loads(path.read_text())
    doc["results"]["Reward Trend"]["cells_detail"][1]["auroc"] = 0.999999
    path.write_text(json.dumps(doc))
    assert check.main(argv) == 1
    out = capsys.readouterr().out
    assert "[FAIL] replicate 0 is the board" in out
    assert "Reward Trend / fam_a @ 0.8: replicate_000 0.999999" in out
    assert "REFUSED: 1 of" in out

    assert check.main(argv + ["--max-board-mismatch", "1"]) == 0
    assert "allowed by --max-board-mismatch 1" in capsys.readouterr().out


def test_the_command_line_refuses_an_unreadable_board(check, good_set, tmp_path, capsys):
    reps, board, cells = _write_set(tmp_path, *good_set)
    board.write_text("{not json")
    argv = ["--from-dir", str(reps), "--board", str(board), "--board-cells", str(cells),
            "--expect-commit", _COMMIT]
    assert check.main(argv) == 1
    assert "--board does not parse" in capsys.readouterr().out


# ======================================================== the scripts, end to end
class _StubFamily(BaseFamily):
    """A family whose runs cost microseconds, built from its layout seed.

    The layout seed moves the runs (``offset``), and every seed the family was built at
    is recorded, so a test can tell that the scripts' layout seed reached it.
    """

    mechanism = HackingMechanism.CAMPING_EXPLOIT
    complexity = EnvironmentComplexity.TABULAR

    def __init__(self, name: str, n_episodes: int) -> None:
        self._name = name
        self.n_episodes = n_episodes
        self.seeds_built: list[int] = []

    @property
    def name(self) -> str:
        return self._name

    def difficulty_range(self) -> tuple[float, float]:
        return (0.7, 0.8)

    def generate_pair(self, difficulty: float, seed: int = 0) -> MatchedPair:
        self.seeds_built.append(seed)
        offset = float(np.random.default_rng(seed).normal())
        n = self.n_episodes

        def variant(hacking: bool):
            def rollout(s: int):
                rng = np.random.default_rng([s, seed, int(hacking)])
                proxy = rng.normal(size=n)
                drift = (0.5 if hacking else 0.0) * np.linspace(0.0, 1.0, n)
                behav = rng.normal(
                    loc=(difficulty - 0.5) * (1 if hacking else -1) + 0.2 * offset, size=n
                )
                run = RunData(proxy_rewards=proxy, true_rewards=proxy - drift + 0.1 * offset,
                              state_counts=None, behav_trace=behav)
                return run, (n // 2 if hacking else -1)
            return rollout

        return MatchedPair(
            family=self._name, mechanism=self.mechanism, complexity=self.complexity,
            difficulty=difficulty, n_episodes=n, rollout_hacking=variant(True),
            rollout_legit=variant(False), params={},
        )


#: Fast detectors, one per level, plus one that declares min_episodes (Reward Skewness,
#: 100: not applicable on the 30-episode stub) and one whose channel the stubs do not
#: ship (Bimodal Occupancy, state_counts: not applicable everywhere).
_E2E_DETECTORS = [
    "Reward Threshold", "Reward Skewness", "Bimodal Occupancy",
    "Behavioral Threshold", "True Reward Oracle",
]
_E2E_COMMIT = "a1b2c3d4" * 5


@pytest.fixture
def stub_world(monkeypatch):
    families = {n: _StubFamily(n, ep) for n, ep in (("stub_long", 120), ("stub_short", 30))}
    monkeypatch.setattr(FamilyRegistry, "_families", families)
    monkeypatch.setattr(Benchmark, "_rollout_cache", {})
    # A clean checkout at a known commit, whatever the tree running the test looks like.
    monkeypatch.setattr(provenance_module, "git_provenance", lambda: {
        "commit": _E2E_COMMIT, "short_commit": _E2E_COMMIT[:12], "branch": "test",
        "dirty": False, "dirty_files": [], "source": "git",
    })
    # And a pinned process. The pins act when BLAS and ATen load, so a test process cannot
    # take them on after the fact; the gate's pin check has its own cases above.
    monkeypatch.setattr(provenance_module, "numerics_block", _pinned_numerics)
    return families


def _run_board(board_script, out: Path) -> tuple[dict, dict]:
    board_script.main([
        "--detectors", *_E2E_DETECTORS, "--n-seeds", "4",
        "--out", str(out / "board.json"), "--cells-out", str(out / "cells.json"),
    ])
    return (json.loads((out / "board.json").read_text()),
            json.loads((out / "cells.json").read_text()))


def _run_replicates(replicate_script, out: Path, ids) -> dict[str, dict]:
    for rid in ids:
        replicate_script.main([
            "--replicate-id", str(rid), "--detectors", *_E2E_DETECTORS, "--n-seeds", "4",
            "--out-dir", str(out / "reps"),
        ])
    return {p.name: json.loads(p.read_text()) for p in sorted((out / "reps").glob("*.json"))}


def test_the_board_writes_its_cells_sidecar(board_script, stub_world, tmp_path):
    board, cells = _run_board(board_script, tmp_path)

    assert cells["provenance"] == board["provenance"], "sidecar and board are not one run's"
    assert list(cells["cells"]) == list(board["results"])
    assert set(cells["cells"]) == set(_E2E_DETECTORS)
    assert board["sampling"]["layout_seeds"] == [0]
    for det, detail in cells["cells"].items():
        assert [(c["family"], c["difficulty"]) for c in detail] == [
            ("stub_long", 0.8), ("stub_long", 0.7), ("stub_short", 0.8), ("stub_short", 0.7)
        ]
        per_family = board["results"][det]["per_family"]
        for fam in ("stub_long", "stub_short"):
            scored = [c["auroc"] for c in detail if c["family"] == fam and c["auroc"] is not None]
            assert per_family[fam] == (round(float(np.mean(scored)), 3) if scored else None)
    skew = {c["family"]: c["auroc"] for c in cells["cells"]["Reward Skewness"]}
    assert skew["stub_short"] is None and skew["stub_long"] is not None
    assert all(c["auroc"] is None for c in cells["cells"]["Bimodal Occupancy"])
    # Built at layout 0 and nowhere else.
    assert set(stub_world["stub_long"].seeds_built) == {0}


def test_replicates_record_their_draw_and_replicate_0_is_the_board(
    board_script, replicate_script, check, stub_world, tmp_path
):
    board, cells = _run_board(board_script, tmp_path)
    for fam in stub_world.values():
        fam.seeds_built.clear()
    reps = _run_replicates(replicate_script, tmp_path, range(3))

    for name, rep in reps.items():
        assert rep["draw_schema"] == replicate_script.DRAW_SCHEMA == 2
        assert (rep["layout_seed"], rep["seed_base"]) == replicate_script.draw_for(
            rep["replicate_id"]
        )
        assert rep["detectors_evaluated"] == _E2E_DETECTORS
    # The layout seed reached the family: draws 1 and 2 were built at layouts 1 and 2.
    assert set(stub_world["stub_long"].seeds_built) == {0, 1, 2}

    rep0 = reps["replicate_000.json"]["results"]
    for det in _E2E_DETECTORS:
        assert rep0[det]["cells_detail"] == cells["cells"][det], det
        summary = {k: v for k, v in rep0[det].items() if k != "cells_detail"}
        assert summary == board["results"][det], det
    rep1 = reps["replicate_001.json"]["results"]
    assert any(rep1[d]["cells_detail"] != cells["cells"][d] for d in _E2E_DETECTORS), (
        "draw 1 reproduced draw 0 exactly: the draw coordinates did not reach the rollouts"
    )


def test_a_real_redraw_passes_the_gate_and_the_aggregator(
    board_script, replicate_script, check, aggregate_script, stub_world, tmp_path
):
    """What the scripts write is what the gate and ``--require-single-source`` accept."""
    board, cells = _run_board(board_script, tmp_path)
    reps = _run_replicates(replicate_script, tmp_path, range(3))
    kwargs = dict(expect_commit=_E2E_COMMIT, n_replicates=3,
                  expected_detectors=_E2E_DETECTORS,
                  registered_families=["stub_long", "stub_short"])
    checks = check.run_checks(reps, board, cells, **kwargs)
    assert [c.name for c in checks if not c.ok] == [], check.render(checks, "e2e")

    aggregate_script.main([
        "--in-dir", str(tmp_path / "reps"), "--out", str(tmp_path / "agg.json"),
        "--out-md", str(tmp_path / "agg.md"), "--require-single-source",
    ])
    agg = json.loads((tmp_path / "agg.json").read_text())
    assert agg["replicates"]["single_source"] is True
    assert agg["replicates"]["layouts_resampled"] is True
    assert agg["provenance"]["script"] == "scripts/aggregate_replication.py"
    assert "layout seed and rollout seeds resampled together" in (tmp_path / "agg.md").read_text()

    # And one changed cell in replicate 0 is caught on the real files too.
    reps["replicate_000.json"]["results"]["True Reward Oracle"]["cells_detail"][0]["auroc"] = -1.0
    failed = [c.name for c in check.run_checks(reps, board, cells, **kwargs) if not c.ok]
    assert failed == ["replicate 0 is the board"]
