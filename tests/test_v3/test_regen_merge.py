"""``scripts/admission_ledger.py --merge-shards`` publishes only a complete, uniform set.

``.github/workflows/regenerate.yml`` certifies the ledger one cell per runner and commits
what the merge writes in place of the published artifact. A merge that accepted a missing
cell, a duplicate, a shard from a dirty tree or from different package versions would
publish exactly the kind of artifact the repository's provenance blocks exist to rule
out, labelled complete. These tests build shards by hand, so they need no rollouts, and
check each refusal, through the function and through the command line the workflow runs.

That a merged ledger equals an unsharded run of the same scope needs rollouts and is not
repeated here; REPRODUCIBILITY.md ("The Admission Ledger") records the check.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from rhob.v3.admission_gate import CRITERIA

_REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def shipped_families_only(monkeypatch):
    """Other test modules register mock families into the process-wide registry
    (test_v3_core, test_sign_randomization); the full grid is the shipped families'."""
    from rhob.v3.registry import FamilyRegistry

    shipped = {
        name: fam for name, fam in FamilyRegistry._families.items()
        if type(fam).__module__.startswith("rhob.v3.families.")
    }
    monkeypatch.setattr(FamilyRegistry, "_families", shipped)


@pytest.fixture(scope="module")
def ledger_script():
    return _load("_admission_ledger_under_test", _REPO / "scripts" / "admission_ledger.py")


FAMILIES = ["eval_probe_sandbagging", "monitored_sandbagging"]


def _provenance(commit: str = "a" * 40) -> dict:
    return {
        "generated_utc": "2026-11-19T00:00:00Z",
        "git": {"commit": commit, "short_commit": commit[:12], "branch": "b",
                "dirty": False, "dirty_files": []},
        "python": "3.12.16",
        "python_implementation": "CPython",
        "platform": "Linux-x86_64",
        "packages": {"numpy": "2.5.4", "mujoco": "3.17.0"},
        "argv": [],
    }


def _ledger_shard(family: str, difficulty: float, status: str = "ADMITTED") -> dict:
    prov = _provenance()
    prov["scope"] = {"families_run": [family], "families_registered": 33,
                     "complete": False, "difficulty_override": [difficulty]}
    prov["cpu"] = "Test CPU"
    prov["gate"] = {"n_layouts": 12, "min_seeds_per_layout": 24, "n_bootstrap": 2000,
                    "equivalence_margin": 0.1, "informativeness_floor": 1e-4, "alpha": 0.05,
                    "target_power": 0.9, "behavioral_floor": 0.6, "onset_std_frac": 0.25}
    degenerate = status == "DEGENERATE"
    outcomes = {c: ("degenerate" if degenerate and c == "proxy_matched" else "pass")
                for c in CRITERIA}
    row = {
        "family": family, "difficulty": difficulty, "passed": status == "ADMITTED",
        "status": status, "criteria": {c: outcomes[c] == "pass" for c in CRITERIA},
        "outcomes": outcomes,
        "degenerate_criteria": ["proxy_matched"] if degenerate else [],
        "details": {}, "metrics": {}, "design": {"n_layouts": 12}, "error": None,
    }
    return {"schema": "rhob.admission_ledger/1", "provenance": prov,
            "design": {"n_layouts": 12}, "summary": {}, "results": [row],
            "timing_seconds": {family: 2.0}}


def _write_ledger_shards(root: Path, ledger_script, mutate=None, families=FAMILIES) -> list[Path]:
    paths = []
    # Written in reverse so the merge has to restore grid order itself.
    for i, (family, difficulty) in enumerate(reversed(ledger_script.ledger_grid(families))):
        shard = _ledger_shard(family, difficulty,
                              "DEGENERATE" if family == "monitored_sandbagging" else "ADMITTED")
        # The first grid cell, written last, is the merge's reference; give it the
        # earliest stamp so taking the reference's stamp would be caught.
        shard["provenance"]["generated_utc"] = f"2026-11-19T00:{(999 - i) // 60:02d}:{(999 - i) % 60:02d}Z"
        if mutate is not None:
            mutate(i, shard)
        path = root / f"ledger-{i:03d}" / "admission_ledger.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(shard))
        paths.append(path)
    return paths


def test_a_complete_set_merges_in_grid_order(tmp_path, ledger_script):
    _write_ledger_shards(tmp_path, ledger_script)
    payload = ledger_script.merge_shards(tmp_path, families=FAMILIES)
    grid = ledger_script.ledger_grid(FAMILIES)
    assert [(r["family"], r["difficulty"]) for r in payload["results"]] == grid
    s = payload["summary"]
    assert (s["n_cells"], s["n_admitted"], s["n_degenerate"]) == (10, 5, 5)
    assert s["degenerate_families"] == ["monitored_sandbagging"]
    assert payload["timing_seconds"] == {f: 10.0 for f in FAMILIES}
    scope = payload["provenance"]["scope"]
    assert scope["families_run"] == FAMILIES and scope["difficulty_override"] is None
    assert scope["families_registered"] == 33
    # Two of 33 families is not the full grid, and must not be labelled as one.
    assert scope["complete"] is False
    sharding = payload["provenance"]["sharding"]
    assert sharding["n_shards"] == 10
    assert "--families F --difficulties D --out-dir" in sharding["shard_command"]
    stamps = sorted(f"2026-11-19T00:{(999 - i) // 60:02d}:{(999 - i) % 60:02d}Z" for i in range(10))
    assert payload["provenance"]["generated_utc"] == stamps[-1] == sharding["generated_utc_last"]
    assert sharding["generated_utc_first"] == stamps[0]
    # The single-machine fields would name one shard's machine; the lists are in sharding.
    assert payload["provenance"]["cpu"] is None and payload["provenance"]["platform"] is None


def test_the_merge_records_its_own_command_and_ci_run(tmp_path, ledger_script, monkeypatch):
    _write_ledger_shards(tmp_path, ledger_script)
    argv = ["scripts/admission_ledger.py", "--merge-shards", "SHARDS"]
    monkeypatch.setattr(ledger_script.sys, "argv", argv)
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    payload = ledger_script.merge_shards(tmp_path, families=FAMILIES)
    assert payload["provenance"]["argv"] == argv
    url = "https://github.com/o/r/actions/runs/42"
    assert payload["provenance"]["sharding"]["ci_run"] == url
    assert f"; CI run {url}" in ledger_script.render_markdown(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda i, s: s["provenance"]["git"].update(dirty=True), "dirty"),
        (lambda i, s: i == 3 and s["provenance"]["git"].update(dirty=True), "dirty"),
        (lambda i, s: i == 3 and s["provenance"]["git"].update(dirty=None), "dirty"),
        (lambda i, s: i == 3 and s["provenance"]["git"].update(commit="b" * 40), "commit"),
        (lambda i, s: i == 3 and s["provenance"]["packages"].update(mujoco="4.0.0"), "packages"),
        (lambda i, s: i == 3 and s["provenance"]["gate"].update(n_layouts=6), "gate"),
        (lambda i, s: i == 3 and s["results"][0]["design"].update(n_layouts=6), "design"),
        (lambda i, s: i == 3 and s["provenance"]["scope"].update(difficulty_override=None),
         "scope"),
        (lambda i, s: i == 3 and s["results"].append(copy.deepcopy(s["results"][0])),
         "exactly one cell"),
        (lambda i, s: i == 3 and s["provenance"].update(python="3.12.17"), "python differs"),
        (lambda i, s: i == 3 and s["provenance"].update(python_implementation="PyPy"),
         "python_implementation differs"),
        (lambda i, s: i == 3 and s.update(schema="rhob.admission_ledger/2"), "schema"),
        (lambda i, s: i == 3 and s["provenance"]["scope"].update(families_run=["orbit_chirality"]),
         "scope"),
        (lambda i, s: s["provenance"]["git"].update(commit=None), "no git commit"),
    ],
)
def test_a_nonuniform_set_is_refused(tmp_path, ledger_script, mutate, message):
    _write_ledger_shards(tmp_path, ledger_script, mutate)
    with pytest.raises(ledger_script.ShardError, match=message):
        ledger_script.merge_shards(tmp_path, families=FAMILIES)


def test_a_missing_or_duplicated_cell_is_refused(tmp_path, ledger_script):
    paths = _write_ledger_shards(tmp_path, ledger_script)
    paths[0].unlink()
    with pytest.raises(ledger_script.ShardError, match="1 missing"):
        ledger_script.merge_shards(tmp_path, families=FAMILIES)
    paths[0].write_text(paths[1].read_text())
    with pytest.raises(ledger_script.ShardError, match="already supplied"):
        ledger_script.merge_shards(tmp_path, families=FAMILIES)


def test_a_cell_outside_the_grid_is_refused(tmp_path, ledger_script):
    _write_ledger_shards(tmp_path, ledger_script)
    stray = tmp_path / "stray" / "admission_ledger.json"
    stray.parent.mkdir()
    stray.write_text(json.dumps(_ledger_shard("orbit_chirality", 0.9)))
    with pytest.raises(ledger_script.ShardError, match="1 not in the grid"):
        ledger_script.merge_shards(tmp_path, families=FAMILIES)


def _as_raised(shard: dict) -> None:
    """Turn a shard's row into what run_ledger writes for a cell that raised."""
    shard["results"][0].update(
        passed=False, status="NOT ADMITTED", criteria={c: False for c in CRITERIA},
        outcomes={c: "fail" for c in CRITERIA}, degenerate_criteria=[], details={},
        metrics={}, design={}, error="ImportError: no mujoco",
    )
    shard["design"] = {}


def test_a_cell_that_raised_needs_an_explicit_allowance(tmp_path, ledger_script):
    # Shards are written in reverse grid order, so the last index is the first grid cell:
    # the one the merge would read `design` off if it did not skip empty designs.
    first = len(ledger_script.ledger_grid(FAMILIES)) - 1
    _write_ledger_shards(tmp_path, ledger_script, lambda i, s: i == first and _as_raised(s))
    with pytest.raises(ledger_script.ShardError, match="1 cell"):
        ledger_script.merge_shards(tmp_path, families=FAMILIES)
    payload = ledger_script.merge_shards(tmp_path, families=FAMILIES, allow_cell_errors=True)
    assert payload["summary"]["n_not_admitted"] == 1
    assert payload["results"][0]["error"] == "ImportError: no mujoco"
    assert payload["design"] == {"n_layouts": 12}


def test_the_merged_ledger_passes_the_published_schema(tmp_path, ledger_script):
    """The same top-level keys the committed artifact is pinned to."""
    _write_ledger_shards(tmp_path, ledger_script)
    payload = ledger_script.merge_shards(tmp_path, families=FAMILIES)
    assert set(payload) == {"schema", "provenance", "design", "summary", "results",
                            "timing_seconds"}
    json_path, md_path = ledger_script.write_artifacts(payload, tmp_path / "out")
    assert json.loads(json_path.read_text()) == payload
    assert md_path.read_text().startswith("# RHOB Admission Ledger")


def test_a_differing_machine_is_recorded_not_refused(tmp_path, ledger_script):
    """Cells land on whatever runner is free; the merge lists the machines it saw."""
    _write_ledger_shards(
        tmp_path, ledger_script,
        lambda i, s: i == 3 and s["provenance"].update(cpu="Other CPU", platform="Linux-aarch64"),
    )
    payload = ledger_script.merge_shards(tmp_path, families=FAMILIES)
    sharding = payload["provenance"]["sharding"]
    assert sharding["cpus"] == ["Other CPU", "Test CPU"]
    assert sharding["platforms"] == ["Linux-aarch64", "Linux-x86_64"]
    md = ledger_script.render_markdown(payload)
    assert "on Linux-aarch64, Linux-x86_64" in md and "CPUs: Other CPU, Test CPU" in md


_PINS = {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_NUM_THREADS": "1", "MKL_CBWR": "COMPATIBLE",
         "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "ATEN_CPU_CAPABILITY": "avx2"}


def _with_numerics(i: int, shard: dict, vendor: str = "GenuineIntel") -> None:
    shard["provenance"]["numerics"] = {
        "env": dict(_PINS),
        "cpu": {"vendor": vendor, "model": f"{vendor} model"},
        "torch_imported": True, "torch_cpu_capability": "AVX2", "blas": [],
    }


def test_shards_under_different_numerics_pins_are_refused(tmp_path, ledger_script):
    """The pins select the kernels; cells certified under different ones are not merged."""
    def mutate(i, s):
        _with_numerics(i, s)
        if i == 3:
            s["provenance"]["numerics"]["env"]["OPENBLAS_CORETYPE"] = None
    _write_ledger_shards(tmp_path, ledger_script, mutate)
    with pytest.raises(ledger_script.ShardError, match="numerics_env differs"):
        ledger_script.merge_shards(tmp_path, families=FAMILIES)


def test_the_reference_shards_cpu_is_not_published_as_the_ledgers(tmp_path, ledger_script):
    """numerics.cpu is one machine's; the merge empties it and lists every vendor seen."""
    _write_ledger_shards(
        tmp_path, ledger_script,
        lambda i, s: _with_numerics(i, s, "AuthenticAMD" if i == 3 else "GenuineIntel"),
    )
    payload = ledger_script.merge_shards(tmp_path, families=FAMILIES)
    numerics = payload["provenance"]["numerics"]
    assert numerics["cpu"] is None
    assert numerics["env"] == _PINS
    assert payload["provenance"]["sharding"]["cpu_vendors"] == ["AuthenticAMD", "GenuineIntel"]


def test_the_markdown_says_the_ledger_was_sharded(tmp_path, ledger_script):
    _write_ledger_shards(tmp_path, ledger_script)
    payload = ledger_script.merge_shards(tmp_path, families=FAMILIES)
    md = ledger_script.render_markdown(payload)
    assert "- Sharded: merged from 10 single-cell runs" in md
    assert "- Merge command:" in md and "- Each shard ran:" in md
    assert "neither the run's elapsed time nor the cost of one unsharded re-run" in md
    sharding = payload["provenance"]["sharding"]
    assert f"generated {sharding['generated_utc_first']} to {sharding['generated_utc_last']}" in md
    assert "2 of 33 registered families" in md
    unsharded = copy.deepcopy(payload)
    del unsharded["provenance"]["sharding"]
    unsharded["provenance"].update(platform="Linux-x86_64", cpu="Test CPU")
    md = ledger_script.render_markdown(unsharded)
    assert "Sharded" not in md and "- Command:" in md
    assert "one cell per machine" not in md
    assert "the cost of a full re-run is knowable in advance" in md


@pytest.fixture
def full_grid(tmp_path_factory, ledger_script) -> Path:
    """Shards for all 123 cells, as the workflow's merge job receives them."""
    root = tmp_path_factory.mktemp("full")
    _write_ledger_shards(root, ledger_script, families=None)
    return root


def test_the_cli_merges_the_full_grid_and_labels_it_complete(full_grid, tmp_path, ledger_script):
    out = tmp_path / "out"
    assert ledger_script.main(["--merge-shards", str(full_grid), "--out-dir", str(out)]) == 0
    payload = json.loads((out / "admission_ledger.json").read_text())
    assert payload["summary"]["n_cells"] == len(ledger_script.ledger_grid()) == 123
    assert payload["provenance"]["scope"]["complete"] is True
    assert (out / "ADMISSION_LEDGER.md").exists()


def test_the_cli_refuses_and_writes_nothing(full_grid, tmp_path, ledger_script):
    """The workflow commits on the exit code, so a refusal must be non-zero and leave no file."""
    broken = tmp_path / "broken"
    shutil.copytree(full_grid, broken)
    first = sorted(broken.rglob("admission_ledger.json"))[0]
    shard = json.loads(first.read_text())
    _as_raised(shard)
    first.write_text(json.dumps(shard))
    out = tmp_path / "out"
    assert ledger_script.main(["--merge-shards", str(broken), "--out-dir", str(out)]) == 1
    assert not out.exists()
    assert ledger_script.main(
        ["--merge-shards", str(broken), "--out-dir", str(out), "--allow-cell-errors"]
    ) == 0
    assert json.loads((out / "admission_ledger.json").read_text())["summary"]["n_not_admitted"] >= 1
    first.unlink()
    assert ledger_script.main(["--merge-shards", str(broken), "--out-dir", str(tmp_path / "o2")]) == 1


@pytest.mark.parametrize("extra", [["--families", "orbit_chirality"], ["--difficulties", "0.5"]])
def test_the_cli_merge_takes_no_scope(tmp_path, ledger_script, extra):
    with pytest.raises(SystemExit) as exc:
        ledger_script.main(["--merge-shards", str(tmp_path), *extra])
    assert exc.value.code == 2


def test_the_cli_lists_the_full_grid(capsys, ledger_script):
    assert ledger_script.main(["--list-cells"]) == 0
    cells = json.loads(capsys.readouterr().out)
    assert [(c["family"], c["difficulty"]) for c in cells] == ledger_script.ledger_grid()


def test_the_out_dir_exists_before_any_cell_is_certified(tmp_path, ledger_script, monkeypatch):
    """An unwritable --out-dir must fail before hours of certification, not after."""
    out = tmp_path / "made-first"

    def fake_run_ledger(*args, **kwargs):
        assert out.is_dir()
        raise SystemExit(99)

    monkeypatch.setattr(ledger_script, "run_ledger", fake_run_ledger)
    with pytest.raises(SystemExit) as exc:
        ledger_script.main(["--families", "eval_probe_sandbagging", "--quiet", "--out-dir", str(out)])
    assert exc.value.code == 99
