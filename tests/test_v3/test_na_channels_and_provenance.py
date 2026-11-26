"""A detector's missing input channel must produce N/A, not a fabricated 0.5.

The audit found that 8 L1 detectors were being scored on the 25 of 33 families that
ship ``state_counts=None`` -- 88 of the 123 leaderboard cells. Every one of them returns
a hardcoded 0.5 in that case, so the published "L1 access level" number was 71% constant
averaged in as if it were a measurement. These tests pin the corrected behaviour: an
absent channel is excluded from every aggregate, and is distinguishable in the artifact
from a real 0.5.

Also covers the provenance block that every result JSON now carries.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import numpy as np
import pytest

import rhob.v3.provenance as provenance_module
from rhob.detectors.posthoc import PosthocDetector, RunData
from rhob.v3.benchmark import (
    BenchmarkResults,
    CellResult,
    _evaluate_cell,
    missing_channels,
    required_channels,
)
from rhob.v3.provenance import (
    _TRACKED_PACKAGES,
    _git,
    git_provenance,
    package_versions,
    provenance_block,
    sampling_block,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The tracked list as it stood until 2026-11-16. Every committed artifact carries these.
_PRE_2026_11_16_PACKAGES = ("numpy", "scipy", "scikit-learn", "torch", "rhob")

#: Added on 2026-11-16; see docs/postmortems/2026-11-mujoco-calibration-cache.md.
_SIMULATOR_PACKAGES = ("mujoco", "mujoco-core", "gymnasium", "pettingzoo", "mpe2")

#: Where committed result artifacts live.
_ARTIFACT_DIRS = ("admission", "results", "docs/figures", "leaderboard")


class _FakeL1(PosthocDetector):
    """An L1 detector with the same 0.5 no-signal fallback every real one has."""

    @property
    def access_level(self) -> str:
        return "L1"

    @property
    def name(self) -> str:
        return "Fake L1"

    def classify(self, run: RunData) -> float:
        if run.state_counts is None:
            return 0.5
        return float(run.state_counts[-1].sum())

    def detect_onset(self, run: RunData) -> int:
        return -1


def _run(state_counts) -> RunData:
    return RunData(
        proxy_rewards=np.linspace(0.0, 1.0, 20),
        true_rewards=np.linspace(0.0, 1.0, 20),
        state_counts=state_counts,
        behav_trace=np.zeros(20),
    )


def test_required_channels_defaults_to_the_access_level_signal():
    assert required_channels(_FakeL1()) == ("state_counts",)


def test_required_channels_respects_a_detector_declared_override():
    detector = _FakeL1()
    detector.required_channels = ("behav_trace",)
    assert required_channels(detector) == ("behav_trace",)


def test_missing_channels_flags_none_and_empty():
    present = [_run(np.ones((20, 3)))]
    absent = [_run(None)]
    empty = [_run(np.array([]))]
    assert missing_channels(present, ("state_counts",)) == []
    assert missing_channels(absent, ("state_counts",)) == ["state_counts"]
    assert missing_channels(empty, ("state_counts",)) == ["state_counts"]


def test_absent_channel_yields_nan_cell_with_a_reason_not_a_half():
    runs_a = [_run(None) for _ in range(4)]
    runs_b = [_run(None) for _ in range(4)]
    auroc, mae, reason = _evaluate_cell(_FakeL1(), runs_a, runs_b, [5] * 4, "L1", 20)
    assert np.isnan(auroc)
    assert np.isnan(mae)
    assert reason is not None and "state_counts" in reason


def test_present_channel_is_scored_normally():
    runs_a = [_run(np.ones((20, 3)) * 2) for _ in range(4)]
    runs_b = [_run(np.ones((20, 3))) for _ in range(4)]
    auroc, _, reason = _evaluate_cell(_FakeL1(), runs_a, runs_b, [5] * 4, "L1", 20)
    assert reason is None
    assert auroc == pytest.approx(1.0)


def test_na_cells_are_excluded_from_overall_auroc_not_averaged_as_chance():
    """The whole point: a 0.9 measurement plus an N/A must not report 0.7."""
    results = BenchmarkResults(
        detector_name="Fake L1",
        access_level="L1",
        cells=[
            CellResult("fam_with_counts", "camping", 0.9, 0.9, 5),
            CellResult(
                "fam_without_counts", "camping", 0.9, float("nan"), 5,
                na_reason="family does not provide state_counts",
            ),
        ],
    )
    assert results.overall_auroc == pytest.approx(0.9)
    assert [c.family for c in results.scored_cells] == ["fam_with_counts"]
    assert results.na_families == ["fam_without_counts"]
    assert len(results.na_cells) == 1


def test_overall_auroc_is_nan_when_nothing_was_measurable():
    results = BenchmarkResults(
        detector_name="Fake L1",
        access_level="L1",
        cells=[
            CellResult(
                "fam", "camping", 0.9, float("nan"), 5,
                na_reason="family does not provide state_counts",
            )
        ],
    )
    assert np.isnan(results.overall_auroc)
    assert results.scored_cells == []


def test_summary_renders_na_rather_than_a_number():
    results = BenchmarkResults(
        detector_name="Fake L1",
        access_level="L1",
        cells=[
            CellResult(
                "fam", "camping", 0.9, float("nan"), 5,
                na_reason="family does not provide state_counts",
            )
        ],
    )
    text = results.summary()
    assert "N/A" in text
    assert "0.500" not in text


def test_provenance_block_carries_commit_versions_and_timestamp():
    block = provenance_block(script="test")
    assert block["script"] == "test"
    assert block["generated_utc"].endswith("Z")
    assert block["python"].count(".") == 2
    assert set(block["packages"]) >= {"numpy", "scipy", "scikit-learn", "rhob"}
    assert block["packages"]["numpy"] is not None
    # git may legitimately be unavailable (sdist install); the keys must exist either way.
    # Superset, not equality: the block is additive, and asserting equality made adding a
    # field a test failure rather than a reviewable change. The five below are what any
    # consumer may rely on.
    assert set(block["git"]) >= {"commit", "short_commit", "branch", "dirty", "dirty_files"}
    # Where the revision came from is load-bearing: "git" means git answered, "env:..."
    # means the operator asserted it and it is unverified, None means neither was
    # available. A consumer that treats a declared commit as a verified one is wrong, so
    # the discriminator has to be present.
    assert "source" in block["git"]
    assert block["git"]["source"] in {"git", "env:RHOB_SOURCE_COMMIT", None}


def test_provenance_records_the_simulators_that_step_the_rollouts():
    """MuJoCo 4.0.0 moved a family out of its admission band, and no artifact said which
    MuJoCo it had run under. Every tracked name is a key, installed or not."""
    packages = package_versions()
    assert tuple(packages) == _TRACKED_PACKAGES
    assert set(_SIMULATOR_PACKAGES) <= set(packages)
    assert set(_PRE_2026_11_16_PACKAGES) <= set(packages)
    for name, value in packages.items():
        assert value is None or (isinstance(value, str) and value), (name, value)


def test_an_absent_package_is_recorded_as_null_not_dropped(monkeypatch):
    """A run without MuJoCo must say so. Omitting the key would make it look like an
    artifact written before the key existed, which is a different fact."""
    real = provenance_module._pkg_version

    def without_mujoco(name: str) -> str:
        if name in {"mujoco", "mujoco-core"}:
            raise PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(provenance_module, "_pkg_version", without_mujoco)
    packages = json.loads(json.dumps(provenance_block()))["packages"]
    assert "mujoco" in packages and packages["mujoco"] is None
    assert "mujoco-core" in packages and packages["mujoco-core"] is None
    # The calibration cache hashes the string "absent"; provenance writes JSON null, as
    # it always has for torch. The two spellings must not leak into each other.
    assert "absent" not in packages.values()
    assert packages["numpy"] == real("numpy")


def test_every_distribution_the_calibration_cache_keys_on_is_recorded():
    """The cache invalidates on these because they move a calibration. A version that can
    move a number must also be readable from the artifact that published it."""
    from rhob.v3.calibration_cache import PHYSICS_DISTRIBUTIONS

    missing = sorted(set(PHYSICS_DISTRIBUTIONS) - set(_TRACKED_PACKAGES))
    assert not missing, f"calibration_cache keys on {missing}, provenance does not record it"


def _committed_package_blocks() -> list[tuple[str, dict]]:
    """Every provenance ``packages`` block in a committed artifact, nested ones included
    (a re-scored replicate carries its source run's provenance inside its own)."""
    found: list[tuple[str, dict]] = []
    for folder in _ARTIFACT_DIRS:
        for path in sorted((_REPO_ROOT / folder).rglob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            where = path.relative_to(_REPO_ROOT).as_posix()
            stack = [data]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if isinstance(node.get("packages"), dict) and "generated_utc" in node:
                        found.append((where, node["packages"]))
                    stack.extend(node.values())
                elif isinstance(node, list):
                    stack.extend(node)
    return found


def test_artifacts_written_under_the_old_list_still_read():
    """Artifacts committed before 2026-11-16 carry the five-name list (the admission
    ledger of 2026-08-03 also the four simulators its script added); the ledger
    regenerated on 2026-11-19 and the v5 board re-scored on CI on 2026-11-26 carry all
    ten. Nothing may require the new names, and no recorded name may be renamed or
    dropped, or old and new artifacts stop comparing key by key."""
    blocks = _committed_package_blocks()
    assert len(blocks) >= 23, "the scan found fewer provenance blocks than are committed"
    for where, packages in blocks:
        assert set(_PRE_2026_11_16_PACKAGES) <= set(packages), where
        assert set(packages) <= set(_TRACKED_PACKAGES), (where, sorted(packages))
        for name, value in packages.items():
            assert value is None or isinstance(value, str), (where, name, value)


@pytest.mark.parametrize(
    ("script", "artifact", "report"),
    [
        (
            "scripts/admission_ledger.py",
            "admission/admission_ledger.json",
            "admission/ADMISSION_LEDGER.md",
        ),
        (
            "scripts/measure_difficulty_calibration.py",
            "results/difficulty_calibration/difficulty_calibration.json",
            "results/difficulty_calibration/DIFFICULTY_CALIBRATION.md",
        ),
    ],
)
def test_the_report_renderers_still_read_an_old_packages_block(script, artifact, report):
    """The two scripts that render a packages block, run on their committed artifacts,
    must reproduce the committed report's line. The calibration artifact is still written
    under the old five-name list, so a renderer that indexed the new names would raise
    KeyError here. The ledger has carried the full list since 2026-11-19; the next test
    keeps its old-list case."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_renderer_under_test", _REPO_ROOT / script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = json.loads((_REPO_ROOT / artifact).read_text(encoding="utf-8"))
    rendered = [
        line for line in module.render_markdown(payload).splitlines()
        if line.startswith("- Packages:")
    ]
    committed = [
        line for line in (_REPO_ROOT / report).read_text(encoding="utf-8").splitlines()
        if line.startswith("- Packages:")
    ]
    assert rendered and rendered == committed


#: The ``packages`` block of the admission ledger of 2026-08-03 (``git show
#: acd35c2:admission/admission_ledger.json``), written under the old list plus the four
#: simulators, and the line its committed report rendered from it (line 18 of
#: ``git show acd35c2:admission/ADMISSION_LEDGER.md``).
_LEDGER_2026_08_03_PACKAGES = {
    "numpy": "2.4.3", "scipy": "1.17.1", "scikit-learn": "1.8.0", "torch": "2.11.0",
    "rhob": "1.8.0", "gymnasium": "1.3.0", "mujoco": "3.10.0", "pettingzoo": "1.26.1",
    "mpe2": "1.1.0",
}
_LEDGER_2026_08_03_PACKAGES_LINE = (
    "- Packages: gymnasium 1.3.0, mpe2 1.1.0, mujoco 3.10.0, numpy 2.4.3, "
    "pettingzoo 1.26.1, rhob 1.8.0, scikit-learn 1.8.0, scipy 1.17.1, torch 2.11.0"
)


def test_the_ledger_renderer_still_reads_the_2026_08_packages_block():
    """The ledger's renderer, on a ledger whose packages block predates ``mujoco-core``.

    The committed ledger no longer has one, so the block is substituted into it here.
    """
    import importlib.util

    script = _REPO_ROOT / "scripts" / "admission_ledger.py"
    spec = importlib.util.spec_from_file_location("_ledger_renderer_under_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = json.loads((_REPO_ROOT / "admission" / "admission_ledger.json").read_text("utf-8"))
    payload["provenance"]["packages"] = dict(_LEDGER_2026_08_03_PACKAGES)
    rendered = [
        line for line in module.render_markdown(payload).splitlines()
        if line.startswith("- Packages:")
    ]
    assert rendered == [_LEDGER_2026_08_03_PACKAGES_LINE]


def test_dirty_file_paths_are_not_truncated():
    """Regression: ``git status --porcelain`` starts an unstaged line with a space.

    Stripping the whole stdout ate that space and therefore the first character of the
    first path, silently writing ``ocs/figures/...`` into the provenance block.
    """
    info = git_provenance()
    if not info["dirty_files"]:
        pytest.skip("clean working tree: nothing to check")
    raw = _git("status", "--porcelain")
    assert raw is not None
    # The invariant that the bug broke: every porcelain line keeps both status columns,
    # so column 2 is always the separating space. Under the old whole-output ``.strip()``
    # the first line arrived as "M docs/..." instead of " M docs/...", column 2 was "d",
    # and slicing [3:] produced "ocs/figures/...".
    for line in raw.splitlines():
        assert line[2] == " ", f"porcelain line lost a leading column: {line!r}"
    assert all(not path.startswith(" ") for path in info["dirty_files"])


def test_sampling_block_marks_a_single_draw_as_such():
    block = sampling_block(
        n_seeds=5,
        n_layouts=1,
        layout_seeds=[0],
        rollout_seeds_hacking=[0, 1, 2, 3, 4],
        rollout_seeds_legit=[1000, 1001, 1002, 1003, 1004],
        n_replicates=1,
    )
    assert block["single_draw"] is True
    assert block["confidence_intervals"] is None
    assert block["n_seeds_per_variant"] == 5
    assert block["rollout_seeds_legit"] == [1000, 1001, 1002, 1003, 1004]
    assert "Single unreplicated draw" in block["note"]


def test_sampling_block_replicated_run_is_not_marked_single_draw():
    block = sampling_block(
        n_seeds=10,
        n_layouts=1,
        layout_seeds=[0],
        rollout_seeds_hacking=list(range(10)),
        rollout_seeds_legit=[1000 + s for s in range(10)],
        n_replicates=5,
    )
    assert block["single_draw"] is False
    assert block["note"] is None
