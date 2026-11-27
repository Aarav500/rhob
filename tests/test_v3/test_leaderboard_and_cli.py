"""Tests for the v3 leaderboard and CLI."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from rhob.v3.benchmark import BenchmarkResults, CellResult
from rhob.v3.cli.__main__ import build_parser, main
from rhob.v3.leaderboard.board import Leaderboard, LeaderboardEntry, submission_problems

_REPO = Path(__file__).resolve().parents[2]
_LEADERBOARD = _REPO / "leaderboard"
_COMMUNITY_HEADER = "# RHOB community submissions (self-reported, single draw)"

#: The documented minimum submission (submissions/README.md): four keys, no author or
#: timestamp.
_MINIMAL_SUBMISSION = {
    "detector_name": "My Detector",
    "access_level": "L2",
    "overall_auroc": 0.87,
    "n_cells": 40,
}

#: A board in the ``results`` schema scripts/v5_leaderboard_and_transfer.py writes (and
#: the retired v3.2 ``leaderboard/leaderboard.json`` used). ``Leaderboard.load`` used to
#: read this as an empty board, and ``rhob submit`` then saved one entry over it.
_RESULTS_SCHEMA_BOARD = {
    "timestamp": "2026-07-07T19:57:00",
    "total_detectors": 1,
    "results": {"Reward CUSUM": {"access_level": "L0", "overall_auroc": 0.6, "cells": 25}},
}


def _entry(**overrides) -> dict:
    """A complete, valid board entry as JSON would hold it, with ``overrides`` applied."""
    entry = asdict(
        LeaderboardEntry("E", "L2", "a", "2026-11-27T00:00:00+00:00", 0.8, 3, {"f": 0.8})
    )
    entry.update(overrides)
    return entry


def _fake_results() -> BenchmarkResults:
    return BenchmarkResults(
        detector_name="fake_det",
        access_level="L2",
        cells=[
            CellResult("gridworld_camping", "camping", 1.0, 1.000, 10),
            CellResult("continuous_camping", "camping", 0.90, 0.850, 10),
            CellResult("continuous_camping", "camping", 0.70, 0.600, 10),
        ],
    )


def test_entry_from_results_aggregates_correctly():
    entry = LeaderboardEntry.from_results(_fake_results(), author="tester")
    assert entry.detector_name == "fake_det"
    assert entry.access_level == "L2"
    assert entry.n_cells == 3
    assert entry.family_auroc["gridworld_camping"] == 1.0
    assert abs(entry.family_auroc["continuous_camping"] - 0.725) < 1e-6
    assert "camping" in entry.mechanism_auroc


def test_leaderboard_standings_sorted_descending():
    board = Leaderboard()
    board.submit(_fake_results(), author="a")
    weaker = BenchmarkResults("weak_det", "L0", [CellResult("gridworld_camping", "camping", 1.0, 0.5, 10)])
    board.submit(weaker, author="b")
    standings = board.standings()
    assert standings[0].detector_name == "fake_det"
    assert standings[1].detector_name == "weak_det"


def test_standings_handles_none_overall_auroc():
    """A detector that errored during evaluation can be written through with
    overall_auroc=None (seen as "Reward Skewness" in the retired v3.2 board,
    ``git show b7324ec:leaderboard/leaderboard.json``, from an older population script),
    and :meth:`LeaderboardEntry.from_results` now writes None for a run with no measured
    cell. standings() must sort it last, not crash -- render_standings_md() must not
    crash formatting it either."""
    board = Leaderboard()
    board.submit(_fake_results(), author="a")
    broken = LeaderboardEntry(
        detector_name="broken_det",
        access_level="L0",
        author="c",
        timestamp="",
        overall_auroc=None,
        n_cells=0,
    )
    board.add(broken)

    standings = board.standings()
    assert standings[-1].detector_name == "broken_det"

    rendered = board.render_standings_md()
    assert "broken_det" in rendered
    assert "-" in rendered  # the None entry's AUROC column


def test_leaderboard_save_and_load_roundtrip(tmp_path: Path):
    board = Leaderboard()
    board.submit(_fake_results(), author="tester")
    path = tmp_path / "leaderboard.json"
    board.save(path)

    reloaded = Leaderboard.load(path)
    assert len(reloaded.entries) == 1
    assert reloaded.entries[0].detector_name == "fake_det"


def test_leaderboard_load_missing_file_returns_empty(tmp_path: Path):
    board = Leaderboard.load(tmp_path / "does_not_exist.json")
    assert board.entries == []


def test_load_refuses_a_board_in_another_schema(tmp_path: Path):
    """Not an empty board: the caller that loads a board saves it back."""
    path = tmp_path / "board.json"
    path.write_text(json.dumps(_RESULTS_SCHEMA_BOARD))
    with pytest.raises(ValueError, match="load_any_leaderboard_json"):
        Leaderboard.load(path)


@pytest.mark.parametrize("name", ["v5_leaderboard.json", "v5_replicated.json"])
def test_load_refuses_the_benchmarks_own_boards(name: str):
    """The committed boards are in the ``results`` and ``detectors`` schemas, and
    ``Leaderboard.load`` must refuse both rather than return nothing to save over them."""
    with pytest.raises(ValueError, match="not an entries-schema board"):
        Leaderboard.load(_LEADERBOARD / name)


@pytest.mark.parametrize(
    "text, match",
    [
        ("{not json", "not valid JSON"),
        ("[]", "not an entries-schema board"),
        ('{"entries": {}}', "'entries' is not a list"),
        ('{"entries": [{"detector_name": "x"}]}', "does not match LeaderboardEntry"),
        ('{"entries": [1]}', "entry 0 is not an object"),
        (json.dumps({"entries": [_entry(overall_auroc="0.9")]}), "entry 0 is invalid"),
        (json.dumps({"entries": [_entry(n_cells="3")]}), "n_cells"),
        (json.dumps({"entries": [_entry(family_auroc={"f": None})]}), "family_auroc"),
        ('{"entries": [], "x": NaN}', "not valid JSON"),
    ],
)
def test_load_refuses_a_malformed_board(tmp_path: Path, text: str, match: str):
    path = tmp_path / "board.json"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        Leaderboard.load(path)


def test_save_refuses_nan_and_writes_nothing(tmp_path: Path):
    """JSON has no NaN; ``save`` raises rather than write a file strict readers reject."""
    board = Leaderboard([LeaderboardEntry(**_entry(overall_auroc=float("nan")))])
    path = tmp_path / "sub" / "board.json"
    with pytest.raises(ValueError):
        board.save(path)
    assert not path.parent.exists()


def test_an_evaluated_entry_is_a_valid_submission():
    """What ``rhob evaluate`` writes passes ``rhob validate``, with or without a measured
    cell."""
    assert submission_problems(asdict(LeaderboardEntry.from_results(_fake_results()))) == []
    none_measured = BenchmarkResults(
        "na_det", "L1",
        [CellResult("continuous_camping", "camping", 0.9, float("nan"), 10, na_reason="n/a")],
    )
    assert submission_problems(asdict(LeaderboardEntry.from_results(none_measured))) == []


def test_n_cells_counts_measured_cells_only():
    """A not-applicable cell is neither averaged nor counted: ``n_cells`` is the
    denominator of ``overall_auroc``, as the adapters read it from ``cells_measured``."""
    results = BenchmarkResults(
        detector_name="l1_det",
        access_level="L1",
        cells=[
            CellResult("gridworld_camping", "camping", 1.0, 0.900, 10),
            CellResult(
                "continuous_camping", "camping", 0.90, float("nan"), 10,
                na_reason="horizon 40 is below the 50 episodes this detector's windows need",
            ),
            CellResult(
                "continuous_camping", "camping", 0.70, float("nan"), 10,
                na_reason="horizon 40 is below the 50 episodes this detector's windows need",
            ),
        ],
    )
    entry = LeaderboardEntry.from_results(results)
    assert entry.n_cells == len(results.scored_cells) == 1
    assert entry.overall_auroc == 0.9
    assert entry.family_auroc == {"gridworld_camping": 0.9}


def test_a_run_with_no_measured_cell_has_no_overall_auroc():
    """None, not NaN: the entry is written to JSON, which has no NaN."""
    results = BenchmarkResults(
        "na_det", "L1",
        [CellResult("continuous_camping", "camping", 0.9, float("nan"), 10, na_reason="n/a")],
    )
    entry = LeaderboardEntry.from_results(results)
    assert entry.overall_auroc is None
    assert entry.n_cells == 0
    json.dumps(asdict(entry), allow_nan=False)


def test_render_standings_md_contains_detector_name():
    board = Leaderboard()
    board.submit(_fake_results(), author="tester")
    md = board.render_standings_md()
    assert "fake_det" in md
    assert "tester" in md


def test_render_by_mechanism_and_difficulty():
    board = Leaderboard()
    board.submit(_fake_results(), author="tester")
    assert "camping" in board.render_by_mechanism_md()
    assert "TRIVIAL" in board.render_by_difficulty_md()


def test_render_keeps_a_name_with_a_pipe_in_one_cell():
    board = Leaderboard([LeaderboardEntry(**_entry(detector_name="A|B\nC", author="x|y"))])
    row = board.render_standings_md().splitlines()[2]
    assert "A\\|B C" in row and "x\\|y" in row
    assert row.replace("\\|", "").count("|") == 7


# ------------------------------------------------------------------------------- CLI
# In-process through ``main(argv)``, which is what ``python -m rhob`` calls: each
# subprocess pays the whole ``rhob`` import. ``families``, ``evaluate`` and one ``submit``
# test still go through ``python -m rhob`` to cover that wiring.


def _run_cli(*args: str, cwd: Path = _REPO) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "rhob", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def _cli(capsys: pytest.CaptureFixture, *args: object) -> tuple[int, str]:
    """``python -m rhob *args`` in-process: ``(exit code, stdout)``."""
    code = main([str(a) for a in args])
    return code, capsys.readouterr().out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(directory: Path) -> dict[str, str]:
    """Every file in ``directory`` by name, with its sha256."""
    return {p.name: _sha256(p) for p in sorted(directory.iterdir())}


def _write_submission(tmp_path: Path, text: str | None = None, **overrides) -> Path:
    path = tmp_path / "submission.json"
    path.write_text(text if text is not None else json.dumps({**_MINIMAL_SUBMISSION, **overrides}))
    return path


def test_cli_families_lists_builtin_families():
    result = _run_cli("families")
    assert result.returncode == 0
    assert "gridworld_camping" in result.stdout
    assert "continuous_camping" in result.stdout


def test_cli_validate_rejects_missing_keys(tmp_path: Path, capsys):
    bad = _write_submission(tmp_path, json.dumps({"detector_name": "x"}))
    code, out = _cli(capsys, "validate", bad)
    assert code == 1
    assert "INVALID: missing keys" in out


def test_cli_validate_accepts_well_formed_submission(tmp_path: Path, capsys):
    good = _write_submission(tmp_path, detector_name="x", overall_auroc=0.9, n_cells=4)
    code, out = _cli(capsys, "validate", good)
    assert code == 0
    assert out.startswith("VALID")


def test_cli_validate_accepts_no_measured_cell(tmp_path: Path, capsys):
    """``null`` is how ``rhob evaluate`` writes the AUROC of a run with no measured cell."""
    good = _write_submission(tmp_path, overall_auroc=None, n_cells=0)
    assert _cli(capsys, "validate", good)[0] == 0


@pytest.mark.parametrize(
    "text, match",
    [
        ("[]", "a submission is a JSON object"),
        ('{"detector_name": "x", "access_level": "L2", "overall_auroc": NaN, "n_cells": 1}', "NaN"),
        (
            '{"detector_name": "x", "access_level": "L2", "overall_auroc": Infinity, "n_cells": 1}',
            "Infinity",
        ),
        (json.dumps({**_MINIMAL_SUBMISSION, "overall_auroc": "0.9"}), "overall_auroc"),
        (json.dumps({**_MINIMAL_SUBMISSION, "overall_auroc": True}), "overall_auroc"),
        (json.dumps({**_MINIMAL_SUBMISSION, "overall_auroc": 1.5}), "overall_auroc"),
        # Valid JSON that parses to inf: refused by the range check, not the parser.
        ('{"detector_name": "x", "access_level": "L2", "overall_auroc": 1e999, "n_cells": 1}',
         "overall_auroc must be a number"),
        (json.dumps({**_MINIMAL_SUBMISSION, "n_cells": -1}), "n_cells"),
        (json.dumps({**_MINIMAL_SUBMISSION, "n_cells": 3.0}), "n_cells"),
        (json.dumps({**_MINIMAL_SUBMISSION, "n_cells": True}), "n_cells"),
        (json.dumps({**_MINIMAL_SUBMISSION, "detector_name": ""}), "detector_name"),
        (json.dumps({**_MINIMAL_SUBMISSION, "access_level": ["L2"]}), "access_level"),
        (json.dumps({**_MINIMAL_SUBMISSION, "author": 5}), "author"),
        (json.dumps({**_MINIMAL_SUBMISSION, "family_auroc": [0.9]}), "family_auroc"),
        (json.dumps({**_MINIMAL_SUBMISSION, "family_auroc": {"f": "0.9"}}), "family_auroc"),
        (json.dumps({**_MINIMAL_SUBMISSION, "mechanism_auroc": {"m": None}}), "mechanism_auroc"),
        (json.dumps({**_MINIMAL_SUBMISSION, "tier_auroc": {"Easy": 0.9}}), "difficulty tiers"),
    ],
)
def test_cli_validate_rejects_a_value_submit_could_not_store(
    tmp_path: Path, capsys, text: str, match: str
):
    """These passed ``validate`` (and so leaderboard_validate.yml) when only the keys
    were checked; ``submit`` then saved them and crashed rendering, or wrote ``NaN``."""
    code, out = _cli(capsys, "validate", _write_submission(tmp_path, text))
    assert code == 1
    assert out.startswith("INVALID") and match in out, out


def test_cli_evaluate_end_to_end_on_gridworld(tmp_path: Path):
    detector_file = tmp_path / "det.py"
    detector_file.write_text(
        "from rhob.detectors.posthoc import PosthocDetector\n"
        "class D(PosthocDetector):\n"
        "    @property\n"
        "    def access_level(self): return 'L2'\n"
        "    @property\n"
        "    def name(self): return 'toy'\n"
        "    def classify(self, run): return float(run.behav_trace[-1])\n"
        "    def detect_onset(self, run): return len(run.proxy_rewards)//2\n"
    )
    out = tmp_path / "submission.json"
    result = _run_cli(
        "evaluate", "--detector", str(detector_file), "--families", "gridworld_camping",
        "--n-seeds", "3", "--output", str(out),
    )
    assert result.returncode == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["detector_name"] == "toy"
    assert data["overall_auroc"] > 0.9
    assert submission_problems(data) == []


def test_submit_defaults_to_the_community_board_dir():
    args = build_parser().parse_args(["submit", "x.json"])
    assert args.board_dir == Path("leaderboard") / "community"
    assert build_parser().parse_args(["submit", "x.json", "--board-dir", "b"]).board_dir == Path("b")


def test_cli_submit_minimal_submission_into_board_dir(tmp_path: Path):
    """Through ``python -m rhob``. The documented four-key minimum used to crash
    ``submit`` (LeaderboardEntry needs an author and a timestamp); it now records the entry
    with defaults for both."""
    community = _LEADERBOARD / "community"
    existed_before = community.exists()
    sub = _write_submission(tmp_path)
    board_dir = tmp_path / "b"

    result = _run_cli("submit", str(sub), "--board-dir", str(board_dir))
    assert result.returncode == 0, result.stdout + result.stderr

    entries = json.loads((board_dir / "board.json").read_text())["entries"]
    assert len(entries) == 1
    assert entries[0]["detector_name"] == "My Detector"
    assert entries[0]["author"] == "anonymous"
    assert entries[0]["timestamp"]
    for name in ("README.md", "by_mechanism.md", "by_difficulty.md"):
        assert (board_dir / name).read_text().splitlines()[0] == _COMMUNITY_HEADER
    assert "My Detector" in (board_dir / "README.md").read_text()
    # --board-dir is honoured: the repository's own leaderboard/ is not written.
    assert community.exists() == existed_before


def test_cli_submit_appends_to_its_own_board(tmp_path: Path, capsys):
    """A second submission adds to the board and re-renders the tables ``submit`` wrote
    the first time; the refusal of foreign Markdown does not block its own files."""
    board_dir = tmp_path / "b"
    sub = _write_submission(tmp_path)
    assert _cli(capsys, "submit", sub, "--board-dir", board_dir)[0] == 0
    code, out = _cli(
        capsys, "submit", sub, "--board-dir", board_dir, "--name", "Second", "--author", "me"
    )
    assert code == 0, out
    entries = json.loads((board_dir / "board.json").read_text())["entries"]
    assert [(e["detector_name"], e["author"]) for e in entries] == [
        ("My Detector", "anonymous"),
        ("Second", "me"),
    ]
    assert "Second" in (board_dir / "README.md").read_text()


def test_cli_submit_writes_leaderboard_community_by_default(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
):
    """Without --board-dir, ``submit`` writes ``leaderboard/community/`` under the working
    directory, never ``leaderboard/`` itself. Run from a scratch directory so the
    repository's tree is not touched."""
    sub = _write_submission(tmp_path)
    monkeypatch.chdir(tmp_path)
    code, out = _cli(capsys, "submit", sub)
    assert code == 0, out
    assert (tmp_path / "leaderboard" / "community" / "board.json").exists()
    assert sorted(p.name for p in (tmp_path / "leaderboard").iterdir()) == ["community"]


def test_cli_submit_refuses_another_schema(tmp_path: Path, capsys):
    """A ``results``-schema board.json is refused, byte for byte untouched, and nothing
    else is written next to it."""
    board_dir = tmp_path / "b"
    board_dir.mkdir()
    (board_dir / "board.json").write_text(json.dumps(_RESULTS_SCHEMA_BOARD, indent=2))
    before = _snapshot(board_dir)

    code, out = _cli(capsys, "submit", _write_submission(tmp_path), "--board-dir", board_dir)
    assert code == 1
    assert "REFUSED" in out and "load_any_leaderboard_json" in out
    assert _snapshot(board_dir) == before


def test_cli_submit_refuses_a_value_of_the_wrong_type(tmp_path: Path, capsys):
    """A string AUROC used to be saved to board.json and then crash rendering, leaving
    the tables stale and every later ``submit`` crashing on the mixed types."""
    board_dir = tmp_path / "b"
    assert _cli(capsys, "submit", _write_submission(tmp_path), "--board-dir", board_dir)[0] == 0
    before = _snapshot(board_dir)

    bad = _write_submission(tmp_path, overall_auroc="0.9")
    code, out = _cli(capsys, "submit", bad, "--board-dir", board_dir)
    assert code == 1
    assert out.startswith("INVALID") and "overall_auroc" in out
    assert _snapshot(board_dir) == before


def test_cli_submit_refuses_an_invalid_name_override(tmp_path: Path, capsys):
    board_dir = tmp_path / "b"
    code, out = _cli(
        capsys, "submit", _write_submission(tmp_path), "--board-dir", board_dir, "--name", " "
    )
    assert code == 1
    assert "detector_name" in out
    assert not board_dir.exists()


def test_cli_submit_refuses_a_board_holding_an_invalid_entry(tmp_path: Path, capsys):
    """A board an older ``submit`` corrupted is refused, not crashed on and half-rewritten."""
    board_dir = tmp_path / "b"
    board_dir.mkdir()
    (board_dir / "board.json").write_text(json.dumps({"entries": [_entry(overall_auroc="0.9")]}))
    before = _snapshot(board_dir)

    code, out = _cli(capsys, "submit", _write_submission(tmp_path), "--board-dir", board_dir)
    assert code == 1
    assert "REFUSED" in out and "entry 0 is invalid" in out
    assert _snapshot(board_dir) == before


def test_cli_submit_refuses_to_overwrite_markdown_it_did_not_write(tmp_path: Path, capsys):
    """``--board-dir leaderboard`` would otherwise replace the hand-written index README
    with a standings table: a README that does not start with the community header is
    refused, and nothing is written, board.json included."""
    board_dir = tmp_path / "b"
    board_dir.mkdir()
    readme = board_dir / "README.md"
    readme.write_text((_LEADERBOARD / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
    before = _snapshot(board_dir)

    code, out = _cli(capsys, "submit", _write_submission(tmp_path), "--board-dir", board_dir)
    assert code == 1
    assert "REFUSED" in out
    assert _snapshot(board_dir) == before


def test_cli_submit_leaves_other_markdown_alone(tmp_path: Path, capsys):
    """Only the three files ``submit`` renders are checked for the header; another
    Markdown file in the directory neither blocks it nor is written."""
    board_dir = tmp_path / "b"
    board_dir.mkdir()
    notes = board_dir / "NOTES.md"
    notes.write_text("# Maintainer notes\n")
    before = _sha256(notes)

    code, out = _cli(capsys, "submit", _write_submission(tmp_path), "--board-dir", board_dir)
    assert code == 0, out
    assert _sha256(notes) == before
    assert sorted(p.name for p in board_dir.iterdir()) == [
        "NOTES.md", "README.md", "board.json", "by_difficulty.md", "by_mechanism.md",
    ]


# ------------------------------------------------------------- leaderboard/ index


def _index() -> str:
    return (_LEADERBOARD / "README.md").read_text(encoding="utf-8")


def test_the_retired_v3_tables_are_gone():
    for name in ("leaderboard.json", "by_mechanism.md", "by_difficulty.md"):
        assert not (_LEADERBOARD / name).exists(), name
    for name in ("populate_v3_leaderboard.py", "populate_leaderboard_v32.py"):
        assert not (_REPO / "scripts" / name).exists(), name


def test_the_leaderboard_index_names_every_committed_board():
    """A board added to leaderboard/ without an entry in the index fails here."""
    index = _index()
    for path in sorted(_LEADERBOARD.glob("*.json")):
        assert f"`{path.name}`" in index, f"{path.name} is not in leaderboard/README.md"


def test_the_leaderboard_index_links_resolve():
    links = re.findall(r"\]\(([^)#\s]+)\)", _index())
    assert links
    for link in links:
        assert (_LEADERBOARD / link).resolve().exists(), link


def test_the_leaderboard_index_quotes_no_figures():
    """The index is numberless on purpose: a figure quoted here would go stale the next
    time a board is regenerated, as the v3.2 standings did."""
    figures = re.findall(r"(?<![\w.-])\d?\.\d{2,}\b", _index())
    assert not figures, figures
