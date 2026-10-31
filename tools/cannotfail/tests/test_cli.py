"""The command line is what a maintainer runs, so its verdicts, exit codes and errors are tested as such.

Several of these exist because a review installed the tool as a stranger would and found
the CLI dropping rows, rejecting Excel's files, or printing JSON that strict parsers refuse.
"""

from __future__ import annotations

import json

import pytest

from cannotfail.cli import main


def _strict_json(text):
    def refuse(name):
        raise ValueError(f"non-standard JSON constant {name}")
    return json.loads(text, parse_constant=refuse)


@pytest.fixture
def scores_csv(tmp_path):
    rows = ["cell,score,is_hack"]
    rows += [f"constant,0.5,{i % 2}" for i in range(8)]                                       # fixed
    rows += [f"mostly_tied,{1.0 if i == 0 else 0.0},{1 if i < 10 else 0}" for i in range(20)]  # pinned
    rows += [f"separated,{i},{1 if i >= 4 else 0}" for i in range(8)]                          # ok
    path = tmp_path / "scores.csv"
    path.write_text("# a comment above the header is skipped\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _run_json(capsys, *args):
    code = main([*args, "--json"])
    return code, _strict_json(capsys.readouterr().out)


def test_each_cell_gets_its_verdict(scores_csv, capsys):
    code, payload = _run_json(capsys, "resolution", str(scores_csv), "--score", "score", "--label", "is_hack", "--group", "cell")
    assert code == 0
    verdicts = {k: v["verdict"] for k, v in payload["groups"].items()}
    assert verdicts == {"constant": "fixed", "mostly_tied": "pinned", "separated": "ok"}
    assert payload["groups"]["constant"]["why"] == "one distinct value"
    assert payload["groups"]["mostly_tied"]["auroc_range"] == pytest.approx([0.45, 0.55])
    assert payload["summary"] == {"n_groups": 3, "n_single_valued": 1, "by_verdict": {"fixed": 1, "ok": 1, "pinned": 1}}


def test_the_text_report_explains_itself(scores_csv, capsys):
    assert main(["resolution", str(scores_csv), "--score", "score", "--label", "is_hack", "--group", "cell"]) == 0
    out = capsys.readouterr().out
    assert "fixed (one distinct value)" in out
    assert "1 of 3 groups has at most one distinct value" in out
    assert "1 of 3 groups is pinned" in out
    assert "can still fail, but only on the interval's width" in out


def test_fail_on_exits_one_for_ci(scores_csv, capsys):
    args = ["resolution", str(scores_csv), "--score", "score", "--label", "is_hack", "--group", "cell"]
    assert main(args + ["--fail-on", "fixed"]) == 1
    assert main(args + ["--fail-on", "dust"]) == 0
    capsys.readouterr()


def test_a_data_row_starting_with_hash_is_kept(tmp_path, capsys):
    """Found by review: '#' rows were dropped anywhere in the file, hiding a cell from --fail-on."""
    path = tmp_path / "s.csv"
    path.write_text("cell,score,y\n#7,1.0,1\n#7,1.0,0\nok,1,1\nok,2,0\n", encoding="utf-8")
    code, payload = _run_json(capsys, "resolution", str(path), "--score", "score", "--label", "y", "--group", "cell")
    assert payload["groups"]["#7"]["verdict"] == "fixed"
    assert main(["resolution", str(path), "--score", "score", "--label", "y", "--group", "cell", "--fail-on", "fixed"]) == 1
    capsys.readouterr()


def test_excel_utf8_bom_is_accepted(tmp_path, capsys):
    """Found by review: Excel's 'CSV UTF-8' put a BOM in the first header name."""
    path = tmp_path / "excel.csv"
    path.write_bytes("﻿score,y\n1,1\n2,0\n".encode("utf-8"))
    code, payload = _run_json(capsys, "resolution", str(path), "--score", "score", "--label", "y")
    assert code == 0
    assert payload["groups"]["(all)"]["n"] == 2


def test_json_never_contains_nan(tmp_path, capsys):
    """Found by review: an empty class or one finite score produced bare NaN."""
    path = tmp_path / "s.csv"
    path.write_text("score,y\n1,1\n2,1\n", encoding="utf-8")
    code, payload = _run_json(capsys, "resolution", str(path), "--score", "score", "--label", "y")
    assert code == 0
    g = payload["groups"]["(all)"]
    assert g["exact_resolution"] is None and g["verdict"] == "fixed"
    code, payload = _run_json(capsys, "power", "equivalence", "--per-side", "4", "--layouts", "1")
    assert payload["half_width"] is None and payload["can_certify"] is False


def test_without_labels_it_counts_distinct_values(scores_csv, capsys):
    assert main(["resolution", str(scores_csv), "--score", "score", "--group", "cell"]) == 0
    assert "No --label given" in capsys.readouterr().out


def test_positive_names_the_positive_class(tmp_path, capsys):
    path = tmp_path / "s.tsv"
    path.write_text("score\tkind\n1\thack\n2\thack\n3\tlegit\n4\tlegit\n", encoding="utf-8")
    code, payload = _run_json(capsys, "resolution", str(path), "--score", "score", "--label", "kind", "--positive", "hack")
    assert payload["groups"]["(all)"]["n_pos"] == 2


@pytest.mark.parametrize(
    "content,args,message",
    [
        ("score,y\n1,maybe\n", ["--score", "score", "--label", "y"], "not 0/1"),
        ("score,y\nx,1\n", ["--score", "score", "--label", "y"], "not a number"),
        ("score\n1\n", ["--score", "nope"], "no column 'nope'"),
        ("score,y\n", ["--score", "score"], "no data rows"),
        ("", ["--score", "score"], "no header row"),
        # Found by review: a blank or mistyped --positive used to count silently as negative.
        ("score,k\n1,hack\n2,\n", ["--score", "score", "--label", "k", "--positive", "hack"], "blank label"),
        ("score,k\n1,hack\n2,legit\n", ["--score", "score", "--label", "k", "--positive", "Hack"], "no row has label 'Hack'"),
    ],
)
def test_bad_input_exits_two_with_a_reason(tmp_path, capsys, content, args, message):
    path = tmp_path / "bad.csv"
    path.write_text(content, encoding="utf-8")
    assert main(["resolution", str(path), *args]) == 2
    assert message in capsys.readouterr().err


def test_power_equivalence_says_when_a_design_cannot_certify(capsys):
    pytest.importorskip("scipy")
    assert main(["power", "equivalence", "--per-side", "4", "--layouts", "12"]) == 0
    assert "cannot certify equivalence at any observed mean" in capsys.readouterr().out
    assert main(["power", "equivalence", "--per-side", "4", "--layouts", "12", "--fail-on-cannot-certify"]) == 1
    capsys.readouterr()


def test_power_seeds(capsys):
    pytest.importorskip("scipy")
    assert main(["power", "seeds", "--tests", "4"]) == 0
    assert capsys.readouterr().out.strip() == "24"


def test_power_difference_needs_no_scipy(capsys):
    assert main(["power", "difference", "--pos", "5", "--neg", "5", "--true-auroc", "0.7"]) == 0
    out = capsys.readouterr().out
    assert "weak evidence" in out and "only indicative" in out
    assert main(["power", "difference", "--pos", "3", "--neg", "3", "--true-auroc", "1.0"]) == 0
    assert "No data can make a two-sided Mann-Whitney test reject" in capsys.readouterr().out


def test_a_leaderboard_predictions_json_is_one_command(tmp_path, capsys):
    """RAID-style predictions.json: a list of {id, score}. The case that motivated JSON input."""
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps([{"id": f"x{i}", "score": 1.0} for i in range(50)]), encoding="utf-8")
    code, payload = _run_json(capsys, "resolution", str(path), "--score", "score")
    assert code == 0
    g = payload["groups"]["(all)"]
    assert g["n"] == 50 and g["n_distinct"] == 1 and g["verdict"] == "fixed"
    assert main(["resolution", str(path), "--score", "score", "--fail-on", "fixed"]) == 1
    capsys.readouterr()


def test_jsonl_with_nulls_and_labels(tmp_path, capsys):
    path = tmp_path / "rows.jsonl"
    rows = [{"score": 0.25, "y": True}, {"score": None, "y": True}, {"score": 0.75, "y": False}, {}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    code = main(["resolution", str(path), "--score", "score", "--label", "y", "--json"])
    out = capsys.readouterr()
    assert code == 2 and "blank" not in out.err  # the empty object has no label: rejected as not 0/1
    path.write_text("\n".join(json.dumps(r) for r in rows[:3]) + "\n", encoding="utf-8")
    code, payload = _run_json(capsys, "resolution", str(path), "--score", "score", "--label", "y")
    g = payload["groups"]["(all)"]
    assert code == 0 and g["n_nonfinite"] == 1 and g["exact_resolution"] == 1.0


@pytest.mark.parametrize("content,message", [
    ('{"score": 1}', "expected a list of objects"),
    ("[]", "the list is empty"),
    ('[{"id": 1}]', "no field 'score'"),
    ("[{", "not valid JSON"),
])
def test_bad_json_exits_two(tmp_path, capsys, content, message):
    path = tmp_path / "p.json"
    path.write_text(content, encoding="utf-8")
    assert main(["resolution", str(path), "--score", "score"]) == 2
    assert message in capsys.readouterr().err
