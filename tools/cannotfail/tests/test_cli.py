"""The command line is what a maintainer runs, so its verdicts, exit codes and errors are tested as such."""

from __future__ import annotations

import json

import pytest

from cannotfail.cli import main


@pytest.fixture
def scores_csv(tmp_path):
    rows = ["cell,score,is_hack"]
    rows += [f"constant,0.5,{i % 2}" for i in range(8)]                     # fixed
    rows += [f"mostly_tied,{1.0 if i == 0 else 0.0},{1 if i < 10 else 0}" for i in range(20)]  # pinned
    rows += [f"separated,{i},{1 if i >= 4 else 0}" for i in range(8)]        # ok
    path = tmp_path / "scores.csv"
    path.write_text("# a comment line is skipped\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_resolution_report_names_each_verdict(scores_csv, capsys):
    code = main(["resolution", str(scores_csv), "--score", "score", "--label", "is_hack", "--group", "cell"])
    out = capsys.readouterr().out
    assert code == 0
    lines = {line.split()[0]: line.split()[-1] for line in out.splitlines() if line and line.split()[0] in {"constant", "mostly_tied", "separated"}}
    assert lines == {"constant": "fixed", "mostly_tied": "pinned", "separated": "ok"}
    assert "1 of 3 groups have a score vector with a single distinct value" in out
    assert "1 of 3 groups are pinned" in out


def test_fail_on_exits_one_for_ci(scores_csv, capsys):
    args = ["resolution", str(scores_csv), "--score", "score", "--label", "is_hack", "--group", "cell"]
    assert main(args + ["--fail-on", "fixed"]) == 1
    assert main(args + ["--fail-on", "dust"]) == 0
    capsys.readouterr()


def test_json_output_is_machine_readable(scores_csv, capsys):
    code = main(["resolution", str(scores_csv), "--score", "score", "--label", "is_hack", "--group", "cell", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["groups"]["constant"]["verdict"] == "fixed"
    assert payload["groups"]["mostly_tied"]["auroc_range"] == pytest.approx([0.45, 0.55])
    assert payload["summary"] == {"n_groups": 3, "n_single_valued": 1, "by_verdict": {"fixed": 1, "ok": 1, "pinned": 1}}


def test_without_labels_it_counts_distinct_values(scores_csv, capsys):
    assert main(["resolution", str(scores_csv), "--score", "score", "--group", "cell"]) == 0
    out = capsys.readouterr().out
    assert "No --label given" in out


def test_positive_names_the_positive_class(tmp_path, capsys):
    path = tmp_path / "s.tsv"
    path.write_text("score\tkind\n1\thack\n2\thack\n3\tlegit\n4\tlegit\n", encoding="utf-8")
    assert main(["resolution", str(path), "--score", "score", "--label", "kind", "--positive", "hack", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["groups"]["(all)"]["n_pos"] == 2


@pytest.mark.parametrize(
    "content,args,message",
    [
        ("score,y\n1,maybe\n", ["--score", "score", "--label", "y"], "not 0/1"),
        ("score,y\nx,1\n", ["--score", "score", "--label", "y"], "not a number"),
        ("score\n1\n", ["--score", "nope"], "no column 'nope'"),
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


def test_power_seeds_and_difference(capsys):
    pytest.importorskip("scipy")
    assert main(["power", "seeds", "--tests", "4"]) == 0
    assert capsys.readouterr().out.strip() == "24"
    assert main(["power", "difference", "--pos", "5", "--neg", "5", "--true-auroc", "0.7"]) == 0
    out = capsys.readouterr().out
    assert "weak evidence" in out and "only indicative" in out
