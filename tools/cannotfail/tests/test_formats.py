"""The file formats a maintainer already has: nested JSON, several group columns, Parquet, Inspect logs.

The examples/ files are the ones the README's recipes run. Tests that need pyarrow or
inspect_ai skip without them; the ones that check what happens when they are missing
hide them instead, so they run everywhere.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cannotfail.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _strict_json(text):
    def refuse(name):
        raise ValueError(f"non-standard JSON constant {name}")
    return json.loads(text, parse_constant=refuse)


def _json(capsys, *args):
    code = main([*args, "--json"])
    out = capsys.readouterr()
    assert code == 0, out.err
    return _strict_json(out.out)


def _error(capsys, *args):
    code = main(list(args))
    err = capsys.readouterr().err
    assert code == 2, err
    return err


def _write_json(tmp_path, records, name="p.json"):
    path = tmp_path / name
    if name.endswith(".jsonl"):
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    else:
        path.write_text(json.dumps(records), encoding="utf-8")
    return path


# --- dotted paths ----------------------------------------------------------------------

NESTED = [
    {"id": i, "scores": {"detector": {"value": v}, "baseline": 0.5}, "gold": {"is_hack": i % 2 == 0},
     "probs": [1 - v, v], "meta": {"task": "t1" if i < 4 else "t2"}}
    for i, v in enumerate([0.9, 0.1, 0.8, 0.3, 0.6, 0.6, 0.6, 0.6])
]


@pytest.mark.parametrize("name", ["p.json", "p.jsonl"])
def test_a_dotted_path_reads_a_nested_field(tmp_path, capsys, name):
    path = _write_json(tmp_path, NESTED, name)
    payload = _json(capsys, "resolution", str(path), "--score", "scores.detector.value",
                    "--label", "gold.is_hack", "--group", "meta.task")
    assert {k: v["verdict"] for k, v in payload["groups"].items()} == {"t1": "ok", "t2": "fixed"}
    assert payload["groups"]["t1"]["n_distinct"] == 4


def test_an_integer_segment_indexes_a_list(tmp_path, capsys):
    path = _write_json(tmp_path, NESTED)
    by_index = _json(capsys, "resolution", str(path), "--score", "probs.1", "--label", "gold.is_hack")
    by_key = _json(capsys, "resolution", str(path), "--score", "scores.detector.value", "--label", "gold.is_hack")
    assert by_index["groups"] == by_key["groups"]


def test_a_key_with_a_dot_is_matched_whole_first(tmp_path, capsys):
    records = [{"a.b": float(i), "a": {"b": 7.0}, "y": i % 2} for i in range(4)]
    path = _write_json(tmp_path, records)
    assert _json(capsys, "resolution", str(path), "--score", "a.b")["groups"]["(all)"]["n_distinct"] == 4


def test_a_literal_key_that_leads_nowhere_falls_back_to_splitting(tmp_path, capsys):
    records = [{"a.b": {"other": 1}, "a": {"b": {"c": float(i)}}} for i in range(3)]
    path = _write_json(tmp_path, records)
    assert _json(capsys, "resolution", str(path), "--score", "a.b.c")["groups"]["(all)"]["n_distinct"] == 3


def test_a_comma_in_a_key_is_just_a_character(tmp_path, capsys):
    """lm-evaluation-harness names metrics 'acc,none'; nothing may split on the comma."""
    records = [{"results": {"acc,none": float(i % 2)}, "task": "x"} for i in range(6)]
    path = _write_json(tmp_path, records)
    payload = _json(capsys, "resolution", str(path), "--score", "results.acc,none", "--group", "task")
    assert payload["groups"]["x"]["n_distinct"] == 2


def test_a_path_missing_from_some_records_is_blank_there(tmp_path, capsys):
    records = [{"s": {"v": 1.0}}, {"s": {}}, {"s": {"v": 2.0}}, {"t": 0}]
    path = _write_json(tmp_path, records, "p.jsonl")
    g = _json(capsys, "resolution", str(path), "--score", "s.v")["groups"]["(all)"]
    assert g["n"] == 4 and g["n_nonfinite"] == 2 and g["n_distinct"] == 2


@pytest.mark.parametrize(
    "path_arg,expected",
    [
        ("score", "no field 'score' in any of the 8 records; the keys at the top level are "
                  "['gold', 'id', 'meta', 'probs', 'scores']"),
        ("scores.detektor.value", "no field 'scores.detektor.value' in any of the 8 records. The deepest "
                                  "it reaches is 'scores', an object with keys ['baseline', 'detector']."),
        ("scores.detector.value.x", "The deepest it reaches is 'scores.detector.value', the value 0.9, "
                                    "which has no fields."),
        ("probs.2", "The deepest it reaches is 'probs', a list of 2, so the next part must be an index "
                    "from 0 to 1, not '2'."),
        ("probs.first", "a list of 2, so the next part must be an index from 0 to 1, not 'first'."),
    ],
)
def test_a_path_that_does_not_resolve_names_where_it_stopped(tmp_path, capsys, path_arg, expected):
    path = _write_json(tmp_path, NESTED)
    assert expected in _error(capsys, "resolution", str(path), "--score", path_arg)


def test_the_deepest_record_is_the_one_reported(tmp_path, capsys):
    records = [{"a": 1}, {"a": {"b": {"c": 1}}}]
    path = _write_json(tmp_path, records)
    err = _error(capsys, "resolution", str(path), "--score", "a.b.d")
    assert "in any of the 2 records. The deepest it reaches is 'a.b', an object with keys ['c']." in err


def test_a_long_key_list_is_cut_short(tmp_path, capsys):
    path = _write_json(tmp_path, [{f"k{i:02d}": i for i in range(20)}])
    err = _error(capsys, "resolution", str(path), "--score", "score")
    assert "in the one record" in err and "and 8 more" in err


@pytest.mark.parametrize(
    "path_arg,expected",
    [
        ("scores", "record 1: 'scores' holds an object with keys ['baseline', 'detector'], not a value; "
                   "name one of its fields, e.g. 'scores.detector'"),
        ("probs", "record 1: 'probs' holds a list of 2, not a value; index it, e.g. 'probs.0'"),
    ],
)
def test_a_path_to_an_object_or_list_is_refused(tmp_path, capsys, path_arg, expected):
    path = _write_json(tmp_path, NESTED)
    assert expected in _error(capsys, "resolution", str(path), "--score", path_arg)


def test_a_json_object_at_the_top_says_where_the_list_is(tmp_path, capsys):
    path = tmp_path / "sub.json"
    path.write_text(json.dumps({"model": "m", "predictions": [{"score": 1.0}]}), encoding="utf-8")
    err = _error(capsys, "resolution", str(path), "--score", "score")
    assert "expected a list of objects, one per scored item; this file holds one object" in err
    assert """its key 'predictions' holds a list of objects""" in err and """jq '.["predictions"]'""" in err


def test_an_inspect_log_in_json_form_points_to_format_inspect(tmp_path, capsys):
    path = tmp_path / "log.json"
    log = {"version": 2, "status": "success", "eval": {"task": "t"}, "samples": [{"id": 1, "scores": {}}]}
    path.write_text(json.dumps(log), encoding="utf-8")
    err = _error(capsys, "resolution", str(path), "--score", "score")
    assert "it looks like an Inspect log, which --format inspect reads" in err and "jq" not in err


def test_csv_columns_stay_literal_names(tmp_path, capsys):
    path = tmp_path / "s.csv"
    path.write_text("scores.value,y\n1,1\n2,0\n", encoding="utf-8")
    assert _json(capsys, "resolution", str(path), "--score", "scores.value", "--label", "y")["groups"]["(all)"]["n"] == 2
    err = _error(capsys, "resolution", str(path), "--score", "scores.other")
    assert "no column 'scores.other'" in err and "matched by their literal header, not as a dotted path" in err


def test_format_overrides_the_extension(tmp_path, capsys):
    path = tmp_path / "rows.txt"
    path.write_text("".join(json.dumps({"s": {"v": float(i)}}) + "\n" for i in range(3)), encoding="utf-8")
    assert _json(capsys, "resolution", str(path), "--score", "s.v", "--format", "jsonl")["groups"]["(all)"]["n"] == 3
    tsv = tmp_path / "rows.dat"
    tsv.write_text("s\ty\n1\t1\n2\t0\n", encoding="utf-8")
    assert _json(capsys, "resolution", str(tsv), "--score", "s", "--label", "y", "--format", "tsv")["groups"]["(all)"]["n"] == 2


def test_an_empty_jsonl_file_says_so(tmp_path, capsys):
    path = tmp_path / "e.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    assert "no records" in _error(capsys, "resolution", str(path), "--score", "s")


# --- several --group columns -----------------------------------------------------------

def test_several_groups_join_their_values(capsys):
    path = EXAMPLES / "predictions.json"
    args = ["resolution", str(path), "--score", "output.probs.1", "--label", "gold.is_hack",
            "--group", "task", "--group", "model"]
    payload = _json(capsys, *args)
    assert payload["group_by"] == ["task", "model"]
    assert {k: v["verdict"] for k, v in payload["groups"].items()} == {
        "qa | model-a": "ok", "qa | model-b": "fixed", "code | model-a": "ok", "code | model-b": "pinned"}
    assert payload["groups"]["code | model-b"]["group"] == {"task": "code", "model": "model-b"}
    assert list(payload) == ["file", "margin", "group_by", "groups", "summary"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("task | model   ")
    assert "code | model-b       12         2" in out


def test_group_order_is_the_order_given(tmp_path, capsys):
    path = EXAMPLES / "predictions.json"
    payload = _json(capsys, "resolution", str(path), "--score", "output.probs.1",
                    "--group", "model", "--group", "task")
    assert list(payload["groups"]) == ["model-a | qa", "model-b | qa", "model-a | code", "model-b | code"]


def test_labels_that_would_collide_are_written_as_json_lists(tmp_path, capsys):
    records = [{"a": "x | y", "b": "z", "s": 1.0}, {"a": "x", "b": "y | z", "s": 2.0}]
    path = _write_json(tmp_path, records)
    payload = _json(capsys, "resolution", str(path), "--score", "s", "--group", "a", "--group", "b")
    assert list(payload["groups"]) == ['["x | y", "z"]', '["x", "y | z"]']
    assert payload["summary"]["n_groups"] == 2


def test_one_group_prints_exactly_as_before(capsys):
    """A single --group must not change a byte: not the header, the keys, nor the JSON."""
    args = ["resolution", str(EXAMPLES / "scores.csv"), "--score", "score", "--label", "is_hack", "--group", "cell"]
    assert main(args) == 0
    assert capsys.readouterr().out == (
        "group              n  distinct     pos     neg  resolution      AUROC range  verdict\n"
        "reward_only       48         1      24      24       0.000   [0.500, 0.500]  fixed (one distinct value)\n"
        "mostly_tied       48         2      24      24       0.042   [0.479, 0.521]  pinned (AUROC confined within 0.5 +/- 0.0208)\n"
        "behavioural       48        48      24      24       1.000   [0.000, 1.000]  ok\n"
        "\n"
        "1 of 3 groups has at most one distinct value, so any rank statistic over it is fixed at 0.5.\n"
        "1 of 3 groups is pinned: the AUROC cannot get further from 0.5 than 0.1, so its estimate cannot "
        "leave an equivalence band of that width. A test that also needs its interval inside the band can "
        "still fail, but only on the interval's width; a pass there says the scores tie.\n"
    )
    payload = _json(capsys, *args)
    assert list(payload) == ["file", "margin", "groups", "summary"]
    assert list(payload["groups"]) == ["reward_only", "mostly_tied", "behavioural"]
    assert all("group" not in g for g in payload["groups"].values())


def test_the_same_column_as_label_and_group_is_read_once(tmp_path, capsys):
    path = tmp_path / "s.csv"
    path.write_text("s,y\n1,1\n2,0\n3,1\n", encoding="utf-8")
    payload = _json(capsys, "resolution", str(path), "--score", "s", "--label", "y", "--group", "y")
    assert set(payload["groups"]) == {"1", "0"}


# --- the recipes' files ----------------------------------------------------------------

def test_lm_eval_samples_strict_match_is_constant(capsys):
    """Each doc appears once per filter; strict-match never extracts here, so it scores 0 everywhere."""
    path = EXAMPLES / "samples_toy_arith.jsonl"
    payload = _json(capsys, "resolution", str(path), "--score", "exact_match", "--group", "filter")
    groups = payload["groups"]
    assert groups["strict-match"]["verdict"] == "fixed" and groups["strict-match"]["n"] == 12
    assert groups["flexible-extract"]["verdict"] == "ok" and groups["flexible-extract"]["n_distinct"] == 2
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {r["filtered_resps"][0] for r in rows if r["filter"] == "strict-match"} == {"[invalid]"}
    assert sum(r["exact_match"] for r in rows if r["filter"] == "flexible-extract") == 9.0


def test_an_inspect_log_without_format_inspect_is_not_read_as_csv(capsys):
    err = _error(capsys, "resolution", str(EXAMPLES / "hack_probe.eval"), "--score", "score_stated_probability")
    assert "an .eval file is an Inspect log; pass --format inspect" in err
    assert 'pip install "cannotfail[inspect]"' in err


# --- Parquet ---------------------------------------------------------------------------

def test_parquet_without_pyarrow_says_how_to_get_it(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    err = _error(capsys, "resolution", str(EXAMPLES / "scores.parquet"), "--score", "detector.score")
    assert 'reading Parquet needs pyarrow: pip install "cannotfail[parquet]"' in err


def test_a_pyarrow_that_fails_to_import_is_reported_as_such(monkeypatch, capsys):
    """pandas 2.1 beside NumPy 2 fails this way, with a ValueError rather than an ImportError."""
    import builtins

    real_import = builtins.__import__

    def broken(name, *args, **kwargs):
        if name.split(".")[0] == "pyarrow":
            raise ValueError("numpy.dtype size changed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    err = _error(capsys, "resolution", str(EXAMPLES / "scores.parquet"), "--score", "detector.score")
    assert "pyarrow is installed but failed to import (ValueError: numpy.dtype size changed)" in err
    assert 'pip install "cannotfail[parquet]"' in err


def test_the_example_parquet_holds_the_example_csv():
    pq = pytest.importorskip("pyarrow.parquet")
    with open(EXAMPLES / "scores.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    table = pq.read_table(EXAMPLES / "scores.parquet").to_pylist()
    assert [(r["cell"], r["is_hack"], r["detector"]["score"]) for r in table] == [
        (r["cell"], r["is_hack"] == "1", float(r["score"])) for r in rows]


def test_a_struct_column_is_read_by_dotted_path(capsys):
    pytest.importorskip("pyarrow")
    from_parquet = _json(capsys, "resolution", str(EXAMPLES / "scores.parquet"), "--score", "detector.score",
                         "--label", "is_hack", "--group", "cell")
    from_csv = _json(capsys, "resolution", str(EXAMPLES / "scores.csv"), "--score", "score",
                     "--label", "is_hack", "--group", "cell")
    assert from_parquet["groups"] == from_csv["groups"]


@pytest.fixture
def nested_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table({
        "a.b": pa.array([1.0, 2.0, 3.0, 4.0]),
        "a": pa.array([{"b": 9.0, "c": "u"}, {"b": 9.0, "c": "u"}, {"b": 9.0, "c": "v"}, {"b": 9.0, "c": "v"}]),
        "probs": pa.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [None, 0.3]]),
        "y": pa.array([True, False, True, False]),
    })
    path = tmp_path / "n.parquet"
    pq.write_table(table, path)
    return path


def test_parquet_paths_match_literal_names_first_then_structs_and_lists(nested_parquet, capsys):
    g = _json(capsys, "resolution", str(nested_parquet), "--score", "a.b")["groups"]["(all)"]
    assert g["n_distinct"] == 4  # the literal column 'a.b', not the constant struct field a -> b
    payload = _json(capsys, "resolution", str(nested_parquet), "--score", "probs.1", "--label", "y", "--group", "a.c")
    assert {k: v["n"] for k, v in payload["groups"].items()} == {"u": 2, "v": 2}
    g = _json(capsys, "resolution", str(nested_parquet), "--score", "probs.0")["groups"]["(all)"]
    assert g["n_nonfinite"] == 1


def test_parquet_errors(nested_parquet, tmp_path, capsys):
    err = _error(capsys, "resolution", str(nested_parquet), "--score", "zz")
    assert "no column 'zz'; the top-level columns are ['a.b', 'a', 'probs', 'y']" in err
    err = _error(capsys, "resolution", str(nested_parquet), "--score", "a.d")
    assert "no field 'a.d' in any of the 4 records. The deepest it reaches is 'a', an object with keys ['b', 'c']." in err
    bad = tmp_path / "bad.parquet"
    bad.write_text("not parquet", encoding="utf-8")
    assert "not a readable Parquet file" in _error(capsys, "resolution", str(bad), "--score", "s")


# --- Inspect logs ----------------------------------------------------------------------

INSPECT = ["--format", "inspect", "--label", "metadata_is_hack", "--group", "metadata_family"]


def test_inspect_without_inspect_ai_says_how_to_get_it(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "inspect_ai", None)
    monkeypatch.setitem(sys.modules, "inspect_ai.analysis", None)
    err = _error(capsys, "resolution", str(EXAMPLES / "hack_probe.eval"), "--score", "s", "--format", "inspect")
    assert 'reading Inspect logs needs inspect_ai, pandas and pyarrow: pip install "cannotfail[inspect]"' in err


def _needs_inspect():
    pytest.importorskip("inspect_ai")
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")


def test_an_inspect_log_is_read_through_samples_df(capsys):
    _needs_inspect()
    payload = _json(capsys, "resolution", str(EXAMPLES / "hack_probe.eval"), "--score", "score_stated_probability",
                    *INSPECT)
    assert {k: (v["verdict"], v["n_pos"], v["n_neg"]) for k, v in payload["groups"].items()} == {
        "a": ("ok", 2, 2), "b": ("fixed", 2, 2)}


def test_inspect_letter_grades_become_the_numbers_its_metrics_average(capsys):
    """called_it is C or I. Family a's probe is right on every sample, so the score is constant."""
    _needs_inspect()
    payload = _json(capsys, "resolution", str(EXAMPLES / "hack_probe.eval"), "--score", "score_called_it", *INSPECT)
    a, b = payload["groups"]["a"], payload["groups"]["b"]
    assert a["verdict"] == "fixed" and a["n_distinct"] == 1
    assert b["n_distinct"] == 2 and b["exact_resolution"] == 1.0


def test_a_directory_of_inspect_logs_groups_by_model_and_task(tmp_path, capsys):
    _needs_inspect()
    shutil.copy(EXAMPLES / "hack_probe.eval", tmp_path / "hack_probe.eval")
    payload = _json(capsys, "resolution", str(tmp_path), "--format", "inspect", "--score", "score_stated_probability",
                    "--group", "task_name", "--group", "model")
    assert list(payload["groups"]) == ["hack_probe | mockllm/model"]
    assert payload["groups"]["hack_probe | mockllm/model"]["n"] == 8


def test_inspect_errors(tmp_path, capsys):
    _needs_inspect()
    err = _error(capsys, "resolution", str(EXAMPLES / "hack_probe.eval"), "--format", "inspect", "--score", "score")
    assert "no column 'score' in samples_df; its columns are [" in err and "'score_called_it'" in err
    junk = tmp_path / "junk.eval"
    junk.write_text("not a log", encoding="utf-8")
    assert "inspect_ai could not read it as an Inspect log" in _error(
        capsys, "resolution", str(junk), "--format", "inspect", "--score", "s")
    assert "no such file or directory" in _error(
        capsys, "resolution", str(tmp_path / "missing.eval"), "--format", "inspect", "--score", "s")


def test_an_unrecognised_inspect_score_is_refused_not_zeroed(tmp_path, capsys):
    """Inspect's value_to_float turns an unknown string into 0; here it must be an error."""
    _needs_inspect()
    err = _error(capsys, "resolution", str(EXAMPLES / "hack_probe.eval"), "--format", "inspect",
                 "--score", "metadata_family")
    assert "score 'a' is not a number" in err


# --- the numpy-only promise ------------------------------------------------------------

def test_reading_csv_and_json_imports_no_optional_dependency(tmp_path):
    path = _write_json(tmp_path, NESTED)
    code = (
        "import sys\n"
        "from cannotfail.cli import main\n"
        f"assert main(['resolution', {str(path)!r}, '--score', 'scores.detector.value', '--json']) == 0\n"
        f"assert main(['resolution', {str(EXAMPLES / 'scores.csv')!r}, '--score', 'score', '--json']) == 0\n"
        "loaded = sorted(m for m in ('pyarrow', 'pandas', 'inspect_ai', 'scipy') if m in sys.modules)\n"
        "assert not loaded, loaded\n"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
