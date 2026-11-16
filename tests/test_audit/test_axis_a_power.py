"""The axis-A power script must apply the pre-registered rules, and keep the rows private.

``scripts/survey_axis_a_power.py`` computes D2 power for the paper's sub-class A claims
by the rules in ``docs/SURVEY_AXIS_A_PREREG.md``. Every row built here is synthetic, and no
row names a real frame member: the benchmark names are invented (``SYNTH-...``), and the
runs that must reconcile use an invented frame. One test reads the real frame's names, to
check that the aggregate output prints none of them, and reports a failure as a count.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "survey_axis_a_power.py"
PREREG = REPO / "docs" / "SURVEY_AXIS_A_PREREG.md"
FRAME = REPO / "audit" / "survey_s1_frame.json"
TOKEN = "zq7tok"  # placed in free-text cells; no error message may print it, in any case


def _load():
    spec = importlib.util.spec_from_file_location("_axis_a_power", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve string annotations through it
    spec.loader.exec_module(module)
    return module


ax = _load()


def _claim(**kw):
    base = dict(
        line=2,
        claim_id="S-1",
        benchmark="SYNTH-BENCH-ALPHA",
        criterion="synthetic criterion",
        source_location="synthetic source",
        sub_class_a=True,
        test_used="mann_whitney",
        n_a=12,
        n_b=12,
        unit="item",
        alpha=0.05,
        one_sided=False,
        clustered="no",
        coder_1="c1",
        coder_2="c2",
        disagreed=(),
        notes="",
        defaults=frozenset(),
        exact_test="yes",
        scores="continuous",
        other_tests=(),
    )
    base.update(kw)
    return ax.Claim(**base)


def _row(i: int = 1, **kw) -> dict:
    row = dict(
        claim_id=f"S-{i}",
        benchmark="SYNTH-BENCH-ALPHA",
        criterion="synthetic criterion text",
        source_location="synthetic source, line 1",
        sub_class_a="yes",
        test_used="mann_whitney",
        n_arm_a="12",
        n_arm_b="12",
        unit_of_replication="item",
        alpha="0.05",
        sidedness="two_sided",
        defaults_applied="",
        exact_test="yes",
        scores="continuous",
        clustered="no",
        other_tests="",
        coder_1="c1",
        coder_2="c2",
        disagreed_fields="",
        notes="",
    )
    row.update(kw)
    return row


def _write(tmp_path: Path, rows, columns=None, name: str = "sheet.csv") -> Path:
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns or ax.COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_lines(tmp_path: Path, lines, name: str = "sheet.csv") -> Path:
    """A sheet written cell by cell, for malformed shapes ``csv.DictWriter`` cannot make."""
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(lines)
    return path


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )


def _main(capsys, *args):
    """The script in-process, so that a test can stand in a synthetic frame and record."""
    code = ax.main([str(a) for a in args])
    out, err = capsys.readouterr()
    return code, out, err


SYNTH_BENCHES = ("SYNTH-BENCH-A", "SYNTH-BENCH-B", "SYNTH-BENCH-C", "SYNTH-BENCH-D")
SALT = "5a" * 32


def _synthetic_frame(tmp_path: Path) -> Path:
    """A frame of four invented benchmarks with the paper's totals, so that a synthetic
    sheet can reconcile without naming a real frame member."""
    frame = {
        "support_totals": {"difference_test": 15},
        "benchmarks_with_difference_test_claims": 4,
        "frame": [{"benchmark": b, "axis2_claims": 5} for b in SYNTH_BENCHES]
        + [{"benchmark": "SYNTH-UNAUDITED", "axis2_claims": None}],
    }
    path = tmp_path / "frame.json"
    path.write_text(json.dumps(frame), encoding="utf-8")
    return path


def _fifteen(kinds) -> list[dict]:
    """Fifteen rows over the four synthetic benchmarks, row i taking ``kinds[i]``."""
    assert len(kinds) == 15
    return [
        _row(i, benchmark=SYNTH_BENCHES[i % 4], **kind) for i, kind in enumerate(kinds)
    ]


def _write_record(tmp_path: Path, rows, paper=None, expected="W=0 j=0", salt=SALT) -> Path:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(ax.RECORD_COLUMNS)
    for row in rows:
        writer.writerow([*(row[k] for k in ax.IDENTIFICATION),
                         (paper or {}).get(row["claim_id"], "no")])
    path = tmp_path / "sealed.txt"
    path.write_text(
        "axis-A sealed record\n"
        f"salt: {salt}\n"
        "paper_underpowered_bar: synthetic bar\n"
        f"expected: {expected}\n"
        "claims:\n" + buf.getvalue(),
        encoding="utf-8",
    )
    return path


_OTHER_VALUE = {
    "sub_class_a": lambda v: "no" if v == "yes" else "yes",
    "test_used": lambda v: "chi_square" if v != "chi_square" else "ks",
    "n_arm_a": lambda v: str(int(v) + 1) if v else v,
    "n_arm_b": lambda v: str(int(v) + 1) if v else v,
    "unit_of_replication": lambda v: v + " group",
    "alpha": lambda v: "0.01" if v != "0.01" else "0.05",
    "sidedness": lambda v: "one_sided" if v == "two_sided" else "two_sided",
    "defaults_applied": lambda v: "conflict" if not v else "",
    "exact_test": lambda v: "no" if v != "no" else "yes",
    "scores": lambda v: "ordinal" if v != "ordinal" else "continuous",
    "clustered": lambda v: "yes" if v != "yes" else "no",
    "other_tests": lambda v: "ks:::0.05:two_sided" if not v else "",
}


def _coder_sheets(tmp_path: Path, rows, flips=None):
    """Each coder's own sheet: the reconciled values, with the second coder differing on
    the fields the row's ``disagreed_fields`` names (or on ``flips``, row by row)."""
    first, second = [], []
    for i, row in enumerate(rows):
        own = dict(row, disagreed_fields="", notes="")
        first.append(own)
        fields = flips[i] if flips is not None else ax._names(row["disagreed_fields"])
        second.append({**own, **{f: _OTHER_VALUE[f](row[f]) for f in fields}})
    return (_write(tmp_path, first, name="coder_1.csv"),
            _write(tmp_path, second, name="coder_2.csv"))


@pytest.fixture
def registered(tmp_path, monkeypatch, capsys):
    """Run the script in-process on synthetic inputs that stand in for the registered ones:
    a synthetic frame pinned in place of the real one, a sealed record whose digest stands
    in for the one section 13 records, and the two coders' own sheets."""
    frame = _synthetic_frame(tmp_path)
    monkeypatch.setattr(ax, "FRAME", frame)
    monkeypatch.setattr(ax, "REGISTERED_FRAME_SHA256", ax.lf_sha256(frame))

    def run(rows, *extra, record=True, coders=True, paper=None, record_rows=None,
            register=True, flips=None, expected="W=0 j=0", own_rows=None):
        """``register`` True stands in the record's own digest for section 13's, False
        stands in none, and a string stands in that digest."""
        args = [_write(tmp_path, rows)]
        if record:
            path = _write_record(tmp_path, record_rows or rows, paper=paper, expected=expected)
            digest = register if isinstance(register, str) else (
                hashlib.sha256(path.read_bytes()).hexdigest() if register else None)
            monkeypatch.setattr(ax, "registered_record_digest", lambda: digest)
            args += ["--sealed-record", path]
        if coders:
            args += ["--coder-sheets", *_coder_sheets(tmp_path, own_rows or rows, flips)]
        return _main(capsys, *args, *extra)

    return run


def _last_line(text: str) -> str:
    return text.strip().splitlines()[-1]


# --- the per-claim rule at its boundaries -----------------------------------------------


def test_a_floor_equal_to_alpha_can_still_reject():
    """The rule is min p > alpha. A p-value equal to alpha rejects."""
    assert ax.decide(0.05, 0.05, 0.0) == ax.UNDERPOWERED
    assert ax.decide(math.nextafter(0.05, 1.0), 0.05, 1.0) == ax.CANNOT_REJECT


def test_one_against_thirty_nine_sits_exactly_on_the_floor():
    """2 / C(40, 1) is exactly 0.05, so this design can reject at 0.05; a floor computed
    inexactly would call it 'cannot reject at any data' and zero its power."""
    result = ax.assess(_claim(n_a=1, n_b=39))
    assert result.min_p == 0.05
    assert result.verdict == ax.UNDERPOWERED
    assert result.power[0.7] > 0.0


def test_power_exactly_at_the_target_is_adequate():
    assert ax.decide(None, 0.05, 0.8) == ax.ADEQUATE
    assert ax.decide(None, 0.05, math.nextafter(0.8, 0.0)) == ax.UNDERPOWERED


def test_an_unreachable_rejection_is_reported_whatever_the_power_figure():
    assert ax.decide(0.1, 0.05, 0.99) == ax.CANNOT_REJECT


def test_the_primary_threshold_falls_between_31_and_32_a_side():
    """Two-sided, alpha 0.05, balanced: 0.796 at 31 a side, 0.810 at 32."""
    under = ax.assess(_claim(n_a=31, n_b=31))
    over = ax.assess(_claim(n_a=32, n_b=32))
    assert under.verdict == ax.UNDERPOWERED
    assert over.verdict == ax.ADEQUATE
    assert under.power[0.7] < 0.8 <= over.power[0.7]
    assert under.power[0.7] == pytest.approx(0.796, abs=5e-4)
    assert over.power[0.7] == pytest.approx(0.810, abs=5e-4)


def test_each_power_is_the_larger_of_the_two_orders_of_the_arms():
    """``difference_power`` is not symmetric in its arms (prereg section 5)."""
    one, other = ax.difference_power(5, 40, 0.7), ax.difference_power(40, 5, 0.7)
    assert one == pytest.approx(0.302, abs=5e-4) and other == pytest.approx(0.259, abs=5e-4)
    assert ax.power_grid(5, 40, 0.05, False)[0.7] == ax.power_grid(40, 5, 0.05, False)[0.7] == one


def test_three_against_three_cannot_reject_at_any_data():
    result = ax.assess(_claim(n_a=3, n_b=3))
    assert result.verdict == ax.CANNOT_REJECT
    assert result.min_p == pytest.approx(0.1)
    assert all(p == 0.0 for p in result.power.values())


def test_three_against_three_on_the_command_line(tmp_path):
    sheet = _write(tmp_path, [_row(n_arm_a="3", n_arm_b="3")])
    proc = _run(sheet)
    assert proc.returncode == 0, proc.stderr
    assert "cannot reject at any data: 1" in proc.stdout
    assert "0.100" in proc.stdout, "the per-row table prints the floor"


def test_three_against_three_at_alpha_ten_percent_is_decided_on_power():
    result = ax.assess(_claim(n_a=3, n_b=3, alpha=0.1))
    assert result.min_p == pytest.approx(0.1)
    assert result.verdict == ax.UNDERPOWERED


def test_three_against_four_cannot_reject_at_five_percent():
    result = ax.assess(_claim(n_a=3, n_b=4))
    assert result.min_p == pytest.approx(2 / 35)
    assert result.verdict == ax.CANNOT_REJECT


def test_one_sided_halves_the_floor_and_doubles_the_level():
    result = ax.assess(_claim(n_a=3, n_b=3, one_sided=True))
    assert result.min_p == pytest.approx(0.05)
    assert result.verdict == ax.UNDERPOWERED, "an exact one-sided p of 0.05 rejects at 0.05"
    assert ax.power_grid(20, 20, 0.05, True)[0.7] == ax.difference_power(20, 20, 0.7, alpha=0.1)


@pytest.mark.parametrize(
    "row",
    [
        dict(n_arm_a="1000", n_arm_b="1000"),
        dict(test_used="t_test", n_arm_a="600", n_arm_b="600"),
        dict(test_used="t_test", n_arm_a="491", n_arm_b="491"),
    ],
)
def test_a_large_design_is_computed(tmp_path, row):
    """The exact floor once overflowed from 515 a side, and the stand-in reaches that
    through the efficiency-adjusted n from 491 a side."""
    proc = _run(_write(tmp_path, [_row(**row)]), "--aggregate-only")
    assert proc.returncode == 0, proc.stderr
    assert "adequately powered: 1" in proc.stdout


# --- rule 3 is a statement about an exact test, at a stated level ---------------------


def test_rule_three_needs_an_exact_test_shown():
    """A library default switches to a normal approximation on tied scores, which can go
    below the exact floor (3 vs 3 tied: 0.047), so the row is underpowered, not 'cannot
    reject'. It stays among the claims that could not reliably fail."""
    for exact in ("no", "unknown"):
        result = ax.assess(_claim(n_a=3, n_b=3, exact_test=exact))
        assert (result.verdict, result.blocked) == (ax.UNDERPOWERED, "exact")
        assert result.min_p == pytest.approx(0.1)


@pytest.mark.parametrize("default", ["alpha", "sidedness"])
def test_rule_three_does_not_rest_on_a_default(default):
    result = ax.assess(_claim(n_a=3, n_b=3, defaults=frozenset({default})))
    assert (result.verdict, result.blocked) == (ax.UNDERPOWERED, "default")


def test_rule_three_needs_every_test_of_the_claim_unable_to_reject():
    rejects = ax.DiffTest("t_test", 3, 3, 0.05, False)  # a t-test on six values can reject
    result = ax.assess(_claim(n_a=3, n_b=3, other_tests=(rejects,)))
    assert (result.verdict, result.blocked) == (ax.UNDERPOWERED, "other_tests")
    cannot = ax.DiffTest("mann_whitney", 3, 4, 0.05, False)
    assert ax.assess(_claim(n_a=3, n_b=3, other_tests=(cannot,))).verdict == ax.CANNOT_REJECT


def test_a_rank_row_rule_three_does_not_reach_is_counted(tmp_path):
    proc = _run(_write(tmp_path, [_row(n_arm_a="3", n_arm_b="3", exact_test="unknown")]))
    assert proc.returncode == 0, proc.stderr
    assert "cannot reject at any data: 0" in proc.stdout
    assert (
        "underpowered rows whose exact floor is above alpha, where rule 3 did not apply: 1 "
        "(the source does not show an exact test 1, alpha or sidedness rests on a default 0, "
        "another test of the claim could reject 0)"
    ) in proc.stdout


# --- tests that are not rank-based --------------------------------------------------------


MEAN_DIFFERENCE = ("t_test", "bootstrap_mean_difference")  # mapped as section 4.2 says
NOT_MAPPED = ("paired_t", "wilcoxon_signed_rank", "chi_square", "fisher_exact", "ks", "other")


def test_the_mapping_is_the_registered_one():
    assert ax.RANK_TESTS == {"mann_whitney"}
    assert ax.MEAN_TESTS == set(MEAN_DIFFERENCE)
    assert ax.UNCOMPUTED_TESTS == set(NOT_MAPPED)


@pytest.mark.parametrize("test", MEAN_DIFFERENCE)
def test_a_mean_difference_test_is_not_computed_where_its_rank_floor_is_above_alpha(test):
    result = ax.assess(_claim(test_used=test, n_a=3, n_b=3))
    assert (result.verdict, result.reason) == (ax.NOT_COMPUTED, "floor")
    assert result.adjusted_verdict is None


@pytest.mark.parametrize("test", MEAN_DIFFERENCE)
def test_a_mean_difference_test_is_decided_on_the_stand_in_power_with_no_floor(test):
    result = ax.assess(_claim(test_used=test, n_a=32, n_b=32))
    assert result.verdict == ax.ADEQUATE
    assert result.min_p is None
    assert result.adjusted_verdict == ax.ADEQUATE


def test_a_one_sided_mean_difference_test_is_computed_at_twice_alpha():
    result = ax.assess(_claim(test_used="t_test", n_a=20, n_b=30, one_sided=True))
    expected = max(ax.difference_power(20, 30, 0.7, alpha=0.1),
                   ax.difference_power(30, 20, 0.7, alpha=0.1))
    assert result.power[0.7] == expected


def test_the_efficiency_adjustment_is_reported_when_it_changes_a_verdict(tmp_path):
    result = ax.assess(_claim(test_used="t_test", n_a=31, n_b=31))
    assert ax.efficiency_adjusted(31) == 33
    assert (result.verdict, result.adjusted_verdict) == (ax.UNDERPOWERED, ax.ADEQUATE)

    sheet = _write(tmp_path, [_row(test_used="t_test", n_arm_a="31", n_arm_b="31")])
    proc = _run(sheet, "--aggregate-only")
    assert proc.returncode == 0, proc.stderr
    assert "verdict changes at the efficiency-adjusted n: 1" in proc.stdout


@pytest.mark.parametrize("n_a, n_b", [(22, 41), (41, 22)])
def test_the_efficiency_adjustment_uses_each_arms_own_n(n_a, n_b):
    """22/41 is underpowered (0.771) and 24/43 adequate (0.804); adjusting both arms from
    either one alone would give 24/24 (0.678) or 43/43."""
    assert (ax.efficiency_adjusted(22), ax.efficiency_adjusted(41)) == (24, 43)
    result = ax.assess(_claim(test_used="t_test", n_a=n_a, n_b=n_b))
    assert (result.verdict, result.adjusted_verdict) == (ax.UNDERPOWERED, ax.ADEQUATE)


@pytest.mark.parametrize("test", NOT_MAPPED)
def test_tests_without_a_registered_mapping_are_not_computed(test):
    result = ax.assess(_claim(test_used=test))
    assert (result.verdict, result.reason) == (ax.NOT_COMPUTED, "test")
    assert result.power is None


def test_an_unstated_n_is_not_computed_rather_estimated():
    result = ax.assess(_claim(n_a=None, n_b=None))
    assert (result.verdict, result.reason) == (ax.NOT_COMPUTED, "n")


def test_a_row_the_coders_reject_from_sub_class_a_is_not_computed():
    """Rule 1 comes first, before any test or n rule."""
    assert ax.assess(_claim(sub_class_a=False, n_a=3, n_b=3)).verdict == ax.NOT_MEMBER
    assert ax.assess(_claim(sub_class_a=False, test_used="ks")).verdict == ax.NOT_MEMBER
    assert ax.assess(_claim(sub_class_a=False, n_a=None, n_b=None)).verdict == ax.NOT_MEMBER


# --- a claim resting on several tests (section 2, analysis (d)) ------------------------


def test_the_coded_test_must_be_the_most_powerful(tmp_path):
    weaker = "mann_whitney:10:10:0.05:two_sided"
    stronger = "mann_whitney:30:30:0.05:two_sided"
    ok = _run(_write(tmp_path, [_row(other_tests=f"{weaker}; ks:::0.05:two_sided")]))
    assert ok.returncode == 0, ok.stderr
    bad = _run(_write(tmp_path, [_row(other_tests=f"{weaker};{stronger}")]))
    assert bad.returncode != 0
    assert "line 2: other_tests: entry 2 is more powerful at AUROC 0.7" in bad.stderr
    coded_uncomputed = _run(_write(tmp_path, [_row(test_used="ks", other_tests=weaker)]))
    assert "entry 1 is more powerful" in coded_uncomputed.stderr


def test_the_most_powerful_test_is_chosen_by_power_before_its_floor():
    """3 vs 1000 has the smaller floor (1.19e-8 against 1.29e-8) but 15 vs 15 the higher
    power (0.459 against 0.228), so 15 vs 15 is the one to code."""
    small_arm = ax.DiffTest("mann_whitney", 3, 1000, 0.05, False)
    balanced = ax.DiffTest("mann_whitney", 15, 15, 0.05, False)
    assert ax.floor_p(3, 1000, False) < ax.floor_p(15, 15, False)
    assert ax.rank_key(balanced) > ax.rank_key(small_arm)


def test_analysis_d_combines_every_test_of_the_claim():
    """Two rank tests at 30 vs 30 (0.782 each) combine to 0.952; a count test with no
    registered computation counts at its alpha, 1 - 0.218 * 0.95 = 0.793."""
    twin = ax.DiffTest("mann_whitney", 30, 30, 0.05, False)
    both = ax.assess(_claim(n_a=30, n_b=30, other_tests=(twin,)))
    assert both.verdict == ax.UNDERPOWERED
    assert both.combined == pytest.approx(1 - (1 - both.power[0.7]) ** 2)
    assert both.combined >= 0.8
    count = ax.DiffTest("chi_square", None, None, 0.05, False)
    with_count = ax.assess(_claim(n_a=30, n_b=30, other_tests=(count,)))
    assert with_count.combined == pytest.approx(1 - (1 - with_count.power[0.7]) * 0.95)
    assert with_count.combined < 0.8


def test_analysis_d_never_counts_a_test_below_its_alpha(monkeypatch):
    """p_i is the larger of the power and alpha, so (d) stays at or above the confined case
    of section 2 even where the approximation gives a power below the level."""
    test = ax.DiffTest("mann_whitney", 30, 30, 0.05, False)
    monkeypatch.setattr(ax, "test_power", lambda t: 0.001)
    assert ax.chance_term(test, True) == 0.05
    monkeypatch.setattr(ax, "test_power", lambda t: 0.3)
    assert ax.chance_term(test, True) == 0.3


def test_analysis_d_counts_a_test_the_exact_floor_stops_at_its_alpha_unless_rule_three_holds():
    small = ax.DiffTest("mann_whitney", 3, 3, 0.05, False)
    held = ax.assess(_claim(n_a=3, n_b=3, other_tests=(small,)))
    assert (held.verdict, held.combined) == (ax.CANNOT_REJECT, 0.0)
    loose = ax.assess(_claim(n_a=3, n_b=3, exact_test="unknown", other_tests=(small,)))
    assert loose.combined == pytest.approx(1 - 0.95 * 0.95)


# --- the overall outcome --------------------------------------------------------------


def _verdicts(*verdicts, clustered=(), adjusted=None, **fields):
    """Assessments with the given verdicts; ``fields`` maps an Assessment field to
    {row index: value}."""
    adjusted = adjusted or {}
    return [
        ax.Assessment(
            _claim(claim_id=f"S-{i}", clustered="yes" if i in clustered else "no"),
            v,
            adjusted_verdict=adjusted.get(i),
            **{name: by_row.get(i) for name, by_row in fields.items()},
        )
        for i, v in enumerate(verdicts)
    ]


def test_outcome_weak_majority():
    rows = _verdicts(ax.CANNOT_REJECT, ax.UNDERPOWERED, ax.ADEQUATE)
    assert ax.outcome(rows) == ax.MOSTLY_WEAK


def test_outcome_adequate_majority():
    rows = _verdicts(ax.ADEQUATE, ax.ADEQUATE, ax.UNDERPOWERED)
    assert ax.outcome(rows) == ax.MOSTLY_ADEQUATE


def test_outcome_split():
    assert ax.outcome(_verdicts(ax.ADEQUATE, ax.UNDERPOWERED)) == ax.SPLIT


def test_outcome_is_inconclusive_when_half_or_fewer_are_computed():
    rows = _verdicts(ax.UNDERPOWERED, ax.UNDERPOWERED, ax.NOT_COMPUTED, ax.NOT_COMPUTED)
    assert ax.outcome(rows) == ax.INCONCLUSIVE
    rows = _verdicts(ax.UNDERPOWERED, ax.UNDERPOWERED, ax.UNDERPOWERED, ax.NOT_COMPUTED)
    assert ax.outcome(rows) == ax.MOSTLY_WEAK
    rows = _verdicts(ax.UNDERPOWERED, ax.ADEQUATE, ax.NOT_COMPUTED, ax.NOT_COMPUTED)
    assert ax.outcome(rows) == ax.INCONCLUSIVE


def test_most_adequately_powered_stands_whatever_the_coverage():
    """The outcome that weakens the paper is not hidden by the coverage rule: 7 computed
    and all adequately powered, 8 not computed, is 'most adequately powered' over the 7."""
    rows = _verdicts(*[ax.ADEQUATE] * 7, *[ax.NOT_COMPUTED] * 8)
    assert ax.outcome(rows) == ax.MOSTLY_ADEQUATE
    assert ax.outcome(_verdicts(ax.ADEQUATE, *[ax.NOT_COMPUTED] * 14)) == ax.MOSTLY_ADEQUATE


def test_rows_outside_sub_class_a_leave_the_denominator():
    rows = _verdicts(ax.UNDERPOWERED, ax.NOT_MEMBER, ax.NOT_MEMBER, ax.NOT_MEMBER)
    assert ax.outcome(rows) == ax.MOSTLY_WEAK
    assert ax.outcome(_verdicts(ax.NOT_MEMBER)) == ax.INCONCLUSIVE


def test_a_weak_majority_that_rests_on_clustered_rows_is_not_robust():
    rows = _verdicts(ax.UNDERPOWERED, ax.UNDERPOWERED, ax.ADEQUATE, clustered={0, 1})
    primary = ax.outcome(rows)
    sens = ax.sensitivity_outcomes(rows)
    assert primary == ax.MOSTLY_WEAK
    assert sens["clustered or unknown rows dropped"] == ax.MOSTLY_ADEQUATE
    assert ax.reported_outcome(primary, sens) == ax.NOT_ROBUST


def test_a_weak_majority_that_rests_on_the_t_test_stand_in_is_not_robust():
    rows = _verdicts(
        ax.UNDERPOWERED, ax.UNDERPOWERED, ax.ADEQUATE, adjusted={0: ax.ADEQUATE, 1: ax.ADEQUATE}
    )
    sens = ax.sensitivity_outcomes(rows)
    assert sens["t-test family at the efficiency-adjusted n"] == ax.MOSTLY_ADEQUATE
    assert ax.reported_outcome(ax.outcome(rows), sens) == ax.NOT_ROBUST


@pytest.mark.parametrize(
    "name, field",
    [
        ("claims resting on several tests at their combined chance to fail", "combined"),
        ("binary or unknown scores at the two-proportion power", "binary_power"),
        ("a defaulted sidedness taken as one-sided", "one_sided_power"),
    ],
)
def test_analyses_d_e_and_f_can_withdraw_a_weak_majority(name, field):
    rows = _verdicts(ax.UNDERPOWERED, ax.UNDERPOWERED, ax.ADEQUATE, **{field: {0: 0.85, 1: 0.9}})
    sens = ax.sensitivity_outcomes(rows)
    assert sens[name] == ax.MOSTLY_ADEQUATE
    others = {k: v for k, v in sens.items() if k != name}
    assert set(others.values()) == {ax.MOSTLY_WEAK}
    assert ax.reported_outcome(ax.outcome(rows), sens) == ax.NOT_ROBUST


def test_sensitivity_analyses_move_rows_only_toward_adequately_powered():
    """A figure below 0.8 keeps the (a) verdict, and a row not computed in (a) stays not
    computed whatever the analysis finds for it."""
    rows = _verdicts(ax.ADEQUATE, ax.NOT_COMPUTED, ax.UNDERPOWERED,
                     combined={0: 0.1, 1: 0.99}, binary_power={0: 0.1, 1: 0.99},
                     one_sided_power={0: 0.1, 1: 0.99})
    for name, value in ax.sensitivity_outcomes(rows).items():
        assert value == ax.outcome(rows) == ax.SPLIT, name


def test_analyses_e_and_f_on_real_designs():
    binary = ax.assess(_claim(n_a=24, n_b=24, scores="binary"))
    assert binary.verdict == ax.UNDERPOWERED
    assert binary.power[0.7] == pytest.approx(0.678, abs=5e-4)
    assert binary.binary_power == pytest.approx(0.812, abs=5e-4)
    assert ax.assess(_claim(n_a=24, n_b=24, scores="unknown")).binary_power == binary.binary_power
    assert ax.assess(_claim(n_a=24, n_b=24, scores="ordinal")).binary_power is None
    # Unequal arms: the pooled proportion is (0.3 * 10 + 0.7 * 40) / 50, not one half.
    assert ax.binary_power(10, 40, 0.05, False) == pytest.approx(0.6528, abs=5e-5)
    assert ax.binary_power(40, 10, 0.05, False) == pytest.approx(0.6528, abs=5e-5)
    defaulted = ax.assess(_claim(n_a=25, n_b=25, defaults=frozenset({"sidedness"})))
    assert defaulted.verdict == ax.UNDERPOWERED
    assert defaulted.one_sided_power >= 0.8


def test_a_floor_row_stays_not_computed_in_analysis_c():
    """3 vs 4 is not computed for a t-test at its stated n, though 4 vs 5 would clear the
    floor; it is not brought in at the adjusted n."""
    row = ax.assess(_claim(test_used="t_test", n_a=3, n_b=4))
    assert (row.verdict, row.adjusted_verdict) == (ax.NOT_COMPUTED, None)
    rows = [row, ax.assess(_claim(claim_id="S-2", n_a=40, n_b=40))]
    sens = ax.sensitivity_outcomes(rows)
    assert sens["t-test family at the efficiency-adjusted n"] == ax.outcome(rows)


def test_a_sensitivity_analysis_cannot_withdraw_the_outcome_that_weakens_the_paper():
    rows = _verdicts(ax.ADEQUATE, ax.ADEQUATE, ax.UNDERPOWERED, clustered={0, 1})
    primary = ax.outcome(rows)
    sens = ax.sensitivity_outcomes(rows)
    assert primary == ax.MOSTLY_ADEQUATE
    assert sens["clustered or unknown rows dropped"] != ax.MOSTLY_ADEQUATE
    assert ax.reported_outcome(primary, sens) == ax.MOSTLY_ADEQUATE


@pytest.mark.parametrize("primary", [ax.INCONCLUSIVE, ax.SPLIT, ax.MOSTLY_ADEQUATE])
def test_every_other_outcome_of_a_stands_on_a(primary):
    sens = {name: ax.MOSTLY_WEAK for _, name in ax.SENSITIVITIES}
    assert ax.reported_outcome(primary, sens) == primary


# --- what the last line may report ------------------------------------------------------


def test_most_adequately_powered_at_low_coverage_is_reported(registered):
    """7 rows at 100 vs 100, and 8 KS rows, which are not computed."""
    kinds = [dict(n_arm_a="100", n_arm_b="100")] * 7 + [dict(test_used="ks")] * 8
    code, out, err = registered(_fifteen(kinds), "--aggregate-only")
    assert code == 0, err
    assert "reconciles with the paper's counts: yes" in out
    assert "adequately powered: 7" in out
    assert "not computed: 8" in out
    assert "outcome, (a) all rows: most adequately powered" in out
    assert _last_line(out) == "outcome as it may be reported: most adequately powered"


def test_a_full_confirmed_sheet_reports_its_outcome(tmp_path, registered):
    frame = ax.load_frame(_synthetic_frame(tmp_path))
    assert ax.membership_correction(15, 4, frame) is None
    assert ax.membership_correction(14, 4, frame).startswith(
        "membership correction: 14 of 15 confirmed, over 4 of 4 benchmarks"
    )
    code, out, err = registered(_fifteen([dict(n_arm_a="5", n_arm_b="5")] * 15),
                                "--aggregate-only")
    assert code == 0, err
    assert "membership correction" not in out
    assert _last_line(out) == "outcome as it may be reported: most could not reliably fail"


def test_a_weak_majority_resting_on_unknown_clustering_is_not_robust(registered):
    """Analysis (b) drops `unknown` rows as it drops `yes` rows."""
    kinds = ([dict(n_arm_a="10", n_arm_b="10", clustered="unknown")] * 8
             + [dict(n_arm_a="40", n_arm_b="40")] * 7)
    code, out, err = registered(_fifteen(kinds), "--aggregate-only")
    assert code == 0, err
    assert "outcome, (a) all rows: most could not reliably fail" in out
    assert "outcome, (b) clustered or unknown rows dropped: most adequately powered" in out
    assert _last_line(out) == "outcome as it may be reported: not robust"


def test_a_sheet_that_does_not_reconcile_reports_no_outcome(tmp_path):
    """Every row underpowered, so the counts alone would read as strengthening, but the
    sheet is not the paper's sub-class A."""
    proc = _run(_write(tmp_path, [_row(i) for i in range(3)]), "--aggregate-only")
    assert proc.returncode == 0, proc.stderr
    assert "reconciles with the paper's counts: NO" in proc.stdout
    assert "outcome, (a) all rows: most could not reliably fail" in proc.stdout
    assert _last_line(proc.stdout).startswith(
        "outcome as it may be reported: none: the sheet is not the paper's sub-class A "
        "(prereg 7.1)"
    )
    assert "membership correction" not in proc.stdout


def test_a_membership_correction_comes_first_and_is_reported(registered):
    kinds = [dict(sub_class_a="no")] + [dict(n_arm_a="5", n_arm_b="5")] * 14
    code, out, err = registered(_fifteen(kinds), "--aggregate-only")
    assert code == 0, err
    lines = out.splitlines()
    correction = (
        "membership correction: 14 of 15 confirmed, over 4 of 4 benchmarks; the sub-class A "
        "count is wrong (prereg 7.1; weakens the paper)"
    )
    assert ("reconciles with the paper's counts: yes (15 claims, Appendix F; over 4 "
            "benchmarks, section 5)") in lines
    assert correction in lines
    first_power_figure = lines.index("  underpowered (power below 0.8 at AUROC 0.7): 14")
    assert lines.index(correction) < first_power_figure < lines.index(
        "outcome, (a) all rows: most could not reliably fail"
    )
    assert _last_line(out) == (
        "outcome as it may be reported: membership correction, 14 of 15 confirmed, over 4 of 4 "
        "benchmarks (prereg 7.1; weakens the paper); then, over the 14 confirmed: most could "
        "not reliably fail"
    )


def test_a_membership_correction_gives_the_benchmark_count(registered):
    """Every row of one benchmark re-coded `no`: the correction says 3 of 4 benchmarks, and
    the benchmarks-with-a-weak-claim line counts over those 3."""
    kinds = [dict(sub_class_a="no") if i % 4 == 3 else dict(n_arm_a="5", n_arm_b="5")
             for i in range(15)]
    code, out, err = registered(_fifteen(kinds), "--aggregate-only")
    assert code == 0, err
    lines = out.splitlines()
    correction = next(line for line in lines if line.startswith("membership correction"))
    assert "12 of 15 confirmed, over 3 of 4 benchmarks" in correction
    first_power_figure = lines.index("  underpowered (power below 0.8 at AUROC 0.7): 12")
    assert lines.index(correction) < first_power_figure
    assert "over 3 of 4 benchmarks" in _last_line(out)
    assert "benchmarks with at least one claim that could not reliably fail: 3 of 3" in lines


def test_a_run_on_another_power_py_reports_no_outcome(monkeypatch, registered):
    monkeypatch.setattr(ax, "power_source_digest", lambda: "0" * 64)
    code, out, err = registered(_fifteen([dict(n_arm_a="5", n_arm_b="5")] * 15),
                                "--aggregate-only")
    assert code == 0, err
    assert "(DIFFERS FROM the registered source)" in out
    assert _last_line(out) == (
        "outcome as it may be reported: none: power.py is not the registered source (prereg 5)"
    )


def test_a_run_on_another_frame_reports_no_outcome(monkeypatch, registered):
    monkeypatch.setattr(ax, "REGISTERED_FRAME_SHA256", "0" * 64)
    code, out, err = registered(_fifteen([dict(n_arm_a="5", n_arm_b="5")] * 15),
                                "--aggregate-only")
    assert code == 0, err
    assert "(DIFFERS FROM the registered transcription)" in out
    assert _last_line(out) == (
        "outcome as it may be reported: none: the frame is not the registered transcription "
        "(prereg 5)"
    )


def test_the_frame_cannot_be_swapped_from_the_command_line():
    proc = _run("--help")
    assert proc.returncode == 0, proc.stderr
    assert "--frame" not in proc.stdout


# --- the sealed record and the coders' own sheets ----------------------------------------


FIVE = _fifteen([dict(n_arm_a="5", n_arm_b="5")] * 15)


def test_an_outcome_needs_the_sealed_record_and_the_coders_sheets(registered):
    code, out, _ = registered(FIVE, "--aggregate-only", record=False, coders=False)
    assert code == 0
    assert _last_line(out) == (
        "outcome as it may be reported: none: no sealed record was given (prereg 13); the "
        "coders' own sheets were not given (prereg 9)"
    )


def test_a_sealed_record_section_13_does_not_record_reports_no_outcome(registered):
    code, out, _ = registered(FIVE, "--aggregate-only", register=False)
    assert code == 0
    assert "(section 13 records no digest yet)" in out
    assert "section 13 records no sealed-record digest yet (prereg 13)" in _last_line(out)


def test_a_sealed_record_other_than_section_13s_reports_no_outcome(registered):
    code, out, _ = registered(FIVE, "--aggregate-only", register="0" * 64)
    assert code == 0
    assert "(DIFFERS FROM the digest section 13 records)" in out
    assert "the sealed record is not the one section 13 records (prereg 13)" in _last_line(out)


def test_rows_that_are_not_the_sealed_records_report_no_outcome(registered):
    swapped = [dict(r) for r in FIVE]
    swapped[3]["criterion"] = "another synthetic criterion"
    code, out, _ = registered(FIVE, "--aggregate-only", record_rows=swapped)
    assert code == 0
    assert "rows whose identification columns are not in the sealed record: 1 of 15" in out
    assert "the sheet's rows are not the sealed record's (prereg 13)" in _last_line(out)


def test_an_invalid_sealed_record_is_refused_without_echo(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ax, "FRAME", _synthetic_frame(tmp_path))
    rows = [dict(r, criterion=f"{TOKEN} criterion") for r in FIVE]
    record = _write_record(tmp_path, rows, salt=TOKEN)
    code, out, err = _main(capsys, _write(tmp_path, rows), "--sealed-record", record)
    assert code == 1
    assert "sealed record: salt must be 64 lowercase hexadecimal digits" in err
    assert TOKEN not in (out + err).lower()


def test_the_sealed_record_splits_the_counts_by_the_papers_20(registered):
    kinds = [dict(n_arm_a="3", n_arm_b="3")] + [dict(n_arm_a="40", n_arm_b="40")] * 2 + [
        dict(n_arm_a="5", n_arm_b="5")] * 12
    rows = _fifteen(kinds)
    paper = {"S-0": "not_underpowered", "S-1": "underpowered", "S-2": "not_underpowered",
             "S-3": "unknown"}
    code, out, err = registered(rows, "--aggregate-only", paper=paper, expected="W=12 j=1")
    assert code == 0, err
    lines = out.splitlines()
    assert ("confirmed claims among the paper's 20 designs (sealed record): 3; of those, the "
            "paper found underpowered: 1") in lines
    assert ("  among the paper's 20: cannot reject 1, underpowered 0, adequately powered 2, "
            "not computed 0") in lines
    assert ("  possibly among them (paper_20 unknown): cannot reject 0, underpowered 1, "
            "adequately powered 0, not computed 0") in lines
    assert ("  not among them (paper_20 no): cannot reject 0, underpowered 11, adequately "
            "powered 0, not computed 0") in lines
    assert ("  cannot reject at any data among the paper's 20, each against its '0 met the "
            "CANNOT_FAIL bar' (prereg 7.2): 1") in lines
    assert "  verdicts among the paper's 20 that differ from the paper's: 2 of 3 compared" in lines
    assert ("the author's expected counts (sealed record): W 12, j 1; found: W 13, j 1") in lines


def test_disagreed_fields_must_match_the_coders_own_sheets(registered):
    rows = [dict(r) for r in FIVE]
    rows[0]["disagreed_fields"] = "sidedness"
    code, out, _ = registered(rows, "--aggregate-only")
    assert "value both own sheets hold) 0" in out
    assert "none" not in _last_line(out)

    code, out, _ = registered(rows, "--aggregate-only", flips=[()] * 15)
    assert code == 0
    assert "value both own sheets hold) 1" in out
    assert "this sheet does not match the coders' own sheets (prereg 9)" in _last_line(out)


def test_the_second_coders_own_sheet_is_reported_beside(registered):
    rows = [dict(r) for r in FIVE]
    for row in rows[:3]:
        row["disagreed_fields"] = "sub_class_a"
    code, out, err = registered(rows, "--aggregate-only")
    assert code == 0, err
    assert ("on the second coder's own sheet: confirmed 12; outcome, all rows: most could not "
            "reliably fail") in out.splitlines()


def test_the_salted_digests_are_the_ones_section_13_describes(tmp_path, registered):
    code, out, err = registered(FIVE, "--aggregate-only")
    assert code == 0, err

    def salted(name):
        data = (tmp_path / name).read_bytes()
        return hashlib.sha256(SALT.encode() + b"\n" + data).hexdigest()

    lines = out.splitlines()
    assert f"this sheet, salted sha256 {salted('sheet.csv')}" in lines
    assert f"first coder's sheet, salted sha256 {salted('coder_1.csv')}" in lines
    assert f"second coder's sheet, salted sha256 {salted('coder_2.csv')}" in lines
    record = hashlib.sha256((tmp_path / "sealed.txt").read_bytes()).hexdigest()
    assert f"sealed record sha256 {record} (matches the digest section 13 records)" in lines


def test_the_record_digest_is_read_from_its_line_in_section_13():
    digest = "ab" * 32
    assert ax.record_digest_in(f"- **Sealed record: SHA-256 `{digest}`.** Made") == digest
    assert ax.record_digest_in("- **Sealed record: not yet made.**") is None
    assert ax.record_digest_in("**Sealed record: SHA-256 `<digest>`**") is None
    # Section 13 either still says the record is not made, or holds a line the script
    # reads. A mistyped line fails here, before the deposit, not at the registered run.
    current = ax.registered_record_digest()
    unmade = "**Sealed record: not yet made.**" in PREREG.read_text(encoding="utf-8")
    assert (current is None) == unmade


def test_a_run_after_author_responses_is_labelled_as_such(registered):
    code, out, err = registered(FIVE, "--aggregate-only", "--after-responses")
    assert code == 0, err
    assert _last_line(out) == (
        "outcome after author responses, reported beside the registered outcome; it leads "
        "only if it weakens the paper (prereg 7.4): most could not reliably fail"
    )


# --- an invalid sheet stops with its reason ---------------------------------------------


def _token_row(i: int, **kw) -> dict:
    """A row whose every free-text cell carries the token."""
    cells = dict(
        claim_id=f"{TOKEN}-{i}",
        benchmark=f"SYNTH-{TOKEN}",
        criterion=f"{TOKEN} criterion",
        source_location=f"{TOKEN} source",
        unit_of_replication=f"{TOKEN} item",
        coder_1=f"{TOKEN}-c1",
        coder_2=f"{TOKEN}-c2",
        notes=f"{TOKEN} note",
    )
    return _row(i, **{**cells, **kw})


def _no_token(proc) -> bool:
    return TOKEN not in (proc.stdout + proc.stderr).lower()


@pytest.mark.parametrize(
    "change, column, reason",
    [
        ({"n_arm_a": "0"}, "n_arm_a", "whole number of at least 1"),
        ({"n_arm_b": "12.5"}, "n_arm_b", "whole number of at least 1"),
        ({"n_arm_b": "1²"}, "n_arm_b", "whole number of at least 1"),
        ({"n_arm_b": "١٢"}, "n_arm_b", "in ASCII digits"),
        ({"n_arm_b": ""}, "n_arm_a/n_arm_b", "give both"),
        ({"alpha": "0.7"}, "alpha", "from 1e-10 to 0.5"),
        ({"alpha": "1e-11"}, "alpha", "from 1e-10 to 0.5"),
        ({"alpha": f"five percent {TOKEN}"}, "alpha", "from 1e-10 to 0.5"),
        ({"test_used": f"Welch t-test, Table 3 {TOKEN.upper()}"}, "test_used", "must be one of"),
        ({"sidedness": "both"}, "sidedness", "two_sided"),
        ({"clustered": "maybe"}, "clustered", "must be one of no, unknown, yes"),
        ({"sub_class_a": "probably"}, "sub_class_a", "must be one of no, yes"),
        ({"exact_test": "probably"}, "exact_test", "must be one of no, unknown, yes"),
        ({"scores": "integers"}, "scores", "must be one of binary"),
        ({"defaults_applied": f"{TOKEN}"}, "defaults_applied", "1 entry(ies) not among"),
        ({"defaults_applied": "alpha;alpha"}, "defaults_applied", "names a default twice"),
        ({"other_tests": f"mann_whitney:10:{TOKEN}"}, "other_tests", "entry 1 must be"),
        ({"other_tests": "mann_whitney:10::0.05:two_sided"}, "other_tests", "entry 1 must be"),
        ({"coder_2": f"{TOKEN.upper()}-C1"}, "coder_1/coder_2", "two different coders"),
        ({"coder_2": ""}, "coder_1/coder_2", "two coders are required"),
        ({"disagreed_fields": f"benchmark;{TOKEN} note"}, "disagreed_fields",
         "2 entry(ies) not among the coded fields"),
        ({"disagreed_fields": "alpha;alpha"}, "disagreed_fields", "names a field twice"),
        ({"benchmark": ""}, "benchmark", "required"),
        ({"source_location": ""}, "source_location", "required"),
        ({"unit_of_replication": " "}, "unit_of_replication", "required"),
    ],
)
def test_an_invalid_row_exits_non_zero_with_its_reason(tmp_path, change, column, reason):
    sheet = _write(tmp_path, [_token_row(1), _token_row(2, **change)])
    proc = _run(sheet)
    assert proc.returncode != 0
    assert f"line 3: {column}: " in proc.stderr
    assert reason in proc.stderr
    assert "Traceback" not in proc.stderr
    assert proc.stdout == "", "nothing may be computed from an invalid sheet"
    assert _no_token(proc), "errors must not echo any cell"


def test_a_repeated_claim_id_is_refused(tmp_path):
    proc = _run(_write(tmp_path, [_row(1), _row(1)]))
    assert proc.returncode != 0
    assert "line 3: claim_id: duplicates the claim_id on line 2" in proc.stderr


@pytest.mark.parametrize(
    "columns, reason",
    [
        ([c for c in ax.COLUMNS if c != "clustered"], "missing column(s) clustered"),
        ([*ax.COLUMNS, f"{TOKEN}_size"], "header: 1 unexpected column name(s)"),
        ([*ax.COLUMNS, "notes"], "repeated column(s) notes"),
    ],
)
def test_a_wrong_header_is_refused(tmp_path, columns, reason):
    proc = _run(_write(tmp_path, [_token_row(1)], columns=columns))
    assert proc.returncode != 0
    assert reason in proc.stderr
    assert _no_token(proc)


def test_a_sheet_without_its_header_line_prints_none_of_its_first_row(tmp_path):
    """The first data row is read as the header; none of it may reach the output."""
    rows = [[_token_row(i)[c] for c in ax.COLUMNS] for i in (1, 2)]
    proc = _run(_write_lines(tmp_path, rows))
    assert proc.returncode != 0
    assert "line 1 is not the header" in proc.stderr
    assert _no_token(proc)


@pytest.mark.parametrize("drop", ["n_arm_b", "clustered", "disagreed_fields"])
def test_a_row_with_a_cell_missing_is_refused_without_echo(tmp_path, drop):
    """A missing cell would shift every later cell one column left, into columns whose
    checks report on them."""
    row = _token_row(1)
    cells = [row[c] for c in ax.COLUMNS if c != drop]
    proc = _run(_write_lines(tmp_path, [list(ax.COLUMNS), cells]))
    assert proc.returncode != 0
    assert "line 2: fewer cells than the header has columns" in proc.stderr
    assert _no_token(proc)


def test_a_row_with_a_cell_too_many_is_refused_without_echo(tmp_path):
    row = _token_row(1)
    cells = [row[c] for c in ax.COLUMNS]
    cells.insert(3, f"{TOKEN} stray")  # an unquoted comma in the criterion, say
    proc = _run(_write_lines(tmp_path, [list(ax.COLUMNS), cells]))
    assert proc.returncode != 0
    assert "line 2: more cells than the header has columns" in proc.stderr
    assert _no_token(proc)


def test_a_header_only_sheet_is_refused(tmp_path):
    proc = _run(_write(tmp_path, []))
    assert proc.returncode != 0
    assert "the sheet has a header and no claim rows" in proc.stderr


def test_the_smallest_registered_alpha_is_computed():
    result = ax.assess(_claim(alpha=1e-10, n_a=100, n_b=100))
    assert result.verdict in ax.COMPUTED


def test_a_member_the_paper_did_not_audit_is_out_of_scope(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ax, "FRAME", _synthetic_frame(tmp_path))
    code, out, err = _main(capsys, _write(tmp_path, [_row(1, benchmark="SYNTH-UNAUDITED")]))
    assert code != 0
    assert "line 2: benchmark: a member the paper did not audit" in err
    assert "SYNTH-UNAUDITED" not in err


# --- privacy ----------------------------------------------------------------------------


def _private_rows():
    kinds = [
        dict(n_arm_a="3", n_arm_b="3"),
        dict(n_arm_a="40", n_arm_b="40"),
        dict(test_used="t_test", n_arm_a="31", n_arm_b="31", clustered="unknown"),
        dict(test_used="chi_square"),
        dict(sub_class_a="no", disagreed_fields="sub_class_a;test_used"),
    ]
    return [
        _row(
            i,
            claim_id=f"CLAIM-ID-{i}",
            benchmark=SYNTH_BENCHES[i % 4] if i < 4 else "SYNTH-BENCH-EXTRA",
            criterion=f"criterion text {i}",
            source_location=f"source location {i}",
            coder_1=f"first-coder-{i}",
            coder_2=f"second-coder-{i}",
            notes=f"private note {i}",
            **kind,
        )
        for i, kind in enumerate(kinds)
    ]


def test_aggregate_only_prints_no_row_content(tmp_path, monkeypatch, capsys):
    rows = _private_rows()
    sheet = _write(tmp_path, rows)
    monkeypatch.setattr(ax, "FRAME", _synthetic_frame(tmp_path))

    code, full, _ = _main(capsys, sheet)
    assert code == 0
    for row in rows:
        assert row["benchmark"] in full, "control: the per-row table does name benchmarks"
    assert "END OF PRIVATE TABLE" in full

    code, out, err = _main(capsys, sheet, "--aggregate-only")
    assert code == 0, err
    for row in rows:
        for column in ("claim_id", "benchmark", "criterion", "source_location", "coder_1",
                       "coder_2", "notes"):
            assert row[column] not in out + err, f"--aggregate-only printed a {column}"
    assert "confirmed sub-class A: 4" in out
    assert "PRIVATE:" not in out
    assert "Private until every group's outreach window has closed" in out


def test_aggregate_only_prints_no_real_frame_members_name(tmp_path):
    """Against the real frame, which the synthetic rows do not name. A failure prints a
    count, not the names."""
    proc = _run(_write(tmp_path, _private_rows()), "--aggregate-only")
    assert proc.returncode == 0
    members = json.loads(FRAME.read_text(encoding="utf-8"))["frame"]
    hits = sum(1 for m in members if m["benchmark"] in proc.stdout + proc.stderr)
    assert hits == 0, f"--aggregate-only printed {hits} frame member name(s)"


def test_the_aggregate_lines_on_a_mixed_sheet(tmp_path):
    claims = [
        _claim(claim_id="c0", n_a=3, n_b=3),
        _claim(claim_id="c1", n_a=40, n_b=40, clustered="yes"),
        _claim(claim_id="c2", test_used="t_test", n_a=31, n_b=31),
        _claim(claim_id="c3", test_used="chi_square"),
        _claim(claim_id="c4", n_a=None, n_b=None),
        _claim(claim_id="c5", test_used="t_test", n_a=3, n_b=3),
        _claim(claim_id="c6", sub_class_a=False, disagreed=("sub_class_a", "test_used")),
        _claim(claim_id="c7", n_a=3, n_b=3, exact_test="unknown", defaults=frozenset({"alpha"})),
        _claim(claim_id="c8", n_a=24, n_b=24, scores="binary"),
    ]
    frame = ax.load_frame(_synthetic_frame(tmp_path))
    lines = ax.aggregate_lines([ax.assess(c) for c in claims], ax.Context(frame=frame))
    expected = [
        "rows with a coder disagreement: 1 of 9 (per field: sub_class_a 1, test_used 1)",
        "rows resting on a rule for a silent or conflicting source: alpha 1, sidedness 0, "
        "test 0, conflict 0",
        "    test neither rank-based nor a two-sample mean difference: 1",
        "    n per arm not stated in the source: 1",
        "    mean-difference test whose exact rank floor is above alpha: 1",
        "underpowered rows whose exact floor is above alpha, where rule 3 did not apply: 1 "
        "(the source does not show an exact test 1, alpha or sidedness rests on a default 0, "
        "another test of the claim could reject 0)",
        "computed rows with power below 0.8: 5 at AUROC 0.6, 4 at AUROC 0.7, 2 at AUROC 0.8, of 5",
        "computed rows by test: rank-based 4, t-test stand-in 1",
        "computed rows by scores: continuous 4, ordinal 0, binary 1, unknown 0",
        "computed rows whose units are clustered or unknown: 1 of 5",
        "adequately powered rows: clustered or unknown 1, t-test stand-in 0, of 1",
        "t-test stand-in rows whose verdict changes at the efficiency-adjusted n: 1",
        "benchmarks with at least one claim that could not reliably fail: 1 of 1",
    ]
    for line in expected:
        assert line in lines, line


def test_the_help_keeps_the_counts_with_the_coders_until_the_windows_close():
    """Aggregate counts included, nothing goes beyond the two coders before every window
    has closed."""
    proc = _run("--help")
    assert proc.returncode == 0, proc.stderr
    text = " ".join(proc.stdout.split())
    assert "safe to publish" not in text
    assert "shared privately" not in text
    assert "stay with the two coders until every outreach window has closed" in text


def test_one_benchmarks_rows_print_alone(tmp_path):
    rows = _fifteen([dict(n_arm_a="5", n_arm_b="5")] * 15)
    proc = _run(_write(tmp_path, rows), "--benchmark", "synth-bench-b")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.count("SYNTH-BENCH-B") == 4
    for other in ("SYNTH-BENCH-A", "SYNTH-BENCH-C", "SYNTH-BENCH-D"):
        assert other not in proc.stdout
    assert "outcome" not in proc.stdout and "confirmed" not in proc.stdout
    assert proc.stdout.strip().endswith("END OF PRIVATE ROWS")
    missing = _run(_write(tmp_path, rows), "--benchmark", f"SYNTH-{TOKEN}")
    assert missing.returncode != 0 and _no_token(missing)


def test_no_input_may_sit_in_the_checkout_outside_the_ignored_directory(tmp_path):
    assert ax.tracked_path_refusal(REPO / "audit" / "axis_a_sheet.csv")
    assert ax.tracked_path_refusal(REPO / "axis_a_sheet.csv")
    assert ax.tracked_path_refusal(ax.TEMPLATE), "the template is refused in place"
    assert ax.tracked_path_refusal(REPO / "audit" / "private" / "axis_a_sheet.csv") is None
    assert ax.tracked_path_refusal(tmp_path / "axis_a_sheet.csv") is None

    proc = _run(REPO / "audit" / "axis_a_sheet_filled.csv")
    assert proc.returncode != 0
    assert "refusing to read an input" in proc.stderr
    sheet = _write(tmp_path, [_row(1)])
    for option in (["--sealed-record", REPO / "audit" / "axis_a_sealed.txt"],
                   ["--coder-sheets", REPO / "audit" / "c1.csv", sheet]):
        proc = _run(sheet, *option)
        assert proc.returncode != 0
        assert "refusing to read an input" in proc.stderr


def test_the_ignored_directory_is_ignored():
    lines = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/audit/private/" in lines
    assert ax.PRIVATE_DIR == REPO / "audit" / "private"
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("git, or this checkout's git directory, is not available")
    proc = subprocess.run(
        ["git", "check-ignore", "-q", "audit/private/axis_a_sheet.csv"], cwd=REPO, timeout=60
    )
    assert proc.returncode == 0, "git does not ignore audit/private/"


# --- the template and the pre-registration agree with the script ---------------------


def test_the_template_is_one_synthetic_row_that_runs_and_does_not_reconcile(tmp_path):
    with ax.TEMPLATE.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # Plain values first, so that a template filled in place cannot have its rows printed
    # by a failing assertion.
    n_rows = len(rows)
    assert n_rows == 1, f"the template holds {n_rows} rows; it must hold one"
    header_ok = tuple(rows[0].keys()) == ax.COLUMNS
    assert header_ok, "the template's header is not COLUMNS"
    is_synthetic = "SYNTHETIC" in rows[0]["benchmark"]
    assert is_synthetic, "the template's row is not the synthetic one"

    copy = tmp_path / "template_copy.csv"
    shutil.copy(ax.TEMPLATE, copy)
    proc = _run(copy)
    assert proc.returncode == 0, proc.stderr
    assert "reconciles with the paper's counts: NO" in proc.stdout
    assert _last_line(proc.stdout).startswith(
        "outcome as it may be reported: none: the sheet is not the paper's sub-class A"
    )


def test_the_reconciliation_reads_its_totals_from_the_frame():
    frame = ax.load_frame(FRAME)
    published = json.loads(FRAME.read_text(encoding="utf-8"))
    assert frame.expected_rows == published["support_totals"]["difference_test"]
    assert frame.expected_benchmarks == published["benchmarks_with_difference_test_claims"]


def _claims_over(counts: dict[str, int]):
    return [
        _claim(claim_id=f"{bench}-{k}", benchmark=bench)
        for bench, n in counts.items() for k in range(n)
    ]


@pytest.mark.parametrize(
    "counts, message",
    [
        ({"SYNTH-BENCH-A": 4, "SYNTH-BENCH-B": 4, "SYNTH-BENCH-C": 3, "SYNTH-BENCH-D": 3},
         "14 row(s); Appendix F counts 15 claims"),
        ({"SYNTH-BENCH-A": 5, "SYNTH-BENCH-B": 5, "SYNTH-BENCH-C": 5},
         "3 benchmark(s); the paper's section 5 counts 4"),
        ({"SYNTH-BENCH-A": 4, "SYNTH-BENCH-B": 4, "SYNTH-BENCH-C": 4, "SYNTH-ELSEWHERE": 3},
         "1 benchmark name(s) not in the frame"),
        ({"SYNTH-BENCH-A": 6, "SYNTH-BENCH-B": 3, "SYNTH-BENCH-C": 3, "SYNTH-BENCH-D": 3},
         "1 benchmark(s) with more rows than Appendix F enumerates for them"),
    ],
)
def test_each_reconciliation_check_fails_alone(tmp_path, counts, message):
    frame = ax.load_frame(_synthetic_frame(tmp_path))
    assert ax.reconcile(_claims_over(counts), frame) == [message]


def test_the_registered_files_are_the_ones_in_this_checkout():
    """A change to power.py or the frame fails here until it is re-pinned, which after
    registration is an amendment (prereg sections 5 and 13)."""
    assert ax.power_source_digest() == ax.REGISTERED_POWER_SHA256
    assert ax.lf_sha256(ax.FRAME) == ax.REGISTERED_FRAME_SHA256


def test_the_preregistration_pins_the_script_and_the_template():
    """Section 5 names the digest of every file the registration rests on, so a change to
    the script's rules or to the template fails here until section 5 says so."""
    text = PREREG.read_text(encoding="utf-8")
    for digest in (ax.REGISTERED_POWER_SHA256, ax.REGISTERED_FRAME_SHA256,
                   ax.lf_sha256(SCRIPT), ax.lf_sha256(ax.TEMPLATE)):
        assert digest in text, digest


def test_the_preregistration_names_every_column_test_and_rule():
    text = PREREG.read_text(encoding="utf-8")
    for column in ax.COLUMNS:
        assert f"`{column}`" in text, column
    for test in ax.TESTS:
        assert f"`{test}`" in text, test
    for value in (*ax.DEFAULTS, *ax.SCORES, *ax.PAPER_20):
        assert f"`{value}`" in text, value
    assert "An AI assistant does not count as an independent coder" in " ".join(text.split())
    for key in ax.RECORD_KEYS:
        assert f"{key}:" in text, key


# --- round 4: what defaults_applied describes, own-sheet checks, and coding rules -------


@pytest.mark.parametrize(
    "coded, other, default",
    [
        (dict(alpha="0.01"), "mann_whitney:3:3:0.05:two_sided", "alpha"),
        (dict(), "other:3:3:0.05:two_sided", "test"),
        (dict(sidedness="one_sided"), "mann_whitney:3:3:0.05:two_sided", "sidedness"),
        (dict(alpha="0.0125"), "", "alpha"),  # Bonferroni over 4 tests of an unstated 0.05
    ],
)
def test_defaults_applied_describes_the_claim_not_the_coded_test(tmp_path, coded, other, default):
    """A default that applied to another test of the claim, or a corrected default, is
    listed without the coded test's own value having to be the default."""
    row = _row(n_arm_a="3", n_arm_b="3", defaults_applied=default, other_tests=other, **coded)
    proc = _run(_write(tmp_path, [row]), "--aggregate-only")
    assert proc.returncode == 0, proc.stderr
    defaults_line = next(
        line for line in proc.stdout.splitlines() if line.startswith("rows resting on a rule")
    )
    assert f"{default} 1" in defaults_line
    assert "cannot reject at any data: 0" in proc.stdout


def test_a_listed_sidedness_takes_the_coded_test_one_sided_in_f():
    """Even where the default applied only to another test: (f) can only withdraw."""
    other = ax.DiffTest("mann_whitney", 10, 10, 0.05, False)
    row = ax.assess(_claim(n_a=25, n_b=25, defaults=frozenset({"sidedness"}),
                           other_tests=(other,)))
    assert row.one_sided_power == ax.power_grid(25, 25, 0.05, True)[0.7]


def test_a_line_break_in_a_criterion_matches_the_sealed_record(registered):
    """A sheet saved with CRLF keeps a CRLF inside a quoted cell; the record reads LF."""
    rows = [dict(r) for r in FIVE]
    rows[0]["criterion"] = "line one\r\nline two"
    code, out, err = registered(rows, "--aggregate-only")
    assert code == 0, err
    assert "rows whose identification columns are not in the sealed record: 0 of 15" in out
    assert "none" not in _last_line(out)


def test_a_sheet_that_is_not_utf8_is_refused_whole(tmp_path):
    sheet = _write(tmp_path, [_token_row(1, criterion=f"{TOKEN} critérion")])
    sheet.write_bytes(sheet.read_bytes().decode("utf-8").encode("latin-1"))
    proc = _run(sheet)
    assert proc.returncode == 1
    assert "the file is not UTF-8 text" in proc.stderr
    assert "Traceback" not in proc.stderr and _no_token(proc)


def test_a_cell_past_the_csv_field_limit_is_refused_whole(tmp_path):
    proc = _run(_write(tmp_path, [_token_row(1, notes=TOKEN * 30000)]))
    assert proc.returncode == 1
    assert "the file is not a readable CSV" in proc.stderr
    assert "Traceback" not in proc.stderr and _no_token(proc)


def test_check_validates_one_sheet_and_computes_nothing(tmp_path):
    proc = _run(_write(tmp_path, FIVE), "--check")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "valid: 15 rows; nothing computed"
    bad = _run(_write(tmp_path, [_token_row(1, sidedness="two-sided")], name="bad.csv"),
               "--check")
    assert bad.returncode == 1
    assert "line 2: sidedness: must be one of" in bad.stderr
    assert bad.stdout == "" and _no_token(bad)
    both = _run(_write(tmp_path, FIVE), "--check", "--coder-sheets", tmp_path / "bad.csv",
                tmp_path / "bad.csv")
    assert both.returncode == 2 and "--check reads one sheet" in both.stderr


def test_check_counts_rows_that_are_not_the_sealed_records(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ax, "FRAME", _synthetic_frame(tmp_path))
    record = _write_record(tmp_path, FIVE)
    code, out, err = _main(capsys, _write(tmp_path, FIVE), "--check", "--sealed-record", record)
    assert code == 0, err
    assert "rows whose identification columns are not in the sealed record: 0 of 15" in out
    moved = [dict(r) for r in FIVE]
    moved[2]["source_location"] = "another synthetic source"
    code, out, _ = _main(capsys, _write(tmp_path, moved, name="moved.csv"), "--check",
                         "--sealed-record", record)
    assert code == 1
    assert "not in the sealed record: 1 of 15" in out
    for computed in ("underpowered", "cannot reject", "outcome", "power", "paper_20"):
        assert computed not in out


def test_one_file_cannot_stand_for_two_sheets(tmp_path):
    sheet = _write(tmp_path, FIVE)
    other = _write(tmp_path, FIVE, name="other.csv")
    for args in ([sheet, "--coder-sheets", sheet, other],
                 [sheet, "--coder-sheets", other, other]):
        proc = _run(*args)
        assert proc.returncode == 1
        assert "are the same file" in proc.stderr
        assert proc.stdout == ""


def test_the_sheet_must_keep_the_values_both_coders_hold(registered):
    """A value both own sheets hold, changed in the reconciled sheet, gives no outcome; the
    copy after author responses changes such values by design (prereg 7.4)."""
    changed = [dict(r) for r in FIVE]
    changed[0].update(n_arm_a="40", n_arm_b="40")
    code, out, _ = registered(changed, "--aggregate-only", own_rows=FIVE)
    assert code == 0
    assert "value both own sheets hold) 1" in out
    assert "this sheet does not match the coders' own sheets (prereg 9)" in _last_line(out)
    code, out, _ = registered(changed, "--aggregate-only", "--after-responses", own_rows=FIVE)
    assert "value both own sheets hold) 0" in out
    assert "none" not in _last_line(out)


def test_the_order_of_other_tests_is_not_a_disagreement():
    four = ax.DiffTest("mann_whitney", 4, 4, 0.05, False)
    count = ax.DiffTest("ks", None, None, 0.05, False)
    one, two = _claim(other_tests=(four, count)), _claim(other_tests=(count, four))
    assert ax.check_coders([one], [one], [two], None, [Path("/dev/null")] * 2).rows_mismatched == 0


GAP_FRAME = ax.Frame(expected_rows=15, expected_benchmarks=4,
                     audited={f"synth-bench-{x}": 10 for x in "abcde"}, unaudited=frozenset())
D = "claims resting on several tests at their combined chance to fail"
C = "t-test family at the efficiency-adjusted n"


def _parsed(**kw):
    problems = []
    return ax._parse_row(_row(**kw), 2, GAP_FRAME, problems), problems


def _c(i=1, bench="SYNTH-BENCH-A", **kw):
    claim, problems = _parsed(claim_id=f"S-{i}", benchmark=bench, **kw)
    assert claim is not None, problems
    return claim


def test_too_many_rows_or_benchmarks_do_not_reconcile():
    assert ax.reconcile([_c(i, f"SYNTH-BENCH-{'ABCD'[i % 4]}") for i in range(16)],
                        GAP_FRAME) == ["16 row(s); Appendix F counts 15 claims"]
    assert ax.reconcile([_c(i, f"SYNTH-BENCH-{'ABCDE'[i % 5]}") for i in range(15)],
                        GAP_FRAME) == ["5 benchmark(s); the paper's section 5 counts 4"]


def test_rule_three_needs_all_other_tests_unable_to_reject():
    row = ax.assess(_c(n_arm_a="3", n_arm_b="3",
                       other_tests="mann_whitney:3:3:0.05:two_sided;ks:3:3:0.05:two_sided"))
    assert (row.verdict, row.blocked) == (ax.UNDERPOWERED, "other_tests")


def test_another_tests_floor_is_taken_at_its_own_sidedness():
    """A one-sided 3 vs 3 at 0.05 has minimum p 0.05, equal to alpha: it can reject."""
    row = ax.assess(_c(n_arm_a="31", n_arm_b="31", other_tests="mann_whitney:3:3:0.05:one_sided"))
    assert row.verdict == ax.UNDERPOWERED and row.combined >= 0.8
    assert ax.sensitivity_outcomes([row])[D] == ax.MOSTLY_ADEQUATE


def test_analysis_d_takes_alpha_for_a_floor_test_on_a_defaulted_row():
    row = ax.assess(_c(n_arm_a="31", n_arm_b="31", defaults_applied="alpha",
                       other_tests="mann_whitney:3:3:0.05:two_sided"))
    assert row.combined == pytest.approx(1 - (1 - row.power[0.7]) * 0.95)


def test_analysis_c_keeps_a_row_underpowered_at_both_n():
    row = ax.assess(_c(test_used="t_test", n_arm_a="10", n_arm_b="10"))
    assert row.adjusted_verdict == ax.UNDERPOWERED
    assert ax.sensitivity_outcomes([row])[C] == ax.MOSTLY_WEAK


@pytest.mark.parametrize(
    "coded, other, refused",
    [
        (dict(n_arm_a="3", n_arm_b="3"), "mann_whitney:3:4:0.05:two_sided", True),
        (dict(test_used="ks", n_arm_a="3", n_arm_b="3"), "mann_whitney:3:3:0.05:two_sided", True),
        (dict(test_used="t_test", n_arm_a="3", n_arm_b="3"), "mann_whitney:3:3:0.05:two_sided",
         True),
        (dict(), "mann_whitney:12:12:0.05:two_sided", False),
        (dict(n_arm_a="30", n_arm_b="30", sidedness="one_sided"),
         "mann_whitney:31:31:0.05:two_sided", False),
    ],
)
def test_the_coding_order_of_section_2(coded, other, refused):
    claim, problems = _parsed(other_tests=other, **coded)
    assert (claim is None) is refused, problems


def test_one_sided_levels_in_c_e_and_f():
    row = ax.assess(_c(test_used="t_test", n_arm_a="23", n_arm_b="23", sidedness="one_sided",
                       scores="binary"))
    assert row.adjusted_verdict == ax.ADEQUATE
    assert row.binary_power == pytest.approx(ax.binary_power(23, 23, 0.05, True))
    assert ax.binary_power(20, 20, 0.05, True) == pytest.approx(ax.binary_power(20, 20, 0.1, False))
    f = ax.assess(_c(n_arm_a="20", n_arm_b="20", defaults_applied="sidedness"))
    assert f.one_sided_power == pytest.approx(ax.power_grid(20, 20, 0.1, False)[0.7])


@pytest.mark.parametrize(
    "field, value",
    [("n_arm_a", "13"), ("exact_test", "no"), ("other_tests", "mann_whitney:3:3:0.05:two_sided")],
)
def test_every_coded_field_is_compared(field, value):
    base = [_c(i) for i in range(3)]
    second = [_c(0, **{field: value}), *base[1:]]
    paths = [Path("/dev/null")] * 2
    assert ax.check_coders(base, base, second, None, paths).rows_mismatched == 1
    listed = [_c(0, disagreed_fields=field, **{field: value}), *base[1:]]
    assert ax.check_coders(listed, base, second, None, paths).rows_mismatched == 0


def test_a_row_missing_from_an_own_sheet_is_counted():
    base = [_c(i) for i in range(3)]
    check = ax.check_coders(base, base, base[:2], None, [Path("/dev/null")] * 2)
    assert check.rows_not_shared == 1


def test_weak_benchmarks_count_only_benchmarks_with_a_weak_row():
    rows = [ax.assess(_c(0, "SYNTH-BENCH-A", n_arm_a="10", n_arm_b="10")),
            ax.assess(_c(1, "SYNTH-BENCH-B", n_arm_a="40", n_arm_b="40"))]
    lines = ax.aggregate_lines(rows, ax.Context(frame=GAP_FRAME))
    assert "benchmarks with at least one claim that could not reliably fail: 1 of 2" in lines


def test_own_sheets_that_lack_a_row_give_no_outcome(registered):
    code, out, _ = registered(FIVE, "--aggregate-only", own_rows=FIVE[:14])
    assert code == 0
    assert "coders' own sheets: rows not shared with this sheet 1;" in out
    assert "the coders' own sheets do not hold this sheet's rows (prereg 9)" in _last_line(out)
