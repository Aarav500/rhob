"""The difficulty-calibration artifact must be internally sound and match the prose.

Why this file exists
--------------------
``results/difficulty_calibration/difficulty_calibration.json`` publishes the first
measurement RHOB has ever made of the thing its difficulty argument claims to be: the
*achieved* L2-AUROC at each *requested* difficulty. Its headline is a count -- how many
families' knobs move nothing at all -- and a count in prose is exactly the kind of number
that goes stale silently. ``tests/test_v3/test_ledger_doc_consistency.py`` exists because
that already happened once to the admission ledger: the ledger went from "30 of 35 cells
admitted" to a three-way split and every doc quoting the old number kept quoting it,
because nothing in the repo read the artifact and the prose together.

So this file reads them together, in both directions:

1. **The artifact is self-consistent.** Its ``summary`` counts are recomputed from the
   per-family blocks. A summary that disagrees with its own data is worse than no summary.
2. **Each classification is justified by the data it claims to rest on.** A family called
   ``structurally_decoupled`` must actually have a bit-identical ``behav_trace`` across
   tiers; one called ``closed_form_calibrated`` must actually invert a closed form in its
   source; one listed as span-zero must actually measure a span of exactly 0.000. The
   classifier lives in the script and could be changed there; these assertions are the
   independent re-derivation.
3. **The Markdown states the artifact's numbers.** Both the generated report and any
   other repo document that cites this measurement.
4. **A partial artifact says so.** The measurement is expensive and resumable, so a
   partial artifact is a normal state -- but one that reads as complete is the selection
   effect the whole audit exists to remove.

What is deliberately not asserted is any *particular value*: not "27 families are dead",
not "the worst error is 0.40". Pinning the measurement itself would turn a finding into a
fixture, and the numbers are supposed to move when someone fixes a family. What is pinned
is that whatever the artifact says, the docs say the same thing and the classification
follows from the data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_ARTIFACT_DIR = _REPO / "results" / "difficulty_calibration"
_JSON = _ARTIFACT_DIR / "difficulty_calibration.json"
_MD = _ARTIFACT_DIR / "DIFFICULTY_CALIBRATION.md"
_SCRIPT = _REPO / "scripts" / "measure_difficulty_calibration.py"
_FAMILY_DIR = _REPO / "src" / "rhob" / "v3" / "families"

_INT = re.compile(r"\d+")

#: The four ways a family can be classified as *not* answering to its difficulty knob,
#: plus the two ways it can be excused. Kept here rather than imported so that adding a
#: class to the script forces a decision about this test rather than silently widening it.
_CLASSES = {
    "closed_form_calibrated",
    "structurally_decoupled",
    "saturated",
    "flat_off_ceiling",
    "responsive",
    "single_tier",
}


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert _JSON.exists(), (
        f"{_JSON.relative_to(_REPO)} is missing. It is a published artifact -- regenerate "
        "it with `python scripts/measure_difficulty_calibration.py` (it is resumable and "
        "may be run one family group at a time) before the docs can be checked against it."
    )
    return json.loads(_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def measured(artifact) -> dict:
    """Only the families that were actually measured; errors and skips are not data."""
    return {n: b for n, b in artifact["families"].items() if b.get("status") == "measured"}


def test_summary_counts_are_recomputable_from_the_family_blocks(artifact, measured):
    """The headline is arithmetic about the rest of the file, so it has to be checkable."""
    s = artifact["summary"]
    assert s["n_families_measured"] == len(measured)
    assert s["n_cells_measured"] == sum(len(b["tiers"]) for b in measured.values())

    multi = {n: b for n, b in measured.items() if b["summary"]["n_tiers"] > 1}
    assert s["n_families_with_more_than_one_tier"] == len(multi)

    zero = sorted(n for n, b in multi.items() if b["summary"]["span_is_exactly_zero"])
    assert s["families_separation_span_exactly_zero"] == zero
    assert s["n_families_separation_span_exactly_zero"] == len(zero)

    by_class: dict[str, list[str]] = {}
    for name, block in measured.items():
        by_class.setdefault(block["summary"]["classification"], []).append(name)
    assert s["families_by_classification"] == {k: sorted(v) for k, v in sorted(by_class.items())}
    assert s["n_by_classification"] == {k: len(v) for k, v in sorted(by_class.items())}
    assert sum(s["n_by_classification"].values()) == len(measured), (
        "every measured family must land in exactly one class"
    )
    assert set(s["n_by_classification"]) <= _CLASSES, (
        f"unknown classification(s): {set(s['n_by_classification']) - _CLASSES}. A new "
        "class needs a reader-facing gloss in the Markdown and a decision here about what "
        "evidence justifies it."
    )


def test_span_zero_means_the_measured_span_is_exactly_zero(artifact, measured):
    """The headline count is only worth anything if 0.000 means 0.000 and not 'small'."""
    for name in artifact["summary"]["families_separation_span_exactly_zero"]:
        seps = [
            t["achieved_separation"]
            for t in measured[name]["tiers"]
            if t["achieved_separation"] is not None
        ]
        assert len(seps) > 1, f"{name} is counted as span-zero on fewer than two tiers"
        assert max(seps) == min(seps), (
            f"{name} is listed as separation span exactly 0.000 but its achieved "
            f"separations across tiers are {seps}"
        )


def test_structural_decoupling_is_backed_by_a_bit_identical_behavioural_channel(measured):
    """``structurally_decoupled`` is a claim about bytes; check the bytes.

    The whole force of this class is that it is a *proof*, not an inference from a flat
    curve: layout 0 is generated at the same layout seed and rolled out at the same seed
    base at every difficulty, so an identical ``behav_trace`` digest across tiers means no
    L2 detector could have responded to the difficulty argument whatever it did.
    """
    for name, block in measured.items():
        summary = block["summary"]
        digests = {t["layout_0_channel_sha256"]["behav_trace"] for t in block["tiers"]}
        if summary["classification"] == "structurally_decoupled":
            assert summary["channel_varies_with_difficulty"]["behav_trace"] is False
            assert len(digests) == 1, (
                f"{name} is classified structurally_decoupled but its layout-0 behav_trace "
                f"digests differ across tiers: {sorted(digests)}"
            )
            assert summary["separation_span"] == 0.0, (
                f"{name} has a bit-identical behavioural channel across tiers yet a "
                f"non-zero separation span of {summary['separation_span']} -- that is "
                "arithmetically impossible and means the two measurements disagree"
            )
        elif summary["n_tiers"] > 1:
            assert len(digests) > 1, (
                f"{name} has a bit-identical layout-0 behav_trace at every tier but is "
                f"classified {summary['classification']!r}, not structurally_decoupled"
            )


def test_closed_form_classification_is_re_derived_from_the_family_source(measured):
    """A family is closed-form iff its module inverts the normal CDF; check the tree.

    Re-derived here rather than trusted, because the script's detector is a source scan
    and a source scan is exactly the kind of thing that silently stops matching. The
    marker is ``scipy.special.ndtri``: it is the inverse standard-normal CDF, and the only
    reason a family module contains it is to invert ``L2 = Phi(d / (sqrt(2) sigma_a))``.
    """
    with_ndtri = {
        path.stem
        for path in _FAMILY_DIR.glob("*.py")
        if "ndtri" in path.read_text(encoding="utf-8")
    }
    for name, block in measured.items():
        summary = block["summary"]
        is_closed_form = name in with_ndtri
        assert bool(summary["closed_form_source"]) == is_closed_form, (
            f"{name}: the artifact says closed_form_source="
            f"{summary['closed_form_source']!r} but scanning "
            f"{_FAMILY_DIR.relative_to(_REPO).as_posix()}/{name}.py for `ndtri` says "
            f"{is_closed_form}"
        )
        if is_closed_form and summary["n_tiers"] > 1:
            assert summary["classification"] == "closed_form_calibrated"


def test_every_measured_family_records_the_design_it_was_measured_under(artifact, measured):
    """Cells measured under different designs must not be tabulated as one measurement."""
    design = artifact["design"]
    for name, block in measured.items():
        assert block["design"]["n_layouts"] == design["n_layouts"], name
        assert block["design"]["seeds_per_layout"] == design["seeds_per_layout"], name
        for tier in block["tiers"]:
            assert len(tier["per_layout_auroc"]) == design["n_layouts"], (
                f"{name} @ {tier['requested_difficulty']} recorded "
                f"{len(tier['per_layout_auroc'])} layouts under a "
                f"{design['n_layouts']}-layout design"
            )
    assert design["randomize_behav_sign"] is False, (
        "this measurement is of a family's separability, not of a detector's ability to "
        "orient itself in the benchmark's randomized coordinate -- the same reason the "
        "admission gate certifies un-randomized. Turning randomization on here changes "
        "what the achieved-vs-requested error means and the artifact must say so."
    )


def test_the_error_column_is_achieved_minus_requested(measured):
    """The arithmetic of the one column a reader will quote."""
    for name, block in measured.items():
        for tier in block["tiers"]:
            if tier["achieved_l2_auroc"] is None:
                continue
            expected = tier["achieved_l2_auroc"] - tier["requested_difficulty"]
            assert tier["error_achieved_minus_requested"] == pytest.approx(
                expected, abs=1e-6
            ), f"{name} @ {tier['requested_difficulty']}"
            assert tier["achieved_separation"] == pytest.approx(
                abs(tier["achieved_l2_auroc"] - 0.5), abs=1e-6
            ), f"{name} @ {tier['requested_difficulty']}"


def test_the_gate_crosscheck_reproduces_the_admission_ledger(artifact):
    """At the gate's design this harness must be reading the gate's own number.

    The achieved-vs-requested table is only a re-reading of the admission gate's
    measurement if the draw is the gate's draw. When the design matches, every cell the
    two artifacts share must agree to the published precision; a disagreement means one of
    them is measuring something else and the comparison in the report is not what it says.
    """
    check = artifact.get("gate_crosscheck") or {}
    if not check.get("checked"):
        pytest.skip(f"cross-check not applicable: {check.get('reason')}")
    if check["n_cells_compared"] == 0:
        pytest.skip("no cells are present in both the ledger and this artifact")
    assert check["n_exact"] == check["n_cells_compared"], (
        "this harness no longer reproduces the admission ledger's mean L2 AUROC on the "
        f"cells they share: {check['n_cells_compared'] - check['n_exact']} of "
        f"{check['n_cells_compared']} differ, worst by {check['max_abs_delta']}. Either "
        "the gate's design/draw changed or this script drifted from it; the artifact's "
        "claim to be measuring the same statistic depends on this.\n"
        f"  {check['differing_cells']}"
    )


def _headline_lines(text: str) -> list[str]:
    """Lines that state the span-zero headline: the phrase, and at least two numbers.

    Narrow on purpose. The report discusses spans in several places -- a legend, a column
    header, a per-family row -- and only a sentence that gives the count out of the total
    is making the claim this test polices.
    """
    out = []
    for line in text.splitlines():
        low = line.lower()
        if "separation span of exactly" in low and len(_INT.findall(line)) >= 2:
            out.append(line.strip())
    return out


def _citing_docs() -> list[Path]:
    """Every repo Markdown file that points at this measurement.

    Discovered rather than listed, so a new document that starts quoting these numbers is
    covered the day it is written instead of the day someone remembers to add it here.
    """
    needles = (
        "measure_difficulty_calibration.py",
        "difficulty_calibration.json",
        "DIFFICULTY_CALIBRATION.md",
    )
    docs = []
    for path in sorted(_REPO.rglob("*.md")):
        if any(part in {".git", ".worktrees", "node_modules"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(n in text for n in needles):
            docs.append(path)
    return docs


def test_the_span_zero_headline_in_every_citing_doc_matches_the_artifact(artifact):
    """The count of dead knobs, as the docs state it, against the artifact."""
    s = artifact["summary"]
    expected = {
        s["n_families_separation_span_exactly_zero"],
        s["n_families_with_more_than_one_tier"],
    }
    assert _MD.exists(), f"{_MD.relative_to(_REPO)} is missing next to its JSON"

    docs = {_MD, *_citing_docs()}
    checked_any = False
    for doc in sorted(docs):
        for line in _headline_lines(doc.read_text(encoding="utf-8")):
            checked_any = True
            found = {int(n) for n in _INT.findall(line)}
            missing = expected - found
            assert not missing, (
                f"{doc.relative_to(_REPO)} states a span-zero headline that disagrees "
                f"with {_JSON.relative_to(_REPO)}.\n"
                f"  doc says : {line}\n"
                f"  artifact : {s['n_families_separation_span_exactly_zero']} of "
                f"{s['n_families_with_more_than_one_tier']} multi-tier families have a "
                "separation span of exactly 0.000\n"
                f"  missing from the line: {sorted(missing)}"
            )
    assert checked_any, (
        f"{_MD.relative_to(_REPO)} never states the span-zero headline. It is the "
        "artifact's central finding and the report has to lead with it."
    )


def test_the_report_states_the_classification_counts_the_json_records(artifact):
    """The class table is the finding's structure; a stale one misdescribes the failure."""
    md = _MD.read_text(encoding="utf-8")
    for klass, n in artifact["summary"]["n_by_classification"].items():
        row = re.search(rf"^\|\s*`{re.escape(klass)}`\s*\|\s*(\d+)\s*\|", md, re.MULTILINE)
        assert row, f"the report has no classification row for `{klass}`"
        assert int(row.group(1)) == n, (
            f"the report says {row.group(1)} families are `{klass}`; the artifact says {n}"
        )


def test_a_partial_artifact_is_labelled_partial(artifact):
    """An incomplete measurement that reads as complete is the defect, not the gap."""
    s = artifact["summary"]
    md = _MD.read_text(encoding="utf-8")
    if s["complete"]:
        assert not s["families_not_measured"]
        return
    assert "PARTIAL" in md, (
        f"{s['n_families_measured']} of {s['n_families_registered']} families are "
        "measured, but the report does not say it is partial. A reader must not be able "
        "to mistake an unmeasured family for a calibrated one."
    )
    for name in s["families_not_measured"]:
        assert f"`{name}`" in md, (
            f"`{name}` was not measured and is not named in the report's unmeasured list"
        )


def test_the_script_that_produces_the_artifact_is_the_one_the_artifact_names(artifact):
    """Provenance has to point at a file that exists, or it is decoration."""
    assert _SCRIPT.exists()
    assert artifact["provenance"]["script"] == "scripts/measure_difficulty_calibration.py"
    assert artifact["schema"] == "rhob.difficulty_calibration/1"
