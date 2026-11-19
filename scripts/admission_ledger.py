"""Admission ledger: run the AdmissionGate over every (family, difficulty) and publish it.

Why this exists
---------------
RHOB's headline L0 result -- "reward-only detectors sit at chance" -- is a *negative
control*, not a finding: it is the manipulation check confirming the proxy rewards were
matched as intended. A negative control is only worth anything if it is visibly
falsifiable, and until this script existed nothing in the repo published the check. The
2026-08 audit found that 90 of 123 evaluated leaderboard cells had never been through
the gate at all, and that the cells that had were certified at a *single* difficulty
(``default_difficulties()[0]``) which in at least one family (``rlhf_sparse_coverage_gaming``,
certified at 0.95) is not even a difficulty the benchmark scores.

So this script writes the full grid -- every registered family x every difficulty in
``family.default_difficulties()``, every criterion, pass *and* fail, with the numbers --
to a committed JSON + Markdown artifact. Failures are the point: a ledger that only
listed admitted cells would reproduce exactly the selection effect it is meant to expose.

Degenerate cells are counted separately, and should be
------------------------------------------------------
A cell is ``DEGENERATE`` when a criterion could not be evaluated at all: either its
statistic had no resolution to begin with (every detector score ties, so the AUROC is
0.5 by the tie convention rather than by matching) or the proxy stream it reads carries
no signal above numerical dust (see ``rhob.v3.admission_gate.degeneracy_reason``).

**Six** of the 33 registered families are in that state at every tier they are scored
at, 30 (family, difficulty) cells in all -- 5 tiers each: ``distributional_shift``,
``monitored_sandbagging``, ``orbit_chirality``, ``physics_exploitation``,
``rlhf_reward_model_overopt`` and ``shortcut_exploitation``. The full ledger of
2026-11-19, over all 33 families, finds these six and no others. Five of them measure
resolution exactly 0.000 on all four L0 statistics and ``physics_exploitation`` measures
0.003 to 0.007, and they are unmeasurable in three ways:

- **Nothing in the proxy at all** -- ``distributional_shift`` (relative dispersion
  exactly 0.0), ``orbit_chirality`` (1.1e-16) and ``rlhf_reward_model_overopt``
  (1.6e-16 to 2.4e-16), i.e. a constant to within one ULP, below the informativeness
  floor.
- **Almost nothing** -- ``physics_exploitation`` (3.3e-4): above the floor, but so
  nearly constant that the L0 statistics order under 1% of the cross-variant pairs.
- **A varied proxy that is the same in every run** -- ``monitored_sandbagging``
  (dispersion 1.00) and ``shortcut_exploitation`` (3.0-4.4). There is plenty in the
  stream; none of it distinguishes one run from another, so every cross-variant pair
  ties anyway. An informativeness criterion alone would pass these two and
  ``physics_exploitation``, which is why the gate asks both questions.

Is such a family a legitimate RHOB family at all? Our answer, and the reason the ledger
segregates rather than merely fails these cells: **it is a legitimate hacking family and
an illegitimate L0 negative control, and those are separate questions.** The behavioral
side is real -- all six separate at L2 and diverge in true reward, which is what makes
them useful benchmark items -- but RHOB's L0 headline is "reward-only detectors sit at
chance *on a carefully matched proxy*", and on a constant proxy every detector is at
chance for a reason that has nothing to do with matching. Averaging those cells into the
L0 statistic inflates the negative control with cells that could not have come out any
other way; on the first equivalence ledger they were 15 of 35 cells, i.e. 43% of the
evidence was arithmetic. So: keep them as families, exclude them from the L0
negative-control statistics, and label them loudly enough that nobody re-includes them
by accident. Making the proxy informative-but-matched (a matched random proxy rather
than a constant one) is what would move them back into the control group, and that is a
change to the families, not to the gate.

That exclusion is not a promise this document makes and nothing keeps: it is
implemented in ``rhob.v3.leaderboard.access_summary.summarize_access_levels``, whose
``degenerate_families`` argument holds those six out of the L0 rung and reports the
withheld set and the with-degenerate figure alongside, and it is fed from the
``degenerate_families`` block this script writes. DEGENERATE is the only status it holds
out: the cells this ledger measures NOT ADMITTED (51 of 123 on 2026-11-19, against 42
ADMITTED and 30 DEGENERATE) stay in the L0 mean, and whether they should is an open
question for the author (``ROADMAP.md``).

Usage
-----
Full grid (slow -- see the timing table the run prints; budget hours, not minutes)::

    python scripts/admission_ledger.py

A subset, e.g. to smoke-test a change to the gate::

    python scripts/admission_ledger.py --families orbit_chirality,distributional_shift

Determinism: the gate seeds every layout and bootstrap from a fixed root seed
(``rhob.v3.admission_gate._RNG_SEED``), so re-running with the same flags on the same
commit reproduces the ``results`` block byte-for-byte. The ``timing_seconds`` block and
the provenance timestamp are machine-dependent by construction and are the only parts
that will differ.

The artifact is rewritten after every family, so an interrupted run still leaves a
complete, correctly-labelled ledger for the families it finished.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib import metadata
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Run against *this* checkout's sources, not whatever `rhob` happens to be installed.
# The ledger records a git commit as provenance, so it has to be the commit whose code
# actually ran -- which matters in a worktree, where the editable install points
# somewhere else entirely.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import rhob.v3.families  # noqa: F401,E402  (import for registration side effects)
from rhob.v3.admission_gate import (  # noqa: E402
    CRITERIA,
    AdmissionCertificate,
    AdmissionGate,
    CriterionOutcome,
)
from rhob.v3.provenance import provenance_block  # noqa: E402
from rhob.v3.registry import FamilyRegistry  # noqa: E402

DEFAULT_OUT_DIR = _REPO_ROOT / "admission"

#: Simulator packages that can move a rollout. A mujoco or pettingzoo bump changes the
#: physics/agent dynamics underneath a third of the families, so a ledger that omitted
#: them could not explain a disagreement with a later re-run. Since 2026-11-16
#: :func:`rhob.v3.provenance.package_versions` records these itself (with mujoco-core);
#: the loop below is kept so the ledger states them even if that list changes again.
_SIMULATOR_PACKAGES = ("gymnasium", "mujoco", "pettingzoo", "mpe2")


def collect_provenance(
    gate: AdmissionGate, families: list[str], difficulties: list[float] | None
) -> dict[str, Any]:
    """Everything needed to know what produced these numbers.

    Built on the repo-wide :func:`provenance_block` so the ledger's provenance is the
    same shape every other RHOB artifact publishes, then extended with the two things
    only this artifact has: which families were in scope, and how the gate was
    configured.
    """
    all_families = FamilyRegistry.list_families()
    block = provenance_block(
        scope={
            # A partial ledger is still worth publishing, but it must never be mistaken
            # for the full grid: an incomplete admission ledger that looks complete is
            # the same selection effect the ledger exists to remove.
            "families_run": list(families),
            "families_registered": len(all_families),
            "complete": sorted(families) == all_families and difficulties is None,
            "difficulty_override": difficulties,
        },
        gate={
            "n_layouts": gate.n_layouts,
            "min_seeds_per_layout": gate.min_seeds_per_layout,
            "n_bootstrap": gate.n_bootstrap,
            "equivalence_margin": gate.equivalence_margin,
            "informativeness_floor": gate.informativeness_floor,
            "alpha": gate.alpha,
            "target_power": gate.target_power,
            "behavioral_floor": gate.behavioral_floor,
            "onset_std_frac": gate.onset_std_frac,
        },
    )
    for name in _SIMULATOR_PACKAGES:
        try:
            block["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            block["packages"][name] = None
    block["cpu"] = _cpu_model()
    return block


def _cpu_model() -> str | None:
    """The CPU model, recorded because floating-point results can differ between models.

    A sharded ledger certifies its cells on different machines; this is what lets a
    reader see whether they were the same kind of machine.
    """
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform

    return platform.processor() or None


def run_ledger(
    gate: AdmissionGate,
    family_names: list[str],
    difficulties: list[float] | None,
    n_seeds_per_variant: int | None,
    verbose: bool = True,
    on_family_done: Callable[[list[dict[str, Any]], dict[str, float]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Certify every (family, difficulty) cell; return (rows, per-family wall-clock).

    ``on_family_done`` is invoked after each family with the rows collected so far. The
    full grid takes hours (a single MuJoCo cell is ~50 minutes at the default design),
    so the artifact is flushed as it is produced: a run interrupted at family 30 of 33
    must leave 30 families of usable ledger behind, not nothing.
    """
    rows: list[dict[str, Any]] = []
    timing: dict[str, float] = {}
    for name in family_names:
        family = FamilyRegistry.get(name)
        tiers = difficulties if difficulties is not None else family.default_difficulties()
        started = time.perf_counter()
        n_cells = 0
        for difficulty in tiers:
            lo, hi = family.difficulty_range()
            if not (lo - 1e-9 <= difficulty <= hi + 1e-9):
                continue  # an explicit --difficulties this family cannot generate
            n_cells += 1
            try:
                cert: AdmissionCertificate = gate.certify(
                    family, difficulty=difficulty, n_seeds_per_variant=n_seeds_per_variant
                )
                row = cert.to_dict()
                row["error"] = None
            except Exception as exc:  # noqa: BLE001 - a broken family is a ledger entry
                # An optional-dependency failure (mujoco, pettingzoo) or a genuinely
                # broken generator must appear in the ledger, not silently shrink it.
                # A cell that raised is NOT ADMITTED, never DEGENERATE: the gate did not
                # decline to measure the statistic, it never got to run.
                row = {
                    "family": name,
                    "difficulty": difficulty,
                    "passed": False,
                    "status": "NOT ADMITTED",
                    "criteria": {c: False for c in CRITERIA},
                    "outcomes": {c: CriterionOutcome.FAIL.value for c in CRITERIA},
                    "degenerate_criteria": [],
                    "details": {},
                    "metrics": {},
                    "design": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            if verbose:
                failed = [c for c in CRITERIA if row["outcomes"][c] == "fail"]
                degen = row["degenerate_criteria"]
                notes = []
                if failed:
                    notes.append(f"failed: {', '.join(failed)}")
                if degen:
                    notes.append(f"unmeasurable: {', '.join(degen)}")
                suffix = f" ({'; '.join(notes)})" if notes else ""
                print(f"  {name} @ {difficulty:.2f}: {row['status']}{suffix}", flush=True)
        timing[name] = time.perf_counter() - started
        if verbose:
            print(f"  -> {name}: {timing[name]:.1f}s for {n_cells} tier(s)", flush=True)
        if on_family_done is not None:
            on_family_done(rows, timing)
    return rows, timing


def _fmt(values: dict[str, Any], key: str, spec: str = ".3f") -> str:
    """Format a recorded number, or ``-`` when the cell never produced one."""
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    return format(float(value), spec)


def _shape_cell(metrics: dict[str, Any], detector_key: str) -> str:
    """``mean [lo, hi]`` for one shape detector's TOST interval."""
    return (
        f"{_fmt(metrics, detector_key + '_auroc', '.3f')} "
        f"[{_fmt(metrics, detector_key + '_ci_lo', '.3f')}, "
        f"{_fmt(metrics, detector_key + '_ci_hi', '.3f')}]"
    )


#: How each criterion outcome is rendered in the Markdown pass/fail grid. Degenerate
#: gets its own mark rather than being folded into FAIL: a reader scanning the grid has
#: to be able to tell "we measured this and it is wrong" from "we could not measure it".
_GRID_MARK = {
    CriterionOutcome.PASS.value: "PASS",
    CriterionOutcome.FAIL.value: "**FAIL**",
    CriterionOutcome.DEGENERATE.value: "*DEGEN*",
}

_STATUS_MARK = {
    "ADMITTED": "ADMITTED",
    "NOT ADMITTED": "**NOT ADMITTED**",
    "DEGENERATE": "**DEGENERATE**",
}


def render_markdown(payload: dict[str, Any]) -> str:
    prov = payload["provenance"]
    rows = payload["results"]
    gate = prov["gate"]
    n_pass = sum(1 for r in rows if r["passed"])
    degenerate_rows = [r for r in rows if r["degenerate_criteria"]]

    scope = prov["scope"]
    # `design` is empty only if *every* cell raised (e.g. an uninstalled optional
    # dependency across the whole scope). The ledger still has to render, because a
    # ledger of nothing but errors is exactly the state a reader needs to see.
    design = payload["design"]

    out: list[str] = []
    out.append("# RHOB Admission Ledger")
    out.append("")
    out.append(
        "Every family in scope at every difficulty the benchmark scores, run through "
        "the `AdmissionGate`. **Failing cells are listed, not filtered** -- this table is "
        "the falsifiable form of RHOB's L0-at-chance negative control. A cell that fails "
        "`proxy_matched` is a cell where the proxy reward was *not* demonstrably matched, "
        "and any L0 result on it means something other than \"detectors cannot see the "
        "proxy\". A cell that is *degenerate* on `proxy_matched` is one where the "
        "question could not be asked at all, and an L0 result on it means nothing in "
        "either direction."
    )
    out.append("")
    n_not_admitted = sum(1 for r in rows if r["status"] == "NOT ADMITTED")
    out.append(
        f"**{n_pass} / {len(rows)} cells admitted, {len(degenerate_rows)} degenerate, "
        f"{n_not_admitted} not admitted.**"
    )
    out.append("")
    out.append(
        "Those three add up to the cell count and are the headline numbers any prose "
        "about this ledger has to quote; `tests/test_v3/test_ledger_doc_consistency.py` "
        "reads them out of `admission/admission_ledger.json` and fails if the repo's "
        "Markdown says something else."
    )
    out.append("")
    if degenerate_rows:
        out.append(
            f"**{len(degenerate_rows)} / {len(rows)} cells are DEGENERATE**: at least one "
            "criterion could not be measured at all -- either the statistic it reads has "
            "no resolution on this family (every cross-variant detector score ties, so "
            "the AUROC is 0.5 by the tie convention rather than by matching) or the "
            "proxy stream is a constant to within numerical dust (so whatever the "
            "detectors ordered was rounding error). A degenerate cell is **not** a pass "
            "and **must not be counted** as evidence for RHOB's L0-at-chance negative "
            "control -- see [Degenerate cells](#degenerate-cells) for the list and the "
            "reasoning."
        )
        out.append("")
    if not scope["complete"]:
        out.append(
            f"> **Partial ledger.** In scope: {len(scope['families_run'])} of "
            f"{scope['families_registered']} registered families"
            + (
                f", restricted to difficulties {scope['difficulty_override']}"
                if scope["difficulty_override"]
                else ""
            )
            + ". Families absent from the table below are **uncertified**, which is not "
            "the same as admitted."
        )
        out.append("")

    git = prov["git"]
    out.append("## Provenance")
    out.append("")
    out.append(f"- Generated: {prov['generated_utc']}")
    out.append(f"- Commit: `{git['commit']}` (branch `{git['branch']}`, "
               f"working tree {'dirty' if git['dirty'] else 'clean'})")
    sharding = prov.get("sharding")
    platforms = ", ".join(sharding["platforms"]) if sharding else prov["platform"]
    out.append(f"- Python: {prov['python']} on {platforms}")
    if sharding:
        cpus = ", ".join(c or "unknown" for c in sharding.get("cpus", [])) or "unknown"
        out.append(
            f"- Sharded: merged from {sharding['n_shards']} single-cell runs, generated "
            f"{sharding['generated_utc_first']} to {sharding['generated_utc_last']}, on "
            f"CPUs: {cpus}" + (f"; CI run {sharding['ci_run']}" if sharding.get("ci_run") else "")
        )
        out.append(f"- Each shard ran: `{sharding['shard_command']}`")
    out.append(
        "- Packages: "
        + ", ".join(
            f"{k} {v if v else 'not installed'}" for k, v in sorted(prov["packages"].items())
        )
    )
    out.append(f"- {'Merge command' if sharding else 'Command'}: `{' '.join(prov['argv'])}`")
    out.append("")

    out.append("## Gate configuration")
    out.append("")
    out.append(f"- Layouts per cell: {gate['n_layouts']}; "
               f"seeds per side per layout: {gate['min_seeds_per_layout']}")
    out.append(f"- Root RNG seed: {design.get('rng_seed', '-')} (fixed; the ledger is "
               f"reproducible on this commit)")
    out.append(f"- `proxy_matched`: TOST, equivalence margin +/-{gate['equivalence_margin']}, "
               f"alpha={gate['alpha']} per one-sided test, a-priori power at true "
               f"AUROC 0.5 = {_fmt(design, 'a_priori_power', '.2f')}")
    out.append(f"- `proxy_distribution_matched`: the same TOST applied to each of "
               f"{', '.join(design.get('shape_detectors', [])) or '-'} "
               f"({design.get('n_equivalence_tests', '-')} equivalence tests in total, "
               f"all of which must pass)")
    out.append(f"- `behavioral_separated`: mean L2 AUROC >= {gate['behavioral_floor']}")
    out.append(
        f"- Degeneracy guard: a proxy criterion is *not measurable* when its statistic "
        f"orders <= {2 * gate['equivalence_margin']:.2f} of the cross-variant pairs "
        f"(the equivalence band then contains its whole attainable range) **or** when "
        f"the proxy stream's dispersion relative to its own magnitude is below "
        f"{gate['informativeness_floor']:.0e}. The second is what stops a constant "
        f"proxy plus numerically irrelevant jitter from certifying."
    )
    out.append(f"- Bootstrap resamples: {gate['n_bootstrap']}")
    out.append("")

    out.append("## Per-criterion outcomes")
    out.append("")
    n_pm_degen = sum(
        1 for r in rows if r["outcomes"]["proxy_matched"] == CriterionOutcome.DEGENERATE
    )
    out.append(
        "Measured failures and unmeasurable cells are counted in separate columns, and "
        f"must stay that way: a criterion with 0 measured failures and {n_pm_degen} "
        "degenerate cells is telling a maintainer something completely different from "
        f"one with {n_pm_degen} measured failures, and a single 'not established' count "
        "says neither."
    )
    out.append("")
    out.append("| criterion | PASS | **FAIL** (measured) | *DEGEN* (not measurable) |")
    out.append("|---|---|---|---|")
    for c in CRITERIA:
        n_fail = sum(1 for r in rows if r["outcomes"][c] == CriterionOutcome.FAIL)
        n_degen = sum(1 for r in rows if r["outcomes"][c] == CriterionOutcome.DEGENERATE)
        out.append(f"| `{c}` | {len(rows) - n_fail - n_degen} | {n_fail} | {n_degen} |")
    out.append("")

    out.append("## Pass/fail grid")
    out.append("")
    short = {
        "proxy_matched": "proxy",
        "proxy_distribution_matched": "proxy-dist",
        "behavioral_separated": "behav",
        "true_reward_diverges": "true-rwd",
        "onset_localizable": "onset",
        "camping_quality": "camping",
    }
    header = ["family", "diff"] + [short[c] for c in CRITERIA] + ["overall"]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        cells = [_GRID_MARK[r["outcomes"][c]] for c in CRITERIA]
        overall = _STATUS_MARK[r["status"]]
        out.append(
            "| " + " | ".join([r["family"], f"{r['difficulty']:.2f}", *cells, overall]) + " |"
        )
    out.append("")
    out.append(
        "`PASS` = measured and established. `**FAIL**` = measured and not met. "
        "`*DEGEN*` = **not measurable**: the statistic has no resolution on this family, "
        "so neither a pass nor a fail is available from it."
    )
    out.append("")

    out.append("## Measurements")
    out.append("")
    out.append(
        "`L0 AUROC [CI]` is the TOST interval on the mean across layouts; it must lie "
        "entirely inside the equivalence band. The three shape columns are the same "
        "interval for each shape-sensitive L0 detector. `res` is the L0 statistic's "
        "resolution -- the fraction of cross-variant run pairs the detector strictly "
        "orders. At `res` 0.000 every pair ties, the AUROC is 0.5 by convention, and "
        "the interval next to it is not a measurement of anything. `info` is the proxy "
        "stream's dispersion relative to its own magnitude; below the floor the proxy "
        "is a constant to within numerical dust and `res` says nothing, however high "
        "it reads."
    )
    out.append("")
    mheader = [
        "family", "diff", "L0 AUROC [90% CI]", "res", "info", "var-ratio [CI]",
        "KDE [CI]", "skew [CI]", "L2 AUROC", "true-reward CI", "onset std", "late/early",
    ]
    out.append("| " + " | ".join(mheader) + " |")
    out.append("|" + "|".join(["---"] * len(mheader)) + "|")
    for r in rows:
        m = r["metrics"]
        pm, pd = m.get("proxy_matched", {}), m.get("proxy_distribution_matched", {})
        bs, tr = m.get("behavioral_separated", {}), m.get("true_reward_diverges", {})
        ol, cq = m.get("onset_localizable", {}), m.get("camping_quality", {})
        if r["error"]:
            out.append(
                f"| {r['family']} | {r['difficulty']:.2f} | "
                + " | ".join(["ERROR: " + r["error"]] + ["-"] * (len(mheader) - 3))
                + " |"
            )
            continue
        out.append(
            "| "
            + " | ".join(
                [
                    r["family"],
                    f"{r['difficulty']:.2f}",
                    f"{_fmt(pm, 'mean_l0_auroc', '.4f')} "
                    f"[{_fmt(pm, 'ci_lo', '.4f')}, {_fmt(pm, 'ci_hi', '.4f')}]",
                    _fmt(pm, "statistic_resolution"),
                    _fmt(pm, "proxy_informativeness", ".2e"),
                    _shape_cell(pd, "reward_variance_ratio"),
                    _shape_cell(pd, "reward_kde"),
                    _shape_cell(pd, "reward_skewness"),
                    _fmt(bs, "mean_l2_auroc"),
                    f"[{_fmt(tr, 'diff_ci_lo', '.4f')}, {_fmt(tr, 'diff_ci_hi', '.4f')}]",
                    f"{_fmt(ol, 'onset_std', '.2f')} / {_fmt(ol, 'threshold', '.2f')}",
                    _fmt(cq, "late_over_early_ratio"),
                ]
            )
            + " |"
        )
    out.append("")

    out.append("## Degenerate cells")
    out.append("")
    out.append(
        "A criterion is degenerate when the gate could not evaluate it. Either its "
        "statistic cannot leave the acceptance band no matter what the family does -- "
        "every cross-variant detector score ties, so the AUROC is 0.5 by the tie "
        "convention and a TOST against it would certify against any margin, including "
        "0.001 -- or the proxy stream it reads is a constant to within numerical dust, "
        "in which case whatever the detectors ordered was rounding error. These cells "
        "are listed apart from the failures because the two are different findings: a "
        "failure says the proxy leaks and points at the environment to fix, a degenerate "
        "cell says the proxy carries no information and there is nothing to read either "
        "way. A failure on a proxy criterion also covers an equivalence test that could "
        "not establish matching within the margin at this design (a TOST failure): the "
        "point estimate is inside the band and its interval is not."
    )
    out.append("")
    # The split the sentence above implies, counted rather than asserted. Same test as
    # ``admission_gate._check_proxy_matched`` uses to mark a failure underpowered. The
    # ``admission/ADMISSION_LEDGER.md`` committed in d4b700d (2026-11-19) was rendered
    # before this sentence and this split were added. It was later re-rendered from the
    # same committed JSON with this function, with no new rollouts, and now reads 24
    # measured failures, 8 outside the band and 16 inside.
    pm_fails = [
        r["metrics"]["proxy_matched"]
        for r in rows
        if r["outcomes"].get("proxy_matched") == "fail"
        and "mean_l0_auroc" in r["metrics"].get("proxy_matched", {})
    ]
    if pm_fails:
        margin, alpha = gate["equivalence_margin"], gate["alpha"]
        n_inside = sum(
            1 for m in pm_fails if abs(m["mean_l0_auroc"] - 0.5) < m.get("margin", margin)
        )
        out.append(
            f"Of the {len(pm_fails)} measured `proxy_matched` failures here, "
            f"{len(pm_fails) - n_inside} have a point estimate outside "
            f"[{0.5 - margin:.2f}, {0.5 + margin:.2f}] and {n_inside} a point estimate "
            f"inside it with a {100 * (1 - 2 * alpha):.0f}% CI that is not."
        )
        out.append("")
    if not degenerate_rows:
        out.append("None.")
        out.append("")
    else:
        out.append(
            "**These cells are excluded from RHOB's L0-at-chance negative control.** "
            "The control's claim is that reward-only detectors sit at chance on a "
            "*carefully matched* proxy; on a constant proxy they sit at chance for a "
            "reason that has nothing to do with matching, so counting these cells "
            "inflates the control with results that could not have come out otherwise. "
            "The families remain valid hacking families -- they separate at L2 and "
            "diverge in true reward -- and would rejoin the control group if their "
            "proxies were made informative-but-matched rather than constant."
        )
        out.append("")
        out.append(
            "That exclusion is implemented, not just asserted here: "
            "`rhob.v3.leaderboard.access_summary.summarize_access_levels` takes the "
            "`degenerate_families` list this file publishes and holds those families "
            "out of the **L0 rung only** -- L1/L2/L3 do not read the proxy reward, so a "
            "degenerate-proxy family is an ordinary benchmark item for them -- and "
            "reports the withheld set and the with-degenerate figure alongside so the "
            "adjustment can be checked rather than taken."
        )
        out.append("")
        out.append(
            "| family | diff | degenerate criteria | L0 statistic resolution | "
            "proxy informativeness |"
        )
        out.append("|---|---|---|---|---|")
        for r in degenerate_rows:
            pm = r["metrics"].get("proxy_matched", {})
            out.append(
                f"| {r['family']} | {r['difficulty']:.2f} | "
                f"{', '.join('`' + c + '`' for c in r['degenerate_criteria'])} | "
                f"{_fmt(pm, 'statistic_resolution')} | "
                f"{_fmt(pm, 'proxy_informativeness', '.2e')} |"
            )
        out.append("")
        by_family: dict[str, int] = {}
        for r in degenerate_rows:
            by_family[r["family"]] = by_family.get(r["family"], 0) + 1
        out.append(
            "By family: "
            + ", ".join(f"`{k}` ({v} tier{'s' if v != 1 else ''})" for k, v in by_family.items())
            + "."
        )
        out.append("")

    failing = [r for r in rows if not r["passed"]]
    out.append("## Failing cells, in full")
    out.append("")
    out.append(
        "Every cell that was not admitted, degenerate ones included, with each "
        "criterion's verdict verbatim."
    )
    out.append("")
    if not failing:
        out.append("None.")
    for r in failing:
        out.append(f"### `{r['family']}` @ difficulty {r['difficulty']:.2f}")
        out.append("")
        if r["error"]:
            out.append(f"- Certification raised: `{r['error']}`")
        for c in CRITERIA:
            mark = r["outcomes"][c].upper()
            out.append(f"- **{mark}** `{c}`: {r['details'].get(c, '')}")
        out.append("")

    out.append("## Wall clock")
    out.append("")
    if prov.get("sharding"):
        out.append(
            "Machine-dependent; the only non-reproducible part of this artifact. This ledger "
            "was certified one cell per machine, so each figure sums a family's per-cell "
            "process wall-clock on separate runners. Every shard repeats per-process setup, "
            "so this is neither the run's elapsed time nor the cost of one unsharded re-run."
        )
    else:
        out.append(
            "Machine-dependent; the only non-reproducible part of this artifact. Recorded so "
            "the cost of a full re-run is knowable in advance."
        )
    out.append("")
    out.append("| family | seconds | tiers |")
    out.append("|---|---|---|")
    tiers_per_family: dict[str, int] = {}
    for r in rows:
        tiers_per_family[r["family"]] = tiers_per_family.get(r["family"], 0) + 1
    for name, secs in payload["timing_seconds"].items():
        out.append(f"| {name} | {secs:.1f} | {tiers_per_family.get(name, 0)} |")
    total = sum(payload["timing_seconds"].values())
    out.append(f"| **total** | **{total:.1f}** | **{len(rows)}** |")
    out.append("")
    return "\n".join(out)


def _display_path(path: Path) -> str:
    """Repo-relative when the artifact landed inside the repo, absolute otherwise.

    ``--out-dir`` is routinely pointed outside the checkout (scratch runs, CI caches),
    and a path that cannot be made relative is not a reason to fail after the work is
    already written to disk.
    """
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--families",
        default="all",
        help="comma-separated family names, or 'all' (default)",
    )
    p.add_argument(
        "--difficulties",
        default=None,
        help="comma-separated difficulties to override each family's default_difficulties()",
    )
    p.add_argument("--n-layouts", type=int, default=12, help="independent layouts per cell")
    p.add_argument(
        "--min-seeds-per-layout",
        type=int,
        default=None,
        help="override the power-derived seed floor; lowering it produces an "
        "underpowered design that cannot certify proxy_matched (the certificate says so)",
    )
    p.add_argument(
        "--seeds-per-variant",
        type=int,
        default=None,
        help="total seeds per variant across all layouts (floored at the seed floor)",
    )
    p.add_argument("--n-bootstrap", type=int, default=2000, help="bootstrap resamples")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="artifact directory")
    p.add_argument("--quiet", action="store_true", help="suppress per-cell progress")
    p.add_argument(
        "--list-cells",
        action="store_true",
        help="print the full grid as a JSON list of {family, difficulty} and exit "
        "(the shard list for .github/workflows/regenerate.yml)",
    )
    p.add_argument(
        "--merge-shards",
        type=Path,
        default=None,
        metavar="DIR",
        help="assemble the full ledger from single-cell runs under DIR (each one "
        "`--families F --difficulties D`) instead of certifying anything; see merge_shards",
    )
    p.add_argument(
        "--allow-cell-errors",
        action="store_true",
        help="with --merge-shards, publish cells whose certification raised as NOT ADMITTED "
        "rows instead of refusing the merge",
    )
    return p


def ledger_grid(names: list[str] | None = None) -> list[tuple[str, float]]:
    """Every ``(family, difficulty)`` cell a full run certifies, in the order it writes them.

    The same iteration as :func:`run_ledger` with no ``--difficulties`` override: registry
    order, then each family's ``default_difficulties()``, skipping a tier outside the
    family's own range exactly as ``run_ledger`` does.
    """
    cells: list[tuple[str, float]] = []
    for name in names if names is not None else FamilyRegistry.list_families():
        family = FamilyRegistry.get(name)
        lo, hi = family.difficulty_range()
        for difficulty in family.default_difficulties():
            if lo - 1e-9 <= difficulty <= hi + 1e-9:
                cells.append((name, float(difficulty)))
    return cells


def build_payload(
    rows: list[dict[str, Any]], timing: dict[str, float], provenance: dict[str, Any]
) -> dict[str, Any]:
    """The ledger artifact, from its cells, their per-family wall-clock and a provenance block."""
    # The design block is identical across cells (the gate is configured once), so read it
    # off the first successful certificate rather than duplicating the derivation.
    return {
        "schema": "rhob.admission_ledger/1",
        "provenance": provenance,
        "design": next((r["design"] for r in rows if r["design"]), {}),
        # Three per-criterion counts, not one, because the tri-state exists precisely so
        # that "we measured this and it is wrong" and "we could not measure it" stay
        # apart -- and a field called `n_failing_by_criterion` that counted both re-merged
        # them at the last step, publishing `proxy_matched: 15` for a criterion with zero
        # measured failures. Each name now says exactly which question it answers, and
        # `n_not_established_by_criterion` (the sum of the other two) is the one that maps
        # to `criteria[c] is False`.
        "summary": {
            "n_cells": len(rows),
            "n_admitted": sum(1 for r in rows if r["passed"]),
            "n_degenerate": sum(1 for r in rows if r["degenerate_criteria"]),
            "n_not_admitted": sum(1 for r in rows if r["status"] == "NOT ADMITTED"),
            "n_failed_by_criterion": {
                c: sum(1 for r in rows if r["outcomes"][c] == CriterionOutcome.FAIL)
                for c in CRITERIA
            },
            "n_degenerate_by_criterion": {
                c: sum(1 for r in rows if r["outcomes"][c] == CriterionOutcome.DEGENERATE)
                for c in CRITERIA
            },
            "n_not_established_by_criterion": {
                c: sum(1 for r in rows if not r["criteria"][c]) for c in CRITERIA
            },
            "degenerate_families": sorted(
                {r["family"] for r in rows if r["degenerate_criteria"]}
            ),
        },
        "results": rows,
        "timing_seconds": timing,
    }


def write_artifacts(payload: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Write ``admission_ledger.json`` and ``ADMISSION_LEDGER.md`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "admission_ledger.json"
    md_path = out_dir / "ADMISSION_LEDGER.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return json_path, md_path


class ShardError(ValueError):
    """A set of shards that does not add up to one ledger."""


def _shard_environment(prov: dict[str, Any]) -> dict[str, Any]:
    """The parts of a shard's provenance that must agree across every shard."""
    git = prov.get("git") or {}
    return {
        "commit": git.get("commit"),
        "python": prov.get("python"),
        "python_implementation": prov.get("python_implementation"),
        "packages": prov.get("packages"),
        "gate": prov.get("gate"),
    }


def merge_shards(
    shard_dir: Path, families: list[str] | None = None, allow_cell_errors: bool = False
) -> dict[str, Any]:
    """Assemble the full-grid ledger from single-cell runs under ``shard_dir``.

    Each shard is the ``admission_ledger.json`` of one ``--families F --difficulties D``
    run, so the grid can be certified on many machines at once
    (``.github/workflows/regenerate.yml``). The result is the artifact a single full run
    would have written, because a cell does not depend on what else was in the run:
    :meth:`AdmissionGate.certify` draws every layout seed from a generator re-seeded at
    ``_RNG_SEED`` for each cell, and nothing else carries state between cells.

    Refused, rather than merged, unless the shards are exactly the full grid -- every
    cell once, none extra -- and were all produced from one clean commit under one Python
    and one set of package versions with one gate configuration. A partial merge would
    publish a ledger labelled complete that is not, and a mixed-environment merge one
    whose cells were not measured under the same physics (the MuJoCo 4 failure,
    docs/postmortems/2026-11-mujoco-calibration-cache.md).

    What does differ from a single run, and is recorded under ``provenance.sharding``:
    ``generated_utc`` is the latest shard's; cells may be certified on different CPU
    models and platforms, all listed there (``provenance.cpu`` and ``.platform`` are then
    null); and ``timing_seconds`` sums each family's per-cell wall-clock measured on
    separate machines. "The artifact a single full run would have written" holds on one
    CPU model: numpy's SIMD dispatch and OpenBLAS's kernels depend on the CPU, so a cell
    certified on another model can differ in its last digits.

    ``families`` narrows the expected grid to those families (tests use it to merge a
    cheap subset); the ledger is then labelled incomplete unless they are all registered.

    A cell whose certification raised is a legitimate ledger row (``run_ledger`` records
    it as NOT ADMITTED with its ``error``), but on a sharded run it is as likely to be one
    machine's failure as the family's, so it is refused unless ``allow_cell_errors``.
    """
    import copy
    import os

    paths = sorted(shard_dir.rglob("admission_ledger.json"))
    if not paths:
        raise ShardError(f"no admission_ledger.json under {shard_dir}")

    by_cell: dict[tuple[str, float], tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        shard = json.loads(path.read_text(encoding="utf-8"))
        if shard.get("schema") != "rhob.admission_ledger/1":
            raise ShardError(f"{path}: schema {shard.get('schema')!r}, not a ledger")
        if len(shard.get("results", [])) != 1:
            raise ShardError(
                f"{path}: {len(shard.get('results', []))} cells; a shard is exactly one cell"
            )
        row = shard["results"][0]
        key = (row["family"], round(float(row["difficulty"]), 6))
        scope = shard["provenance"].get("scope") or {}
        if scope.get("families_run") != [row["family"]] or [
            round(float(d), 6) for d in scope.get("difficulty_override") or []
        ] != [key[1]]:
            raise ShardError(f"{path}: its scope does not name its own cell {key}")
        if key in by_cell:
            raise ShardError(f"{path}: cell {key} already supplied by {by_cell[key][0]}")
        by_cell[key] = (path, shard)

    expected = [(name, round(d, 6)) for name, d in ledger_grid(families)]
    missing = [k for k in expected if k not in by_cell]
    extra = sorted(set(by_cell) - set(expected))
    if missing or extra:
        raise ShardError(
            f"shards are not the full grid: {len(missing)} missing {missing[:5]}, "
            f"{len(extra)} not in the grid {extra[:5]}"
        )

    ordered = [by_cell[k] for k in expected]
    ref_path, ref = ordered[0]
    ref_env = _shard_environment(ref["provenance"])
    if not ref_env["commit"]:
        raise ShardError(f"{ref_path}: no git commit recorded")
    for path, shard in ordered:
        prov = shard["provenance"]
        if (prov.get("git") or {}).get("dirty") is not False:
            raise ShardError(f"{path}: produced from a dirty or unknown tree")
        env = _shard_environment(prov)
        for field, value in env.items():
            if value != ref_env[field]:
                raise ShardError(
                    f"{path}: {field} differs from {ref_path} ({value!r} vs {ref_env[field]!r})"
                )
    designs = {
        json.dumps(s["results"][0]["design"], sort_keys=True)
        for _, s in ordered
        if s["results"][0]["design"]
    }
    if len(designs) > 1:
        raise ShardError(f"shards disagree on the sampling design: {len(designs)} variants")

    rows = [shard["results"][0] for _, shard in ordered]
    errors = [(r["family"], r["difficulty"], r["error"]) for r in rows if r.get("error")]
    if errors and not allow_cell_errors:
        raise ShardError(f"{len(errors)} cell(s) raised, e.g. {errors[0]}; rerun those "
                         "shards, or pass --allow-cell-errors to publish them as rows")
    families = list(dict.fromkeys(name for name, _ in expected))
    timing = {name: 0.0 for name in families}
    for _, shard in ordered:
        for name, seconds in shard["timing_seconds"].items():
            timing[name] += float(seconds)

    stamps = sorted(s["provenance"]["generated_utc"] for _, s in ordered)
    provenance = copy.deepcopy(ref["provenance"])
    provenance["generated_utc"] = stamps[-1]
    provenance["argv"] = list(sys.argv)
    # One shard's machine is not the ledger's: every cell's machine is listed under
    # `sharding`, so the single-machine fields are emptied rather than left naming one.
    provenance["platform"] = None
    provenance["cpu"] = None
    provenance["scope"] = {
        "families_run": families,
        "families_registered": len(FamilyRegistry.list_families()),
        "complete": families is None or sorted(families) == FamilyRegistry.list_families(),
        "difficulty_override": None,
    }
    ci_run = None
    if os.environ.get("GITHUB_RUN_ID"):
        ci_run = (
            f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
            f"{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/"
            f"{os.environ['GITHUB_RUN_ID']}"
        )
    provenance["sharding"] = {
        "unit": "one (family, difficulty) cell per run",
        "n_shards": len(ordered),
        "generated_utc_first": stamps[0],
        "generated_utc_last": stamps[-1],
        "platforms": sorted({s["provenance"].get("platform") for _, s in ordered}),
        "cpus": sorted({s["provenance"].get("cpu") or "" for _, s in ordered}),
        "shard_command": (
            "python scripts/admission_ledger.py --families F --difficulties D --out-dir DIR"
        ),
        "timing_seconds": "sum of each family's per-cell wall-clock, on separate machines",
        "ci_run": ci_run,
    }
    return build_payload(rows, timing, provenance)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_cells:
        print(json.dumps([{"family": f, "difficulty": d} for f, d in ledger_grid()]))
        return 0
    if args.merge_shards is not None:
        if args.families != "all" or args.difficulties is not None:
            parser.error("--merge-shards takes no --families or --difficulties")
        try:
            payload = merge_shards(args.merge_shards, allow_cell_errors=args.allow_cell_errors)
        except ShardError as exc:
            print(f"refusing to merge: {exc}", file=sys.stderr)
            return 1
        json_path, md_path = write_artifacts(payload, args.out_dir)
        summary = payload["summary"]
        print(
            f"merged {payload['provenance']['sharding']['n_shards']} shards: "
            f"{summary['n_admitted']}/{summary['n_cells']} cells admitted, "
            f"{summary['n_degenerate']} degenerate; wrote {_display_path(json_path)} and "
            f"{_display_path(md_path)}"
        )
        return 0

    gate = AdmissionGate(
        n_layouts=args.n_layouts,
        n_bootstrap=args.n_bootstrap,
        min_seeds_per_layout=args.min_seeds_per_layout,
    )
    if args.families == "all":
        names = FamilyRegistry.list_families()
    else:
        names = [n.strip() for n in args.families.split(",") if n.strip()]
        for n in names:
            FamilyRegistry.get(n)  # fail fast on a typo
    difficulties = (
        None
        if args.difficulties is None
        else [float(d) for d in args.difficulties.split(",") if d.strip()]
    )

    if not args.quiet:
        print(
            f"Admission ledger: {len(names)} families, "
            f"{gate.n_layouts} layouts x {gate.min_seeds_per_layout} seeds/side per cell",
            flush=True,
        )

    # Before certifying, so an unwritable --out-dir fails now rather than after the first
    # family's work.
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def write(rows: list[dict[str, Any]], timing: dict[str, float]) -> dict[str, Any]:
        # `families_run` reflects what has actually been certified so far, so a
        # partially-written ledger is labelled partial rather than overclaiming.
        payload = build_payload(rows, timing, collect_provenance(gate, list(timing), difficulties))
        write_artifacts(payload, args.out_dir)
        return payload

    rows, timing = run_ledger(
        gate,
        names,
        difficulties,
        args.seeds_per_variant,
        verbose=not args.quiet,
        on_family_done=write,
    )
    payload = write(rows, timing)

    summary = payload["summary"]
    print(
        f"\n{summary['n_admitted']}/{summary['n_cells']} cells admitted, "
        f"{summary['n_degenerate']} degenerate (unmeasurable, not counted as evidence); "
        f"wrote {_display_path(args.out_dir / 'admission_ledger.json')} and "
        f"{_display_path(args.out_dir / 'ADMISSION_LEDGER.md')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
