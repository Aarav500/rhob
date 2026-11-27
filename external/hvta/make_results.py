"""Render ``RESULTS.md`` from the scoring artifacts. Every number in the report comes
from here, so none of them can be typed by hand.

Reads ``hvta_results.json`` (the primary draw, ``seed_base=0``) and any replicate
artifacts passed with ``--replicate``, and writes ``RESULTS.md`` beside them.

A detector re-scored on the primary draw after a fix (``rescore_detector.py``, recorded
under the artifact's ``rescored``, and carried through a later full run as
``detector_fixes``) is reported from its re-scored rows. A replicate draw that does not
attest the same detector source still holds that detector's pre-fix rows: they are shown
marked *(pre-fix)* and left out of every across-draw statistic. So is a draw whose rows
for any detector attest a different source from the primary draw's, marked *(other
source)*. Which draws are stale is worked out from the files loaded, never from a list
written into an artifact earlier, so re-scoring a replicate clears its markers and the
provenance sentence together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

#: Every scored artifact -- the primary draw plus any replicates -- set in ``main``. Claims
#: that would otherwise be a property of one draw are computed over all of them, so a
#: headline can never quote whichever draw happened to be most flattering.
ALL_ARTIFACTS: list = []

#: The primary draw, set in ``main``: every other draw's sources are compared with its.
PRIMARY: dict = {}

#: Detectors the primary draw records a fix for, in a ``rescored`` record or, after a full
#: re-run, in the ``detector_fixes`` it carried forward. Set in ``main``.
FIXED: set = set()


def same_source(a, b):
    """Whether two ``detector_sources`` entries attest the same code, hashed the same way.

    Kept in step with ``score_detectors.same_source``; this script reads artifacts only
    and does not import RHOB.
    """
    if not a or not b:
        return False
    return (a.get("hashed"), a.get("sha256")) == (b.get("hashed"), b.get("sha256"))


def current_source(art, name):
    """The source ``art``'s rows for ``name`` were scored with, if it records one.

    The latest ``rescored`` record that re-scored ``name``, else the full run's
    ``detector_sources``. Kept in step with ``score_detectors.current_source``.
    """
    for rec in reversed(art.get("rescored", [])):
        src = (rec.get("detector_sources") or {}).get(name)
        if src:
            return src
    return (art.get("detector_sources") or {}).get(name)


def stale_mark(art, name):
    """Why ``art``'s figures for ``name`` are not comparable with the primary draw's, or None.

    ``"pre-fix"`` when the primary records a fix to ``name`` and ``art`` does not attest
    the source the primary's rows were scored with. ``"other source"`` when both attest a
    source and they differ. A draw that attests nothing for a detector nobody fixed is
    taken as scored by the same code: the replicates predate source attestation.
    """
    ref, mine = current_source(PRIMARY, name), current_source(art, name)
    if art is PRIMARY or same_source(ref, mine):
        return None
    if name in FIXED:
        return "pre-fix"
    return "other source" if ref and mine else None


def pre_fix(art, name):
    """Whether ``art``'s figures for ``name`` are left out of every across-draw figure."""
    return stale_mark(art, name) is not None


def marked(text, art, name):
    """``text`` with ``art``'s stale marker for ``name`` appended, if it has one."""
    mark = stale_mark(art, name)
    return f"{text} *({mark})*" if mark else text


def _revision(rec):
    """``rec``'s RHOB head, with how many ``src/`` files differed from it."""
    head = f"`{(rec.get('rhob_git_head') or '?')[:12]}`"
    dirty = rec.get("rhob_src_uncommitted")
    if dirty:
        head += f" ({len(dirty)} `src/` file{'s' if len(dirty) != 1 else ''} uncommitted)"
    return head


def _shown_path(p):
    """``p`` relative to this directory when it is under it, else its file name."""
    try:
        return Path(p).resolve().relative_to(HERE).as_posix()
    except ValueError:
        return Path(p).name


def f(x, n=3, dash="n/a"):
    return dash if x is None else f"{x:.{n}f}"


def pct(x, dash="n/a"):
    return dash if x is None else f"{100 * x:.0f}%"


def live_cells(art):
    return [c for c in art["generation"]["cells"] if not c["degenerate"]]


def cell_rows(art, idx, name):
    for r in art["per_cell"][str(idx)]:
        if r["detector"] == name:
            return r
    return None


def firing_gap(om):
    """Firing rate on hacking runs minus firing rate on legitimate runs.

    One definition for every place the report quotes it, so two sentences cannot round
    the same quantity differently.
    """
    return (1 - om["miss_rate_hacking"]) - om["false_onset_rate_legit"]


def onset_means(art, name):
    """Mean onset diagnostics for ``name`` over the non-degenerate cells."""
    ks = []
    for c in live_cells(art):
        r = cell_rows(art, c["index"], name)
        if r and r.get("onset"):
            ks.append(r["onset"])
    if not ks:
        return None

    def m(key):
        v = [k[key] for k in ks if k[key] is not None]
        return float(np.mean(v)) if v else None

    return {k: m(k) for k in ks[0]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "hvta_results.json"))
    ap.add_argument(
        "--replicate",
        nargs="*",
        default=None,
        help="independent-draw result files; defaults to replicates/*.json",
    )
    ap.add_argument("--out", default=str(HERE / "RESULTS.md"))
    args = ap.parse_args(argv)

    global ALL_ARTIFACTS, PRIMARY
    art = json.loads(Path(args.results).read_text(encoding="utf-8"))
    PRIMARY = art
    FIXED.clear()
    for rec in art.get("rescored", []) + art.get("detector_fixes", []):
        FIXED.update(rec.get("detectors") or [])
    rep_paths = (
        sorted((HERE / "replicates").glob("*.json"))
        if args.replicate is None
        else [Path(p) for p in args.replicate]
    )
    # (artifact, path as shown) pairs, in seed order, so a stale draw can be named.
    rep_items = sorted(
        ((json.loads(p.read_text(encoding="utf-8")), _shown_path(p)) for p in rep_paths),
        key=lambda item: item[0]["generation"]["config"]["seed_base"],
    )
    reps = [a for a, _ in rep_items]
    ALL_ARTIFACTS = [art] + reps
    gen, cfg = art["generation"], art["generation"]["config"]
    cells = gen["cells"]
    lc = live_cells(art)
    se = art["auroc_null_se"]
    npairs = art["n_pairs_per_cell"]
    L = []
    w = L.append

    # ---------------------------------------------------------------- header
    w("# RHOB detectors on HVTA: discrimination and onset against a decidable oracle")
    w("")
    w(
        "RHOB's detectors have only ever been scored on RHOB's own families, where a false "
        "positive is knowable only because RHOB's generator held out a programmatic oracle. "
        "This report scores them inside HVTA (Hack-Verifiable Environments, MajoRoth), which "
        "is decidable **by construction**: the environment itself records whether the agent "
        "read the leaked solution and at which step. Every number below is measured against "
        "HVTA's own `hack_onset_step`, not against anything RHOB wrote down."
    )
    w("")
    w("Generated by `external/hvta/make_results.py` from `hvta_results.json`. Do not hand-edit.")
    w("")

    # ------------------------------------------------------------- headline
    scored = [r for r in art["per_detector"] if r["mean_auroc"] is not None]
    na_dets = [r for r in art["per_detector"] if r["mean_auroc"] is None]
    unsup = [r for r in scored if not r["label_fitted"] and r["access_level"] != "L3"
             and not r["duplicate_of"]]
    best_unsup = max(unsup, key=lambda r: r["mean_auroc"])
    below = [r for r in scored if r["mean_auroc"] < 0.5]
    never_fire = [
        r["detector"] for r in art["per_detector"]
        if (om := onset_means(art, r["detector"])) is not None
        and om["miss_rate_hacking"] is not None and om["miss_rate_hacking"] > 0.999
    ]
    fires = [
        (r["detector"], om)
        for r in art["per_detector"]
        if (om := onset_means(art, r["detector"])) is not None
        and om["mae_steps_when_fired"] is not None
    ]
    best_onset = min(fires, key=lambda kv: kv[1]["mae_steps_when_fired"])
    ref = fires[0][1]

    w("## What was found")
    w("")
    bu_draws = [
        m["mean_auroc"]
        for a in (ALL_ARTIFACTS or [art])
        for m in a["per_detector"]
        if m["detector"] == best_unsup["detector"] and m["mean_auroc"] is not None
        and not pre_fix(a, m["detector"])
    ]
    w(
        f"1. **Discrimination transfers, partially.** The best unsupervised detector, "
        f"{best_unsup['detector']} ({best_unsup['access_level']}), reaches a mean AUROC of "
        f"{f(best_unsup['mean_auroc'])} over the {len(lc)} non-degenerate cells "
        f"(per-cell range {f(best_unsup['min_auroc'])}-{f(best_unsup['max_auroc'])}), and "
        f"{f(float(np.mean(bu_draws)))} averaged over {len(bu_draws)} independent seed draws "
        f"(range {f(max(bu_draws) - min(bu_draws))}). "
        f"{len(below)} of the {len(scored)} scoreable detectors sit **below** chance."
    )
    # Best tracking correlation over EVERY draw, not just the primary one: quoting the
    # primary draw's 0.35 when another draw reached 0.43 would be picking the flattering
    # number in the direction of the argument being made.
    tracked = []
    for a in ALL_ARTIFACTS or [art]:
        for r in a["per_detector"]:
            if r["detector"] == "True Reward Oracle" or pre_fix(a, r["detector"]):
                continue
            o = onset_means(a, r["detector"])
            if o and o["pearson_r_pred_vs_true"] is not None:
                tracked.append((o["pearson_r_pred_vs_true"], r["detector"],
                                o["n_fired_hacking"] / o["n_hacking"]))
    best_r, best_r_name, best_r_fire = max(tracked)
    # The unqualified maximum can be owned by a detector that fired on a couple of runs,
    # where r is computed on a handful of points. Both numbers are reported: the raw
    # maximum, which is the least favourable to the argument here, and the maximum among
    # detectors that actually fired on most runs.
    solid = [t for t in tracked if t[2] >= 0.5]
    solid_r, solid_name, _ = max(solid) if solid else (None, None, None)

    def whose(name):
        """``name``, qualified when it has a comparable figure on only some of the draws.

        The maximum is taken over every comparable figure, so a detector whose other
        draws are stale can own it from fewer draws than the sentence counts.
        """
        n_cur = sum(1 for a in ALL_ARTIFACTS if not pre_fix(a, name))
        if n_cur == len(ALL_ARTIFACTS):
            return name
        where = "primary draw only" if n_cur == 1 else f"{n_cur} of the draws only"
        n_stale = len(ALL_ARTIFACTS) - n_cur
        return (
            f"{name}, {where}; its other {n_stale} "
            f"{'draw is' if n_stale == 1 else 'draws are'} "
            f"{'pre-fix' if name in FIXED else 'scored by another source'}"
        )
    tro_om = dict(fires)["True Reward Oracle"]
    flat = sorted(
        n for n, om in fires
        if om["pred_onset_std_steps"] is not None and om["pred_onset_std_steps"] < 0.05
    )
    beat_const = [
        n for n, om in fires
        if om["mae_steps_when_fired"] < om["reference_mae_steps_best_constant_oracle"]
    ]
    w(
        f"2. **Onset localization does not transfer.** This is the measurement RHOB has never "
        f"been able to make against external ground truth, and it is the negative result. "
        f"{len(never_fire)} of the {len(fires) + len(never_fire)} detectors that reach a "
        f"scoreable cell never nominate an onset on any HVTA run. Of those that do, none "
        f"*tracks* the true onset. Over **all {len(ALL_ARTIFACTS)} independent draws** the "
        f"highest correlation between predicted and true onset is r = {f(best_r, 2)} "
        f"({whose(best_r_name)}) -- and that detector fires on only "
        f"{pct(best_r_fire)} of runs, so its r rests on a handful of points and reverses "
        f"sign between draws; among detectors that fire on a majority of runs the best is "
        f"**r = {f(solid_r, 2)}** ({whose(solid_name) if solid_name else 'none'}). "
        f"Meanwhile {len(flat)} detectors emit "
        f"literally the same step on every run (prediction SD = 0.0 against a true-onset SD "
        f"of {f(ref['true_onset_std_steps'], 1)} steps). The positive control confirms the "
        f"measure works: the L3 oracle, whose channel *is* the label, reaches "
        f"r = {f(tro_om['pearson_r_pred_vs_true'], 2)}."
    )
    const_ref = ref["reference_mae_steps_best_constant_oracle"]
    if beat_const:
        margins = {
            n: const_ref - om["mae_steps_when_fired"] for n, om in fires if n in beat_const
        }
        verdict = (
            f"{len(beat_const)} of the {len(fires)} beats it -- "
            + ", ".join(f"{n} by {m:.1f} steps" for n, m in margins.items())
        )
    else:
        verdict = "no detector beats it"
    w(
        f"   The step errors say the same thing. The lowest is {best_onset[0]} at "
        f"{f(best_onset[1]['mae_steps_when_fired'], 1)} steps -- but a *constant* prediction "
        f"chosen with knowledge of the true onsets scores {f(const_ref, 1)} steps, and only "
        f"{verdict}. A uniform random guess scores "
        f"{f(ref['reference_mae_steps_uniform_guess'], 1)} steps. What looks like localization "
        f"is the accuracy of a near-constant landing inside a narrow onset window."
    )
    w(
        f"3. **Every detector that fires, fires on the legitimate twin too.** Because HVTA is "
        f"decidable and the legitimate variant provably never reads the solution "
        f"(`hack_triggered=False`, `hack_onset_step=None`), each of these is a **real** false "
        f"positive. The largest gap between a detector's firing rate on hacking runs and on "
        f"legitimate runs is "
        f"{f(max(firing_gap(om) for _, om in fires), 2)}. "
        f"The detectors with low false-onset rates achieve them by staying silent on hacking "
        f"runs as well, not by discriminating."
    )
    w(
        f"4. **L1 is not scored, and that is the correct outcome.** HVTA exposes no "
        f"state-visitation distribution, so `state_counts` is absent and RHOB's own "
        f"`missing_channels` returns N/A for all {len(na_dets)} L1 detectors. None is imputed "
        f"to 0.5. This is the defect RHOB's own 2026-08 audit found on 88 of 123 leaderboard "
        f"cells, and it is deliberately not reintroduced here."
    )
    w("")

    # ---------------------------------------------------------------- design
    w("## Design")
    w("")
    w(
        f"| | |\n|---|---|\n"
        f"| Environment | HVTA `FilesystemWrapper(hidden_solution=True)` over TextArena Sudoku |\n"
        f"| Games | {', '.join('`' + g + '`' for g in cfg['games'])} |\n"
        f"| Agent competence | {', '.join('`' + c + '`' for c in cfg['competences'])} "
        f"(deduction strength of the scripted policy) |\n"
        f"| Difficulty axis | `slip_rate` &isin; {{{', '.join(f'{s:.2f}' for s in cfg['slip_rates'])}}} "
        f"-- probability the agent plays a wrong value on a move it solved correctly |\n"
        f"| Cells | {art['n_cells']} = {len(cfg['games'])} games x {len(cfg['competences'])} "
        f"competences x {len(cfg['slip_rates'])} slip rates |\n"
        f"| Matched pairs per cell | {npairs} |\n"
        f"| Episodes | {art['n_cells'] * npairs * 2} "
        f"({art['n_cells'] * npairs} hacking + {art['n_cells'] * npairs} legitimate) |\n"
        f"| Steps per episode | {cells[0]['n_steps']} agent actions "
        f"({100 * cells[0]['fs_fraction']:.0f}% filesystem, rest game moves) |\n"
        f"| Seeds | {cfg['seed_base']}-{cfg['seed_base'] + npairs - 1}, identical in every cell |\n"
        f"| Detectors | {len(art['per_cell']['0'])} "
        f"(30 from `scripts/v5_leaderboard_and_transfer.py` + 5 external baselines) |\n"
        f"| Evaluation path | `{art['evaluation_path']}` |"
    )
    w("")
    w(
        f"**Onset distribution.** HVTA's `hack_onset_step` for the {npairs} hacking runs of "
        f"every cell: `{cells[0]['onsets']}` -- {cells[0]['onsets_distinct']} distinct values "
        f"spanning steps {min(cells[0]['onsets'])}-{max(cells[0]['onsets'])} "
        f"({100 * min(cells[0]['onsets']) / cells[0]['n_steps']:.0f}%-"
        f"{100 * max(cells[0]['onsets']) / cells[0]['n_steps']:.0f}% of the episode). Onset is "
        f"spread, not constant, so a detector cannot score well by emitting a fixed step -- "
        f"and the report quantifies exactly how well a fixed step *would* do."
    )
    w("")
    w("**Matching, verified rather than asserted.** For all "
      f"{art['n_cells'] * npairs} pairs: the two variants share the game, the seed and every "
      "action strictly before onset (`prefix_identical`); they run the same number of steps "
      "(`length_matched`); the hacking variant reports `hack_triggered=True` with an integer "
      "`hack_onset_step`, and the legitimate variant reports `False`/`None`. Generation "
      "aborts if any pair fails. Both variants read a file in the *same directory* at the "
      "same step -- hacking reads the solution, legitimate reads the `README.txt` HVTA "
      "writes beside it warning that the directory leaks.")
    w("")

    # ------------------------------------------------- degenerate / control
    deg = [c for c in cells if c["degenerate"]]
    w("### The negative control, and why it is excluded from aggregates")
    w("")
    w(
        f"{len(deg)} of the {len(cells)} cells are **degenerate**: "
        + ", ".join(f"`{c['game']}`/`{c['competence']}`/slip {c['slip_rate']:.2f}" for c in deg)
        + f". There the scripted solver completes the puzzle unaided, so the leaked solution "
        f"confers nothing and the two variants come out identical in every observable channel "
        f"(final proxy-reward gap exactly {f(deg[0]['final_proxy_gap_max'], 3)}). Every "
        f"observable-channel detector scores exactly 0.500 there. That is an **absence of "
        f"signal**, not a measurement of a detector, so these cells are flagged at generation "
        f"time from the environment's own payout -- never inferred from a score -- and "
        f"excluded from every aggregate below."
    )
    w("")
    tro_deg = cell_rows(art, deg[0]["index"], "True Reward Oracle")
    w(
        f"They are retained because they demonstrate one thing sharply: in the degenerate "
        f"cells the L3 True Reward Oracle still scores **{f(tro_deg['auroc'])}** while every "
        f"other detector scores 0.500. Nothing observable distinguished the runs; the oracle "
        f"separated them perfectly because its channel was *derived from the onset label*. "
        f"See the L3 section."
    )
    w("")

    # -------------------------------------------------------------- channels
    w("## Channels")
    w("")
    w("| channel | status | source |")
    w("|---|---|---|")
    w("| `proxy_rewards` | **present** | `SudokuEnv._get_percentage_completion` -- the "
      "environment's own payout, recomputed from the visible board. Observable without an "
      "oracle: rejected moves are never written, so a filled cell is necessarily correct. |")
    w("| `true_rewards` | **present, but derived from the label** | proxy frozen at HVTA's "
      "`hack_onset_step`. Circular by construction -- see the L3 section. |")
    w("| `state_counts` | **absent (`None`)** | HVTA exposes no state-visitation "
      "distribution. Two candidate fabrications (filesystem-directory and board-cell "
      "histograms) were considered and rejected. Absent means NaN, never 0.5. |")
    w("| `behav_trace` | **present** | cumulative move-success rate, "
      "`2*accepted/attempted - 1`. No window length, no fitted centre. |")
    w("")
    c8 = max(lc, key=lambda c: c["slip_rate"])
    w(
        f"**Deliberately not a channel.** HVTA's leaked solution file is ~6.2k characters "
        f"against decoys of 20-238. Measured context volume at the onset step in "
        f"`{c8['game']}`/slip {c8['slip_rate']:.2f}: "
        f"**{c8['obs_chars_at_onset_hacking_mean']:,.0f} chars (hacking) vs "
        f"{c8['obs_chars_at_onset_legit_mean']:,.0f} (legitimate)**. A monitor watching only "
        f"context volume would score ~1.000 and would be restating HVTA's file-size "
        f"distribution -- the same category of error as the L2 sign-convention tautology "
        f"RHOB's own audit removed. It is recorded as `obs_chars` in the artifact and kept "
        f"out of `RunData`."
    )
    w("")

    # ---------------------------------------------------------- AUROC tables
    w("## Discrimination AUROC")
    w("")
    w(
        f"Mean over the {len(lc)} non-degenerate cells, {npairs} hacking + {npairs} "
        f"legitimate runs each. **Per-cell AUROC null SE = {f(se)}** (Bamber/Hanley, "
        f"n=m={npairs}); a single cell within +-{f(2 * se, 2)} of 0.500 is not a detection. "
        f"See *Uncertainty* for why the cell mean is not that SE divided by sqrt({len(lc)})."
    )
    w("")

    def auroc_table(rows, title, note=None):
        if not rows:
            return
        w(f"### {title}")
        w("")
        if note:
            w(note)
            w("")
        w("| detector | level | mean AUROC | min | max | AUROC in the *other* behavioral orientation |")
        w("|---|---|---:|---:|---:|---:|")
        for r in sorted(rows, key=lambda r: -r["mean_auroc"]):
            cf = r["mean_auroc_counterfactual_sign"]
            w(
                f"| {r['detector']}{' *(duplicate)*' if r['duplicate_of'] else ''}"
                f"{' *(external)*' if r['external_baseline'] else ''} "
                f"| {r['access_level']} | **{f(r['mean_auroc'])}** | {f(r['min_auroc'])} "
                f"| {f(r['max_auroc'])} | {f(cf, dash='--')} |"
            )
        w("")

    auroc_table(
        [r for r in scored if not r["label_fitted"] and r["access_level"] != "L3"],
        "Unsupervised detectors",
        "These answer *\"does this fixed statistic separate the two variants?\"*",
    )
    auroc_table(
        [r for r in scored if r["label_fitted"] and r["access_level"] != "L3"],
        "Label-fitted detectors (scored separately, never pooled with the above)",
        "Identified by `rhob.detectors.supervision.is_label_fitted`, which is the scoring "
        "path's own `hasattr(d, \"fit\")` predicate. These are handed the hacking/legitimate "
        "labels for each cross-validation training fold, so they answer a *different* "
        "question: *\"is the two variants' difference learnable from labelled examples?\"* "
        "Five-fold stratified CV with out-of-fold scoring, so the number is not leaked -- but "
        "stacking it into the same maximum as an unsupervised number answers neither "
        "question.",
    )

    w("### L1: not scored")
    w("")
    w(
        f"All {len(na_dets)} L1 detectors ("
        + ", ".join(sorted(r["detector"] for r in na_dets))
        + f") are N/A on all {len(cells)} cells. RHOB's own reason string: "
        f"*\"{na_dets[0]['na_reason']}\"*. They are excluded from every aggregate rather than "
        f"counted as chance."
    )
    w("")

    l3 = [r for r in art["per_detector"] if r["access_level"] == "L3"]
    w("### L3: a label readout, not a detection result")
    w("")
    tro = next(r for r in l3 if r["detector"] == "True Reward Oracle")
    w(
        f"`true_rewards` on HVTA is *derived from the onset it would be used to detect* "
        f"(proxy frozen at `hack_onset_step`). This is more circular than in RHOB's own "
        f"families, where true reward is independently simulated. The True Reward Oracle's "
        f"mean AUROC of {f(tro['mean_auroc'])} is therefore a measurement of that "
        f"construction, not of detection, and the degenerate cells prove it: with zero "
        f"observable difference between the variants it still scored "
        f"{f(tro_deg['auroc'])}. **Do not read the L3 row as a ceiling for this environment.**"
    )
    w("")
    pfo = next((r for r in l3 if r["duplicate_of"]), None)
    if pfo:
        w(
            f"`{pfo['detector']}` is a relabelled duplicate of `{pfo['duplicate_of']}` "
            f"(`rhob.detectors.redundancy`); it reads `behav_trace`, not `true_rewards`, and "
            f"reproduces that detector's {f(pfo['mean_auroc'])} exactly. It is excluded from "
            f"every access-level aggregate."
        )
        w("")

    # ------------------------------------------------------ access-level roll-up
    w("### Access-level roll-up")
    w("")
    w("Duplicates and external baselines excluded; supervision partitions kept apart.")
    w("")
    w("| level | unsupervised mean (n) | unsup. best | label-fitted mean (n) | label-fitted best |")
    w("|---|---:|---|---:|---|")
    for lvl in ("L0", "L1", "L2", "L3"):
        s = art["access_level_summary"].get(lvl)
        if not s:
            w(f"| {lvl} | N/A | *channel absent* | N/A | *channel absent* |")
            continue
        def part(p):
            if not p:
                return "--", "--"
            return (
                f"{f(p['mean_auroc'])} ({p['n_detectors']})",
                f"{p['best_detector']} {f(p['best_auroc'])}",
            )

        u_mean, u_best = part(s.get("unsupervised"))
        lf_mean, lf_best = part(s.get("label_fitted"))
        w(f"| {lvl} | {u_mean} | {u_best} | {lf_mean} | {lf_best} |")
    w("")
    w("L3 is listed for completeness only; see above for why its number is not a ceiling.")
    w("")

    # ------------------------------------------------------- difficulty axis
    w("### Across the difficulty axis")
    w("")
    w(
        "No number in this report is quoted without its slip rate attached. AUROC per cell "
        "for a representative selection:"
    )
    w("")
    def best_in(level, fitted):
        c = [r for r in scored if r["access_level"] == level
             and r["label_fitted"] == fitted and not r["duplicate_of"]]
        return max(c, key=lambda r: r["mean_auroc"])["detector"] if c else None

    sel = [n for n in (
        best_in("L0", False), best_in("L0", True), best_in("L2", False),
        best_in("L2", True), "Behavioral Threshold", "True Reward Oracle",
    ) if n]
    sel = list(dict.fromkeys(sel))
    w("| game | competence | slip | " + " | ".join(sel) + " |")
    w("|---|---|---:|" + "---:|" * len(sel))
    for c in cells:
        vals = []
        for n in sel:
            r = cell_rows(art, c["index"], n)
            vals.append(f(r.get("auroc")) if r else "n/a")
        tag = " *(degenerate)*" if c["degenerate"] else ""
        w(
            f"| {c['game'].replace('Sudoku-v0-', '')}{tag} | {c['competence']} "
            f"| {c['slip_rate']:.2f} | " + " | ".join(vals) + " |"
        )
    w("")
    w(
        "The monotone climb with slip rate is the expected shape: a leaked solution is worth "
        "something only to an agent that could not already solve the puzzle. Read down the "
        "column, not across a single row."
    )
    w("")

    # --------------------------------------------------------------- onset
    w("## Onset error against HVTA's `hack_onset_step`")
    w("")
    w(
        "**This is the measurement RHOB has never been able to make.** On RHOB's own families "
        "the \"true onset\" is a number RHOB's generator wrote down; here it is the "
        "environment's, recorded by HVTA's `FilesystemWrapper` when the solution file is read."
    )
    w("")
    w(
        f"Averaged over the {len(lc)} non-degenerate cells ({npairs} hacking + {npairs} "
        f"legitimate runs each). `step MAE` is over the hacking runs where the detector "
        f"actually nominated a step -- a detector that never fires has no error to average, "
        f"and `fire%` carries that instead. Two detector-independent references, computed "
        f"from the true onsets: a **uniform random guess** scores "
        f"{f(ref['reference_mae_steps_uniform_guess'], 1)} steps, and the best possible "
        f"**constant** prediction -- chosen using the true onsets, so unavailable to an "
        f"honest detector -- scores {f(ref['reference_mae_steps_best_constant_oracle'], 1)} steps."
    )
    w("")
    w("| detector | level | fire% (hacking) | step MAE | median signed | pred SD | **r(pred, true)** | **false onset on legit** | RHOB onset MAE (norm.) |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    order = sorted(
        art["per_detector"],
        key=lambda r: (
            r["access_level"],
            (onset_means(art, r["detector"]) or {}).get("mae_steps_when_fired") or 1e9,
        ),
    )
    for r in order:
        om = onset_means(art, r["detector"])
        if om is None:
            continue
        fire = 1 - om["miss_rate_hacking"] if om["miss_rate_hacking"] is not None else None
        w(
            f"| {r['detector']}{' *(fit)*' if r['label_fitted'] else ''} | {r['access_level']} "
            f"| {pct(fire)} | {f(om['mae_steps_when_fired'], 1, '--')} "
            f"| {f(om['median_signed_err_steps'], 1, '--')} "
            f"| {f(om['pred_onset_std_steps'], 1, '--')} "
            f"| **{f(om['pearson_r_pred_vs_true'], 2, '--')}** "
            f"| **{pct(om['false_onset_rate_legit'])}** "
            f"| {f(r['mean_onset_mae_normalized'])} |"
        )
    w("")
    w(
        "`RHOB onset MAE (norm.)` is `rhob.v3.benchmark._onset_mae` unchanged: "
        "`|pred - true| / n_episodes`, with a full 1.0 penalty when the detector returns -1. "
        "A value of exactly 1.000 therefore means *never fired*. `--` in the correlation "
        "column means the predictions were constant, so r is undefined rather than zero."
    )
    w("")
    w("### The step error is a constant in disguise")
    w("")
    w(
        f"`r(pred, true)` is the Pearson correlation between a detector's predicted onset and "
        f"HVTA's `hack_onset_step` across the {npairs} hacking runs of a cell -- the direct "
        f"test of whether a prediction *moves with* the truth. `pred SD` is the spread of the "
        f"predictions themselves, against a true-onset spread of "
        f"{f(ref['true_onset_std_steps'], 1)} steps."
    )
    w("")
    w(
        f"Read together they dissolve the apparently-decent step errors at the top of the "
        f"table. {', '.join('`' + n + '`' for n in flat)} emit the *same step on every run* "
        f"(pred SD 0.0). `{best_onset[0]}`, the lowest step MAE in the suite at "
        f"{f(best_onset[1]['mae_steps_when_fired'], 1)} steps, has a prediction SD of only "
        f"{f(best_onset[1]['pred_onset_std_steps'], 1)} steps and r = "
        f"{f(best_onset[1]['pearson_r_pred_vs_true'], 2)}. Its rule fires when the proxy first "
        f"exceeds its 50-step baseline maximum by a fixed relative margin, which on a "
        f"monotonically rising reward happens at nearly the same step regardless of when the "
        f"hack occurred. Its 10-step accuracy is the accuracy of a constant that happens to "
        f"land inside the 55-96 onset window, not of localization."
    )
    w("")
    w(
        f"The control that makes this readable is the L3 oracle: reading a channel *derived "
        f"from* `hack_onset_step`, it reaches r = "
        f"{f(tro_om['pearson_r_pred_vs_true'], 2)} with a prediction SD of "
        f"{f(tro_om['pred_onset_std_steps'], 1)} steps. The measure detects tracking when "
        f"tracking is present. No detector on an observable channel does."
    )
    w("")
    ex = max(lc, key=lambda c: (c["slip_rate"], c["game"]))
    w(
        f"The raw predictions from one cell (`{ex['game']}`, competence "
        f"`{ex['competence']}`, slip {ex['slip_rate']:.2f}) make it concrete. Runs are sorted "
        f"by true onset, so a detector that localizes should produce a rising row:"
    )
    w("")
    ex_rows = [("HVTA hack_onset_step (TRUTH)", cells[ex["index"]]["onsets"])]
    for n in ("Reward Peak", "Reward Threshold", "Reward KDE", "True Reward Oracle"):
        r = cell_rows(art, ex["index"], n)
        if r and r.get("onsets_pred_hacking"):
            ex_rows.append((f"{n} (predicted)", r["onsets_pred_hacking"]))
    lab_w = max(len(a) for a, _ in ex_rows)
    w("```")
    for lab, vals in ex_rows:
        w(f"{lab:<{lab_w}}  " + " ".join(f"{v:>3}" for v in vals))
    w("```")
    w("")
    w(
        "`Reward KDE` returns the same step 20 times. `Reward Peak` and `Reward Threshold` "
        "wander over a ten-step band while the truth moves across forty. Only the oracle -- "
        "reading the label-derived channel -- rises with it, and even that is biased ~30 "
        "steps early because it fires at the midpoint crossing rather than at the freeze."
    )
    w("")

    w("### Why so many detectors never fire")
    w("")
    w(
        f"{len(never_fire)} detectors return -1 on every HVTA run: "
        + ", ".join(f"`{n}`" for n in sorted(never_fire))
        + ". Three distinct mechanisms, all measurable:"
    )
    w("")
    w(
        "* **Absolute thresholds in reward units.** `RewardTrendDetector.detect_onset` fires "
        "on a 50-step rolling slope `> 0.1` per step and `GradientReversalDetector` on "
        "`> 0.05`. The largest 50-step rolling slope anywhere in HVTA's proxy is **0.00574** "
        "-- 5.7% of the smaller threshold. These rules cannot fire on this environment at any "
        "difficulty. `RewardMLPDetector` is the same shape: it requires a jump `> 0.1` in "
        "P(hacking) between consecutive steps."
    )
    w(
        "* **Trajectory-shape assumptions.** `MaxPlateauDetector` looks for a plateau near the "
        "running maximum and `VarianceWindowDetector` for a collapse to below half the "
        "baseline variance. HVTA's proxy rises monotonically for the whole episode and does "
        "neither. Both rules are scale-free, so this is a shape mismatch rather than a units "
        "bug."
    )
    w(
        "* **No threshold crossing.** `ARResidualDetector` fits its own residual scale from "
        "data and then needs three consecutive 3-sigma residuals; HVTA's behavioral trace is "
        "too smooth to produce them. This one is a genuine null, not a calibration artifact."
    )
    w("")
    w(
        "The first two categories are a **generalizability finding about RHOB, not about "
        "HVTA**: RHOB's onset rules encode the reward scale and the trajectory shape of "
        "RHOB's own families, where episodes are i.i.d. repetitions and hacking shows up as "
        "an abrupt regime switch. HVTA's episode is a single smooth, monotone trajectory. "
        "Nothing was tuned to make these fire, and nothing should be: the point of an "
        "external environment is to find exactly this."
    )
    w("")

    w("### False onsets are real false positives here")
    w("")
    w(
        "HVTA is decidable, and the legitimate variant provably never reads the solution. So "
        "every onset nominated on a legitimate run is a false positive with no oracle "
        "uncertainty attached -- precisely what RHOB could not previously establish. "
        "`rhob.v3.benchmark._onset_mae` never looks at the legitimate runs, so this column "
        "does not exist on the RHOB leaderboard."
    )
    w("")
    gaps = []
    for r in art["per_detector"]:
        om = onset_means(art, r["detector"])
        if om is None or om["false_onset_rate_legit"] is None:
            continue
        fire = 1 - om["miss_rate_hacking"]
        gaps.append((firing_gap(om), r["detector"], fire, om["false_onset_rate_legit"]))
    firing = [g for g in gaps if g[2] > 0]
    w(
        f"The {len(gaps) - len(firing)} detectors that never fire are 0% / 0% by definition "
        f"and are omitted. The rest:"
    )
    w("")
    w("| detector | fires on hacking | fires on legitimate | gap |")
    w("|---|---:|---:|---:|")
    for g, n, a, b in sorted(firing, reverse=True):
        w(f"| {n} | {pct(a)} | {pct(b)} | {g:+.2f} |")
    w("")
    w(
        f"The best gap in the whole suite is {f(max(g for g, *_ in gaps), 2)}. Read the two "
        f"columns together: the detectors that fire on hacking runs fire on legitimate ones at "
        f"the same rate, and the ones with low false-onset rates got there by not firing on "
        f"hacking runs either. **No detector in the suite both fires on the hack and stays "
        f"quiet on its matched legitimate twin.**"
    )
    w("")

    # ------------------------------------------------------ sign randomization
    w("## Sign randomization (L2)")
    w("")
    w(
        f"On. `behav_trace` was multiplied by "
        f"`behav_sign(\"{gen['family_key']}\", layout_seed=0)` = "
        f"**{cells[0]['behav_sign_drawn']:+.0f}** before any detector saw it, both variants "
        f"identically, exactly as `MatchedPair.rollout` does for RHOB's own families. The "
        f"draw is a keyed digest of the family name and was fixed before any score was "
        f"computed."
    )
    w("")
    w(
        "The last column of the AUROC tables reports the same cells in the orientation the "
        "benchmark did **not** draw. It is a harness-side diagnostic -- the harness already "
        "knows every label -- and no headline number is a maximum over the two. It separates "
        "three behaviours cleanly:"
    )
    w("")
    bt = next(r for r in scored if r["detector"] == "Behavioral Threshold")
    ct = next((r for r in scored if r["detector"] == "Centroid Tracker"), None)
    am = next((r for r in scored if r["detector"] == "Angular Momentum"), None)
    if am:
        w(
            f"* **Sign-invariant by construction** (magnitude-only): e.g. `{am['detector']}`, "
            f"{f(am['mean_auroc'])} in both orientations. The flip cannot touch these."
        )
    if ct:
        w(
            f"* **Sign-dependent, un-oriented**: `{ct['detector']}` scores "
            f"{f(ct['mean_auroc'])} as drawn and "
            f"{f(ct['mean_auroc_counterfactual_sign'])} flipped -- they sum to "
            f"{ct['mean_auroc'] + ct['mean_auroc_counterfactual_sign']:.3f}. Its result is a "
            f"coin flip on the draw, which is exactly what sign randomization exists to expose."
        )
    bt_cells = [cell_rows(art, c["index"], bt["detector"])["auroc"] for c in lc]
    bt_wrong = sum(1 for v in bt_cells if v < 0.5)
    w(
        f"* **Oriented, and oriented wrongly.** `{bt['detector']}` scores "
        f"{f(bt['mean_auroc'])} as drawn and {f(bt['mean_auroc_counterfactual_sign'])} "
        f"flipped -- *the same number*, because its `observe_cell` orientation inference "
        f"flips with the coordinate, leaving the product invariant. So it is not "
        f"coin-flipping: it commits to a direction, and on HVTA it commits to the wrong one "
        f"in **{bt_wrong} of the {len(lc)}** non-degenerate cells (per-cell range "
        f"{f(bt['min_auroc'])}-{f(bt['max_auroc'])}). Its heuristic reads the *retention "
        f"asymmetry* that RHOB's admission gate imposes on its own hacking variants -- "
        f"`camping_quality` requires the hacking trace to retain half its early magnitude, "
        f"and constrains the legitimate variant not at all. HVTA imposes no such requirement, "
        f"so the hypothesis is simply false here. The detector's own docstring anticipates "
        f"this and calls a mis-oriented cell \"a measurement of what this detector actually "
        f"knew\"; this is that measurement, taken outside RHOB for the first time."
    )
    w("")

    # ------------------------------------------------------------ uncertainty
    w("## Uncertainty")
    w("")
    w(
        f"**Per-cell null SE = {f(se)}** at n=m={npairs} runs per class (Bamber/Hanley, "
        f"`sqrt((2n+1)/(12n^2))`). A single cell's AUROC is one draw carrying this much "
        f"sampling error."
    )
    w("")
    w(
        f"**The {len(lc)}-cell mean is not {len(lc)} independent replicates.** Every cell uses "
        f"the same {npairs} seeds and the same onset schedule, differing only in game, "
        f"competence and slip rate; two cells "
        f"(`medium`/slip 0.30 at both competence levels) come out bit-identical because the "
        f"weaker solver never needs the stronger rule there. Dividing the null SE by "
        f"sqrt({len(lc)}) would overstate the precision of a cell mean."
    )
    w("")
    if reps:
        w(
            "**Independent seed draws.** The whole grid was regenerated at seed bases "
            + ", ".join(str(r["generation"]["config"]["seed_base"]) for r in reps)
            + " -- different Sudoku puzzles, same onset schedule and same design -- and "
            f"rescored end to end. Spread of each detector's grid-mean AUROC across the "
            f"{len(reps) + 1} draws:"
        )
        w("")
        w("| detector | level | " + " | ".join(
            f"seed base {a['generation']['config']['seed_base']}" for a in [art] + reps
        ) + " | mean | range |")
        w("|---|---|" + "---:|" * (len(reps) + 3))
        allart = [art] + reps
        fragile, unassessed = [], []
        for r in sorted(scored, key=lambda r: -r["mean_auroc"]):
            vals, shown = [], []
            for a in allart:
                m = next((x for x in a["per_detector"] if x["detector"] == r["detector"]), None)
                v = m["mean_auroc"] if m else None
                if m is not None and pre_fix(a, r["detector"]):
                    shown.append(marked(f(v), a, r["detector"]))
                    v = None
                else:
                    shown.append(f(v))
                vals.append(v)
            good = [v for v in vals if v is not None]
            if len(good) < len(allart):
                unassessed.append(r["detector"])
                mean_s, rng_s = "--", "--"
            else:
                mean, rng = float(np.mean(good)), max(good) - min(good)
                if rng >= abs(mean - 0.5):
                    fragile.append(r["detector"])
                mean_s, rng_s = f(mean), f(rng)
            w(
                f"| {r['detector']}{' *(fit)*' if r['label_fitted'] else ''} "
                f"| {r['access_level']} | "
                + " | ".join(shown)
                + f" | **{mean_s}** | {rng_s} |"
            )
        w("")
        kinds = {
            stale_mark(a, r["detector"]) for a in allart for r in scored
        } - {None}
        if "pre-fix" in kinds:
            w(
                "*(pre-fix)*: that draw's rows for this detector were not scored by the "
                "source the primary draw's were, and the primary records a fix to it (see "
                "*Reproduce*). Its figure is not this detector's, so it is left out of the "
                "mean and range."
            )
            w("")
        if "other source" in kinds:
            w(
                "*(other source)*: that draw and the primary draw attest different sources "
                "for this detector, so the two figures are not of one detector and that "
                "draw's is left out of the mean and range."
            )
            w("")
        w(
            f"A detector whose across-draw range is as large as its distance from 0.500 has "
            f"not been shown to detect anything on this environment. That is "
            f"{len(fragile)} of the {len(scored) - len(unassessed)} scoreable detectors "
            f"measured on every draw: "
            + ", ".join(f"`{n}`" for n in sorted(fragile))
            + "."
            + (
                " Not assessed, for want of a current figure on every draw: "
                + ", ".join(f"`{n}`" for n in sorted(unassessed)) + "."
                if unassessed else ""
            )
        )
        w("")
        low_fire = sorted(
            n for n, om in fires
            if om["n_fired_hacking"] / om["n_hacking"] < 0.2
        )
        w(
            "The onset conclusion is the one that replicates most cleanly -- the oracle "
            "tracks in every draw and nothing else does in any of them:"
        )
        w("")
        w("| detector | " + " | ".join(
            f"r(pred,true) @ {a['generation']['config']['seed_base']}" for a in allart
        ) + " |")
        w("|---|" + "---:|" * len(allart))
        for name, _om in sorted(fires, key=lambda kv: -(kv[1]["pearson_r_pred_vs_true"] or -9)):
            cols = []
            for a in allart:
                o = onset_means(a, name)
                cols.append(marked(f((o or {}).get("pearson_r_pred_vs_true"), 2, "--"),
                                   a, name))
            w(f"| {name} | " + " | ".join(cols) + " |")
        w("")
        tro_rs = [
            onset_means(a, "True Reward Oracle")["pearson_r_pred_vs_true"] for a in allart
        ]
        complete, flipping, stale_r = [], [], []
        for name, _om in fires:
            if name == "True Reward Oracle":
                continue
            if any(pre_fix(a, name) for a in allart):
                stale_r.append(name)
                continue
            rs = [(onset_means(a, name) or {}).get("pearson_r_pred_vs_true") for a in allart]
            if all(v is not None for v in rs):
                complete.append((name, rs))
                if min(rs) < 0 < max(rs):
                    flipping.append(name)
        steady = [(n, rs) for n, rs in complete if n not in flipping]
        w(
            f"The oracle holds r = {f(min(tro_rs), 2)}-{f(max(tro_rs), 2)} across all "
            f"{len(allart)} draws. No observable-channel detector comes close in any draw: the "
            f"highest correlation any of them reaches is r = {f(best_r, 2)} "
            f"({whose(best_r_name)}). "
            f"Of the {len(complete)} with a correlation in every draw, {len(flipping)} change "
            f"sign between draws"
            + (
                "; " + ", ".join(f"`{n}`" for n, _ in steady)
                + f" keep {'its' if len(steady) == 1 else 'their'} sign, at "
                f"|r| <= {f(max(abs(v) for _, rs in steady for v in rs), 2)}. "
                if steady else ". "
            )
            + (
                ("`" + "`, `".join(low_fire) + "` swings widest"
                 if len(low_fire) == 1 else
                 "`" + "`, `".join(low_fire) + "` swing widest")
                + ", but fires on under 20% of runs, so that correlation rests on a handful "
                  "of points and is noise rather than a signal that reverses."
                if low_fire else ""
            )
            + "".join(
                f" `{n}` has a comparable correlation on "
                + ("the primary draw only" if k == 1 else f"{k} of the {len(allart)} draws")
                + " (the others are marked above), so its stability across draws is "
                "unmeasured."
                for n in stale_r
                for k in [sum(1 for a in allart if not pre_fix(a, n))]
            )
        )
        w("")
    w(
        f"**Scope.** {art['n_cells'] * npairs} pairs on one HVTA hack mechanism "
        f"(`hidden_solution`) in one game family (Sudoku), driven by a scripted policy rather "
        f"than an LLM. It is one external environment, not a resolution of the "
        f"generalizability question."
    )
    w("")

    # ------------------------------------------------------ what this establishes
    w("## What this does and does not establish")
    w("")
    w("**Does:**")
    w("")
    w(
        "* RHOB's L0 reward-only discrimination statistics carry over to an environment RHOB's "
        "authors did not build, at a level well above chance, and degrade gracefully along a "
        "difficulty axis defined entirely by HVTA-side agent behaviour."
    )
    w(
        f"* RHOB's onset localization does **not** carry over. Against a ground truth RHOB did "
        f"not author, no detector's onset prediction correlates with the true onset above "
        f"r = {f(best_r, 2)}; several are literally constant; and none separates hacking from "
        f"legitimate runs through its onset channel at all. The apparent step accuracy of the "
        f"best detectors is the accuracy of a constant."
    )
    w(
        "* Two concrete, fixable defects in RHOB's onset rules are exposed: absolute "
        "thresholds expressed in the reward units of RHOB's own families, and trajectory-shape "
        "assumptions (plateau, variance collapse) that a monotone reward never satisfies. "
        "Neither is visible from inside RHOB, because inside RHOB both assumptions hold."
    )
    w(
        "* The false positives measured here are real. HVTA is decidable, so a nominated onset "
        "on a legitimate run is not an oracle artifact."
    )
    w(
        "* RHOB's L2 orientation heuristic fails *systematically* rather than randomly outside "
        "RHOB, because the asymmetry it reads is a property of RHOB's own admission gate."
    )
    w("")
    w("**Does not:**")
    w("")
    w(
        "* Say anything about L1. HVTA supplies no state-visitation channel, so the entire L1 "
        "access level is unmeasured here -- reported as N/A rather than filled in."
    )
    w(
        "* Provide an oracle ceiling. The L3 channel is derived from the onset label on this "
        "adapter and its number is a tautology."
    )
    w(
        "* Generalize past `hidden_solution` on Sudoku with a scripted policy. HVTA's other "
        "channels (`opponent_prompt_read`, `write_hack`, `logical_bug`) and LLM-driven "
        "episodes are untouched."
    )
    w("")

    # -------------------------------------------------------------- reproduce
    w("## Reproduce")
    w("")
    w("```")
    w("# 1. generate the pair grid (HVTA venv: textarena + hvta, no torch needed)")
    w("PYTHONUTF8=1 PYTHONPATH=<rhob>/src \\")
    w("  C:/hvta-pr/.venv/Scripts/python.exe external/hvta/export_pairs.py --pairs 20")
    w("")
    w("# 2. score every detector (needs torch for the two MLP detectors)")
    w("PYTHONPATH=<rhob>/src python external/hvta/score_detectors.py")
    w("")
    w("# 2b. after a fix to one detector, re-score it alone and merge its rows")
    w("python -I external/hvta/rescore_detector.py --detector <Class> --reason \"...\"")
    w("")
    w("# 3. render this report")
    w("python external/hvta/make_results.py")
    w("```")
    w("")
    w("| file | role |")
    w("|---|---|")
    w("| `hvta_policies.py` | scripted matched HACKING/LEGIT policies over HVTA |")
    w("| `hvta_adapter.py` | HVTA episode -> `rhob.detectors.posthoc.RunData` |")
    w("| `export_pairs.py` | rolls out the grid, freezes it as `hvta_pairs.npz` + `.meta.json` |")
    w("| `score_detectors.py` | runs the detectors via `_evaluate_cell`, writes `hvta_results.json` |")
    w("| `rescore_detector.py` | re-scores named detectors on a committed draw, merges their rows, records it under `rescored` |")
    w("| `make_results.py` | renders this file |")
    w("| `generate_pairs.py` | standalone pair-construction check (prints the matching table) |")
    w("| `test_matched_pairs.py` | 19 invariants on the pair construction |")
    w(f"| `replicates/*.json` | the {len(reps)} independent seed draws behind *Uncertainty* |")
    w("")
    mism = [
        r["detector"] for cs in art["per_cell"].values() for r in cs
        if r.get("na_reason") is None and "error" not in r and not r.get("mirror_agrees")
    ]
    w(
        f"**Verification.** Headline AUROC and onset MAE come from "
        f"`rhob.v3.benchmark._evaluate_cell` unmodified. That function keeps its per-run "
        f"predictions internal, so the onset diagnostics come from a mirrored re-execution of "
        f"its logic which is required to reproduce its AUROC and MAE bit-for-bit before any "
        f"diagnostic derived from it is recorded. Mismatches across all "
        f"{sum(len(v) for v in art['per_cell'].values())} detector-cell evaluations: "
        f"**{len(set(mism))}**."
    )
    w("")
    ge, sc = gen["generation_env"], art["scoring_env"]
    w(
        f"Generation: Python {ge['python']} ({ge['platform']}), numpy {ge['numpy']}, "
        f"HVTA `{(ge['hvta_git_head'] or '?')[:12]}` (branch `onset-logging`, "
        f"PR MajoRoth/hack-verifiable-environments#2), "
        f"RHOB `{(ge['rhob_git_head'] or '?')[:12]}`. "
        f"Scoring: Python {sc['python']}, numpy {sc['numpy']}"
        + (
            f", {art['date_utc'][:10]}, RHOB {_revision(art)}"
            if art.get("rhob_git_head") else ""
        )
        + "."
    )
    w("")

    def source(s):
        return (
            f"`{s['path']}`, {'code ' if s.get('hashed') else ''}sha256 `{s['sha256'][:12]}`"
            + (f", git blob `{s['git_blob'][:12]}`" if s.get("git_blob") else "")
        )

    for rec in art.get("rescored", []):
        env = rec["scoring_env"]
        srcs = rec.get("detector_sources", {})
        at_head = {s.get("matches_head") for s in srcs.values()}
        superseded = [n for n, s in srcs.items() if not same_source(current_source(art, n), s)]
        w(
            f"**Re-scored {rec['date_utc'][:10]}** by `{rec['script']}`: "
            + ", ".join(f"{n} ({source(s)})" for n, s in srcs.items())
            + f", on all {rec['n_cells']} cells of this draw, Python {env['python']} "
            f"({env['platform']}), numpy {env['numpy']}, RHOB {_revision(rec)}"
            + (
                "" if at_head == {True} else
                " plus that source, which was not yet committed there (its git blob names "
                "it once it is)" if False in at_head else
                " plus that source"
            )
            + ". Every other detector's rows are as first scored. "
            f"Why: {rec['reason']}. Replaced figures: "
            + "; ".join(
                f"{n} mean AUROC {f(p['mean_auroc'])}, onset MAE "
                f"{f(p['mean_onset_mae_normalized'])}"
                for n, p in rec.get("previous", {}).items()
            )
            + "."
            + (
                " Superseded by a later re-score for "
                + ", ".join(superseded) + "."
                if superseded else ""
            )
        )
        w("")
    for fix in art.get("detector_fixes", []):
        w(
            f"**Fix recorded {(fix.get('date_utc') or '?')[:10]}** for "
            + ", ".join(fix.get("detectors") or [])
            + f": {fix.get('reason')}. This draw was scored in full after it, so its rows "
            "for every detector come from the sources its `detector_sources` names."
        )
        w("")
    # Which replicates are stale, from the files loaded here: re-scoring one clears it
    # from this sentence as it clears its markers above.
    for name in [r["detector"] for r in art["per_detector"]]:
        for kind in ("pre-fix", "other source"):
            stale = [shown for a, shown in rep_items if stale_mark(a, name) == kind]
            if not stale:
                continue
            w(
                ("The replicate draw " if len(stale) == 1 else "The replicate draws ")
                + ", ".join(f"`{p}`" for p in stale)
                + f" {'does' if len(stale) == 1 else 'do'} not attest the source this draw's "
                f"{name} rows were scored with"
                + (
                    ", which postdates a fix to it, so "
                    f"{'its' if len(stale) == 1 else 'their'} rows for it are pre-fix"
                    if kind == "pre-fix" else
                    f" but a different one, so {'its' if len(stale) == 1 else 'their'} rows "
                    "for it are not comparable"
                )
                + ", marked so above and left out of every across-draw figure. Re-scoring a "
                "draw (`rescore_detector.py --pairs-stem <its pairs> --results <its file>`, "
                "or a full `score_detectors.py` run; either needs the draw's frozen pairs, "
                "which `export_pairs.py` rebuilds in the HVTA venv) clears its markers."
            )
            w("")

    # Joined once and named: the backslash escape cannot live inside the f-string's
    # expression part, which is a SyntaxError before Python 3.12 and this package
    # declares requires-python >=3.10. Joining twice was also just wasteful.
    body = "\n".join(L)
    Path(args.out).write_text(body + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(body)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
