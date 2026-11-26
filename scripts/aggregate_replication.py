"""Combine leaderboard replicates into intervals, and say which reported gaps survive.

Input: ``results/replication/replicate_*.json`` from ``scripts/replicate_leaderboard.py``,
each one a full leaderboard run at an independent ``(layout_seed, seed_base)`` draw.

The unit of resampling is the **replicate**, not the cell and not the run. Replicates are
the independent draws of the whole pipeline -- environment layout, behavioral orientation
and run seeds all resample together -- so a percentile bootstrap over them is an interval
on "what would this number be if the benchmark had been drawn again", which is the
question a reader of the leaderboard is actually asking. Bootstrapping cells instead
would treat 123 cells of one draw as 123 independent observations; they are not, they
share a layout and an orientation per family.

The access-level ladder is recomputed per replicate through
:func:`~rhob.v3.leaderboard.access_summary.summarize_access_levels` rather than
reimplemented here, so the interval obeys exactly the published ladder's rules (duplicate
detectors held out, ledger-degenerate families held out of L0 and only L0). An interval
computed under different rules than the point estimate it decorates is worse than none.

Two guards, because a replication study's characteristic failure is reporting precision
it does not have:

  * **Global degeneracy is fatal.** If no detector's score varies across replicates, the
    replicates are not independent draws -- almost certainly a rollout cache serving one
    draw to all of them (see ``tests/test_v3/test_replication_draws_differ.py``). Every
    interval would be zero-width and look like extraordinary precision. This exits
    non-zero rather than writing an artifact.
  * **Local degeneracy is reported, not fatal.** A single detector scoring 1.000 on every
    replicate is a real finding -- saturation -- and its zero-width interval is honest.
    These are counted and listed so saturation is never mistaken for precision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rhob.detectors.supervision import partition_by_supervision  # noqa: E402
from rhob.v3.leaderboard.access_summary import (  # noqa: E402
    ACCESS_LEVELS,
    degenerate_families_from_ledger,
    summarize_access_levels,
)

BOOTSTRAP_RESAMPLES = 10_000
CI_LEVEL = 0.95
#: Fixed so the published interval is reproducible from the same replicate files.
BOOTSTRAP_SEED = 20260804


def bootstrap_ci(values: list[float], key: str) -> dict:
    """Percentile bootstrap of the mean of ``values``, resampling replicates.

    With R replicates the bootstrap can only resolve the interval to ~1/R granularity;
    it does not manufacture information that R draws do not contain. ``n_replicates`` is
    reported alongside every interval so that limit stays visible.

    ``key`` names the quantity, and its hash seeds this call's generator. Every interval
    is therefore a deterministic function of (its own values, its own name) and nothing
    else. The obvious alternative -- one shared ``Generator`` threaded through every call
    -- makes each interval depend on how many bootstraps ran *before* it, so adding a new
    reported quantity silently moves every previously published one. That happened here:
    adding the mean-column and unsupervised-partition ladders shifted L0's suite-mean
    interval from [0.493366, 0.499950] to [0.493327, 0.499889] while the underlying 20
    draws were untouched, and a paper citing the file printed the superseded pair.
    Nothing was wrong with either number; they were answers to "what does the 10,000-draw
    resample say" at two different points in one RNG stream. A published interval must not
    move because an unrelated statistic was added next to it.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": None, "sd": None, "ci_lo": None, "ci_hi": None, "n_replicates": 0}
    if n == 1:
        # One replicate is the single-draw regime this study exists to replace. Report
        # the point and refuse to decorate it with an interval.
        return {
            "mean": float(arr[0]), "sd": None, "ci_lo": None, "ci_hi": None,
            "n_replicates": 1,
            "note": "single replicate -- no interval is estimable from one draw",
        }
    seed = (BOOTSTRAP_SEED + int(hashlib.blake2b(key.encode(), digest_size=8).hexdigest(), 16)) % (2**63)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    means = arr[idx].mean(axis=1)
    lo_q = (1 - CI_LEVEL) / 2 * 100
    return {
        "mean": float(arr.mean()),
        "sd": float(statistics.stdev(arr.tolist())),
        "ci_lo": float(np.percentile(means, lo_q)),
        "ci_hi": float(np.percentile(means, 100 - lo_q)),
        "n_replicates": n,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def load_replicates(in_dir: Path) -> list[dict]:
    files = sorted(in_dir.glob("replicate_*.json"))
    if not files:
        raise SystemExit(f"no replicate_*.json found in {in_dir}")
    reps = [json.loads(p.read_text()) for p in files]
    draws = [(r["layout_seed"], r["seed_base"]) for r in reps]
    if len(set(draws)) != len(draws):
        raise SystemExit(
            f"replicates share a draw: {draws}. These are not independent samples and "
            f"pooling them would understate the interval."
        )
    return reps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=Path("results/replication"))
    parser.add_argument("--out", type=Path, default=Path("leaderboard/v5_replicated.json"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/figures/replication_summary.md"))
    args = parser.parse_args()

    reps = load_replicates(args.in_dir)
    print(f"loaded {len(reps)} replicates from {args.in_dir}")

    # ---- per-detector overall AUROC ------------------------------------------------
    detector_names = sorted({n for r in reps for n in r["results"]})
    per_detector: dict[str, dict] = {}
    saturated: list[str] = []
    for name in detector_names:
        vals, level = [], None
        cell_counts: set[int] = set()
        for r in reps:
            rec = r["results"].get(name)
            if rec and rec.get("overall_auroc") is not None:
                vals.append(float(rec["overall_auroc"]))
                level = rec.get("access_level", level)
                if rec.get("cells_measured") is not None:
                    cell_counts.add(int(rec["cells_measured"]))
        ci = bootstrap_ci(vals, f"detector::{name}")
        ci["access_level"] = level
        # The denominator, carried through from the replicates so a consumer can print
        # it next to the interval. Without it a board-wide detector (123 cells over 33
        # families) and one whose channel most families do not ship (31 over 7) render
        # as if they had answered the same question -- which is exactly the reading the
        # L1 imputation finding came from. Constant across draws by construction (every
        # draw runs the same suite), so a disagreement means the draws are not
        # comparable and this reports None rather than a figure true of only some.
        ci["cells_measured"] = next(iter(cell_counts)) if len(cell_counts) == 1 else None
        # Per-family intervals, so a consumer can show a cell's uncertainty rather than
        # only the detector's board-wide figure. The leaderboard UI is the reason: it
        # filters by family, and a per-family point estimate with no interval is exactly
        # the single-draw presentation this study exists to replace.
        # A family is included only where the detector actually measured it -- a family
        # whose channel the detector cannot read contributes no cells and must not appear
        # as a number (see the L1 imputation finding).
        fam_vals: dict[str, list[float]] = {}
        for r in reps:
            rec = r["results"].get(name) or {}
            for fam, v in (rec.get("per_family") or {}).items():
                if v is not None:
                    fam_vals.setdefault(fam, []).append(float(v))
        ci["per_family"] = {
            fam: bootstrap_ci(vs, f"detector::{name}::family::{fam}")
            for fam, vs in sorted(fam_vals.items())
        }
        per_detector[name] = ci
        if ci.get("sd") == 0.0:
            saturated.append(name)

    varying = [n for n, c in per_detector.items() if (c.get("sd") or 0) > 0]
    if len(reps) > 1 and not varying:
        raise SystemExit(
            "FATAL: no detector's score varies across replicates. The replicates are not "
            "independent draws -- check that (layout_seed, seed_base) reach the "
            "simulation and appear in Benchmark._rollout_cache's key. Every interval "
            "would be zero-width. Refusing to write an artifact."
        )

    # ---- access-level ladder, recomputed per replicate under the published rules ----
    degenerate = degenerate_families_from_ledger()
    # Keyed by replicate id, NOT appended to flat lists. Flat lists lose the pairing the
    # moment any level is unscoreable on any replicate: if L1 is None on replicate 3 and
    # L2 is None on replicate 7, both lists still have length R-1, a length check still
    # passes, and element i of one no longer refers to the same draw as element i of the
    # other -- so the paired difference below would silently subtract across draws.
    def build_ladder(drop: frozenset[str] = frozenset()):
        """Per-replicate ladder, optionally excluding a set of detectors by name."""
        by_rep: dict[str, dict[str, dict[int, float]]] = {
            lv: {"mean": {}, "max": {}} for lv in ACCESS_LEVELS
        }
        counts: dict[str, dict[str, int]] = {lv: {} for lv in ACCESS_LEVELS}
        for r in reps:
            rid = r["replicate_id"]
            res = {k: v for k, v in r["results"].items() if k not in drop}
            for lv, s in summarize_access_levels(res, degenerate_families=degenerate).items():
                if s.mean_auroc is not None:
                    by_rep[lv]["mean"][rid] = s.mean_auroc
                if s.max_auroc is not None:
                    by_rep[lv]["max"][rid] = s.max_auroc
                if s.best_detector:
                    counts[lv][s.best_detector] = counts[lv].get(s.best_detector, 0) + 1
        return by_rep, counts

    # A detector exposing fit() is handed the labels for each CV training fold
    # (rhob.v3.benchmark._evaluate_cell). Cross-validation keeps that honest, but it
    # answers "is the difference learnable from labelled examples?" rather than "does
    # this fixed statistic separate the variants?". Mixing both into one per-level
    # maximum reports whichever question produced the larger number. On this board that
    # is decisive: the entire published L0->L1 step is one label-fitted detector, and
    # the step reverses sign without it. See rhob.detectors.supervision.
    _unsup, _fitted = partition_by_supervision(sorted({n for r in reps for n in r["results"]}))
    FITTED = frozenset(_fitted)
    ladder_by_rep, best_counts = build_ladder()
    unsup_by_rep, unsup_counts = build_ladder(FITTED)

    def _vals(lv: str, stat: str) -> list[float]:
        """Values ordered by replicate id, for the unpaired per-level intervals."""
        return [ladder_by_rep[lv][stat][k] for k in sorted(ladder_by_rep[lv][stat])]

    def _uvals(lv: str, stat: str) -> list[float]:
        return [unsup_by_rep[lv][stat][k] for k in sorted(unsup_by_rep[lv][stat])]

    ladder_unsup = {
        lv: {
            "mean_auroc": bootstrap_ci(_uvals(lv, "mean"), f"unsup::{lv}::mean"),
            "max_auroc": bootstrap_ci(_uvals(lv, "max"),  f"unsup::{lv}::max"),
            "best_detector_frequency": dict(
                sorted(unsup_counts[lv].items(), key=lambda kv: -kv[1])
            ),
        }
        for lv in ACCESS_LEVELS
    }

    ladder = {
        lv: {
            "mean_auroc": bootstrap_ci(_vals(lv, "mean"), f"ladder::{lv}::mean"),
            "max_auroc": bootstrap_ci(_vals(lv, "max"),  f"ladder::{lv}::max"),
            # How often each detector was the best at its level. A level whose "best
            # detector" changes between draws does not have a stable best detector, and
            # the published single-draw winner was a coin flip.
            "best_detector_frequency": dict(
                sorted(best_counts[lv].items(), key=lambda kv: -kv[1])
            ),
        }
        for lv in ACCESS_LEVELS
    }

    # ---- do adjacent rungs actually separate? --------------------------------------
    # The headline claim is an ordering over access levels. Test it the way it is used:
    # paired by replicate (each draw scores every level), so the comparison is not
    # confounded by between-draw variation shared by both rungs.
    # BOTH statistics, always. Reporting only one is how a one-sided conclusion gets
    # published: on this board both statistics separate L1->L2, in opposite directions.
    # The suite mean puts L2 above L1 (+0.012, CI excludes zero); the best-detector max
    # puts L1 above L2 (-0.055, CI excludes zero, 0/20 draws favouring L2). Either alone
    # reads as a settled answer; together they are the actual finding, which is that the
    # rung's direction depends on the statistic. A verdict is therefore the pair (does it
    # separate, which way): ``disagree`` below lists rungs that separate under one
    # statistic and not the other, ``disagree_direction`` rungs that separate under both
    # with opposite signs -- which is where this one is listed. Comparing only whether
    # each statistic separates printed "Max and mean agree on every rung" on this board.
    # The max is additionally a selection statistic -- the maximum over a level's
    # detectors exceeds the truth by construction -- and at L0, where every detector is
    # at chance, the argmax is a coin flip (see best_detector_frequency).
    def compute_separations(by_rep, tag: str):
        seps = {}
        for stat in ("max", "mean"):
            for lo, hi in zip(ACCESS_LEVELS, ACCESS_LEVELS[1:]):
                lo_by_rep, hi_by_rep = by_rep[lo][stat], by_rep[hi][stat]
                shared = sorted(set(lo_by_rep) & set(hi_by_rep))
                if not shared:
                    continue
                diffs = np.array([hi_by_rep[k] - lo_by_rep[k] for k in shared], dtype=float)
                d_ci = bootstrap_ci(diffs.tolist(), f"{tag}::{hi}_minus_{lo}::{stat}")
                seps[f"{hi}_minus_{lo}_{stat}"] = {
                    **d_ci,
                    "statistic": stat,
                    "excludes_zero": bool(
                        d_ci["ci_lo"] is not None and (d_ci["ci_lo"] > 0 or d_ci["ci_hi"] < 0)
                    ),
                    "replicates_with_hi_above_lo": int((diffs > 0).sum()),
                    # Replicates where BOTH rungs were scoreable. Reported because it
                    # can be smaller than n_replicates, and a paired comparison over a
                    # subset of draws is a different claim than one over all of them.
                    "n_paired": len(shared),
                    # Pairing removes the between-draw variation the two rungs share.
                    # It does NOT equalise denominators: within one replicate L1
                    # aggregates 31 cells over 7 families while L0 aggregates at most 93
                    # over 27 (its detectors are scored on 8 to 27 families) and L2/L3
                    # at most 123 over 33. No resampling fixes that, so any rung
                    # difference involving L1 is across populations as well as across
                    # access levels.
                    "denominators_equal": False,
                }
        return seps

    separations = compute_separations(ladder_by_rep, "all")
    separations_unsup = compute_separations(unsup_by_rep, "unsup")
    rungs = [
        (f"{hi}_minus_{lo}", separations[f"{hi}_minus_{lo}_max"],
         separations[f"{hi}_minus_{lo}_mean"])
        for lo, hi in zip(ACCESS_LEVELS, ACCESS_LEVELS[1:])
        if f"{hi}_minus_{lo}_max" in separations and f"{hi}_minus_{lo}_mean" in separations
    ]
    disagree = [r for r, mx, mn in rungs if mx["excludes_zero"] != mn["excludes_zero"]]
    disagree_direction = [
        r for r, mx, mn in rungs
        if mx["excludes_zero"] and mn["excludes_zero"] and (mx["mean"] > 0) != (mn["mean"] > 0)
    ]

    out = {
        "n_replicates": len(reps),
        "draws": [{"replicate_id": r["replicate_id"], "layout_seed": r["layout_seed"],
                   "seed_base": r["seed_base"], "n_seeds": r["n_seeds"]} for r in reps],
        "method": {
            "resampling_unit": "replicate",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "ci_level": CI_LEVEL,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "interval_type": "percentile bootstrap of the mean over replicates",
        },
        "degeneracy": {
            "detectors_with_zero_variance": sorted(saturated),
            "n_detectors_with_zero_variance": len(saturated),
            "note": (
                "Zero variance across replicates means the detector returned the same "
                "score on every independent draw -- saturation (typically 1.000 or "
                "0.500), not measurement precision. Its interval is a point by "
                "construction and must not be read as a tight estimate."
            ),
        },
        "supervision": {
            "label_fitted_detectors": sorted(FITTED),
            "why_it_matters": (
                "A detector exposing fit() is handed the hacking/legitimate labels for "
                "each cross-validation training fold. That is a legitimate measurement "
                "but a different question from an unsupervised post-hoc detector's: "
                "'is the difference learnable from labelled examples?' rather than 'does "
                "this fixed statistic separate the variants?'. A per-level maximum over "
                "both reports whichever question produced the larger number."
            ),
            "access_levels_unsupervised_only": ladder_unsup,
            "adjacent_level_separation_unsupervised_only": separations_unsup,
        },
        "access_levels": ladder,
        "adjacent_level_separation": separations,
        "statistic_disagreement": {
            "rungs_where_max_and_mean_disagree_on_separation": disagree,
            "rungs_where_max_and_mean_disagree_on_direction": disagree_direction,
            "note": (
                "The max and the mean give these rungs different verdicts: a rung listed "
                "under separation separates under one statistic and not the other, and a "
                "rung listed under direction separates under both with opposite signs. "
                "Reporting either alone would present a statistic-dependent result as a "
                "settled one. Quote both."
            ),
        },
        "detectors": per_detector,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[OK] wrote {args.out}")

    _write_markdown(out, args.out_md)
    print(f"[OK] wrote {args.out_md}")

    print(f"\n{len(reps)} replicates; {len(saturated)} detectors with zero variance")
    for lv in ACCESS_LEVELS:
        m = ladder[lv]["max_auroc"]
        if m["mean"] is not None and m["ci_lo"] is not None:
            print(f"  {lv} max: {m['mean']:.3f}  [{m['ci_lo']:.3f}, {m['ci_hi']:.3f}]")


def _write_markdown(out: dict, path: Path) -> None:
    L = [
        "# Leaderboard replication",
        "",
        f"{out['n_replicates']} independent draws of the full benchmark "
        f"(layout seed and rollout seeds resampled together). Intervals are "
        f"{int(CI_LEVEL * 100)}% percentile bootstrap over replicates.",
        "",
        "## Access-level ladder",
        "",
        "| Level | best-detector AUROC | 95% CI | mean over detectors | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for lv in ACCESS_LEVELS:
        mx, mn = out["access_levels"][lv]["max_auroc"], out["access_levels"][lv]["mean_auroc"]
        if mx["mean"] is None:
            continue
        f = lambda c: (  # noqa: E731
            f"[{c['ci_lo']:.3f}, {c['ci_hi']:.3f}]" if c.get("ci_lo") is not None else "n/a"
        )
        L.append(
            f"| {lv} | {mx['mean']:.3f} | {f(mx)} | "
            f"{mn['mean']:.3f} | {f(mn)} |"
        )
    L += ["", "## Supervised vs unsupervised", "",
          "Five detectors expose `fit()` and are scored by 5-fold cross-validation on labels: "
          + ", ".join(f"`{n}`" for n in out["supervision"]["label_fitted_detectors"]) + ".",
          "Removing them and re-deriving the ladder from the same draws:", "",
          "| Level | best AUROC (all) | 95% CI | best AUROC (unsupervised) | 95% CI | best unsupervised detector |",
          "|---|---|---|---|---|---|"]
    for lv in ACCESS_LEVELS:
        a = out["access_levels"][lv]["max_auroc"]
        b = out["supervision"]["access_levels_unsupervised_only"][lv]["max_auroc"]
        if a["mean"] is None:
            continue
        f = lambda c: (f"[{c['ci_lo']:.4f}, {c['ci_hi']:.4f}]" if c.get("ci_lo") is not None else "n/a")  # noqa: E731
        bd = next(iter(out["supervision"]["access_levels_unsupervised_only"][lv]["best_detector_frequency"]), "--")
        bv = f"{b['mean']:.4f}" if b["mean"] is not None else "--"
        L.append(f"| {lv} | {a['mean']:.4f} | {f(a)} | {bv} | {f(b)} | {bd} |")
    us = out["supervision"]["adjacent_level_separation_unsupervised_only"]
    if "L1_minus_L0_max" in us:
        v = us["L1_minus_L0_max"]
        L += ["", f"Unsupervised L1 minus L0 (max, paired): **{v['mean']:+.4f} "
                  f"[{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]**, "
                  f"{v['replicates_with_hi_above_lo']}/{v['n_paired']} draws favouring L1 "
                  f"-- {'separates' if v['excludes_zero'] else 'does NOT separate'}."]
        both = out["adjacent_level_separation"].get("L1_minus_L0_max")
        best = out["access_levels"]["L1"].get("best_detector_frequency") or {}
        carrier = max(best, key=best.get) if best else None
        if both is not None and carrier in out["supervision"]["label_fitted_detectors"]:
            flips = (both["mean"] > 0) != (v["mean"] > 0)
            if v["excludes_zero"]:
                ending = (" and separates the other way: unsupervised L1 sits below L0"
                          if flips else " and still separates")
            else:
                ending = " and stops separating"
            L += ["",
                  f"The L0->L1 climb under the max ({both['mean']:+.4f}) is carried by a "
                  f"label-fitted detector, `{carrier}`, the best L1 detector on "
                  f"{best[carrier]}/{out['n_replicates']} draws. Without the label-fitted "
                  f"detectors the step {'reverses sign' if flips else 'keeps its sign'}"
                  f"{ending}. Quote the partition alongside any rung difference."]
    degenerate = set(degenerate_families_from_ledger())
    spans = []
    for lv in ACCESS_LEVELS:
        counts = sorted({
            len([f for f in d.get("per_family") or {} if lv != "L0" or f not in degenerate])
            for d in out["detectors"].values()
            if d.get("access_level") == lv
        })
        if counts:
            spans.append(f"{lv} on {counts[0]}" + (f" to {counts[-1]}" if len(counts) > 1 else ""))
    L += ["", "## Do adjacent rungs separate?", "",
          "Paired within each draw, under BOTH statistics. Where the two disagree, the rung's",
          "status depends on the statistic and neither answer may be quoted alone. Pairing does",
          "not equalise denominators. The families each detector is scored on (a family is "
          "left out where it lacks the detector's channel or runs fewer episodes than its "
          "min_episodes, and the ledger-degenerate ones are left out of L0): "
          + ", ".join(spans) + ".", ""]
    if out["adjacent_level_separation"]:
        L += ["| Comparison | statistic | difference | 95% CI | excludes 0? | draws favouring higher |",
              "|---|---|---|---|---|---|"]
        for k, v in out["adjacent_level_separation"].items():
            ci = (f"[{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]" if v.get("ci_lo") is not None else "n/a")
            L.append(
                f"| {k.rsplit('_', 1)[0].replace('_', ' ')} | {v['statistic']} | "
                f"{v['mean']:+.4f} | {ci} | "
                f"{'yes' if v['excludes_zero'] else '**NO**'} | "
                f"{v['replicates_with_hi_above_lo']}/{v['n_paired']} |"
            )
        seps = out["adjacent_level_separation"]
        sd = out["statistic_disagreement"]

        def says(v: dict, hi: str, lo: str) -> str:
            ci = (f"{v['mean']:+.4f} "
                  + (f"[{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]" if v.get("ci_lo") is not None
                     else "[n/a]")
                  + f", {v['replicates_with_hi_above_lo']}/{v['n_paired']} draws favouring {hi}")
            if not v["excludes_zero"]:
                return f"does not separate {hi} from {lo} ({ci})"
            top, bottom = (hi, lo) if v["mean"] > 0 else (lo, hi)
            return f"puts {top} above {bottom} ({ci})"

        for kind, how in (
            ("separation", "One statistic separates it and the other does not"),
            ("direction", "Both statistics separate it, in opposite directions"),
        ):
            for rung in sd.get(f"rungs_where_max_and_mean_disagree_on_{kind}", []):
                hi, _, lo = rung.split("_")
                mx, mn = seps[f"{rung}_max"], seps[f"{rung}_mean"]
                L += ["", f"**Statistic-dependent: {hi} minus {lo}.** {how}: the max "
                          f"{says(mx, hi, lo)}; the mean {says(mn, hi, lo)}. Quote both."]
        if not (sd["rungs_where_max_and_mean_disagree_on_separation"]
                or sd.get("rungs_where_max_and_mean_disagree_on_direction")):
            L += ["", "Max and mean agree on every rung, on whether it separates and in "
                      "which direction."]
    L += [
        "",
        "## Zero-variance detectors",
        "",
        f"{out['degeneracy']['n_detectors_with_zero_variance']} of {len(out['detectors'])} "
        f"detectors returned an identical score on every draw. That is saturation, not "
        f"precision; their intervals are points by construction.",
        "",
    ]
    for n in out["degeneracy"]["detectors_with_zero_variance"]:
        L.append(f"- {n} ({out['detectors'][n]['mean']:.3f})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
