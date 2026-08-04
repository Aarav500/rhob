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
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rhob.v3.leaderboard.access_summary import (  # noqa: E402
    ACCESS_LEVELS,
    degenerate_families_from_ledger,
    summarize_access_levels,
)

BOOTSTRAP_RESAMPLES = 10_000
CI_LEVEL = 0.95
#: Fixed so the published interval is reproducible from the same replicate files.
BOOTSTRAP_SEED = 20260804


def bootstrap_ci(values: list[float], rng: np.random.Generator) -> dict:
    """Percentile bootstrap of the mean of ``values``, resampling replicates.

    With R replicates the bootstrap can only resolve the interval to ~1/R granularity;
    it does not manufacture information that R draws do not contain. ``n_replicates`` is
    reported alongside every interval so that limit stays visible.
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
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    print(f"loaded {len(reps)} replicates from {args.in_dir}")

    # ---- per-detector overall AUROC ------------------------------------------------
    detector_names = sorted({n for r in reps for n in r["results"]})
    per_detector: dict[str, dict] = {}
    saturated: list[str] = []
    for name in detector_names:
        vals, level = [], None
        for r in reps:
            rec = r["results"].get(name)
            if rec and rec.get("overall_auroc") is not None:
                vals.append(float(rec["overall_auroc"]))
                level = rec.get("access_level", level)
        ci = bootstrap_ci(vals, rng)
        ci["access_level"] = level
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
    ladder_by_rep: dict[str, dict[str, dict[int, float]]] = {
        lv: {"mean": {}, "max": {}} for lv in ACCESS_LEVELS
    }
    best_counts: dict[str, dict[str, int]] = {lv: {} for lv in ACCESS_LEVELS}
    for r in reps:
        rid = r["replicate_id"]
        summaries = summarize_access_levels(r["results"], degenerate_families=degenerate)
        for lv, s in summaries.items():
            if s.mean_auroc is not None:
                ladder_by_rep[lv]["mean"][rid] = s.mean_auroc
            if s.max_auroc is not None:
                ladder_by_rep[lv]["max"][rid] = s.max_auroc
            if s.best_detector:
                best_counts[lv][s.best_detector] = best_counts[lv].get(s.best_detector, 0) + 1

    def _vals(lv: str, stat: str) -> list[float]:
        """Values ordered by replicate id, for the unpaired per-level intervals."""
        return [ladder_by_rep[lv][stat][k] for k in sorted(ladder_by_rep[lv][stat])]

    ladder = {
        lv: {
            "mean_auroc": bootstrap_ci(_vals(lv, "mean"), rng),
            "max_auroc": bootstrap_ci(_vals(lv, "max"), rng),
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
    # published: on this board the two disagree about L1->L2, which separates on the
    # suite mean (+0.015, CI excludes zero) and does not on the best-detector max
    # (+0.001, CI spans zero). Either alone reads as a settled answer; together they are
    # the actual finding, which is that the rung's status depends on the statistic.
    # The max is additionally a selection statistic -- the maximum over a level's
    # detectors exceeds the truth by construction -- and at L0, where every detector is
    # at chance, the argmax is a coin flip (see best_detector_frequency).
    separations = {}
    for stat in ("max", "mean"):
        for lo, hi in zip(ACCESS_LEVELS, ACCESS_LEVELS[1:]):
            lo_by_rep, hi_by_rep = ladder_by_rep[lo][stat], ladder_by_rep[hi][stat]
            shared = sorted(set(lo_by_rep) & set(hi_by_rep))
            if not shared:
                continue
            diffs = np.array([hi_by_rep[k] - lo_by_rep[k] for k in shared], dtype=float)
            d_ci = bootstrap_ci(diffs.tolist(), rng)
            separations[f"{hi}_minus_{lo}_{stat}"] = {
                **d_ci,
                "statistic": stat,
                "excludes_zero": bool(
                    d_ci["ci_lo"] is not None and (d_ci["ci_lo"] > 0 or d_ci["ci_hi"] < 0)
                ),
                "replicates_with_hi_above_lo": int((diffs > 0).sum()),
                # Replicates where BOTH rungs were scoreable. Reported because it can be
                # smaller than n_replicates, and a paired comparison over a subset of
                # draws is a different claim than one over all of them.
                "n_paired": len(shared),
                # Pairing removes the between-draw variation the two rungs share. It does
                # NOT equalise denominators: within one replicate L1 aggregates 35 cells
                # over 8 families while L0 aggregates 93 over 27 and L2/L3 aggregate 123
                # over 33. No amount of resampling fixes that, so any rung difference
                # involving L1 is across populations as well as across levels.
                "denominators_equal": False,
            }
    disagree = [
        f"{hi}_minus_{lo}"
        for lo, hi in zip(ACCESS_LEVELS, ACCESS_LEVELS[1:])
        if f"{hi}_minus_{lo}_max" in separations
        and f"{hi}_minus_{lo}_mean" in separations
        and separations[f"{hi}_minus_{lo}_max"]["excludes_zero"]
        != separations[f"{hi}_minus_{lo}_mean"]["excludes_zero"]
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
        "access_levels": ladder,
        "adjacent_level_separation": separations,
        "statistic_disagreement": {
            "rungs_where_max_and_mean_disagree_on_separation": disagree,
            "note": (
                "A rung listed here separates under one statistic and not the other. "
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
    L += ["", "## Do adjacent rungs separate?", "",
          "Paired within each draw, under BOTH statistics. Where the two disagree, the rung's",
          "status depends on the statistic and neither answer may be quoted alone. Pairing does",
          "not equalise denominators: L1 aggregates 35 cells over 8 families, L0 93 over 27,",
          "L2/L3 123 over 33.", ""]
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
        dis = out["statistic_disagreement"]["rungs_where_max_and_mean_disagree_on_separation"]
        L += ["", (f"**Statistic-dependent: {', '.join(dis)}.** Separates under one statistic and "
                   f"not the other." if dis else "Max and mean agree on every rung.")]
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
