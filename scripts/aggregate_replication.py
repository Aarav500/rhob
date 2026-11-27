"""Combine leaderboard replicates into intervals, and say which reported gaps survive.

Input: ``results/replication/replicate_*.json`` from ``scripts/replicate_leaderboard.py``,
each one a full leaderboard run at an independent ``(layout_seed, seed_base)`` draw.

The unit of resampling is the **replicate**, not the cell and not the run. Replicates are
the independent draws of the whole pipeline -- environment layout, behavioral orientation
and run seeds resample together in a replicate of ``draw_schema`` 2 -- so a percentile
bootstrap over them is an interval on "what would this number be if the benchmark had
been drawn again", which is the question a reader of the leaderboard is actually asking.
Bootstrapping cells instead would treat 123 cells of one draw as 123 independent
observations; they are not, they share a layout and an orientation per family.

A replicate written before 2026-11-27 has no ``draw_schema``: ``generate_suite`` dropped
its layout seed, so it was built and oriented at layout 0 and resampled only its rollout
seeds. Its interval then answers the narrower question "drawn again with new rollout
seeds". The artifact's ``replicates`` block records which kind every input was, and the
summary's first sentence is written from it rather than assumed.

**Where the draws came from.** ``provenance`` is this aggregation's own block
(:func:`rhob.v3.provenance.provenance_block`). ``replicates`` tallies the inputs' own
provenance -- source commit, clean tree, Python, package versions, platform, numerics
(the pins and the kernels picked), CPU, draw schema and any rows merged in afterwards by
``scripts/merge_rescored_detector.py`` -- and says whether the set is *single-source*:
one recorded commit, a clean tree, one Python, one package set, one numerics signature,
one draw schema and no re-scored rows. Platform and CPU are reported but not required to
agree: the hosted runners of one CI run differ in kernel build and CPU model, and under
the numerics pins the kernels they pick do not. ``--require-single-source`` refuses to
write anything unless the set is single-source; the redraw workflow passes it, after
``scripts/check_redraw.py`` has checked the set against the board.

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

Run: python scripts/aggregate_replication.py [--in-dir DIR] [--out PATH] [--out-md PATH]
     [--require-single-source]
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
from rhob.v3.provenance import provenance_block  # noqa: E402

BOOTSTRAP_RESAMPLES = 10_000
CI_LEVEL = 0.95
#: Fixed so the published interval is reproducible from the same replicate files.
BOOTSTRAP_SEED = 20260804

#: The ``draw_schema`` from which a replicate's layout seed reaches the families
#: (``scripts/replicate_leaderboard.py``'s ``DRAW_SCHEMA``). A file without the key was
#: written before it, and was built at layout 0.
LAYOUT_DRAW_SCHEMA = 2

STATISTIC_DISAGREEMENT_NOTE = (
    "The max and the mean give these rungs different verdicts: a rung listed under "
    "separation separates under one statistic and not the other, and a rung listed under "
    "direction separates under both with opposite signs. Reporting either alone would "
    "present a statistic-dependent result as a settled one. Quote both."
)


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


# --------------------------------------------------------------------- provenance
def _prov(rep: dict) -> dict:
    prov = rep.get("provenance")
    return prov if isinstance(prov, dict) else {}


def _git(rep: dict) -> dict:
    git = _prov(rep).get("git")
    return git if isinstance(git, dict) else {}


def _commit(rep: dict):
    return _git(rep).get("commit")


def _dirty(rep: dict):
    return _git(rep).get("dirty")


def _python(rep: dict):
    prov = _prov(rep)
    if prov.get("python") is None:
        return None
    impl = prov.get("python_implementation")
    return f"{impl} {prov['python']}" if impl else str(prov["python"])


def _packages(rep: dict):
    packages = _prov(rep).get("packages")
    return packages if isinstance(packages, dict) else None


def _platform(rep: dict):
    return _prov(rep).get("platform")


def _numerics(rep: dict):
    """What selected the kernels: the pins, the ATen level and the BLAS libraries picked.

    ``None`` when the replicate predates the ``numerics`` block. The CPU is left out on
    purpose (see :func:`_cpu`): two runners with different CPUs and the same signature
    here ran the same kernels.
    """
    numerics = _prov(rep).get("numerics")
    if not isinstance(numerics, dict):
        return None
    return {
        "env": numerics.get("env"),
        "torch_cpu_capability": numerics.get("torch_cpu_capability"),
        "blas": numerics.get("blas"),
    }


def _cpu(rep: dict):
    numerics = _prov(rep).get("numerics")
    cpu = numerics.get("cpu") if isinstance(numerics, dict) else None
    if not isinstance(cpu, dict):
        return None
    return {k: cpu.get(k) for k in ("vendor", "model", "family", "model_number", "stepping")}


def _draw_schema(rep: dict):
    schema = rep.get("draw_schema")
    return schema if isinstance(schema, int) and not isinstance(schema, bool) else None


def _rescored(rep: dict) -> list[dict]:
    entries = _prov(rep).get("rescored")
    if not isinstance(entries, list):
        return []
    return [
        {"detector": e.get("detector"), "commit": e.get("commit"), "source": e.get("source")}
        for e in entries
        if isinstance(e, dict)
    ]


def _tally(reps: list[dict], value_of) -> list[dict]:
    """Group replicates by ``value_of(rep)``, in order of first appearance.

    ``[{"value": v, "n": k, "replicate_ids": [...]}]``: a list rather than a mapping,
    because the values (a package set, a numerics block, ``null``) are not JSON keys.
    One entry means every replicate agrees.
    """
    groups: list[dict] = []
    for r in reps:
        v = value_of(r)
        for g in groups:
            if g["value"] == v:
                g["replicate_ids"].append(r["replicate_id"])
                break
        else:
            groups.append({"value": v, "replicate_ids": [r["replicate_id"]]})
    return [
        {"value": g["value"], "n": len(g["replicate_ids"]), "replicate_ids": g["replicate_ids"]}
        for g in groups
    ]


def layouts_resampled(reps: list[dict]) -> bool:
    """Whether every replicate's layout seed reached the families (``draw_schema`` >= 2)."""
    return all((_draw_schema(r) or 1) >= LAYOUT_DRAW_SCHEMA for r in reps)


def single_source_problems(reps: list[dict]) -> list[str]:
    """Why the replicates are not one uniform draw of one source; empty when they are.

    Single-source means: one recorded commit, every tree clean (``dirty`` exactly
    False, not unknown), one Python, one package set, one numerics signature (recorded),
    one draw schema, and no rows merged in after the draw. Platform and CPU are not
    required to agree; see the module docstring.
    """
    problems: list[str] = []
    commits = _tally(reps, _commit)
    if len(commits) > 1:
        problems.append(
            f"{len(commits)} source commits: "
            + "; ".join(f"{g['value']} on {_ids(g['replicate_ids'])}" for g in commits)
        )
    elif commits and commits[0]["value"] is None:
        problems.append("no replicate records the commit it was drawn from")
    unclean = [r["replicate_id"] for r in reps if _dirty(r) is not False]
    if unclean:
        problems.append(
            f"{len(unclean)} replicate(s) not recorded as drawn from a clean tree "
            f"(dirty true or unrecorded): {_ids(unclean)}"
        )
    # A draw schema that is absent everywhere is uniform (every file predates the key);
    # whether a set is of the current schema is check_redraw.py's question, not this one.
    for label, value_of, required in (
        ("Python", _python, True),
        ("package set", _packages, True),
        ("numerics signature", _numerics, True),
        ("draw schema", _draw_schema, False),
    ):
        t = _tally(reps, value_of)
        if len(t) > 1:
            problems.append(
                f"{len(t)} different {label}s: "
                + "; ".join(f"{_ids(g['replicate_ids'])}" for g in t)
            )
        elif required and t and t[0]["value"] is None:
            problems.append(f"no replicate records its {label}")
    rescored = [r["replicate_id"] for r in reps if _rescored(r)]
    if rescored:
        problems.append(
            f"{len(rescored)} replicate(s) carry rows re-scored after the draw "
            f"(provenance.rescored): {_ids(rescored)}"
        )
    return problems


def summarize_replicates(reps: list[dict]) -> dict:
    """The ``replicates`` block: the inputs' provenance, tallied; see the module docstring."""
    problems = single_source_problems(reps)
    return {
        "n_replicate_files": len(reps),
        "commits": _tally(reps, _commit),
        "dirty": _tally(reps, _dirty),
        "pythons": _tally(reps, _python),
        "package_sets": _tally(reps, _packages),
        "platforms": _tally(reps, _platform),
        "numerics": _tally(reps, _numerics),
        "cpus": _tally(reps, _cpu),
        "draw_schemas": _tally(reps, _draw_schema),
        "rescored_rows": _tally(reps, _rescored),
        "layouts_resampled": layouts_resampled(reps),
        "single_source": not problems,
        "single_source_problems": problems,
        "note": (
            "Each list groups the replicate files by one provenance field; one entry means "
            "every replicate agrees, and a null value means the field is not recorded. "
            "draw_schema null means the file predates the key: it was drawn while "
            "FamilyRegistry.generate_suite dropped layout_seed (before 2026-11-27), so it "
            "was built and oriented at layout 0 and resampled only its rollout seeds, "
            "whatever its layout_seed says; layouts_resampled is false if any replicate "
            "is such a draw. single_source requires one recorded commit, clean trees, one "
            "Python, one package set, one numerics signature, one draw schema and no "
            "re-scored rows; platforms and CPUs are reported, not required to agree."
        ),
    }


def _ids(ids: list[int]) -> str:
    """``[0, 1, 2, 5]`` -> ``"replicates 0-2, 5"``."""
    ids = sorted(ids)
    runs: list[str] = []
    start = prev = None
    for i in ids + [None]:
        if start is None:
            start = prev = i
        elif i is not None and i == prev + 1:
            prev = i
        else:
            runs.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = i
    word = "replicate" if len(ids) == 1 else "replicates"
    return f"{word} {', '.join(runs)}"


# ----------------------------------------------------------- statistic disagreement
def disagreement_kind(mx: dict, mn: dict) -> str:
    """How the max and the mean disagree on one rung: ``"separation"``, ``"direction"`` or ``""``.

    ``"separation"``: one statistic separates the rung and the other does not.
    ``"direction"``: both separate it, with opposite signs.
    """
    if mx["excludes_zero"] != mn["excludes_zero"]:
        return "separation"
    if mx["excludes_zero"] and mn["excludes_zero"] and (mx["mean"] > 0) != (mn["mean"] > 0):
        return "direction"
    return ""


def statistic_disagreement(separations: dict) -> dict:
    """The rungs of one partition whose verdict depends on the statistic."""
    kinds: dict[str, list[str]] = {"separation": [], "direction": []}
    for lo, hi in zip(ACCESS_LEVELS, ACCESS_LEVELS[1:]):
        rung = f"{hi}_minus_{lo}"
        mx, mn = separations.get(f"{rung}_max"), separations.get(f"{rung}_mean")
        if mx is None or mn is None:
            continue
        kind = disagreement_kind(mx, mn)
        if kind:
            kinds[kind].append(rung)
    return {
        "rungs_where_max_and_mean_disagree_on_separation": kinds["separation"],
        "rungs_where_max_and_mean_disagree_on_direction": kinds["direction"],
        "note": STATISTIC_DISAGREEMENT_NOTE,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=Path("results/replication"))
    parser.add_argument("--out", type=Path, default=Path("leaderboard/v5_replicated.json"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/figures/replication_summary.md"))
    parser.add_argument(
        "--require-single-source",
        action="store_true",
        help=(
            "Refuse, writing nothing, unless every replicate records one commit, a clean "
            "tree, one Python, one package set, one numerics signature and one draw "
            "schema, and none carries re-scored rows."
        ),
    )
    args = parser.parse_args(argv)

    reps = load_replicates(args.in_dir)
    print(f"loaded {len(reps)} replicates from {args.in_dir}")
    replicates = summarize_replicates(reps)
    if args.require_single_source and not replicates["single_source"]:
        raise SystemExit(
            "refusing (--require-single-source): the replicates are not one uniform draw "
            "of one source:\n  - " + "\n  - ".join(replicates["single_source_problems"])
        )

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
    # separate, which way): statistic_disagreement(), applied below to both partitions,
    # lists under ``rungs_where_max_and_mean_disagree_on_separation`` the rungs that
    # separate under one statistic and not the other, and under
    # ``rungs_where_max_and_mean_disagree_on_direction`` those that separate under both
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
    # The same verdict-pair rule for both partitions. The unsupervised partition is where
    # the L0->L1 step is read against the label-fitted climb, so it needs the rule most:
    # on the 2026-11-26 board its step separates below zero under the max and does not
    # separate under the mean, and a summary reading only the max said it "separates the
    # other way".
    disagreement = statistic_disagreement(separations)
    disagreement_unsup = statistic_disagreement(separations_unsup)

    out = {
        "n_replicates": len(reps),
        "draws": [{"replicate_id": r["replicate_id"], "draw_schema": _draw_schema(r),
                   "layout_seed": r["layout_seed"], "seed_base": r["seed_base"],
                   "n_seeds": r["n_seeds"]} for r in reps],
        # This aggregation's own provenance, stamped once the work above is done.
        "provenance": provenance_block(
            script="scripts/aggregate_replication.py", in_dir=Path(args.in_dir).as_posix()
        ),
        # The inputs' provenance, tallied.
        "replicates": replicates,
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
            "statistic_disagreement_unsupervised_only": disagreement_unsup,
        },
        "access_levels": ladder,
        "adjacent_level_separation": separations,
        "statistic_disagreement": disagreement,
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


def _ci_text(v: dict, missing: str = "n/a") -> str:
    if v.get("ci_lo") is None:
        return missing
    return f"[{v['ci_lo']:+.4f}, {v['ci_hi']:+.4f}]"


def _verdict(v: dict, hi: str, lo: str) -> str:
    """``"puts L0 above L1"`` or ``"does not separate L1 from L0"``, for one statistic."""
    if not v["excludes_zero"]:
        return f"does not separate {hi} from {lo}"
    top, bottom = (hi, lo) if v["mean"] > 0 else (lo, hi)
    return f"puts {top} above {bottom}"


def _says(v: dict, hi: str, lo: str) -> str:
    """:func:`_verdict` with its numbers, for quoting one statistic on one rung."""
    return (f"{_verdict(v, hi, lo)} ({v['mean']:+.4f} {_ci_text(v, '[n/a]')}, "
            f"{v['replicates_with_hi_above_lo']}/{v['n_paired']} draws favouring {hi})")


def _separation_table(seps: dict) -> list[str]:
    L = ["| Comparison | statistic | difference | 95% CI | excludes 0? | draws favouring higher |",
         "|---|---|---|---|---|---|"]
    for k, v in seps.items():
        L.append(
            f"| {k.rsplit('_', 1)[0].replace('_', ' ')} | {v['statistic']} | "
            f"{v['mean']:+.4f} | {_ci_text(v)} | "
            f"{'yes' if v['excludes_zero'] else '**NO**'} | "
            f"{v['replicates_with_hi_above_lo']}/{v['n_paired']} |"
        )
    return L


def _statistic_dependent(seps: dict, sd: dict, where: str = "") -> list[str]:
    """One paragraph per rung whose verdict depends on the statistic, or the all-clear."""
    L: list[str] = []
    for kind, how in (
        ("separation", "One statistic separates it and the other does not"),
        ("direction", "Both statistics separate it, in opposite directions"),
    ):
        for rung in sd.get(f"rungs_where_max_and_mean_disagree_on_{kind}", []):
            hi, _, lo = rung.split("_")
            mx, mn = seps[f"{rung}_max"], seps[f"{rung}_mean"]
            L += ["", f"**Statistic-dependent{where}: {hi} minus {lo}.** {how}: the max "
                      f"{_says(mx, hi, lo)}; the mean {_says(mn, hi, lo)}. Quote both."]
    if not (sd.get("rungs_where_max_and_mean_disagree_on_separation")
            or sd.get("rungs_where_max_and_mean_disagree_on_direction")):
        L += ["", f"Max and mean agree on every rung{where}, on whether it separates and in "
                  "which direction."]
    return L


_NUMBER_WORDS = ("No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
                 "Nine", "Ten")


def _intro(out: dict) -> str:
    n = out["n_replicates"]
    reps = out.get("replicates") or {}
    schemas = reps.get("draw_schemas") or []
    n_layout = sum(
        g["n"] for g in schemas
        if isinstance(g["value"], int) and g["value"] >= LAYOUT_DRAW_SCHEMA
    )
    if n_layout == n:
        what = "(layout seed and rollout seeds resampled together)"
    elif n_layout == 0:
        what = (
            "that resample the rollout seeds only. Every draw was made before the layout "
            "seed reached the families (`draw_schema` absent; see *Provenance of the "
            "draws*), so each was built and oriented at layout 0, whatever layout seed it "
            "records"
        )
    else:
        what = (
            f"of two kinds: {n_layout} resample the layout seed with the rollout seeds, "
            f"and {n - n_layout} were built and oriented at layout 0 and resample the "
            f"rollout seeds only (see *Provenance of the draws*)"
        )
    return (f"{n} independent draws of the full benchmark {what}. Intervals are "
            f"{int(CI_LEVEL * 100)}% percentile bootstrap over replicates.")


def _short_commit(value) -> str:
    return f"`{str(value)[:12]}`" if value else "not recorded"


def _fmt_packages(value) -> str:
    if not isinstance(value, dict):
        return "not recorded"
    return ", ".join(f"{k} {v}" if v is not None else f"{k} (not installed)"
                     for k, v in value.items())


def _fmt_numerics(value) -> str:
    if not isinstance(value, dict):
        return "not recorded (the block dates from 2026-11-27)"
    env = value.get("env") or {}
    pins = ", ".join(f"{k}={v}" for k, v in env.items() if v is not None) or "no pins set"
    bits = [pins]
    if value.get("torch_cpu_capability"):
        bits.append(f"ATen {value['torch_cpu_capability']}")
    for lib in value.get("blas") or []:
        if isinstance(lib, dict) and lib.get("internal_api"):
            arch = lib.get("architecture")
            bits.append(f"{lib['internal_api']}" + (f" {arch}" if arch else ""))
    return "; ".join(bits)


def _fmt_cpu(value) -> str:
    if not isinstance(value, dict):
        return "not recorded"
    return " ".join(str(value[k]) for k in ("vendor", "model") if value.get(k)) or "unknown"


def _fmt_schema(value) -> str:
    if value is None:
        return "absent (drawn before 2026-11-27, at layout 0)"
    if value >= LAYOUT_DRAW_SCHEMA:
        return f"{value} (layout seed forwarded to the families)"
    return str(value)


def _fmt_dirty(value) -> str:
    return {False: "clean tree", True: "uncommitted changes"}.get(value, "tree state unrecorded")


def _fmt_rescored(value) -> str:
    if not value:
        return "none"
    by_commit: dict = {}
    for e in value:
        by_commit.setdefault(e.get("commit"), []).append(e.get("detector"))
    parts = []
    for commit, names in by_commit.items():
        who = ", ".join(str(n) for n in names) if len(names) <= 3 else f"{len(names)} detectors"
        parts.append(f"{who} at {_short_commit(commit)}")
    return f"{len(value)} rows ({'; '.join(parts)})"


def _tally_text(groups: list[dict], fmt, n: int) -> str:
    if not groups:
        return "not recorded"
    if len(groups) == 1:
        return f"{fmt(groups[0]['value'])}" + (f", on all {n}" if n > 1 else "")
    return "; ".join(f"{fmt(g['value'])} on {_ids(g['replicate_ids'])}" for g in groups)


def _provenance_lines(out: dict) -> list[str]:
    reps = out.get("replicates")
    if not reps:
        return []
    n = reps.get("n_replicate_files", out["n_replicates"])
    L = ["", "## Provenance of the draws", "",
         "Read from each replicate file's own provenance (`replicates` in "
         "`leaderboard/v5_replicated.json`).", ""]
    rows = (
        ("Draw schema", "draw_schemas", _fmt_schema),
        ("Source commit", "commits", _short_commit),
        ("Tree", "dirty", _fmt_dirty),
        ("Python", "pythons", lambda v: v or "not recorded"),
        ("Packages", "package_sets", _fmt_packages),
        ("Platform", "platforms", lambda v: v or "not recorded"),
        ("Numerics", "numerics", _fmt_numerics),
        ("CPU", "cpus", _fmt_cpu),
        ("Rows re-scored after the draw", "rescored_rows", _fmt_rescored),
    )
    for label, key, fmt in rows:
        L.append(f"- {label}: {_tally_text(reps.get(key) or [], fmt, n)}.")
    if reps.get("single_source"):
        L.append("- Single source: **yes** -- one commit, clean trees, one Python, one "
                 "package set, one numerics signature, one draw schema, no re-scored rows.")
    else:
        L.append("- Single source: **no**:")
        L += [f"  - {problem}" for problem in reps.get("single_source_problems", [])]
    return L


def _write_markdown(out: dict, path: Path) -> None:
    L = [
        "# Leaderboard replication",
        "",
        _intro(out),
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
    fitted = out["supervision"]["label_fitted_detectors"]
    count = _NUMBER_WORDS[len(fitted)] if len(fitted) < len(_NUMBER_WORDS) else str(len(fitted))
    L += ["", "## Supervised vs unsupervised", "",
          f"{count} detectors expose `fit()` and are scored by 5-fold cross-validation on "
          "labels: " + ", ".join(f"`{n}`" for n in fitted) + ".",
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
    usd = (out["supervision"].get("statistic_disagreement_unsupervised_only")
           or statistic_disagreement(us))
    if us:
        L += ["", "Adjacent rungs without the label-fitted detectors, paired within each "
                  "draw under both statistics:", ""]
        L += _separation_table(us)
        L += _statistic_dependent(us, usd, " without the label-fitted detectors")
    if "L1_minus_L0_max" in us:
        v, m = us["L1_minus_L0_max"], us.get("L1_minus_L0_mean")
        both = out["adjacent_level_separation"].get("L1_minus_L0_max")
        best = out["access_levels"]["L1"].get("best_detector_frequency") or {}
        carrier = max(best, key=best.get) if best else None
        if both is not None and carrier in fitted:
            flips = (both["mean"] > 0) != (v["mean"] > 0)
            sign = "reverses sign" if flips else "keeps its sign"
            # "Separates the other way" is a verdict, and a verdict needs both statistics:
            # it is written only when the mean agrees with the max. Where they disagree
            # the sentence names both, as the rung tables above do.
            kind = disagreement_kind(v, m) if m is not None else ""
            if kind:
                dep = "whether it separates" if kind == "separation" else "which way it separates"
                ending = (f" under the max, and {dep} depends on the statistic: the max "
                          f"{_verdict(v, 'L1', 'L0')} and the mean {_verdict(m, 'L1', 'L0')}, "
                          f"so quote both")
            elif v["excludes_zero"]:
                ending = (" and separates the other way: unsupervised L1 sits below L0"
                          if flips else " and still separates")
            else:
                ending = " and stops separating"
            L += ["",
                  f"The L0->L1 climb under the max ({both['mean']:+.4f}) is carried by a "
                  f"label-fitted detector, `{carrier}`, the best L1 detector on "
                  f"{best[carrier]}/{out['n_replicates']} draws. Without the label-fitted "
                  f"detectors the step {sign}{ending}. Quote the partition alongside any "
                  f"rung difference."]
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
        L += _separation_table(out["adjacent_level_separation"])
        L += _statistic_dependent(out["adjacent_level_separation"], out["statistic_disagreement"])
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
    provenance = _provenance_lines(out)
    if provenance and L[-1] == "":
        provenance = provenance[1:]
    L += provenance
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
