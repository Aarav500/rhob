"""Merge one detector's re-scored replicates into the published replicate set.

Used after a detector fix that changes which cells it can score, not what it scores on
the cells it could. Reward Skewness is the case this was written for: it now declares
``min_episodes = 100``, so 19 families become not-applicable and the other 14 must
come back with exactly the AUROCs they had. This script enforces both halves.

For each ``replicate_NNN.json`` in ``--from-dir`` (produced by
``scripts/replicate_leaderboard.py --detectors <name>``):

* the committed replicate in ``--into-dir`` must record ``draw_schema`` 2 or later
  (:data:`MIN_DRAW_SCHEMA`), or the merge is refused. A file without it was drawn while
  ``FamilyRegistry.generate_suite`` dropped ``layout_seed``, so its draw ``k`` was built
  at layout 0. Re-scoring it now builds layout ``k`` instead and moves every cell, and
  ``--keep-old-values`` would then keep the layout-0 values under the new
  not-applicable set: a draw mixing two layouts while recording one. Such a set is
  re-drawn whole (``scripts/check_redraw.py``), not patched;
* the replicate's ``(replicate_id, draw_schema, layout_seed, seed_base, n_seeds)`` must
  match the file of the same name in ``--into-dir`` -- same draw, or the merge is
  refused;
* on every cell the old record scored AND the new record scored, the AUROCs must agree
  to ``--atol`` -- the fix was supposed to leave scored cells alone, and this is where
  that claim is checked rather than asserted. A cell that nonetheless differs can be
  admitted with ``--allow-changed family:difficulty``, but only by name, and the old
  and new values are written into the replicate's provenance so the artifact says so;
* the old record is replaced by the new one, and a ``rescored`` entry is appended to
  the replicate's provenance naming the detector, the source directory, and the
  commit, so the artifact says which rows were regenerated when.

With ``--keep-old-values`` the merge does not take the re-scored record. It takes the
committed one and applies only the new not-applicable set: a cell the re-score marks
not applicable becomes ``null``, every other cell keeps the value its draw was scored
with, and the record's per-family means, overall AUROC and counts are recomputed. Use it
when the re-score cannot reproduce the draw's rollouts for a reason unrelated to the fix
(the family's code changed after the draw was made), so that every detector in a
replicate stays scored on the same runs. A cell whose value did not reproduce must still
be allowed, by name or by family (``--allow-changed-family``), and both values go into
the replicate's provenance.

Every replicate is compared before any is written, and the report lists every cell that
moved in every replicate, so one run shows the whole picture.

Then run ``scripts/aggregate_replication.py`` to rebuild ``leaderboard/v5_replicated.json``.

Run::

    python scripts/merge_rescored_detector.py --detector "Reward Skewness" \
        --from-dir results/replication_skewness_refit --into-dir results/replication
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

#: The oldest ``draw_schema`` a committed replicate may have and still be merged into:
#: ``scripts/replicate_leaderboard.py``'s ``DRAW_SCHEMA`` as of the layout-seed fix
#: (2026-11-27). Restated here rather than imported, because importing that script loads
#: the whole detector suite; tests/test_v3/test_merge_rescored_detector.py checks the two
#: agree.
MIN_DRAW_SCHEMA = 2


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            cwd=_REPO, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _cells(record: dict) -> dict[tuple[str, float], float | None]:
    return {
        (c["family"], round(float(c["difficulty"]), 4)): c["auroc"]
        for c in record.get("cells_detail", [])
    }


def compare_one(new_path: Path, into_dir: Path, detector: str, atol: float) -> dict:
    """Compare one re-scored replicate with the committed one. Writes nothing."""
    new = json.loads(new_path.read_text(encoding="utf-8"))
    old_path = into_dir / new_path.name
    if not old_path.is_file():
        raise SystemExit(f"{new_path.name}: no counterpart in {into_dir}")
    old = json.loads(old_path.read_text(encoding="utf-8"))

    schema = old.get("draw_schema")
    if not isinstance(schema, int) or isinstance(schema, bool) or schema < MIN_DRAW_SCHEMA:
        raise SystemExit(
            f"{old_path}: draw_schema is {schema!r}, older than {MIN_DRAW_SCHEMA}. This "
            f"draw was made while FamilyRegistry.generate_suite dropped layout_seed, so it "
            f"was built at layout 0 whatever layout_seed {old.get('layout_seed')!r} says; "
            f"a re-score now builds that layout and cannot be merged into it. Re-draw the "
            f"whole set instead (regenerate.yml's redraw mode, gated by "
            f"scripts/check_redraw.py)."
        )
    for key in ("replicate_id", "draw_schema", "layout_seed", "seed_base", "n_seeds"):
        if old.get(key) != new.get(key):
            raise SystemExit(
                f"{new_path.name}: {key} differs ({old.get(key)} vs {new.get(key)}); "
                f"not the same draw, refusing to merge"
            )
    if detector not in new["results"]:
        raise SystemExit(f"{new_path.name}: no record for {detector!r} in the re-scored file")
    if detector not in old["results"]:
        raise SystemExit(f"{new_path.name}: no record for {detector!r} in the published file")

    old_rec, new_rec = old["results"][detector], new["results"][detector]
    old_cells, new_cells = _cells(old_rec), _cells(new_rec)
    if set(old_cells) != set(new_cells):
        raise SystemExit(
            f"{new_path.name}: cell sets differ; "
            f"only-old={sorted(set(old_cells) - set(new_cells))[:5]} "
            f"only-new={sorted(set(new_cells) - set(old_cells))[:5]}"
        )

    unchanged, newly_na, changed = 0, [], []
    for key, old_auroc in old_cells.items():
        new_auroc = new_cells[key]
        if new_auroc is None and old_auroc is not None:
            newly_na.append(key)
        elif new_auroc is not None and old_auroc is not None:
            if abs(new_auroc - old_auroc) > atol:
                changed.append((key, old_auroc, new_auroc))
            else:
                unchanged += 1
        elif new_auroc is not None and old_auroc is None:
            changed.append((key, old_auroc, new_auroc))
    return {
        "file": new_path.name,
        "old": old,
        "old_path": old_path,
        "new": new,
        "new_path": new_path,
        "unchanged": unchanged,
        "newly_na": newly_na,
        "na_families": sorted({fam for fam, _ in newly_na}),
        "changed": changed,
    }


def _allowed(key: tuple[str, float], allow: set[tuple[str, float]], families: set[str]) -> bool:
    return key in allow or key[0] in families


def apply_na_only(old_rec: dict, new_rec: dict) -> dict:
    """The committed record with only the re-score's not-applicable set applied.

    Every cell the re-score marks not applicable is set to ``null``; every other cell
    keeps its committed value. Per-family means and the overall AUROC are recomputed the
    way ``scripts/v5_leaderboard_and_transfer.py`` computes them (mean of the scored
    cells, rounded to 3 places), and the not-applicable bookkeeping is the re-score's.
    """
    new_na = {k for k, v in _cells(new_rec).items() if v is None}
    cells = []
    for c in old_rec["cells_detail"]:
        key = (c["family"], round(float(c["difficulty"]), 4))
        cells.append({**c, "auroc": None if key in new_na else c["auroc"]})
    by_family: dict[str, list[float]] = {}
    families: list[str] = []
    for c in cells:
        if c["family"] not in families:
            families.append(c["family"])
        if c["auroc"] is not None:
            by_family.setdefault(c["family"], []).append(float(c["auroc"]))
    per_family = {
        fam: (round(float(np.mean(by_family[fam])), 3) if fam in by_family else None)
        for fam in sorted(families)
    }
    scored = [float(c["auroc"]) for c in cells if c["auroc"] is not None]
    rec = dict(old_rec)
    rec.update({
        "overall_auroc": round(float(np.mean(scored)), 3) if scored else None,
        "cells": len(cells),
        "cells_measured": len(scored),
        "cells_not_applicable": len(cells) - len(scored),
        "not_applicable_families": sorted(f for f, v in per_family.items() if v is None),
        "not_applicable_reasons": new_rec.get("not_applicable_reasons", {}),
        "per_family": per_family,
        "cells_detail": cells,
    })
    return rec


def write_one(cmp: dict, detector: str, keep_old_values: bool, source: str | None = None) -> None:
    """Write one compared replicate back, recording what was done in its provenance."""
    old, new = cmp["old"], cmp["new"]
    old_rec, new_rec = old["results"][detector], new["results"][detector]
    old["results"][detector] = apply_na_only(old_rec, new_rec) if keep_old_values else new_rec
    prov = old.setdefault("provenance", {})
    entry = {
        "detector": detector,
        "from_dir": str(Path(cmp["new_path"]).parent.as_posix()),
        "source": source,
        "commit": _commit(),
        "cells_newly_not_applicable": len(cmp["newly_na"]),
        "cells_unchanged": cmp["unchanged"],
        "source_provenance": new.get("provenance"),
    }
    changed = [
        {"family": k[0], "difficulty": k[1], "old_auroc": a, "new_auroc": b}
        for k, a, b in cmp["changed"]
    ]
    if keep_old_values:
        entry["mode"] = (
            "kept the committed values; applied only the re-score's not-applicable set"
        )
        entry["cells_that_did_not_reproduce_old_value_kept"] = changed
    else:
        entry["cells_changed_and_allowed"] = changed
    prov.setdefault("rescored", []).append(entry)
    cmp["old_path"].write_text(json.dumps(old, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--from-dir", type=Path, required=True)
    ap.add_argument("--into-dir", type=Path, default=Path("results/replication"))
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument(
        "--allow-changed",
        nargs="*",
        default=[],
        metavar="FAMILY:DIFFICULTY",
        help=(
            "Scored cells permitted to differ, each only after its cause is established. "
            "The old and new values are written into the replicate's provenance, never "
            "silently absorbed."
        ),
    )
    ap.add_argument(
        "--allow-changed-family",
        nargs="*",
        default=[],
        metavar="FAMILY",
        help="As --allow-changed, for every cell of these families, in every replicate.",
    )
    ap.add_argument(
        "--keep-old-values",
        action="store_true",
        help=(
            "Keep each committed cell's value and apply only the re-score's not-applicable "
            "set (see the module docstring)."
        ),
    )
    ap.add_argument(
        "--report-only", action="store_true", help="Compare and report; write nothing."
    )
    ap.add_argument(
        "--source",
        default=None,
        help="Where the re-scored files came from (e.g. a CI run URL), for the provenance.",
    )
    args = ap.parse_args(argv)
    allow: set[tuple[str, float]] = set()
    for spec in args.allow_changed:
        fam, _, diff = spec.rpartition(":")
        allow.add((fam, round(float(diff), 4)))
    allow_families = set(args.allow_changed_family)

    files = sorted(args.from_dir.glob("replicate_*.json"))
    if not files:
        raise SystemExit(f"no replicate_*.json in {args.from_dir}")
    compared = [compare_one(f, args.into_dir, args.detector, args.atol) for f in files]

    changed = [(c["file"], k) for c in compared for k in c["changed"]]
    refused = [(f, k) for f, k in changed if not _allowed(k[0], allow, allow_families)]
    na_sets = {tuple(c["na_families"]) for c in compared}
    print(f"{args.detector!r}: {len(compared)} replicate(s) compared")
    print(f"  scored cells unchanged, total: {sum(c['unchanged'] for c in compared)}")
    print(f"  cells newly not applicable, total: {sum(len(c['newly_na']) for c in compared)}")
    print(f"  scored cells that moved: {len(changed)} ({len(refused)} not allowed)")
    for f, (k, a, b) in changed:
        tag = "" if _allowed(k, allow, allow_families) else "  NOT ALLOWED"
        print(f"    {f} {k}: {a} -> {b}{tag}")
    if len(na_sets) == 1:
        fams = next(iter(na_sets))
        print(f"  not-applicable families, identical across every replicate ({len(fams)}):")
        for fam in fams:
            print(f"    {fam}")
    else:
        print("  WARNING: the not-applicable family set differs between replicates:")
        for c in compared:
            print(f"    {c['file']}: {c['na_families']}")
        return 1
    if refused:
        print(
            "Refusing: a horizon fix must not move a cell it could already score. Pass "
            "--allow-changed family:difficulty (or --allow-changed-family) only after "
            "establishing why; the change is then recorded in the replicate's provenance."
        )
        return 1
    if args.report_only:
        return 0
    for c in compared:
        write_one(c, args.detector, args.keep_old_values, args.source)
    print(f"merged {len(compared)} replicate(s) for {args.detector!r}"
          + (" (committed values kept; not-applicable set applied)" if args.keep_old_values else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
