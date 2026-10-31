"""Merge one detector's re-scored replicates into the published replicate set.

Used after a detector fix that changes which cells it can score, not what it scores on
the cells it could. Reward Skewness is the case this was written for: it now declares
``min_episodes = 100``, so 19 families become not-applicable and the other 14 must
come back with exactly the AUROCs they had. This script enforces both halves.

For each ``replicate_NNN.json`` in ``--from-dir`` (produced by
``scripts/replicate_leaderboard.py --detectors <name>``):

* the replicate's ``(replicate_id, layout_seed, seed_base, n_seeds)`` must match the
  file of the same name in ``--into-dir`` -- same draw, or the merge is refused;
* on every cell the old record scored AND the new record scored, the AUROCs must agree
  to ``--atol`` -- the fix was supposed to leave scored cells alone, and this is where
  that claim is checked rather than asserted. A cell that nonetheless differs can be
  admitted with ``--allow-changed family:difficulty``, but only by name, and the old
  and new values are written into the replicate's provenance so the artifact says so;
* the old record is replaced by the new one, and a ``rescored`` entry is appended to
  the replicate's provenance naming the detector, the source directory, and the
  commit, so the artifact says which rows were regenerated when.

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

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


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


def merge_one(
    new_path: Path,
    into_dir: Path,
    detector: str,
    atol: float,
    allow_changed: set[tuple[str, float]],
) -> dict:
    new = json.loads(new_path.read_text(encoding="utf-8"))
    old_path = into_dir / new_path.name
    if not old_path.is_file():
        raise SystemExit(f"{new_path.name}: no counterpart in {into_dir}")
    old = json.loads(old_path.read_text(encoding="utf-8"))

    for key in ("replicate_id", "layout_seed", "seed_base", "n_seeds"):
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
    not_allowed = [c for c in changed if c[0] not in allow_changed]
    if not_allowed:
        lines = "\n".join(f"  {k}: {a} -> {b}" for k, a, b in not_allowed[:10])
        raise SystemExit(
            f"{new_path.name}: {len(not_allowed)} scored cell(s) changed value; a horizon fix "
            f"must not move a cell it could already score:\n{lines}\n"
            f"Pass --allow-changed family:difficulty only after establishing why; the change "
            f"is then recorded in the replicate's provenance."
        )

    old["results"][detector] = new_rec
    prov = old.setdefault("provenance", {})
    prov.setdefault("rescored", []).append({
        "detector": detector,
        "from_dir": str(Path(new_path).parent.as_posix()),
        "commit": _commit(),
        "cells_newly_not_applicable": len(newly_na),
        "cells_unchanged": unchanged,
        "cells_changed_and_allowed": [
            {"family": k[0], "difficulty": k[1], "old_auroc": a, "new_auroc": b}
            for k, a, b in changed
        ],
        "source_provenance": new.get("provenance"),
    })
    old_path.write_text(json.dumps(old, indent=2), encoding="utf-8")
    return {
        "file": new_path.name,
        "unchanged": unchanged,
        "newly_na": len(newly_na),
        "na_families": sorted({fam for fam, _ in newly_na}),
        "changed": changed,
    }


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
    args = ap.parse_args(argv)
    allow: set[tuple[str, float]] = set()
    for spec in args.allow_changed:
        fam, _, diff = spec.rpartition(":")
        allow.add((fam, round(float(diff), 4)))

    files = sorted(args.from_dir.glob("replicate_*.json"))
    if not files:
        raise SystemExit(f"no replicate_*.json in {args.from_dir}")
    summaries = [merge_one(f, args.into_dir, args.detector, args.atol, allow) for f in files]

    na_sets = {tuple(s["na_families"]) for s in summaries}
    print(f"merged {len(summaries)} replicate(s) for {args.detector!r}")
    print(f"  scored cells unchanged, total: {sum(s['unchanged'] for s in summaries)}")
    print(f"  cells newly not applicable, total: {sum(s['newly_na'] for s in summaries)}")
    changed = [(s["file"], c) for s in summaries for c in s["changed"]]
    print(f"  scored cells changed (allowed by name): {len(changed)}")
    for f, (k, a, b) in changed:
        print(f"    {f} {k}: {a} -> {b}")
    if len(na_sets) == 1:
        fams = next(iter(na_sets))
        print(f"  not-applicable families, identical across every replicate ({len(fams)}):")
        for fam in fams:
            print(f"    {fam}")
    else:
        print("  WARNING: the not-applicable family set differs between replicates:")
        for s in summaries:
            print(f"    {s['file']}: {s['na_families']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
