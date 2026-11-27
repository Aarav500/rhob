"""Refuse a re-draw of the replicated board unless it is complete, uniform and one commit's.

The 20-replicate board (``results/replication/replicate_*.json``, aggregated into
``leaderboard/v5_replicated.json``) is re-drawn whole on CI: 20 jobs, one per draw, each
running ``scripts/replicate_leaderboard.py --replicate-id i``, beside one job running the
single-draw board with ``scripts/v5_leaderboard_and_transfer.py --cells-out``. Before the
new files replace the committed set, this script checks them, and exits non-zero with a
report of every failed check, unless all of these hold:

* **complete ids** -- exactly ``replicate_000.json`` .. ``replicate_{N-1}.json``
  (``--n-replicates``, default 20), each file's ``replicate_id`` matching its name;
* **draw schema** -- every file records ``draw_schema`` equal to
  ``replicate_leaderboard.DRAW_SCHEMA`` (the layout seed reached the families);
* **draws** -- every ``(layout_seed, seed_base)`` is ``draw_for(replicate_id)``;
* **seeds per variant** -- one ``n_seeds`` across the files, equal to the board's;
* **detectors** -- every file, the board and the board's cells sidecar score exactly the
  30 detectors of ``v5_leaderboard_and_transfer.DETECTORS``, in that order: each file's
  ``detectors_evaluated`` and ``results``, the board's ``results`` and the sidecar's
  ``cells`` (all four are written in ``DETECTORS`` order);
* **families** -- every file and the board evaluate exactly the registered families;
* **no errors** -- no detector record is an ``{"error": ...}``, and every replicate
  record carries ``cells_detail``;
* **one commit** -- every file, the board and the sidecar record the same commit, which is
  ``--expect-commit`` (a full SHA, or a prefix of at least 7 characters);
* **clean** -- every file and the board record ``dirty`` exactly ``false``;
* **not re-scored** -- no provenance carries a ``rescored`` entry: a re-draw is drawn,
  never patched (``scripts/merge_rescored_detector.py``);
* **one environment** -- one Python, one package set and one numerics signature (the pins
  and the kernels picked) across the files and the board. Platform and CPU are reported,
  not required to agree: hosted runners differ in both, and under the numerics pins the
  kernels they pick do not;
* **numerics pins** -- every file and the board ran under the pins themselves
  (:data:`PINNED_NUMERICS_ENV`, the values ``regenerate.yml`` sets for every name in
  ``rhob.v3.calibration_cache.NUMERICS_ENV_VARS``), and the kernels show they took
  effect: ATen at ``AVX2`` wherever torch was loaded, every loaded OpenBLAS at
  ``Haswell``. Agreement alone is not enough -- 21 runners that all lost the pins agree
  with each other, and their kernels still follow whichever CPU each one drew;
* **sidecar** -- the cells sidecar's provenance is the board's own, so the two files are
  one run's;
* **cell grid** -- for every detector, every file has the sidecar's cells, no more, no
  fewer, no duplicates;
* **replicate 0 is the board** -- ``replicate_000`` is the draw ``(0, 0)``, the board's, so
  its every cell must equal the sidecar's exactly (both are written by
  ``v5_leaderboard_and_transfer._cells_detail``, so there is no rounding to excuse). At
  most ``--max-board-mismatch`` cells (default 0) may differ, and every difference is
  listed.

Run::

    python scripts/check_redraw.py --from-dir DIR --board leaderboard/v5_leaderboard.json \\
        --board-cells BOARD_CELLS_JSON --expect-commit "$GITHUB_SHA" \\
        [--max-board-mismatch N] [--n-replicates 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Import rhob from THIS checkout, not from whatever tree `pip install -e .` recorded.
# See tests/test_scripts_run_their_own_checkout.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# The detector list, draw mapping and draw schema are the replicate script's own, so this
# check cannot drift from what it checks.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from replicate_leaderboard import DRAW_SCHEMA, draw_for  # noqa: E402
from v5_leaderboard_and_transfer import DETECTORS  # noqa: E402

from rhob.v3.calibration_cache import NUMERICS_ENV_VARS  # noqa: E402
from rhob.v3.registry import FamilyRegistry  # noqa: E402

N_REPLICATES = 20

#: The portable-numerics pins, as ``regenerate.yml`` sets them workflow-wide. The names
#: are ``NUMERICS_ENV_VARS``'s (the calibration cache and provenance read the same list);
#: a name added there with no value here fails the "numerics pins" check rather than
#: passing unchecked. ``tests/test_v3/test_redraw_check.py`` holds these values to
#: ``regenerate.yml``'s.
PINNED_NUMERICS_ENV: dict[str, str] = {
    "OPENBLAS_CORETYPE": "Haswell",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_CBWR": "COMPATIBLE",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "ATEN_CPU_CAPABILITY": "avx2",
}
#: What the pins make the kernels report (``provenance.numerics``): ATen's dispatch level
#: under ``ATEN_CPU_CAPABILITY=avx2``, and OpenBLAS's core under ``OPENBLAS_CORETYPE``.
PINNED_TORCH_CPU_CAPABILITY = "AVX2"
PINNED_OPENBLAS_ARCHITECTURE = "Haswell"

#: How many individual differences a failed check lists before summarising the rest.
_SHOW = 20


class Check:
    """One named check, its verdict and what it found.

    A plain class, not a dataclass: tests load this script by path, without a
    ``sys.modules`` entry, and ``dataclasses`` needs one to resolve the annotations.
    """

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self.problems: list[str] = []
        self.notes: list[str] = []

    @property
    def ok(self) -> bool:
        return not self.problems

    def fail(self, message: str) -> None:
        self.problems.append(message)


# ------------------------------------------------------------------------ accessors
def _prov(doc: dict) -> dict:
    prov = doc.get("provenance")
    return prov if isinstance(prov, dict) else {}


def _git(doc: dict) -> dict:
    git = _prov(doc).get("git")
    return git if isinstance(git, dict) else {}


def _numerics_signature(doc: dict) -> Any:
    numerics = _prov(doc).get("numerics")
    if not isinstance(numerics, dict):
        return None
    return {
        "env": numerics.get("env"),
        "torch_cpu_capability": numerics.get("torch_cpu_capability"),
        "blas": numerics.get("blas"),
    }


def _pin_problems(doc: dict) -> list[str]:
    """How one file's ``numerics`` block falls short of the pins (empty when it does not)."""
    numerics = _prov(doc).get("numerics")
    if not isinstance(numerics, dict):
        return ["no numerics block"]
    problems = []
    env = numerics.get("env") if isinstance(numerics.get("env"), dict) else {}
    unpinned = [
        f"{name}={env.get(name)!r} (pin {PINNED_NUMERICS_ENV[name]!r})"
        for name in NUMERICS_ENV_VARS
        if name in PINNED_NUMERICS_ENV and env.get(name) != PINNED_NUMERICS_ENV[name]
    ]
    if unpinned:
        problems.append("not pinned: " + ", ".join(unpinned))
    if numerics.get("torch_imported") is not False:
        capability = numerics.get("torch_cpu_capability")
        if capability != PINNED_TORCH_CPU_CAPABILITY:
            problems.append(
                f"ATen dispatched at {capability!r}, not {PINNED_TORCH_CPU_CAPABILITY!r}"
            )
    blas = numerics.get("blas")
    if isinstance(blas, list):
        cores = sorted({
            str(row.get("architecture")) for row in blas
            if isinstance(row, dict) and row.get("internal_api") == "openblas"
            and row.get("architecture") != PINNED_OPENBLAS_ARCHITECTURE
        })
        if cores:
            problems.append(
                f"OpenBLAS selected {', '.join(cores)}, not {PINNED_OPENBLAS_ARCHITECTURE}"
            )
    return problems


def _cpu(doc: dict) -> Any:
    numerics = _prov(doc).get("numerics")
    cpu = numerics.get("cpu") if isinstance(numerics, dict) else None
    if not isinstance(cpu, dict):
        return None
    return " ".join(str(cpu[k]) for k in ("vendor", "model") if cpu.get(k)) or None


def _cell_key(cell: dict) -> tuple[Any, Any]:
    return cell.get("family"), cell.get("difficulty")


def _short(value: Any) -> str:
    return str(value)[:12] if value else str(value)


def _listing(items: list[str]) -> list[str]:
    shown = items[:_SHOW]
    if len(items) > _SHOW:
        shown.append(f"... and {len(items) - _SHOW} more")
    return shown


def _commit_matches(commit: Any, expect: str) -> bool:
    if not isinstance(commit, str) or not commit:
        return False
    return commit == expect or (len(expect) >= 7 and commit.startswith(expect))


# --------------------------------------------------------------------------- checks
def run_checks(
    replicates: dict[str, Any],
    board: dict,
    board_cells: dict,
    *,
    expect_commit: str,
    max_board_mismatch: int = 0,
    n_replicates: int = N_REPLICATES,
    expected_detectors: list[str] | None = None,
    registered_families: list[str] | None = None,
) -> list[Check]:
    """Every check on one candidate re-draw. Pure: reads its arguments, writes nothing.

    Args:
        replicates: file name -> parsed replicate JSON, or the error message of a file
            that would not parse.
        board, board_cells: the parsed board and its ``--cells-out`` sidecar.
        expect_commit: the commit the whole set must come from.
        max_board_mismatch: how many cells of replicate 0 may differ from the board.
        n_replicates: how many replicate files make a complete set.
        expected_detectors: the detector names, in order (default: the board script's).
        registered_families: every registered family (default: the registry's).
    """
    if expected_detectors is None:
        expected_detectors = [d.name for d in DETECTORS]
    if registered_families is None:
        registered_families = FamilyRegistry.list_families()

    checks: list[Check] = []
    parsed: dict[str, dict] = {}

    # ---- complete ids ------------------------------------------------------------
    c = Check("complete ids", f"exactly replicate_000 .. replicate_{n_replicates - 1:03d}")
    expected_names = [f"replicate_{i:03d}.json" for i in range(n_replicates)]
    for name in sorted(replicates):
        doc = replicates[name]
        if not isinstance(doc, dict):
            c.fail(f"{name}: does not parse ({doc})")
            continue
        parsed[name] = doc
        stem_id = name[len("replicate_"):-len(".json")]
        if doc.get("replicate_id") != (int(stem_id) if stem_id.isdigit() else None):
            c.fail(f"{name}: records replicate_id {doc.get('replicate_id')!r}")
    missing = [n for n in expected_names if n not in replicates]
    extra = [n for n in sorted(replicates) if n not in expected_names]
    if missing:
        c.fail(f"missing: {', '.join(missing)}")
    if extra:
        c.fail(f"not part of a {n_replicates}-draw set: {', '.join(extra)}")
    checks.append(c)

    docs = [parsed[n] for n in sorted(parsed)]

    # ---- draw schema -------------------------------------------------------------
    c = Check("draw schema", f"every file records draw_schema {DRAW_SCHEMA}")
    for name, doc in sorted(parsed.items()):
        if doc.get("draw_schema") != DRAW_SCHEMA:
            c.fail(f"{name}: draw_schema {doc.get('draw_schema')!r}")
    checks.append(c)

    # ---- draws -------------------------------------------------------------------
    c = Check("draws", "every (layout_seed, seed_base) is draw_for(replicate_id)")
    for name, doc in sorted(parsed.items()):
        rid = doc.get("replicate_id")
        if not isinstance(rid, int) or isinstance(rid, bool) or rid < 0:
            c.fail(f"{name}: no usable replicate_id")
            continue
        want = draw_for(rid)
        got = (doc.get("layout_seed"), doc.get("seed_base"))
        if got != want:
            c.fail(f"{name}: (layout_seed, seed_base) = {got}, draw_for({rid}) = {want}")
    checks.append(c)

    # ---- seeds per variant -------------------------------------------------------
    board_seeds = (board.get("sampling") or {}).get("n_seeds_per_variant")
    c = Check("seeds per variant", f"one n_seeds across the files, equal to the board's ({board_seeds})")
    seeds = {doc.get("n_seeds") for doc in docs}
    if len(seeds) > 1:
        c.fail(f"n_seeds differs between files: {sorted(seeds, key=str)}")
    for name, doc in sorted(parsed.items()):
        if doc.get("n_seeds") != board_seeds:
            c.fail(f"{name}: n_seeds {doc.get('n_seeds')!r}, board {board_seeds!r}")
    checks.append(c)

    # ---- detectors ---------------------------------------------------------------
    c = Check("detectors", f"all {len(expected_detectors)} detectors, in order, everywhere")
    want = set(expected_detectors)

    def _same_detectors(label: str, got: Any) -> None:
        got = list(got) if isinstance(got, (list, dict)) else []
        if got == expected_detectors:
            return
        c.fail(
            f"{label} differs (missing {sorted(want - set(got))}, extra "
            f"{sorted(set(got) - want)}"
            + (", order differs" if set(got) == want and len(got) == len(want) else "")
            + (", duplicates" if len(got) != len(set(got)) else "") + ")"
        )

    for name, doc in sorted(parsed.items()):
        _same_detectors(f"{name}: detectors_evaluated", doc.get("detectors_evaluated"))
        _same_detectors(f"{name}: results", doc.get("results"))
    board_results = board.get("results") if isinstance(board.get("results"), dict) else {}
    _same_detectors("board: results", board_results)
    sidecar = board_cells.get("cells") if isinstance(board_cells.get("cells"), dict) else {}
    _same_detectors("board cells: cells", sidecar)
    checks.append(c)

    # ---- families ----------------------------------------------------------------
    c = Check("families", f"all {len(registered_families)} registered families, everywhere")
    for label, doc in [("board", board)] + sorted(parsed.items()):
        got = doc.get("families_evaluated")
        if got != registered_families:
            got = got or []
            c.fail(
                f"{label}: families_evaluated differs (missing "
                f"{sorted(set(registered_families) - set(got))}, extra "
                f"{sorted(set(got) - set(registered_families))})"
            )
    checks.append(c)

    # ---- errors ------------------------------------------------------------------
    c = Check("no errors", "no detector raised, and every replicate record has cells_detail")
    for label, doc in [("board", board)] + sorted(parsed.items()):
        results = doc.get("results") if isinstance(doc.get("results"), dict) else {}
        for det, rec in results.items():
            if not isinstance(rec, dict):
                c.fail(f"{label}: {det}: not a record")
            elif "error" in rec:
                c.fail(f"{label}: {det} raised: {rec['error']}")
            elif label != "board" and not isinstance(rec.get("cells_detail"), list):
                c.fail(f"{label}: {det}: no cells_detail")
    checks.append(c)

    # ---- one commit --------------------------------------------------------------
    c = Check("one commit", f"every file, the board and its cells record {_short(expect_commit)}")
    sources = [("board", board), ("board cells", board_cells)] + sorted(parsed.items())
    commits = {_git(doc).get("commit") for _, doc in sources}
    if len(commits) > 1:
        c.fail(f"{len(commits)} different commits: {sorted(_short(x) for x in commits)}")
    for label, doc in sources:
        commit = _git(doc).get("commit")
        if not _commit_matches(commit, expect_commit):
            c.fail(f"{label}: commit {_short(commit)}, expected {_short(expect_commit)}")
    checks.append(c)

    # ---- clean -------------------------------------------------------------------
    c = Check("clean", "every file and the board record dirty: false")
    for label, doc in [("board", board)] + sorted(parsed.items()):
        dirty = _git(doc).get("dirty")
        if dirty is not False:
            files = _git(doc).get("dirty_files")
            c.fail(f"{label}: dirty {dirty!r}" + (f" ({', '.join(files[:5])})" if files else ""))
    checks.append(c)

    # ---- not re-scored -----------------------------------------------------------
    c = Check("not re-scored", "no provenance carries a 'rescored' entry")
    for label, doc in [("board", board)] + sorted(parsed.items()):
        if "rescored" in _prov(doc):
            dets = [e.get("detector") for e in _prov(doc)["rescored"] if isinstance(e, dict)]
            c.fail(f"{label}: rows re-scored after the draw: {dets}")
    checks.append(c)

    # ---- one environment ---------------------------------------------------------
    c = Check("one environment", "one Python, one package set, one numerics signature")
    with_board = [("board", board)] + sorted(parsed.items())
    for what, value_of in (
        ("Python", lambda d: (_prov(d).get("python_implementation"), _prov(d).get("python"))),
        ("package set", lambda d: _prov(d).get("packages")),
        ("numerics signature", _numerics_signature),
    ):
        groups: list[tuple[Any, list[str]]] = []
        for label, doc in with_board:
            v = value_of(doc)
            for g in groups:
                if g[0] == v:
                    g[1].append(label)
                    break
            else:
                groups.append((v, [label]))
        if len(groups) > 1:
            c.fail(
                f"{len(groups)} different {what}s: "
                + "; ".join(f"{json.dumps(v, sort_keys=True)[:120]} on {', '.join(ls[:4])}"
                            + (f" (+{len(ls) - 4})" if len(ls) > 4 else "")
                            for v, ls in groups)
            )
        elif groups and groups[0][0] in (None, (None, None)):
            c.fail(f"no {what} recorded")
    platforms = sorted({str(_prov(d).get("platform")) for _, d in with_board})
    cpus = sorted({str(_cpu(d)) for _, d in with_board})
    c.notes.append(f"platforms ({len(platforms)}): {'; '.join(platforms)}")
    c.notes.append(f"CPUs ({len(cpus)}): {'; '.join(cpus)}")
    checks.append(c)

    # ---- numerics pins -----------------------------------------------------------
    c = Check(
        "numerics pins",
        "every file and the board ran under the CI pins, and the kernels show it",
    )
    unknown = [n for n in NUMERICS_ENV_VARS if n not in PINNED_NUMERICS_ENV]
    if unknown or set(PINNED_NUMERICS_ENV) - set(NUMERICS_ENV_VARS):
        c.fail(
            "check_redraw.PINNED_NUMERICS_ENV names "
            f"{sorted(PINNED_NUMERICS_ENV)}, rhob.v3.calibration_cache.NUMERICS_ENV_VARS "
            f"{sorted(NUMERICS_ENV_VARS)}: give every pin its value here"
        )
    # Grouped by fault, so 21 runners that all lost the same pin are one line, not 21.
    by_problem: dict[str, list[str]] = {}
    for label, doc in with_board:
        for problem in _pin_problems(doc):
            by_problem.setdefault(problem, []).append(label)
    for problem, labels in by_problem.items():
        c.fail(
            f"{', '.join(labels[:4])}" + (f" (+{len(labels) - 4})" if len(labels) > 4 else "")
            + f": {problem}"
        )
    checks.append(c)

    # ---- sidecar -----------------------------------------------------------------
    c = Check("sidecar", "the board cells' provenance is the board's own")
    if _prov(board_cells) != _prov(board) or not _prov(board):
        c.fail(
            "the cells file's provenance differs from the board's (generated "
            f"{_prov(board_cells).get('generated_utc')!r} vs "
            f"{_prov(board).get('generated_utc')!r}): not one run's files"
        )
    checks.append(c)

    # ---- cell grid ---------------------------------------------------------------
    c = Check("cell grid", "every file has exactly the board's cells, per detector")
    for det in expected_detectors:
        want_cells = sidecar.get(det)
        if not isinstance(want_cells, list):
            continue  # reported under "detectors"
        want_keys = [_cell_key(x) for x in want_cells]
        if len(set(want_keys)) != len(want_keys):
            c.fail(f"board cells: {det}: duplicate cells")
        for name, doc in sorted(parsed.items()):
            rec = (doc.get("results") or {}).get(det)
            if not isinstance(rec, dict) or not isinstance(rec.get("cells_detail"), list):
                continue  # reported under "detectors" / "no errors"
            got_keys = [_cell_key(x) for x in rec["cells_detail"]]
            if sorted(got_keys, key=str) != sorted(want_keys, key=str):
                only_here = sorted(set(got_keys) - set(want_keys), key=str)[:3]
                only_board = sorted(set(want_keys) - set(got_keys), key=str)[:3]
                c.fail(
                    f"{name}: {det}: {len(got_keys)} cells vs the board's {len(want_keys)}"
                    f" (only here {only_here}, only on the board {only_board})"
                )
    checks.append(c)

    # ---- replicate 0 is the board -------------------------------------------------
    c = Check(
        "replicate 0 is the board",
        "replicate_000 equals the board cell for cell"
        + (f" (up to {max_board_mismatch} allowed to differ)" if max_board_mismatch else ""),
    )
    rep0 = parsed.get("replicate_000.json")
    if rep0 is None:
        c.fail("replicate_000.json is missing")
    else:
        mismatches: list[str] = []
        compared = 0
        for det in expected_detectors:
            want_cells = sidecar.get(det)
            rec = (rep0.get("results") or {}).get(det)
            if not isinstance(want_cells, list) or not isinstance(rec, dict):
                continue
            got = {_cell_key(x): x.get("auroc") for x in rec.get("cells_detail") or []}
            for cell in want_cells:
                key = _cell_key(cell)
                if key not in got:
                    continue  # reported under "cell grid"
                compared += 1
                if got[key] != cell.get("auroc"):
                    mismatches.append(
                        f"{det} / {key[0]} @ {key[1]}: replicate_000 {got[key]!r}, "
                        f"board {cell.get('auroc')!r}"
                    )
        c.notes.append(f"{compared} cells compared, {len(mismatches)} differ")
        if len(mismatches) > max_board_mismatch:
            c.fail(
                f"{len(mismatches)} cells differ from the board "
                f"(--max-board-mismatch {max_board_mismatch}):"
            )
            c.problems += [f"  {m}" for m in _listing(mismatches)]
        elif mismatches:
            c.notes.append(f"allowed by --max-board-mismatch {max_board_mismatch}:")
            c.notes += [f"  {m}" for m in _listing(mismatches)]
        if compared == 0:
            c.fail("no cell of replicate_000 could be compared with the board")
    checks.append(c)

    return checks


def render(checks: list[Check], header: str) -> str:
    lines = [header, ""]
    for c in checks:
        lines.append(f"[{'PASS' if c.ok else 'FAIL'}] {c.name}: {c.description}")
        lines += [f"    {p}" for p in _listing(c.problems)]
        lines += [f"    {n}" for n in c.notes]
    failed = [c.name for c in checks if not c.ok]
    lines.append("")
    if failed:
        lines.append(
            f"REFUSED: {len(failed)} of {len(checks)} checks failed ({', '.join(failed)}). "
            "This set must not replace the committed replicates."
        )
    else:
        lines.append(f"ACCEPTED: all {len(checks)} checks passed.")
    return "\n".join(lines)


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--from-dir", type=Path, required=True,
                    help="Directory holding the re-drawn replicate_NNN.json files.")
    ap.add_argument("--board", type=Path, required=True,
                    help="The board written beside them (v5_leaderboard_and_transfer.py --out).")
    ap.add_argument("--board-cells", type=Path, required=True,
                    help="The board's per-cell sidecar (v5_leaderboard_and_transfer.py --cells-out).")
    ap.add_argument("--expect-commit", required=True,
                    help="The commit every file must come from (e.g. $GITHUB_SHA).")
    ap.add_argument("--max-board-mismatch", type=int, default=0,
                    help="Cells of replicate_000 allowed to differ from the board (default 0).")
    ap.add_argument("--n-replicates", type=int, default=N_REPLICATES,
                    help=f"Replicate files in a complete set (default {N_REPLICATES}).")
    args = ap.parse_args(argv)
    if args.max_board_mismatch < 0:
        ap.error("--max-board-mismatch must be >= 0")
    if args.n_replicates < 1:
        ap.error("--n-replicates must be >= 1")

    replicates = {p.name: _load(p) for p in sorted(args.from_dir.glob("replicate_*.json"))}
    board, board_cells = _load(args.board), _load(args.board_cells)
    for label, doc in (("--board", board), ("--board-cells", board_cells)):
        if not isinstance(doc, dict):
            print(f"REFUSED: {label} does not parse: {doc}")
            return 1

    checks = run_checks(
        replicates,
        board,
        board_cells,
        expect_commit=args.expect_commit,
        max_board_mismatch=args.max_board_mismatch,
        n_replicates=args.n_replicates,
    )
    print(render(
        checks,
        f"check_redraw: {len(replicates)} replicate file(s) in {args.from_dir}, against "
        f"{args.board} and {args.board_cells}, expecting commit {_short(args.expect_commit)}",
    ))
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
