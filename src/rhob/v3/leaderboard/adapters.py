"""Normalize the leaderboard JSON schemas that coexist under ``leaderboard/`` into a
single :class:`~rhob.v3.leaderboard.board.Leaderboard`.

``Leaderboard.load``/``save`` use ``{"entries": [...]}`` (one ``LeaderboardEntry`` per
detector). The bulk-population scripts (``populate_leaderboard_v32.py``,
``v5_leaderboard_and_transfer.py``) instead write ``{"results": {detector_name: {...}}}``
directly, bypassing ``Leaderboard`` entirely -- and that data (``leaderboard/v5_leaderboard.json``)
is validated, paper-cited data that should not be risked by a rewrite. A third schema,
``{"detectors": {detector_name: {mean, ci_lo, ci_hi, per_family: {...}, ...}}}``, is
written by ``scripts/aggregate_replication.py``: every figure there is an interval over
20 independent draws rather than a point from one. This module reads all three without
touching the files on disk.

**The replicated schema loses information here, by construction.** ``LeaderboardEntry``
holds one float per detector and one per family; it has nowhere to put an interval, a
replicate count, or the supervised/unsupervised partition. Flattening to the point
estimates and rendering those alone reproduces exactly the single-draw presentation the
replication exists to replace. Use this function when a caller genuinely only needs the
standings ordering; use :func:`~rhob.v3.leaderboard.replicated.load_replicated_leaderboard`
-- which returns the intervals and the partitions -- for anything that reports numbers to
a reader.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from rhob.v3.leaderboard.board import Leaderboard, LeaderboardEntry
from rhob.v3.leaderboard.replicated import is_replicated_schema


def load_any_leaderboard_json(path: str | Path, author: str = "rhob-team") -> Leaderboard:
    """Load a leaderboard JSON file in any of the three known schemas.

    Args:
        path: Path to a ``leaderboard/*.json`` file.
        author: Attributed author for entries built from the ad hoc ``results`` and
            ``detectors`` schemas, neither of which has a per-detector author field
            (unlike ``LeaderboardEntry``).
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    if "entries" in data:
        return Leaderboard([LeaderboardEntry(**e) for e in data["entries"]])

    if "results" in data:
        entries = []
        for detector_name, info in data["results"].items():
            if "overall_auroc" not in info:
                continue  # detector hit an error (e.g. NaN input); skip, don't crash
            entries.append(
                LeaderboardEntry(
                    detector_name=detector_name,
                    access_level=info.get("access_level", "?"),
                    author=author,
                    timestamp=data.get("timestamp", ""),
                    overall_auroc=info["overall_auroc"],
                    n_cells=info.get("cells", 0),
                    family_auroc=dict(info.get("per_family", {})),
                    mechanism_auroc={},
                    tier_auroc={},
                )
            )
        return Leaderboard(entries)

    # Checked after "results" and keyed on the value's *type*: v5_leaderboard.json also
    # carries a "detectors" key, holding an integer count of them. Dispatching on the key
    # name alone would read that file as replicated and find no detectors at all.
    if is_replicated_schema(data):
        entries = []
        for detector_name, info in data["detectors"].items():
            if not isinstance(info, Mapping):
                continue
            mean = info.get("mean")
            if not isinstance(mean, (int, float)) or isinstance(mean, bool):
                continue  # detector scored on no replicate; skip rather than invent a 0
            per_family = info.get("per_family")
            cells = info.get("cells_measured")
            entries.append(
                LeaderboardEntry(
                    detector_name=detector_name,
                    access_level=info.get("access_level") or "?",
                    author=author,
                    timestamp="",
                    overall_auroc=float(mean),
                    # The measured denominator, not the board's cell count: an L1
                    # detector measures 35 cells, not the 123 an L2 detector does.
                    n_cells=int(cells) if isinstance(cells, int) and not isinstance(cells, bool) else 0,
                    # Only families the detector actually measured are present in the
                    # artifact, and only those are carried over. An absent family means
                    # "no measurement" and must stay absent -- filling it with 0.5 is the
                    # imputation bug the replication was run to remove.
                    family_auroc={
                        fam: float(blob["mean"])
                        for fam, blob in (per_family or {}).items()
                        if isinstance(blob, Mapping)
                        and isinstance(blob.get("mean"), (int, float))
                        and not isinstance(blob.get("mean"), bool)
                    },
                    mechanism_auroc={},
                    tier_auroc={},
                )
            )
        return Leaderboard(entries)

    raise ValueError(
        f"{path}: unrecognized leaderboard JSON schema (no 'entries', 'results', or "
        "'detectors' object)"
    )
