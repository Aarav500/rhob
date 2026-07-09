"""Normalize the two leaderboard JSON schemas that currently coexist under ``leaderboard/``
into a single :class:`~rhob.v3.leaderboard.board.Leaderboard`.

``Leaderboard.load``/``save`` use ``{"entries": [...]}`` (one ``LeaderboardEntry`` per
detector). The bulk-population scripts (``populate_leaderboard_v32.py``,
``v5_leaderboard_and_transfer.py``) instead write ``{"results": {detector_name: {...}}}``
directly, bypassing ``Leaderboard`` entirely -- and that data (``leaderboard/v5_leaderboard.json``)
is validated, paper-cited data that should not be risked by a rewrite. This module reads
either shape without touching the files on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from rhob.v3.leaderboard.board import Leaderboard, LeaderboardEntry


def load_any_leaderboard_json(path: str | Path, author: str = "rhob-team") -> Leaderboard:
    """Load a leaderboard JSON file in either known schema.

    Args:
        path: Path to a ``leaderboard/*.json`` file.
        author: Attributed author for entries built from the ad hoc ``results`` schema,
            which has no per-detector author field (unlike ``LeaderboardEntry``).
    """
    path = Path(path)
    data = json.loads(path.read_text())

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

    raise ValueError(f"{path}: unrecognized leaderboard JSON schema (no 'entries' or 'results' key)")
