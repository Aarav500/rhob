"""Leaderboard: load, update, and render a board of RHOB v3 submissions.

A static, file-based leaderboard (JSON + Markdown), matching the "static files are
fine" guidance -- no server, no auth. Its one writer in the repository is ``python -m
rhob submit``, which records self-reported single-draw submissions under
``leaderboard/community/`` (``board.json`` in the ``{"entries": [...]}`` schema, plus
Markdown tables rendered from it).

This is not where the benchmark's own numbers live. ``leaderboard/v5_replicated.json``
(intervals over 20 draws) and ``leaderboard/v5_leaderboard.json`` (one draw, with
provenance) are written by their scripts in other schemas; read those with
:func:`rhob.v3.leaderboard.replicated.load_replicated_leaderboard` or
:func:`rhob.v3.leaderboard.adapters.load_any_leaderboard_json`. :meth:`Leaderboard.load`
refuses them rather than reading them as an empty board, because the caller that loads
a board usually saves it back, and saving an empty board over one of them would
replace it with the single new entry.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rhob.v3.benchmark import BenchmarkResults
from rhob.v3.taxonomy import DifficultyTier


def _nearest_tier(difficulty: float) -> str:
    tiers = DifficultyTier.all()
    closest = min(tiers, key=lambda t: abs(float(t) - difficulty))
    return closest.name


@dataclass
class LeaderboardEntry:
    """One detector's submission: its results summarized for standings and breakdowns."""

    detector_name: str
    access_level: str
    author: str
    timestamp: str
    overall_auroc: Optional[float]
    n_cells: int
    family_auroc: dict[str, float] = field(default_factory=dict)
    mechanism_auroc: dict[str, float] = field(default_factory=dict)
    tier_auroc: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_results(cls, results: BenchmarkResults, author: str = "anonymous") -> "LeaderboardEntry":
        """Summarize ``results`` over its measured cells only.

        ``n_cells`` is the number of measured cells, the denominator of
        ``overall_auroc`` (as in :mod:`rhob.v3.leaderboard.adapters`, which reads it from
        a board's ``cells_measured``). A not-applicable cell (``na_reason`` set, AUROC
        NaN) is no evidence about the detector, so it is neither averaged nor counted:
        counting it would rank a detector measured on one cell alongside one measured
        on three as "3 cells". A run with no measured cell has ``overall_auroc`` None
        and ``n_cells`` 0, never NaN, which JSON cannot carry.
        """
        by_family: dict[str, list[float]] = {}
        by_mechanism: dict[str, list[float]] = {}
        by_tier: dict[str, list[float]] = {}
        scored = results.scored_cells
        for cell in scored:
            by_family.setdefault(cell.family, []).append(cell.discrimination_auroc)
            by_mechanism.setdefault(cell.mechanism, []).append(cell.discrimination_auroc)
            by_tier.setdefault(_nearest_tier(cell.difficulty), []).append(cell.discrimination_auroc)

        def _mean(d: dict[str, list[float]]) -> dict[str, float]:
            return {k: round(sum(v) / len(v), 4) for k, v in d.items()}

        return cls(
            detector_name=results.detector_name,
            access_level=results.access_level,
            author=author,
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_auroc=round(results.overall_auroc, 4) if scored else None,
            n_cells=len(scored),
            family_auroc=_mean(by_family),
            mechanism_auroc=_mean(by_mechanism),
            tier_auroc=_mean(by_tier),
        )


class Leaderboard:
    """A collection of :class:`LeaderboardEntry` with JSON persistence and Markdown rendering."""

    def __init__(self, entries: list[LeaderboardEntry] | None = None):
        self.entries: list[LeaderboardEntry] = entries or []

    def add(self, entry: LeaderboardEntry) -> None:
        self.entries.append(entry)

    def submit(self, results: BenchmarkResults, author: str = "anonymous") -> LeaderboardEntry:
        entry = LeaderboardEntry.from_results(results, author=author)
        self.add(entry)
        return entry

    def standings(self) -> list[LeaderboardEntry]:
        """Entries sorted by overall AUROC, descending.

        Entries with ``overall_auroc is None`` (a detector that errored during
        evaluation, e.g. a NaN-input failure, written through as ``null`` by an older
        population script) sort last rather than crashing the comparison -- None isn't
        orderable against None either, so it needs a real sentinel, not just a tuple
        with a boolean flag.
        """
        return sorted(
            self.entries,
            key=lambda e: e.overall_auroc if e.overall_auroc is not None else float("-inf"),
            reverse=True,
        )

    # -------------------------------------------------------------- persistence
    @classmethod
    def load(cls, path: Path) -> "Leaderboard":
        """Load an ``{"entries": [...]}`` board; an empty board if ``path`` does not exist.

        Raises:
            ValueError: ``path`` exists but is not an entries-schema board (for example
                ``leaderboard/v5_leaderboard.json``, which is ``{"results": ...}``), or
                it is not valid JSON, or an entry does not match :class:`LeaderboardEntry`.
                It used to read such a file as an empty board, and ``rhob submit`` then
                saved the one new entry over it.
        """
        path = Path(path)
        if not path.exists():
            return cls()
        hint = "read it with rhob.v3.leaderboard.adapters.load_any_leaderboard_json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not valid JSON ({exc})") from exc
        if not isinstance(data, dict) or "entries" not in data:
            raise ValueError(f"{path}: not an entries-schema board; {hint}")
        if not isinstance(data["entries"], list):
            raise ValueError(f"{path}: 'entries' is not a list; {hint}")
        try:
            return cls([LeaderboardEntry(**e) for e in data["entries"]])
        except TypeError as exc:
            raise ValueError(f"{path}: an entry does not match LeaderboardEntry ({exc})") from exc

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"entries": [asdict(e) for e in self.entries]}, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------ render
    def render_standings_md(self) -> str:
        lines = [
            "| Rank | Detector | Access | Overall AUROC | Cells | Author |",
            "|---|---|---|---|---|---|",
        ]
        for i, e in enumerate(self.standings(), 1):
            auroc_str = f"{e.overall_auroc:.3f}" if e.overall_auroc is not None else "-"
            lines.append(f"| {i} | {e.detector_name} | {e.access_level} | {auroc_str} | {e.n_cells} | {e.author} |")
        return "\n".join(lines)

    def render_by_mechanism_md(self) -> str:
        mechanisms = sorted({m for e in self.entries for m in e.mechanism_auroc})
        header = "| Detector | " + " | ".join(mechanisms) + " |"
        sep = "|---|" + "---|" * len(mechanisms)
        lines = [header, sep]
        for e in self.standings():
            row = [f"{e.mechanism_auroc.get(m, float('nan')):.3f}" if m in e.mechanism_auroc else "-" for m in mechanisms]
            lines.append(f"| {e.detector_name} | " + " | ".join(row) + " |")
        return "\n".join(lines)

    def render_by_difficulty_md(self) -> str:
        tiers = [t.name for t in DifficultyTier.all()]
        header = "| Detector | " + " | ".join(tiers) + " |"
        sep = "|---|" + "---|" * len(tiers)
        lines = [header, sep]
        for e in self.standings():
            row = [f"{e.tier_auroc.get(t, float('nan')):.3f}" if t in e.tier_auroc else "-" for t in tiers]
            lines.append(f"| {e.detector_name} | " + " | ".join(row) + " |")
        return "\n".join(lines)
