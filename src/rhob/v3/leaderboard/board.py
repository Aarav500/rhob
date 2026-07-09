"""Leaderboard: load, update, and render RHOB v3 standings.

A static, file-based leaderboard (JSON + Markdown), matching the "static files are
fine" guidance -- no server, no auth, just committed artifacts a CI job or a
maintainer can regenerate.
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
        by_family: dict[str, list[float]] = {}
        by_mechanism: dict[str, list[float]] = {}
        by_tier: dict[str, list[float]] = {}
        for cell in results.cells:
            if cell.discrimination_auroc != cell.discrimination_auroc:  # NaN guard
                continue
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
            overall_auroc=round(results.overall_auroc, 4),
            n_cells=len(results.cells),
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
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls([LeaderboardEntry(**e) for e in data.get("entries", [])])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"entries": [asdict(e) for e in self.entries]}, indent=2))

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
