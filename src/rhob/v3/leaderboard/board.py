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
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rhob.v3.benchmark import BenchmarkResults
from rhob.v3.taxonomy import DifficultyTier

#: The keys every submission must carry (``submissions/README.md``).
SUBMISSION_KEYS = ("detector_name", "access_level", "overall_auroc", "n_cells")
ACCESS_LEVELS = ("L0", "L1", "L2", "L3")
_BREAKDOWN_KEYS = ("family_auroc", "mechanism_auroc", "tier_auroc")


def _nearest_tier(difficulty: float) -> str:
    tiers = DifficultyTier.all()
    closest = min(tiers, key=lambda t: abs(float(t) - difficulty))
    return closest.name


def _refuse_constant(name: str) -> float:
    raise ValueError(f"{name} is not a JSON value")


def strict_json_loads(text: str) -> object:
    """``json.loads``, except that ``NaN``, ``Infinity`` and ``-Infinity`` are refused.

    They are not JSON, but Python's ``json`` reads and writes them by default, so a
    submission carrying ``"overall_auroc": NaN`` would otherwise pass validation and be
    saved as a board no strict JSON reader can parse. Raises ``ValueError`` (which
    ``json.JSONDecodeError`` subclasses) on any of them, or on malformed JSON.
    """
    return json.loads(text, parse_constant=_refuse_constant)


def _is_auroc(value: object) -> bool:
    """A finite real number in [0, 1]; ``bool`` (an ``int`` in Python) is not one."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def submission_problems(data: object) -> list[str]:
    """Why ``data`` is not a valid submission or board entry; empty when it is.

    The four :data:`SUBMISSION_KEYS` are required. ``overall_auroc`` is a finite number in
    [0, 1], or ``null`` for a run with no measured cell; ``n_cells`` a non-negative
    integer. ``author`` and ``timestamp``, when present and not ``null``, are strings.
    ``family_auroc``, ``mechanism_auroc`` and ``tier_auroc``, when present, map names to
    finite numbers in [0, 1], and the tier names are :class:`DifficultyTier` names (any
    other is dropped from the rendered table without a word). Unknown keys are ignored.

    This is what ``python -m rhob validate`` (and so
    ``.github/workflows/leaderboard_validate.yml``) checks, and what
    :meth:`Leaderboard.load` checks of every entry: a value of the wrong type used to
    pass both and then crash rendering, after ``rhob submit`` had saved it.
    """
    if not isinstance(data, dict):
        return [f"a submission is a JSON object, got {type(data).__name__}"]
    missing = sorted(set(SUBMISSION_KEYS) - set(data))
    if missing:
        return [f"missing keys {missing}"]
    problems = []
    name = data["detector_name"]
    if not isinstance(name, str) or not name.strip():
        problems.append(f"detector_name must be a non-empty string, got {name!r}")
    level = data["access_level"]
    if not isinstance(level, str) or level not in ACCESS_LEVELS:
        problems.append(f"access_level must be one of L0/L1/L2/L3, got {level!r}")
    overall = data["overall_auroc"]
    if overall is not None and not _is_auroc(overall):
        problems.append(f"overall_auroc must be a number in [0, 1] or null, got {overall!r}")
    n_cells = data["n_cells"]
    if not isinstance(n_cells, int) or isinstance(n_cells, bool) or n_cells < 0:
        problems.append(f"n_cells must be a non-negative integer, got {n_cells!r}")
    for key in ("author", "timestamp"):
        value = data.get(key)
        if value is not None and not isinstance(value, str):
            problems.append(f"{key} must be a string, got {value!r}")
    tier_names = {t.name for t in DifficultyTier.all()}
    for key in _BREAKDOWN_KEYS:
        if key not in data:
            continue
        mapping = data[key]
        if not isinstance(mapping, dict):
            problems.append(
                f"{key} must be an object of name -> AUROC, got {type(mapping).__name__}"
            )
            continue
        bad = {k: v for k, v in mapping.items() if not isinstance(k, str) or not _is_auroc(v)}
        if bad:
            problems.append(f"{key} values must be numbers in [0, 1], got {bad!r}")
        if key == "tier_auroc":
            unknown = sorted(str(k) for k in set(mapping) - tier_names)
            if unknown:
                problems.append(
                    f"tier_auroc keys must be difficulty tiers {sorted(tier_names)}, "
                    f"got {unknown}"
                )
    return problems


def _md_cell(text: object) -> str:
    """``text`` as one Markdown table cell: a ``|`` or a line break would split the row."""
    return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


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
                it is not valid JSON (``NaN`` included), or an entry does not match
                :class:`LeaderboardEntry`, or one of its values fails
                :func:`submission_problems`. It used to read such a file as an empty
                board, and ``rhob submit`` then saved the one new entry over it.
        """
        path = Path(path)
        if not path.exists():
            return cls()
        hint = "read it with rhob.v3.leaderboard.adapters.load_any_leaderboard_json"
        try:
            data = strict_json_loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError(f"{path}: not valid JSON ({exc})") from exc
        if not isinstance(data, dict) or "entries" not in data:
            raise ValueError(f"{path}: not an entries-schema board; {hint}")
        if not isinstance(data["entries"], list):
            raise ValueError(f"{path}: 'entries' is not a list; {hint}")
        entries = []
        for i, raw in enumerate(data["entries"]):
            if not isinstance(raw, dict):
                raise ValueError(f"{path}: entry {i} is not an object")
            try:
                entry = LeaderboardEntry(**raw)
            except TypeError as exc:
                raise ValueError(
                    f"{path}: an entry does not match LeaderboardEntry ({exc})"
                ) from exc
            problems = submission_problems(raw)
            if problems:
                raise ValueError(f"{path}: entry {i} is invalid: {'; '.join(problems)}")
            entries.append(entry)
        return cls(entries)

    def dumps(self) -> str:
        """The board as JSON text; ``ValueError`` on a NaN or infinite value, never ``NaN``."""
        return json.dumps(
            {"entries": [asdict(e) for e in self.entries]}, indent=2, allow_nan=False
        )

    def save(self, path: Path) -> None:
        """Write :meth:`dumps` to ``path``; on a ``ValueError`` from it nothing is written."""
        text = self.dumps()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    # ------------------------------------------------------------------ render
    def render_standings_md(self) -> str:
        lines = [
            "| Rank | Detector | Access | Overall AUROC | Cells | Author |",
            "|---|---|---|---|---|---|",
        ]
        for i, e in enumerate(self.standings(), 1):
            auroc_str = f"{e.overall_auroc:.3f}" if e.overall_auroc is not None else "-"
            lines.append(
                f"| {i} | {_md_cell(e.detector_name)} | {e.access_level} | {auroc_str} "
                f"| {e.n_cells} | {_md_cell(e.author)} |"
            )
        return "\n".join(lines)

    def render_by_mechanism_md(self) -> str:
        mechanisms = sorted({m for e in self.entries for m in e.mechanism_auroc})
        header = "| Detector | " + " | ".join(_md_cell(m) for m in mechanisms) + " |"
        sep = "|---|" + "---|" * len(mechanisms)
        lines = [header, sep]
        for e in self.standings():
            row = [
                f"{e.mechanism_auroc[m]:.3f}" if m in e.mechanism_auroc else "-"
                for m in mechanisms
            ]
            lines.append(f"| {_md_cell(e.detector_name)} | " + " | ".join(row) + " |")
        return "\n".join(lines)

    def render_by_difficulty_md(self) -> str:
        tiers = [t.name for t in DifficultyTier.all()]
        header = "| Detector | " + " | ".join(tiers) + " |"
        sep = "|---|" + "---|" * len(tiers)
        lines = [header, sep]
        for e in self.standings():
            row = [f"{e.tier_auroc[t]:.3f}" if t in e.tier_auroc else "-" for t in tiers]
            lines.append(f"| {_md_cell(e.detector_name)} | " + " | ".join(row) + " |")
        return "\n".join(lines)
