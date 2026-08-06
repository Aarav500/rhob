"""RHOB interactive leaderboard viewer (Gradio).

A read-only viewer over the committed leaderboard JSON files -- no submission form,
no auth, no write path. Community detector results are submitted via a PR touching
``leaderboard/`` (validated by ``python -m rhob validate``, see
``.github/workflows/leaderboard_validate.yml``), not through this UI. This matches the
project's existing "static files are fine" leaderboard design
(see ``src/rhob/v3/leaderboard/board.py``).

Run locally (from within a full clone of the repo):
    pip install -e ".[space]"
    python space/app.py

When deployed to a Hugging Face Space, only this directory's contents are uploaded
(see .github/workflows/deploy_space.yml), with rhob installed as a regular pip
dependency (space/requirements.txt) rather than imported via a sibling src/ -- and the
deploy step copies leaderboard/*.json alongside this file, so the two candidate
locations below cover "running inside the full repo" and "running as a deployed Space"
respectively.

What this page serves
---------------------
Standings come from ``leaderboard/v5_replicated.json``: 20 independent draws of the whole
benchmark, aggregated by ``scripts/aggregate_replication.py`` into percentile-bootstrap
intervals. They do **not** come from ``leaderboard/v5_leaderboard.json``, which is a
single draw taken before behavioural sign randomization was switched on; its figures are
superseded and are deliberately not rendered anywhere on this page.

The cross-family transfer (RTS) panel is a **separate** experiment and is not covered by
that replication. It is also the one place where the pre-audit sign convention still had
live numbers: ``leaderboard/cross_family_transfer.json`` was generated on 2026-07-09, and
behavioural sign randomization landed on 2026-08-03, so its L2 figures measure compliance
with a house orientation convention rather than transfer (the published RTS of 0.994
re-measures at 0.508 with randomization on). This page therefore renders the *measured
post-randomization* figures from ``docs/figures/sign_randomization_impact.json`` and shows
the published numbers only in a column labelled as superseded. If that measurement file is
absent the panel says so and renders nothing -- it deliberately does not fall back to the
superseded artifact.

Schema knowledge lives in ``rhob.v3.leaderboard.replicated`` and
``rhob.v3.leaderboard.transfer``, not here. This module reads both artifacts through those
loaders and does no JSON archaeology of its own, so a change to an artifact's shape breaks
in one place with a name attached instead of silently producing plausible-looking numbers
in a deployed dashboard.
"""

from __future__ import annotations

import html
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd

from rhob.v3.leaderboard.replicated import (
    CONTROL_LEVEL,
    NOT_MEASURED,
    ChanceControl,
    LadderPartition,
    ReplicatedLeaderboard,
    load_replicated_leaderboard,
)
from rhob.v3.leaderboard.transfer import (
    TransferBoard,
    TransferComparison,
    load_transfer_results,
    transfer_under_sign_randomization,
)

_HERE = Path(__file__).resolve().parent
_CANDIDATE_LEADERBOARD_DIRS = [
    _HERE / "leaderboard",  # deployed Space: deploy step copies data alongside app.py
    _HERE.parent / "leaderboard",  # local dev: running from within the full repo clone
]
_LEADERBOARD_DIR = next((d for d in _CANDIDATE_LEADERBOARD_DIRS if d.is_dir()), _CANDIDATE_LEADERBOARD_DIRS[0])

#: The replicated board is the only source of standings. There is deliberately no
#: fallback to the single-draw v5_leaderboard.json: if this file is missing the page must
#: fail loudly rather than quietly serve superseded numbers under a page that promises
#: confidence intervals.
_REPLICATED_FILE = _LEADERBOARD_DIR / "v5_replicated.json"

#: The published transfer artifact. Pre-sign-randomization (2026-07-09) and pre-dating the
#: fix that stopped absent channels being imputed as 0.5, so it is read only to report
#: which figure each current one supersedes -- never as a current standing.
_PUBLISHED_TRANSFER_FILE = _LEADERBOARD_DIR / "cross_family_transfer.json"

#: The measured before/after sign-randomization artifact, which is where this page's
#: transfer numbers come from. Written by ``scripts/measure_sign_randomization.py``. The
#: Space deploy step stages it next to the app (see .github/workflows/deploy_space.yml).
_CANDIDATE_FIGURE_DIRS = [
    _HERE / "figures",  # deployed Space
    _HERE.parent / "docs" / "figures",  # local dev, full repo clone
]
_FIGURE_DIR = next((d for d in _CANDIDATE_FIGURE_DIRS if d.is_dir()), _CANDIDATE_FIGURE_DIRS[0])
_SIGN_RANDOMIZATION_FILE = _FIGURE_DIR / "sign_randomization_impact.json"

_GITHUB_BLOB = "https://github.com/Aarav500/rhob/blob/main"


@lru_cache(maxsize=1)
def _board() -> ReplicatedLeaderboard:
    """The replicated leaderboard, parsed once and reused across UI callbacks."""
    return load_replicated_leaderboard(_REPLICATED_FILE)


@lru_cache(maxsize=1)
def _transfer_rows() -> tuple[TransferComparison, ...]:
    """Transfer rows valid under sign randomization, with the figures they supersede.

    Empty when the measurement artifact is not present. That is the safe direction: the
    published artifact alone cannot produce a current number, so a missing measurement
    must yield no panel rather than the superseded ones.
    """
    if not _SIGN_RANDOMIZATION_FILE.exists():
        return ()
    measured: TransferBoard = load_transfer_results(_SIGN_RANDOMIZATION_FILE)
    published = (
        load_transfer_results(_PUBLISHED_TRANSFER_FILE)
        if _PUBLISHED_TRANSFER_FILE.exists()
        else None
    )
    return tuple(transfer_under_sign_randomization(measured, published))


@lru_cache(maxsize=1)
def _transfer_board() -> TransferBoard | None:
    if not _SIGN_RANDOMIZATION_FILE.exists():
        return None
    return load_transfer_results(_SIGN_RANDOMIZATION_FILE)


def _esc(text: object) -> str:
    return html.escape(str(text))


_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --rhob-bg: #0a0c10;
    --rhob-panel: #12151b;
    --rhob-panel-2: #171b22;
    --rhob-border: #262b35;
    --rhob-text: #dfe4ea;
    --rhob-text-dim: #8891a0;
    --rhob-amber: #ffa93f;
    --rhob-amber-dim: #7a5a2c;
    --rhob-teal: #2ee6b8;
    --rhob-teal-dim: #1c6656;
    --rhob-red: #ff5c66;
}

.gradio-container {
    background: var(--rhob-bg) !important;
    background-image:
        radial-gradient(circle at 15% 0%, rgba(255,169,63,0.06) 0%, transparent 45%),
        radial-gradient(circle at 85% 15%, rgba(46,230,184,0.05) 0%, transparent 40%) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: var(--rhob-text) !important;
    max-width: 1180px !important;
}

/* ---- Hero header ---- */
#rhob-hero {
    border: 1px solid var(--rhob-border);
    background: linear-gradient(180deg, var(--rhob-panel) 0%, var(--rhob-bg) 100%);
    border-radius: 14px;
    padding: 34px 36px 28px 36px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}
#rhob-hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image: repeating-linear-gradient(
        0deg, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 1px,
        transparent 1px, transparent 3px
    );
    pointer-events: none;
}
.rhob-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--rhob-amber);
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
}
.rhob-eyebrow .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--rhob-teal);
    box-shadow: 0 0 8px 1px var(--rhob-teal);
    animation: rhob-pulse 2.2s ease-in-out infinite;
}
@keyframes rhob-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.45; transform: scale(0.8); }
}
.rhob-title {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 800;
    font-size: 34px;
    letter-spacing: -0.01em;
    color: #f4f6f9;
    margin: 0 0 10px 0;
}
.rhob-title .accent { color: var(--rhob-amber); }
.rhob-sub {
    font-size: 15px;
    line-height: 1.6;
    color: var(--rhob-text-dim);
    max-width: 760px;
    margin-bottom: 18px;
}
.rhob-sub a { color: var(--rhob-teal); text-decoration: none; border-bottom: 1px dotted var(--rhob-teal-dim); }
.rhob-sub a:hover { color: #6ef7d6; }

.rhob-mirrors {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11.5px;
    color: var(--rhob-text-dim);
    letter-spacing: 0.02em;
}
.rhob-mirrors a { color: var(--rhob-text-dim); text-decoration: none; border-bottom: 1px dotted var(--rhob-border); }
.rhob-mirrors a:hover { color: var(--rhob-amber); border-color: var(--rhob-amber); }

/* ---- Provenance line ----
   Deliberately neutral: this states what the numbers are and how they were produced.
   It is not a warning and must not be styled as one. */
#rhob-provenance {
    border: 1px solid var(--rhob-border);
    background: var(--rhob-panel);
    border-radius: 8px;
    padding: 13px 18px;
    margin-bottom: 20px;
}
.rhob-prov-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--rhob-text-dim);
    margin-bottom: 5px;
}
.rhob-prov-body { font-size: 13.5px; line-height: 1.62; color: var(--rhob-text); }
.rhob-prov-body code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px;
    color: var(--rhob-text-dim);
    background: rgba(255, 255, 255, 0.045);
    padding: 1px 5px;
    border-radius: 4px;
}

/* ---- Stat cards ---- */
#rhob-stats { gap: 14px !important; margin-bottom: 22px !important; }
.rhob-stat {
    border: 1px solid var(--rhob-border);
    background: var(--rhob-panel);
    border-radius: 12px;
    padding: 18px 20px;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.rhob-stat:hover { border-color: var(--rhob-amber-dim); transform: translateY(-2px); }
.rhob-stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--rhob-text-dim);
    margin-bottom: 6px;
}
.rhob-stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 28px;
    color: var(--rhob-text);
}
.rhob-stat-value.teal { color: var(--rhob-teal); }
.rhob-stat-value.amber { color: var(--rhob-amber); }
.rhob-stat-note { font-size: 11.5px; color: var(--rhob-text-dim); margin-top: 4px; }

/* ---- Content panels (ladder, transfer, legend) ----
   One shared skin so a panel added later cannot drift into a different visual language.
   #rhob-rts-panel keeps its id for anchoring; the styling now comes from .rhob-panel. */
.rhob-panel {
    border: 1px solid var(--rhob-border);
    background: var(--rhob-panel);
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 22px;
}
.rhob-panel h3 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px;
    letter-spacing: 0.02em;
    color: #f4f6f9;
    margin-top: 0;
}
.rhob-panel h4 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--rhob-text-dim);
    margin: 20px 0 0 0;
}
.rhob-panel table { border-collapse: collapse; width: 100%; margin-top: 10px; }
.rhob-panel th {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--rhob-text-dim);
    text-align: left;
    padding: 8px 10px;
    border-bottom: 1px solid var(--rhob-border);
}
.rhob-panel td {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13.5px;
    padding: 9px 10px;
    border-bottom: 1px solid #1b1f27;
    color: var(--rhob-text);
}
.rhob-panel tr:last-child td { border-bottom: none; }
.rhob-panel p { color: var(--rhob-text-dim); font-size: 13.5px; line-height: 1.62; }
.rhob-panel code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px;
    color: var(--rhob-text-dim);
    background: rgba(255, 255, 255, 0.045);
    padding: 1px 5px;
    border-radius: 4px;
}
.rhob-note { color: var(--rhob-text-dim); font-size: 12.5px; line-height: 1.6; margin-top: 12px; }
.rhob-tablecap {
    font-size: 12px;
    color: var(--rhob-text-dim);
    margin-top: 14px;
}
.rhob-yes { color: var(--rhob-teal); }
.rhob-no { color: var(--rhob-amber); font-weight: 700; }
.rhob-na { color: var(--rhob-text-dim); }

/* Scope tag: says which experiment a panel's numbers come from, so a single-draw
   table cannot be read as if it carried the replication's intervals. */
.rhob-scope {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    border: 1px solid var(--rhob-border);
    border-radius: 999px;
    padding: 2px 9px;
    margin-left: 10px;
    vertical-align: 2px;
    color: var(--rhob-text-dim);
}
.rhob-scope.single { color: var(--rhob-amber); border-color: var(--rhob-amber-dim); }
.rhob-scope.control {
    letter-spacing: 0.10em;
    margin-left: 8px;
    padding: 1px 7px;
    color: var(--rhob-teal);
    border-color: var(--rhob-teal-dim);
}

/* ---- Tabs ---- */
.tab-nav button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 0.03em !important;
}

/* ---- Dataframe ---- */
.gradio-container table.table {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}
.gradio-container .table-wrap {
    border: 1px solid var(--rhob-border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ---- Dropdowns ---- */
.gradio-container label span { font-family: 'JetBrains Mono', monospace !important; font-size: 11.5px !important; letter-spacing: 0.06em !important; text-transform: uppercase; color: var(--rhob-text-dim) !important; }

footer { display: none !important; }
"""


# --------------------------------------------------------------------------- standings
def _auroc_column(board: ReplicatedLeaderboard) -> str:
    return f"Suite AUROC ({board.n_replicates}-draw mean) [{board.ci_percent}% CI]"


def _family_column(board: ReplicatedLeaderboard, family: str) -> str:
    return f"{family} AUROC [{board.ci_percent}% CI]"


def _standings_df(access_level: str, family: str) -> pd.DataFrame:
    """The standings table, filtered by access level and (optionally) family.

    Every AUROC is rendered as ``mean [lo, hi]`` over the replicates. A family the
    detector never measured renders as ``n/a`` -- not 0.5, and not an empty cell that a
    reader would take for zero. That distinction is the whole point of the family view:
    the L1 detectors read a channel 25 of the 33 families do not ship, and a board that
    imputed those cells as chance is what the replication was run to replace.
    """
    board = _board()
    rows = []
    for d in board.standings():
        if access_level != "All" and d.access_level != access_level:
            continue
        row: dict[str, object] = {
            "Detector": d.name,
            "Access": d.access_level,
            _auroc_column(board): d.overall.text(),
        }
        if family != "All":
            measured = d.family(family)
            row[_family_column(board, family)] = (
                measured.text() if measured is not None else NOT_MEASURED
            )
        row["Draws"] = d.overall.n_replicates
        row["Cells measured"] = (
            d.cells_measured if d.cells_measured is not None else NOT_MEASURED
        )
        row["Families measured"] = d.n_families_measured
        row["Scoring"] = "label-fitted (5-fold CV)" if d.label_fitted else "unsupervised"
        row["In level aggregate"] = (
            f"no - duplicate of {d.duplicate_of}" if d.duplicate_of else "yes"
        )
        rows.append(row)

    if not rows:
        columns = ["Detector", "Access", _auroc_column(board)]
        if family != "All":
            columns.append(_family_column(board, family))
        columns += ["Draws", "Cells measured", "Families measured", "Scoring", "In level aggregate"]
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def _all_families() -> list[str]:
    return ["All"] + _board().families()


def _all_access_levels() -> list[str]:
    return ["All"] + _board().access_levels_present()


# ----------------------------------------------------------------------------- panels
def _provenance_html() -> str:
    board = _board()
    seeds = board.seeds_per_variant
    seeds_txt = (
        f"{seeds} rollout seeds per variant per cell, " if seeds is not None else ""
    )
    resamples = board.bootstrap_resamples
    seed = board.bootstrap_seed
    method_bits = []
    if resamples is not None:
        method_bits.append(f"{resamples:,} resamples")
    if seed is not None:
        method_bits.append(f"seed {seed}")
    method_txt = f" ({', '.join(method_bits)})" if method_bits else ""
    return f"""
    <div id="rhob-provenance">
      <div class="rhob-prov-tag">Provenance</div>
      <div class="rhob-prov-body">
        Standings are {board.n_replicates} independent draws of the full benchmark --
        layout seed and rollout seeds resampled together, {seeds_txt}behavioural sign
        randomization on -- aggregated from <code>leaderboard/v5_replicated.json</code>.
        Intervals are {board.ci_percent}% percentile bootstrap over the
        {board.n_replicates} replicates{method_txt}. An interval is on the aggregate over
        draws; an individual cell is noisier than the detector-level figure printed beside
        it, and a zero-width interval means the detector returned the same score on every
        draw (saturation, not precision). Detectors do not share a denominator: see
        <b>Cells measured</b>.
      </div>
    </div>
    """


def _supervision_html() -> str:
    board = _board()
    fitted = board.label_fitted_detectors
    if not fitted:
        return ""
    names = ", ".join(f"<b>{_esc(n)}</b>" for n in fitted)
    unsup = board.unsupervised_only.separation("L0", "L1", "max")
    all_sep = board.all_detectors.separation("L0", "L1", "max")
    step = ""
    if unsup is not None and all_sep is not None and all_sep.separates and not unsup.separates:
        # Name the detector the artifact says carried L1, rather than asserting one: if
        # a future board's L1 winner changes, this sentence must change with it.
        l1 = board.all_detectors.level("L1")
        winner = l1.best_detector if l1 else None
        carried_by = (
            f" The entire climb is one label-fitted detector, <b>{_esc(winner)}</b>, "
            f"which won L1 on {l1.best_detector_draws}/{l1.total_draws} draws."
            if winner and board.detectors.get(winner, None) and board.detectors[winner].label_fitted
            else ""
        )
        step = (
            f" It is load-bearing on this board: the published L0 &rarr; L1 step is "
            f"{all_sep.difference.text(4, signed=True)} "
            f"({all_sep.replicates_with_hi_above_lo}/{all_sep.n_paired} draws) with the "
            f"label-fitted detectors in, and "
            f"{unsup.difference.text(4, signed=True)} "
            f"({unsup.replicates_with_hi_above_lo}/{unsup.n_paired} draws) without them "
            f"-- it reverses sign and stops separating.{carried_by}"
        )
    return f"""
    <div class="rhob-panel">
      <h3>LABEL-FITTED DETECTORS<span class="rhob-scope">{len(fitted)} of {len(board.detectors)}</span></h3>
      <p>
        {names} expose <code>fit()</code> and are scored by 5-fold stratified
        cross-validation <b>on the hacking/legitimate labels</b>. Cross-validation keeps
        that honest -- none is ever scored on runs it was fitted on -- but it answers a
        different question from an unsupervised detector's number:
        <i>&ldquo;is the difference learnable from labelled examples?&rdquo;</i> rather than
        <i>&ldquo;does this fixed statistic separate the variants?&rdquo;</i>. The two are
        not comparable as a ranking, and the <b>Scoring</b> column in the standings says
        which question each row answers.{step}
      </p>
    </div>
    """


def _ladder_table(partition: LadderPartition) -> str:
    rows = ""
    for stats in partition.scored_levels():
        winner = stats.best_detector
        winner_txt = (
            f"{_esc(winner)} <span class='rhob-na'>({stats.best_detector_draws}/{stats.total_draws})</span>"
            if winner
            else f"<span class='rhob-na'>{NOT_MEASURED}</span>"
        )
        # The control rung is tagged in the table itself. Left untagged it reads as the
        # bottom rung of a performance ladder, and its "best detector" cell -- a maximum
        # over 13 detectors, biased upward by selection -- reads as a detection result.
        control = (
            " <span class='rhob-scope control'>control</span>"
            if stats.level == CONTROL_LEVEL
            else ""
        )
        rows += (
            f"<tr><td>{_esc(stats.level)}{control}</td>"
            f"<td>{stats.max_auroc.text()}</td>"
            f"<td>{stats.mean_auroc.text()}</td>"
            f"<td>{winner_txt}</td></tr>\n"
        )
    return (
        "<table><tr><th>Level</th><th>Best detector (max over detectors)</th>"
        "<th>Mean over detectors</th>"
        f"<th>Most frequent winner (draws)</th></tr>{rows}</table>"
    )


def _separation_table(partition: LadderPartition) -> str:
    rows = ""
    for sep in sorted(
        partition.separations.values(), key=lambda s: (s.statistic, s.higher)
    ):
        verdict = (
            "<span class='rhob-yes'>separates</span>"
            if sep.separates
            else "<span class='rhob-no'>does NOT separate</span>"
        )
        rows += (
            f"<tr><td>{_esc(sep.label)}</td><td>{_esc(sep.statistic)}</td>"
            f"<td>{sep.difference.text(4, signed=True)}</td>"
            f"<td>{sep.replicates_with_hi_above_lo}/{sep.n_paired}</td>"
            f"<td>{verdict}</td></tr>\n"
        )
    if not rows:
        return ""
    return (
        "<table><tr><th>Rung</th><th>Statistic</th><th>Paired difference</th>"
        f"<th>Draws favouring higher</th><th>95% CI verdict</th></tr>{rows}</table>"
    )


def _ladder_html() -> str:
    """The access-level ladder, both partitions, with intervals."""
    board = _board()
    disagree = board.statistic_disagreement
    disagree_txt = (
        f"<b>Statistic-dependent: {', '.join(_esc(d.replace('_', ' ')) for d in disagree)}.</b> "
        f"{_esc(board.statistic_disagreement_note)}"
        if disagree
        else "Max and mean agree on every rung."
    )

    unsup_l1 = board.unsupervised_only.separation("L0", "L1", "max")
    unsup_l2 = board.unsupervised_only.separation("L1", "L2", "max")
    finding = ""
    if unsup_l1 is not None and unsup_l2 is not None:
        finding = (
            f"<p class='rhob-note'><b>On the unsupervised partition the ladder is not a "
            f"staircase.</b> {_esc(unsup_l1.label)} (best detector) is "
            f"{unsup_l1.difference.text(4, signed=True)}, "
            f"{unsup_l1.replicates_with_hi_above_lo}/{unsup_l1.n_paired} draws favouring "
            f"L1 -- it does not separate. {_esc(unsup_l2.label)} is "
            f"{unsup_l2.difference.text(4, signed=True)}, "
            f"{unsup_l2.replicates_with_hi_above_lo}/{unsup_l2.n_paired} draws -- it does."
            f"</p>"
        )

    return f"""
    <div class="rhob-panel" id="rhob-ladder-panel">
      <h3>ACCESS-LEVEL LADDER<span class="rhob-scope">{board.n_replicates} draws</span></h3>
      <p>
        Does more access buy more detectability? Recomputed inside every draw and paired
        by draw, under both statistics, and over both partitions of the detector suite.
        Duplicate detectors are held out of every level's aggregate. Pairing does not
        equalise denominators -- L1 aggregates 35 cells over 8 families where L2 and L3
        aggregate 123 over 33 -- so any rung difference involving L1 is across
        populations as well as across access levels.
      </p>
      <p class="rhob-note">
        The <b>best detector</b> column is a maximum taken inside each draw, so it is
        biased upward at every rung: the more detectors a level has, the further the
        expected maximum sits above any one of them. Read it alongside the mean rather
        than instead of it. {CONTROL_LEVEL} is a control rather than a rung -- see the
        panel below, where that bias decides the reading.
      </p>

      <h4>All detectors</h4>
      <div class="rhob-tablecap">Level aggregates</div>
      {_ladder_table(board.all_detectors)}
      <div class="rhob-tablecap">Adjacent-rung differences, paired within each draw</div>
      {_separation_table(board.all_detectors)}

      <h4>Unsupervised only (label-fitted detectors removed)</h4>
      <div class="rhob-tablecap">Level aggregates</div>
      {_ladder_table(board.unsupervised_only)}
      <div class="rhob-tablecap">Adjacent-rung differences, paired within each draw</div>
      {_separation_table(board.unsupervised_only)}

      {finding}
      <p class="rhob-note">{disagree_txt}</p>
    </div>
    """


def _best_transfer() -> TransferComparison | None:
    """The highest transfer score among the rows valid under sign randomization."""
    scored = [c for c in _transfer_rows() if c.rts is not None]
    return max(scored, key=lambda c: c.rts) if scored else None


def _worst_supersession_note() -> str:
    """The largest published-vs-current gap, read off the data rather than written down.

    The headline example of the sign-convention artifact is the Ensemble's 0.994, but
    naming it in prose would pin two constants into the page that nothing checks. This
    derives them, so a regenerated artifact updates the sentence instead of contradicting
    it, and the sentence vanishes if no row is superseded any more.
    """
    superseded = [c for c in _transfer_rows() if c.published_differs]
    if not superseded:
        return ""
    worst = max(superseded, key=lambda c: abs(c.published_delta or 0.0))
    return (
        f" The published <b>{_esc(worst.name)}</b> figure of {worst.published:.3f} "
        f"reproduces with randomization off and re-measures at {worst.rts:.3f} with it on "
        f"&mdash; a drop of {abs(worst.published_delta):.3f}, against a trial-to-trial "
        f"spread of {worst.row.sd:.3f}."
        if worst.published is not None and worst.rts is not None and worst.row.sd is not None
        else ""
    )


def _flat_rows_note() -> str:
    """Name any row that returned the identical value on every family it could score.

    Such a row has not measured a gradient of transfer, and the reason it can happen
    without the detector discriminating anything is documented in the experiment: a frozen
    model that can read a family's channel but cannot consume that family's own
    discretization of it is scored at chance rather than dropped. Computed from the
    artifact so the sentence disappears if a regenerated run stops being flat.
    """
    flat = [
        (c, value)
        for c in _transfer_rows()
        if (value := c.all_scored_families_at) is not None and c.row.n_families_scored > 1
    ]
    if not flat:
        return ""
    described = "; ".join(
        f"<b>{_esc(c.name)}</b> returned {value:.3f} on all "
        f"{c.row.n_families_scored} families it could score"
        for c, value in flat
    )
    return (
        f" {described} &mdash; a flat row, not a gradient. Where a frozen model can read a "
        "family's channel but cannot consume that family's own discretization of it, the "
        "cell is scored at chance rather than dropped, on the grounds that the channel was "
        "there and the model failed to use it "
        "(<code>scripts/cross_family_transfer.py::transfer_eval</code>), so a flat 0.500 "
        "row should be read as transfer failure rather than as a measured near-chance "
        "separation."
    )


def _rts_html() -> str:
    """Render the RHOB Transfer Score (RTS) table under behavioural sign randomization.

    Two things about this panel are not cosmetic.

    First, RTS is a *separate* experiment from the replicated leaderboard: it trains on 6
    mechanisms and tests on 8 held out, at a single benchmark draw. It carries no
    bootstrap interval and none of the replication's uncertainty estimates apply to it.

    Second, the published artifact's numbers are superseded. It was generated before
    behavioural sign randomization, when ``behav_trace``'s sign *was* the label, so its L2
    figures scored compliance with a house convention: the published 0.994 re-measures at
    0.508 once the orientation is randomized. The numbers rendered here are the measured
    post-randomization ones; the published figures appear only in a column that says they
    are superseded. L0 and L1 rows carry over unchanged from the pre-randomization run,
    because ``restrict()`` nulls ``behav_trace`` below L2 and leaves them no axis to flip.
    """
    rows = _transfer_rows()
    if not rows:
        return """
    <div class="rhob-panel" id="rhob-rts-panel">
      <h3>THE RHOB TRANSFER SCORE (RTS)<span class="rhob-scope single">not shown</span></h3>
      <p>
        The cross-family transfer figures are not rendered on this build. The published
        artifact <code>leaderboard/cross_family_transfer.json</code> predates behavioural
        sign randomization, so its L2 numbers measure the old orientation convention
        rather than transfer, and the post-randomization measurement
        (<code>docs/figures/sign_randomization_impact.json</code>) is not present here.
        Showing the superseded figures instead would be the worse of the two options, so
        the panel shows none.
      </p>
    </div>
    """

    board = _transfer_board()
    body = ""
    for c in rows:
        r = c.row
        rts = c.rts
        if rts is None:
            continue
        # No "near-perfect" band any more: under randomization nothing reaches it, and a
        # band that only ever fired on the sign-convention artifact was decoration for it.
        color = "var(--rhob-red)" if rts < 0.55 else "var(--rhob-amber)"
        spread = (
            f" <span class='rhob-na'>&plusmn; {r.sd:.3f}</span>"
            if isinstance(r.sd, float)
            else ""
        )
        scored = (
            f"{r.n_families_scored} of {r.n_families}"
            if r.n_families_not_applicable
            else str(r.n_families_scored)
        )
        na_note = (
            f" <span class='rhob-na'>({r.n_families_not_applicable} {NOT_MEASURED}: "
            f"{_esc(', '.join(r.families_not_applicable))})</span>"
            if r.families_not_applicable
            else ""
        )
        basis = (
            f"<span class='rhob-na'>carried over &mdash; L{r.access_level[1:]} sees no "
            f"behavioural trace</span>"
            if c.carried_over_as_invariant
            else f"re-measured (<code>{_esc(c.measured_config)}</code>)"
        )
        superseded = (
            f"<span class='rhob-no'>{c.published:.3f}</span>"
            if c.published is not None and c.published_differs
            else (
                f"<span class='rhob-na'>{c.published:.3f} (unchanged)</span>"
                if c.published is not None
                else f"<span class='rhob-na'>{NOT_MEASURED}</span>"
            )
        )
        # The surviving L2 maximum under randomization is a label-fitted detector, which
        # is a different question from an unsupervised one's score -- the same distinction
        # the standings table draws, applied here so the two panels agree.
        fitted = (
            " <span class='rhob-na'>&middot; label-fitted</span>"
            if r.name in _board().label_fitted_detectors
            else ""
        )
        body += (
            f"<tr><td>{_esc(r.name)}{fitted}</td><td>{_esc(r.access_level)}</td>"
            f"<td style='color:{color};font-weight:700;'>{rts:.3f}{spread}</td>"
            f"<td>{scored}{na_note}</td>"
            f"<td>{basis}</td>"
            f"<td>{superseded}</td></tr>\n"
        )

    trials = board.n_trials if board else None
    spread_note = (
        f"The &plusmn; figure is the spread across {trials} independently-seeded model "
        "initialisations at that one draw. It measures how much the <i>fit</i> wobbles, "
        "not sampling error in the benchmark draw, and it is not a confidence interval."
        if trials
        else ""
    )
    n_train = len(board.train_families) if board else 6
    n_test = len(board.test_families) if board else 8
    provenance = ""
    if board and board.generated_utc:
        commit = f", commit <code>{_esc(board.git_commit)}</code>" if board.git_commit else ""
        provenance = f" Measured {_esc(board.generated_utc[:10])}{commit}."

    return f"""
    <div class="rhob-panel" id="rhob-rts-panel">
      <h3>THE RHOB TRANSFER SCORE (RTS)<span class="rhob-scope single">single draw &middot; not replicated</span></h3>
      <p>
        Train on {n_train} hacking mechanisms, test on {n_test} never seen. RTS = mean
        AUROC on the held-out mechanisms &mdash; the number every detector submitted to
        RHOB gets scored on.
      </p>
      <table>
        <tr><th>Detector class</th><th>Access</th><th>RTS (transfer AUROC)</th>
            <th>Test families scored</th><th>Under randomization</th>
            <th>Superseded figure</th></tr>
        {body}
      </table>
      <p class="rhob-note">
        <b>These are not the published RTS figures, and the published ones are wrong.</b>
        <code>leaderboard/cross_family_transfer.json</code> was generated on 2026-07-09;
        behavioural sign randomization landed on 2026-08-03. Before it, every family
        emitted its behavioural trace under one fixed global orientation with positive =
        hacking, so the sign of the observation <i>was</i> the label and an L2 detector
        that &ldquo;transferred&rdquo; had learned the repository's house convention. The
        L2 rows above are re-measured with that orientation randomized
        (<a href="{_GITHUB_BLOB}/docs/figures/sign_randomization_impact.md" target="_blank">measured
        before/after</a>).{_worst_supersession_note()}{provenance}
      </p>
      <p class="rhob-note">
        <b>Scope.</b> This is one benchmark draw, so nothing here carries the
        {_board().n_replicates}-draw replication's bootstrap intervals and the confidence
        stated elsewhere on this page does not extend to it. {spread_note}
        A test family that emits no channel the detector reads is <code>{NOT_MEASURED}</code>
        &mdash; excluded from the average, never imputed at 0.5.{_flat_rows_note()}
      </p>
    </div>
    """


def _l0_control_html() -> str:
    """The L0 rung, reported as the construction check it is rather than as a standing.

    The ladder's ``Best detector`` cell at L0 is the mean over draws of whichever detector
    won each draw. Maximising over 13 noisy estimates biases that upward, and on this
    board it lands above *every* individual L0 detector's point estimate -- so a reader
    who takes it as "the best reward-only detector beats chance" reaches the opposite of
    what the data says. Every verdict below is computed from the artifact.
    """
    board = _board()
    control: ChanceControl = board.control_check()
    if control.n_detectors == 0:
        return ""

    ref = f"{control.reference:.3f}"
    verdict = (
        f"<b class='rhob-yes'>all {control.n_detectors}</b>"
        if control.holds
        else (
            f"only <b class='rhob-no'>{control.n_containing} of {control.n_detectors}</b> "
            f"(exceptions: {_esc(', '.join(d.name for d in control.exceptions))})"
        )
    )
    highest = control.highest
    highest_txt = (
        f"The highest is <b>{_esc(highest.name)}</b> at {highest.overall.text()}."
        if highest
        else ""
    )

    selection = ""
    if control.level_max is not None and control.max_exceeds_every_detector and highest:
        excludes = (
            " and excludes it entirely" if control.max_excludes_reference else ""
        )
        selection = (
            f"<p class='rhob-note'><b>Do not read the ladder's "
            f"&ldquo;{_esc(control.level)} best detector&rdquo; cell as a detection "
            f"result.</b> It reports {control.level_max.text()}, which sits above "
            f"{ref}{excludes} &mdash; and above the point estimate of every one of the "
            f"{control.n_detectors} detectors it is a maximum over, the highest of which "
            f"is {highest.overall.mean:.3f}. That gap is selection: the cell is the mean "
            f"over draws of whichever detector won each draw, and a maximum over "
            f"{control.n_detectors} noisy estimates is biased upward whether or not any "
            f"of them has signal. The winner is not even stable: {_l0_winner_share()}.</p>"
        )

    margin = control.mean_margin_from_reference
    mean_txt = ""
    if control.level_mean is not None:
        mean_txt = (
            f"<p class='rhob-note'><b>The control is carried detector by detector, not by "
            f"the suite mean.</b> The {_esc(control.level)} mean over detectors is "
            f"{control.level_mean.text(4)}."
        )
        if margin is not None and control.mean_excludes_reference:
            mean_txt += (
                f" Its interval clears {ref} by {margin:.5f} &mdash; smaller than the "
                f"Monte-Carlo scatter of that endpoint itself, which moves over a range "
                f"about three times as wide when the same bootstrap is replayed under 300 "
                f"different generator seeds, landing at or above {ref} in 18 of them "
                f"(<a href='{_GITHUB_BLOB}/docs/figures/l0_control_robustness.md' "
                f"target='_blank'>l0_control_robustness.md</a>). <b>The interval therefore "
                f"places the suite mean on neither side of chance</b>, and no direction "
                f"may be read from it. The per-detector verdicts above are not "
                f"seed-sensitive in that way."
            )
        mean_txt += "</p>"

    return f"""
    <div class="rhob-panel" id="rhob-l0-control">
      <h3>{_esc(control.level)} IS A CONTROL, NOT A STANDING<span class="rhob-scope">construction check</span></h3>
      <p>
        Detectors at {_esc(control.level)} see only the proxy reward, and every family's
        proxy is certified by the admission gate to be statistically indistinguishable
        between the hacking and legitimate variants. So {_esc(control.level)} is a negative
        control on the <b>benchmark</b>, not a rung of detector performance: a detector
        here that beat chance would be evidence that the proxy matching had failed, not
        that reward-only detection works. The reportable form of that check is per
        detector, over the {board.n_replicates} draws: {verdict}
        {_esc(control.level)} detectors have a {board.ci_percent}% interval containing
        {ref}. {highest_txt}
      </p>
      {selection}
      {mean_txt}
      <p class="rhob-note">
        <b>Robustness.</b> Six of the {len(board.families())} families are marked
        degenerate in the admission ledger because their proxy reward is constant, and on a
        constant proxy a reward-only detector returns {ref} for a reason that has nothing
        to do with proxy matching &mdash; padding the control with cells that could not
        have come out otherwise. Recomputed over the 27 non-degenerate families only,
        family-weighted and bootstrapped over the same {board.n_replicates} replicates,
        <b>zero verdicts flip</b>: all {control.n_detectors} intervals still contain {ref},
        and the largest movers move away from it
        (<a href="{_GITHUB_BLOB}/docs/figures/l0_control_robustness.md" target="_blank">docs/figures/l0_control_robustness.md</a>).
      </p>
    </div>
    """


def _l0_winner_share() -> str:
    """How unstable the control level's per-draw winner is, as a phrase."""
    stats = _board().all_detectors.level(CONTROL_LEVEL)
    if stats is None or not stats.best_detector_frequency:
        return "the artifact records no per-draw winners"
    return (
        f"{len(stats.best_detector_frequency)} different detectors win at least one draw, "
        f"and the most frequent takes only {stats.best_detector_draws} of "
        f"{stats.total_draws}"
    )


def _stat_card(label: str, value: str, note: str = "", color_class: str = "") -> str:
    return f"""
    <div class="rhob-stat">
      <div class="rhob-stat-label">{label}</div>
      <div class="rhob-stat-value {color_class}">{value}</div>
      <div class="rhob-stat-note">{note}</div>
    </div>
    """


_theme = gr.themes.Base(
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#0a0c10",
    body_text_color="#dfe4ea",
    background_fill_primary="#12151b",
    background_fill_secondary="#171b22",
    border_color_primary="#262b35",
    block_background_fill="#12151b",
    block_border_color="#262b35",
    block_label_text_color="#8891a0",
    button_primary_background_fill="#ffa93f",
    button_primary_text_color="#0a0c10",
)

with gr.Blocks(title="RHOB Leaderboard") as demo:
    _board_ref = _board()
    _best_transfer_row = _best_transfer()
    _best_rts_str = (
        f"{_best_transfer_row.rts:.3f}" if _best_transfer_row is not None else "—"
    )
    _best_rts_note = (
        f"{_best_transfer_row.name} — sign randomization on, single draw, no CI"
        if _best_transfer_row is not None
        else "superseded by sign randomization; not shown"
    )

    gr.HTML(
        """
        <div id="rhob-hero">
          <div class="rhob-eyebrow"><span class="dot"></span>LIVE · REWARD-HACKING ONSET BENCHMARK</div>
          <div class="rhob-title">RHOB <span class="accent">/</span> Leaderboard</div>
          <div class="rhob-sub">
            Read-only viewer over the committed leaderboard data in this repository.
            See the <a href="https://github.com/Aarav500/rhob/blob/main/docs/TUTORIAL_DETECTOR.md" target="_blank">Detector Tutorial</a>
            for how to evaluate your own detector, and the <b>How to Submit</b> tab below for how results get added here.
          </div>
          <div class="rhob-mirrors">
            MIRRORS &nbsp;
            <a href="https://huggingface.co/spaces/Aarav500/rhob-leaderboard" target="_blank">hf space</a> &nbsp;·&nbsp;
            <a href="http://54.208.200.139/" target="_blank">aws ec2</a> &nbsp;·&nbsp;
            <a href="https://rhob.aarav-shah.com/" target="_blank">rhob.aarav-shah.com</a>
          </div>
        </div>
        """
    )

    gr.HTML(_provenance_html())

    with gr.Row(elem_id="rhob-stats"):
        gr.HTML(_stat_card("Families", str(len(_board_ref.families())), "matched proxy/legit pairs"))
        gr.HTML(_stat_card("Detectors", str(len(_board_ref.detectors)), "L0 reward-only → L3 oracle"))
        gr.HTML(
            _stat_card(
                "Draws",
                str(_board_ref.n_replicates),
                f"independent replicates, {_board_ref.ci_percent}% CIs",
                "teal",
            )
        )
        gr.HTML(_stat_card("Best RTS", _best_rts_str, _best_rts_note, "amber"))
        gr.HTML(_stat_card("Complexity tiers", "5", "tabular → multi-agent"))

    gr.HTML(_ladder_html())
    gr.HTML(_l0_control_html())
    gr.HTML(_supervision_html())

    _rts_text = _rts_html()
    if _rts_text:
        gr.HTML(_rts_text)

    with gr.Tab("Standings"):
        with gr.Row():
            access_dropdown = gr.Dropdown(
                choices=_all_access_levels(), value="All", label="Access Level"
            )
            family_dropdown = gr.Dropdown(
                choices=_all_families(), value="All", label="Family"
            )
        table = gr.Dataframe(value=_standings_df("All", "All"), interactive=False)
        gr.Markdown(
            f"""
Sorted by the {_board_ref.n_replicates}-draw suite mean, descending, in every view.
**Cells measured** is the denominator that mean was taken over, and it is not the same
for every row: a detector whose channel most families do not ship measures far fewer
cells than a board-wide one. Selecting a family shows that family's own interval, and
`{NOT_MEASURED}` where the detector never measured it -- which is a different statement
from "scored at chance".
"""
        )

        def _refresh(access_level, family):
            return _standings_df(access_level, family)

        access_dropdown.change(_refresh, [access_dropdown, family_dropdown], table)
        family_dropdown.change(_refresh, [access_dropdown, family_dropdown], table)

    with gr.Tab("How to Submit"):
        gr.Markdown(
            """
## Submitting a Detector Result

This viewer is read-only by design (no auth, no write path -- see
`src/rhob/v3/leaderboard/board.py`'s "static files are fine" approach). To get a result
included here:

1. Evaluate your detector: `python -m rhob evaluate --detector your_detector.py`
2. Validate the submission: `python -m rhob validate submission.json`
3. Open a PR adding your submission under `leaderboard/`, or run
   `python -m rhob submit submission.json` to merge it into the tracked leaderboard
   files locally before committing.

A CI workflow (`.github/workflows/leaderboard_validate.yml`) automatically validates
any submission JSON added in a PR.

The standings on this page are the replicated board, `leaderboard/v5_replicated.json`,
built by `scripts/replicate_leaderboard.py` (one run of the full suite per draw) and
`scripts/aggregate_replication.py` (intervals over the draws). A single-draw submission
has no interval to show and appears there once it has been through that harness.

See [CONTRIBUTING.md](https://github.com/Aarav500/rhob/blob/main/CONTRIBUTING.md) for
the full admission-gate requirements if you're also proposing a new family.
"""
        )


if __name__ == "__main__":
    demo.launch(css=_CSS, theme=_theme)
