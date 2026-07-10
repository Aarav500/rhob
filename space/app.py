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
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd

from rhob.v3.leaderboard.adapters import load_any_leaderboard_json
from rhob.v3.leaderboard.board import Leaderboard

_HERE = Path(__file__).resolve().parent
_CANDIDATE_LEADERBOARD_DIRS = [
    _HERE / "leaderboard",  # deployed Space: deploy step copies data alongside app.py
    _HERE.parent / "leaderboard",  # local dev: running from within the full repo clone
]
_LEADERBOARD_DIR = next((d for d in _CANDIDATE_LEADERBOARD_DIRS if d.is_dir()), _CANDIDATE_LEADERBOARD_DIRS[0])
LEADERBOARD_FILES = [
    _LEADERBOARD_DIR / "v5_leaderboard.json",
    _LEADERBOARD_DIR / "leaderboard.json",
]
_TRANSFER_FILE = _LEADERBOARD_DIR / "cross_family_transfer.json"


def _rts_markdown() -> str:
    """Render the RHOB Transfer Score (RTS) headline table from the transfer results.

    RTS is RHOB's designated headline metric: mean AUROC on the 8 held-out,
    mechanistically-unseen test families, after training only on Families 1-6.
    """
    if not _TRANSFER_FILE.exists():
        return ""
    data = json.loads(_TRANSFER_FILE.read_text())
    rows = []
    for name, r in data.get("results", {}).items():
        rts = r.get("avg_transfer_auroc_mean", r.get("avg_transfer_auroc"))
        level = r.get("access_level", "")
        if rts is None:
            continue
        tag = "chance" if rts < 0.55 else ("near-perfect" if rts > 0.98 else "")
        rows.append(f"| {name} | {level} | **{rts:.3f}**{f' — {tag}' if tag else ''} |")
    if not rows:
        return ""
    table = "\n".join(rows)
    return (
        "### The RHOB Transfer Score (RTS)\n"
        "Train on 6 hacking mechanisms, test on 8 never seen. RTS = mean AUROC on the "
        "held-out mechanisms -- the number every detector submitted to RHOB gets scored on.\n\n"
        "| Detector class | Access level | RTS (transfer AUROC) |\n"
        "|---|---|---|\n" + table
    )


def _load_combined() -> Leaderboard:
    """Load every known leaderboard file, skipping any that don't exist or don't parse."""
    combined = Leaderboard()
    for path in LEADERBOARD_FILES:
        if not path.exists():
            continue
        try:
            board = load_any_leaderboard_json(path)
        except (ValueError, OSError):
            continue
        for entry in board.entries:
            combined.add(entry)
    return combined


def _standings_df(access_level: str, family: str) -> pd.DataFrame:
    board = _load_combined()
    rows = []
    for e in board.standings():
        if access_level != "All" and e.access_level != access_level:
            continue
        family_score = e.family_auroc.get(family) if family != "All" else None
        rows.append(
            {
                "Detector": e.detector_name,
                "Access Level": e.access_level,
                "Overall AUROC": round(e.overall_auroc, 3) if e.overall_auroc is not None else "-",
                f"{family} AUROC" if family != "All" else "": (
                    round(family_score, 3) if family_score is not None else "-"
                ) if family != "All" else "",
                "Cells": e.n_cells,
                "Author": e.author,
            }
        )
    df = pd.DataFrame(rows)
    if family == "All" and "" in df.columns:
        df = df.drop(columns=[""])
    return df


def _all_families() -> list[str]:
    board = _load_combined()
    families: set[str] = set()
    for e in board.entries:
        families.update(e.family_auroc.keys())
    return ["All"] + sorted(families)


def _all_access_levels() -> list[str]:
    return ["All", "L0", "L1", "L2", "L3"]


with gr.Blocks(title="RHOB Leaderboard") as demo:
    gr.Markdown(
        "# RHOB — Reward Hacking Onset Benchmark Leaderboard\n"
        "Read-only viewer over the committed leaderboard data in this repository. "
        "See the [Detector Tutorial](https://github.com/Aarav500/rhob/blob/main/docs/TUTORIAL_DETECTOR.md) "
        "for how to evaluate your own detector, and the **How to Submit** tab below for "
        "how results get added here.\n\n"
        "Also mirrored at: "
        "[HF Space](https://huggingface.co/spaces/Aarav500/rhob-leaderboard) | "
        "[AWS EC2](http://54.208.200.139/) | "
        "[rhob.aarav-shah.com](https://rhob.aarav-shah.com/)"
    )

    _rts_text = _rts_markdown()
    if _rts_text:
        gr.Markdown(_rts_text)

    with gr.Tab("Standings"):
        with gr.Row():
            access_dropdown = gr.Dropdown(
                choices=_all_access_levels(), value="All", label="Access Level"
            )
            family_dropdown = gr.Dropdown(
                choices=_all_families(), value="All", label="Family"
            )
        table = gr.Dataframe(value=_standings_df("All", "All"), interactive=False)

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

See [CONTRIBUTING.md](https://github.com/Aarav500/rhob/blob/main/CONTRIBUTING.md) for
the full admission-gate requirements if you're also proposing a new family.
"""
        )


if __name__ == "__main__":
    demo.launch()
