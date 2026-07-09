"""RHOB interactive leaderboard viewer (Gradio).

A read-only viewer over the committed leaderboard JSON files -- no submission form,
no auth, no write path. Community detector results are submitted via a PR touching
``leaderboard/`` (validated by ``python -m rhob validate``, see
``.github/workflows/leaderboard_validate.yml``), not through this UI. This matches the
project's existing "static files are fine" leaderboard design
(see ``src/rhob/v3/leaderboard/board.py``).

Run locally:
    pip install -e ".[space]"
    python space/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rhob.v3.leaderboard.adapters import load_any_leaderboard_json  # noqa: E402
from rhob.v3.leaderboard.board import Leaderboard  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD_FILES = [
    REPO_ROOT / "leaderboard" / "v5_leaderboard.json",
    REPO_ROOT / "leaderboard" / "leaderboard.json",
]


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
        "how results get added here."
    )

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
