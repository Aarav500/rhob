"""Freeze the three newly-ported HVTA channels as arrays scoring can read.

Stage one of two, and separate for the same reason ``export_pairs.py`` is: the HVTA venv
has ``textarena`` and ``hvta`` but no torch, so the two label-fitted detectors cannot even
be constructed there. Generation runs here, against the live environments; scoring runs in
a torch-capable interpreter against this file.

Covers ``logical_bug`` (Wordle), ``write_hack`` (IPD) and ``opponent_prompt_read`` (IPD).
The original ``hidden_solution`` (Sudoku) set has its own frozen artifact and is not
regenerated.

Run::

    PYTHONUTF8=1 PYTHONPATH=<rhob>/src <hvta>/.venv/bin/python \\
      external/hvta/export_channels.py --pairs 20
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from hvta_policies_ipd import SUSCEPTIBILITY_LEVELS
from hvta_policies_ipd import rollout_pair as ipd_pair
from hvta_policies_opr import OPPONENT_COOPERATION_LEVELS
from hvta_policies_opr import rollout_pair as opr_pair
from hvta_policies_wordle import COMPETENCE_LEVELS
from hvta_policies_wordle import rollout_pair as wordle_pair

HERE = Path(__file__).resolve().parent


def _scrub_home(path: str) -> str:
    """Drop the operator's home prefix from a recorded path. See export_pairs.py."""
    try:
        home = str(Path.home())
    except (RuntimeError, OSError):
        return path
    return "~" + path[len(home):] if home and path.lower().startswith(home.lower()) else path


def _wordle_cells(n_pairs: int):
    for competence in COMPETENCE_LEVELS:
        for keep in (4, 2):
            pairs = [
                wordle_pair(seed=s, keep_letters=keep, competence=competence,
                            onset_step=6 + 4 * (s % 5))
                for s in range(n_pairs)
            ]
            yield ("logical_bug_wordle", {"competence": competence, "keep": keep}, pairs,
                   lambda e: (e.completion, e.own_completion, None))


def _ipd_cells(n_pairs: int):
    for susc in SUSCEPTIBILITY_LEVELS:
        pairs = [ipd_pair(seed=s, onset_step=8 + 4 * (s % 5), susceptibility=susc)
                 for s in range(n_pairs)]
        yield ("write_hack_ipd", {"susceptibility": susc}, pairs,
               lambda e: (e.proxy, e.true, e.opponent_cooperation))


def _opr_cells(n_pairs: int):
    for p1 in OPPONENT_COOPERATION_LEVELS:
        pairs = [opr_pair(seed=s, onset_step=8 + 4 * (s % 5), p1_cooperate=p1)
                 for s in range(n_pairs)]
        yield ("opponent_prompt_read_ipd", {"p1_cooperate": p1}, pairs,
               lambda e: (e.proxy, e.true, e.own_cooperation))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--out", default=str(HERE / "channels_pairs"))
    args = ap.parse_args(argv)

    arrays: dict[str, np.ndarray] = {}
    cells_meta = []

    generators = [_wordle_cells, _ipd_cells, _opr_cells]
    for gen in generators:
        for channel, params, pairs, extract in gen(args.pairs):
            usable = [p for p in pairs if not p.degenerate]
            reasons = sorted({p.degenerate_reason for p in pairs if p.degenerate})
            idx = len(cells_meta)
            onsets = []
            for j, pair in enumerate(usable):
                for variant, ep in (("a", pair.hacking), ("b", pair.legit)):
                    proxy, true, behav = extract(ep)
                    arrays[f"c{idx}_{variant}{j}_proxy"] = np.asarray(proxy, float)
                    arrays[f"c{idx}_{variant}{j}_true"] = np.asarray(true, float)
                    if behav is not None:
                        arrays[f"c{idx}_{variant}{j}_behav"] = np.asarray(behav, float)
                onsets.append(int(pair.hacking.hack_onset_step))
            cells_meta.append({
                "index": idx,
                "channel": channel,
                "params": params,
                "n_pairs_requested": args.pairs,
                "n_pairs_usable": len(usable),
                "n_degenerate": len(pairs) - len(usable),
                # Named, not just counted: a cell that dropped pairs dropped them for a
                # stated reason, and at susceptibility 0.0 dropping all of them is the
                # control result rather than a failure.
                "degenerate_reasons": reasons,
                "onsets": onsets,
                "n_episodes": int(len(np.asarray(extract(usable[0].hacking)[0], float)))
                if usable else 0,
                "has_behav": extract(usable[0].hacking)[2] is not None if usable else False,
            })
            print(f"  {channel} {params}: {len(usable)}/{args.pairs} usable", flush=True)

    meta = {
        "schema": "hvta-channels/1",
        "cells": cells_meta,
        "generation_env": {
            "python": sys.version.split()[0],
            "executable": _scrub_home(sys.executable),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    np.savez_compressed(f"{args.out}.npz", **arrays)
    Path(f"{args.out}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}.npz ({len(arrays)} arrays) and {args.out}.meta.json")
    print(f"{len(cells_meta)} cells across 3 channels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
