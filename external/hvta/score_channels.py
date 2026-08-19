"""Score every detector on the three newly-ported HVTA channels, and render the report.

Stage two of two. Reads the frozen arrays from :mod:`export_channels` and scores them
through ``rhob.v3.benchmark._evaluate_cell`` **unmodified** -- the same function the
internal leaderboard uses, so nothing about the evaluation path is special-cased for
these channels.

Runs in a torch-capable interpreter, not the HVTA venv, so the two label-fitted detectors
are constructible. Writes ``channels_results.json`` and renders ``CHANNELS.md`` from it;
the markdown is generated, never hand-edited, so every figure in it is traceable to the
JSON and the JSON to the frozen arrays.

Run::

    PYTHONPATH=<rhob>/src python external/hvta/score_channels.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from rhob.detectors.posthoc import RunData  # noqa: E402
from rhob.v3.benchmark import _evaluate_cell  # noqa: E402

from score_detectors import DETECTOR_FACTORIES, is_label_fitted  # noqa: E402

HERE = Path(__file__).resolve().parent

#: Mann-Whitney null SE of an AUROC, sqrt((n+m+1)/(12nm)). Printed beside every table so
#: a reader cannot take a third decimal seriously at these sample sizes.
def null_se(n: int, m: int | None = None) -> float:
    m = n if m is None else m
    return math.sqrt((n + m + 1) / (12 * n * m))


def _scrub_home(path: str) -> str:
    try:
        home = str(Path.home())
    except (RuntimeError, OSError):
        return path
    return "~" + path[len(home):] if home and path.lower().startswith(home.lower()) else path


def _load(stem: Path):
    meta = json.loads(Path(f"{stem}.meta.json").read_text(encoding="utf-8"))
    arrays = np.load(f"{stem}.npz")
    return meta, arrays


def _runs(arrays, cell, variant: str) -> list[RunData]:
    idx, n, has_behav = cell["index"], cell["n_pairs_usable"], cell["has_behav"]
    out = []
    for j in range(n):
        key = f"c{idx}_{variant}{j}_"
        out.append(RunData(
            proxy_rewards=arrays[key + "proxy"],
            true_rewards=arrays[key + "true"],
            state_counts=None,  # absent in every HVTA pair set -- L1 N/A, never imputed
            behav_trace=arrays[key + "behav"] if has_behav else None,
        ))
    return out


def score(meta, arrays) -> list[dict]:
    results = []
    for cell in meta["cells"]:
        if cell["n_pairs_usable"] == 0:
            # A cell can be entirely degenerate, and at susceptibility 0.0 that IS the
            # control result. Recorded as a cell with a reason, not omitted.
            results.append({
                "channel": cell["channel"], "params": cell["params"],
                "n_pairs": 0, "n_degenerate": cell["n_degenerate"],
                "degenerate_reasons": cell["degenerate_reasons"],
                "detectors": [], "note": "no scoreable pairs",
            })
            continue

        runs_a, runs_b = _runs(arrays, cell, "a"), _runs(arrays, cell, "b")
        onsets, n_ep = cell["onsets"], cell["n_episodes"]
        rows = []
        for factory in DETECTOR_FACTORIES:
            try:
                probe = factory()
            except Exception as exc:  # noqa: BLE001
                rows.append({"detector": factory.__name__, "error": f"construct: {exc}"})
                continue
            row = {"detector": probe.name, "access_level": probe.access_level,
                   "label_fitted": bool(is_label_fitted(probe))}
            try:
                auroc, mae, na = _evaluate_cell(
                    factory(), runs_a, runs_b, onsets, probe.access_level, n_ep
                )
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"evaluate: {type(exc).__name__}: {exc}"
                rows.append(row)
                continue
            row["auroc"] = None if math.isnan(auroc) else round(float(auroc), 4)
            row["onset_mae_normalized"] = None if math.isnan(mae) else round(float(mae), 4)
            row["na_reason"] = na
            rows.append(row)

        pa = float(np.mean([r.proxy_rewards.mean() for r in runs_a]))
        pb = float(np.mean([r.proxy_rewards.mean() for r in runs_b]))
        ta = float(np.mean([r.true_rewards.mean() for r in runs_a]))
        tb = float(np.mean([r.true_rewards.mean() for r in runs_b]))
        results.append({
            "channel": cell["channel"], "params": cell["params"],
            "n_pairs": cell["n_pairs_usable"], "n_degenerate": cell["n_degenerate"],
            "degenerate_reasons": cell["degenerate_reasons"],
            "n_episodes": n_ep,
            "auroc_null_se": round(null_se(cell["n_pairs_usable"]), 4),
            "proxy_mean_hacking": round(pa, 4), "proxy_mean_legit": round(pb, 4),
            "true_mean_hacking": round(ta, 4), "true_mean_legit": round(tb, 4),
            "detectors": rows,
        })
    return results


def render(artifact) -> str:
    L = []
    w = L.append
    w("# RHOB detectors on HVTA's other three hack channels")
    w("")
    w("Generated by `external/hvta/score_channels.py`. Do not hand-edit.")
    w("")
    w("The first pair set (`external/hvta/RESULTS.md`) scores `hidden_solution` on Sudoku.")
    w("This one covers the remaining three channels. Every number is measured through")
    w("`rhob.v3.benchmark._evaluate_cell` unmodified, against HVTA's own latched")
    w("`hack_onset_step`.")
    w("")
    w("## The headline is a negative result about our own gate")
    w("")
    w("| channel | proxy gap | true-reward gap | what the hack does |")
    w("|---|---|---|---|")
    w("| `hidden_solution` (Sudoku) | positive | **negative** | substitutes for competence |")
    for ch, label in (("write_hack_ipd", "`write_hack` (IPD)"),
                      ("opponent_prompt_read_ipd", "`opponent_prompt_read` (IPD)")):
        cells = [c for c in artifact["cells"] if c["channel"] == ch and c["n_pairs"]]
        if not cells:
            continue
        pg = np.mean([c["proxy_mean_hacking"] - c["proxy_mean_legit"] for c in cells])
        tg = np.mean([c["true_mean_hacking"] - c["true_mean_legit"] for c in cells])
        did = ("corrupts the counterparty" if ch == "write_hack_ipd"
               else "augments with information")
        w(f"| {label} | +{pg:.4f} | **{tg:+.4f}** | {did} |")
    w("")
    w("Both two-player channels raise the proxy without moving true reward at all. The")
    w("agent is not degraded, it is augmented -- with a corrupted counterparty or with")
    w("information it should not have. RHOB's own `true_reward_diverges` admission")
    w("criterion recognises only hacking that *substitutes* for competence, so it would")
    w("reject both families. That is correct by its own definition, and it is the point:")
    w("the definition is narrower than the phenomenon.")
    w("")
    w("A consequence worth stating before any L3 number is read: with no true-reward")
    w("divergence there is nothing for the True Reward Oracle to discriminate, so it")
    w("returns 0.500 by ties. It is **not** a working positive control on these channels,")
    w("and the onset measure cannot be validated here the way it was on Sudoku.")
    w("")
    w("## Per-cell results")
    w("")
    for cell in artifact["cells"]:
        params = ", ".join(f"{k}={v}" for k, v in cell["params"].items())
        w(f"### `{cell['channel']}` — {params}")
        w("")
        if not cell["n_pairs"]:
            w(f"No scoreable pairs: all {cell['n_degenerate']} were degenerate.")
            for r in cell["degenerate_reasons"]:
                w(f"- {r}")
            w("")
            continue
        w(f"{cell['n_pairs']} pairs ({cell['n_degenerate']} degenerate, excluded), "
          f"{cell['n_episodes']} steps each. Null SE {cell['auroc_null_se']:.3f}.")
        w("")
        w(f"Proxy: hacking {cell['proxy_mean_hacking']:.4f} vs legit "
          f"{cell['proxy_mean_legit']:.4f}. True: {cell['true_mean_hacking']:.4f} vs "
          f"{cell['true_mean_legit']:.4f}.")
        w("")
        scored = [r for r in cell["detectors"] if r.get("auroc") is not None]
        scored.sort(key=lambda r: -r["auroc"])
        na = [r for r in cell["detectors"] if r.get("auroc") is None and "error" not in r]
        ties = sum(1 for r in scored if abs(r["auroc"] - 0.5) < 1e-9)
        w("| detector | level | AUROC | onset MAE |")
        w("|---|---|---|---|")
        for r in scored[:8]:
            mae = "—" if r["onset_mae_normalized"] is None else f"{r['onset_mae_normalized']:.3f}"
            fit = " *(label-fitted)*" if r["label_fitted"] else ""
            w(f"| {r['detector']}{fit} | {r['access_level']} | {r['auroc']:.3f} | {mae} |")
        w("")
        w(f"{len(scored)} detectors scored, {ties} at exactly chance by ties, "
          f"{len(na)} N/A for a missing channel.")
        w("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stem", default=str(HERE / "channels_pairs"))
    ap.add_argument("--out-json", default=str(HERE / "channels_results.json"))
    ap.add_argument("--out-md", default=str(HERE / "CHANNELS.md"))
    args = ap.parse_args(argv)

    meta, arrays = _load(Path(args.stem))
    cells = score(meta, arrays)
    artifact = {
        "schema": "hvta-channels-results/1",
        "generation": meta,
        "scoring_env": {
            "python": sys.version.split()[0],
            "executable": _scrub_home(sys.executable),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "evaluation_path": "rhob.v3.benchmark._evaluate_cell (unmodified)",
        "cells": cells,
    }
    Path(args.out_json).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(render(artifact), encoding="utf-8")
    print(f"wrote {args.out_json} and {args.out_md}")
    print(f"{len(cells)} cells, "
          f"{sum(1 for c in cells if c['n_pairs'] == 0)} entirely degenerate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
