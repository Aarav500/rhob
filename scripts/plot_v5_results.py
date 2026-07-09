"""Generate full-scope v5 result plots from real experiment data.

Unlike ``scripts/plot_results.py`` (which plots the original 6-detector x
4-variant-set Family 1-2 case study from ``results/detector_evaluation/``),
this script plots the full v5 scope: 30 detectors x 9 families, from
``leaderboard/v5_leaderboard.json``, plus the cross-family transfer results
from ``leaderboard/cross_family_transfer.json`` once that experiment has run.

Outputs (written to ``paper/figures/``):
  v5_heatmap.png        30 detectors x 9 families, per-family AUROC
  v5_access_summary.png Mean AUROC by access level (L0/L1/L2/L3), across all 9 families
  v5_transfer.png       Train vs. per-family-transfer AUROC (Reward MLP, State
                         Divergence, Trajectory MLP, Ensemble) -- only produced
                         if leaderboard/cross_family_transfer.json exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Family order matches Table tab:families-sigma in the paper (Family 1 .. Family 9).
FAMILY_ORDER = [
    "gridworld_camping",
    "continuous_camping",
    "proxy_correlation_gaming",
    "shortcut_exploitation",
    "novelty_farming",
    "orbit_chirality",
    "goal_misgeneralization",
    "physics_exploitation",
    "distributional_shift",
]
FAMILY_LABELS = [
    "1. Gridworld\nCamping",
    "2. Continuous\nCamping",
    "3. Proxy Corr.\nGaming",
    "4. Shortcut\nExploit.",
    "5. Novelty\nFarming",
    "6. Orbit\nChirality",
    "7. Goal\nMisgen.",
    "8. Physics\nExploit.",
    "9. Distrib.\nShift",
]

# Detector order matches Section 5 (Detector Suite) of the paper: L0 (13), L1 (8), L2 (8), L3 (2).
DETECTOR_ORDER = [
    "Reward Peak", "Reward Autocorrelation", "Reward Skewness", "Reward Trend",
    "Reward Threshold", "Reward CUSUM", "Reward Variance Ratio", "Reward KDE",
    "Spectral Reward", "Variance Window", "Max Plateau", "Gradient Reversal", "Reward MLP",
    "State Frequency Anomaly", "Occupancy Polarization", "Centroid Drift", "State Coverage Rate",
    "Visitation Entropy Trend", "Bimodal Occupancy", "Transition Entropy", "State Divergence",
    "Behavioral Threshold", "Angular Momentum", "Centroid Tracker", "Feature Magnitude",
    "Feature Consistency", "Reward-Feature Correlation", "Trajectory MLP",
    "True Reward Oracle", "Perfect Feature Oracle",
]


def load_leaderboard(path: Path) -> dict:
    return json.loads(path.read_text())


def plot_heatmap(board: dict, out_path: Path) -> None:
    results = board["results"]
    detectors = [d for d in DETECTOR_ORDER if d in results]
    matrix = np.full((len(detectors), len(FAMILY_ORDER)), np.nan)
    for i, det in enumerate(detectors):
        per_family = results[det].get("per_family", {})
        for j, fam in enumerate(FAMILY_ORDER):
            if fam in per_family:
                matrix[i, j] = per_family[fam]

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(FAMILY_ORDER)))
    ax.set_xticklabels(FAMILY_LABELS, fontsize=8)
    ax.set_yticks(range(len(detectors)))
    ax.set_yticklabels(detectors, fontsize=8)
    for i in range(len(detectors)):
        for j in range(len(FAMILY_ORDER)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="black", fontsize=6.5)
    ax.set_title(f"RHOB v5: {len(detectors)} Detectors x {len(FAMILY_ORDER)} Families (AUROC)")
    fig.colorbar(im, ax=ax, label="Discrimination AUROC", shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_access_summary(board: dict, out_path: Path) -> None:
    results = board["results"]
    by_level: dict[str, list[float]] = {"L0": [], "L1": [], "L2": [], "L3": []}
    for det, data in results.items():
        level = data.get("access_level")
        auroc = data.get("overall_auroc")
        if level in by_level and auroc is not None:
            by_level[level].append(auroc)

    levels = ["L0", "L1", "L2", "L3"]
    means = [np.mean(by_level[lv]) if by_level[lv] else np.nan for lv in levels]
    stds = [np.std(by_level[lv]) if by_level[lv] else np.nan for lv in levels]
    ns = [len(by_level[lv]) for lv in levels]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(levels))
    ax.bar(x, means, yerr=stds, capsize=6, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
    for i, (m, n) in enumerate(zip(means, ns)):
        if not np.isnan(m):
            ax.text(i, m + 0.03, f"n={n}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("Mean overall AUROC (across all 9 families)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Access-Level Hierarchy Holds at Full v5 Scale")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_transfer(transfer_path: Path, out_path: Path) -> None:
    """Plot train vs. transfer AUROC.

    Neural-net detectors (Reward MLP, Trajectory MLP, Ensemble) are run across
    multiple independently-seeded trials and stored under ``*_mean``/``*_std``
    keys (see ``scripts/cross_family_transfer.py``); deterministic detectors
    (State Divergence) keep the older single-run keys. Handle both.
    """
    data = json.loads(transfer_path.read_text())
    results = data["results"]
    names = list(results.keys())

    def get(n: str, key: str) -> float:
        r = results[n]
        return r.get(f"{key}_mean", r.get(key))

    def get_std(n: str, key: str) -> float:
        return results[n].get(f"{key}_std", 0.0)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(names))
    width = 0.35
    train_vals = [get(n, "train_auroc") for n in names]
    train_errs = [get_std(n, "train_auroc") for n in names]
    transfer_vals = [get(n, "avg_transfer_auroc") for n in names]
    transfer_errs = [get_std(n, "avg_transfer_auroc") for n in names]
    ax.bar(x - width / 2, train_vals, width, yerr=train_errs, capsize=3,
           label="Train (Families 1-6, in-distribution)")
    ax.bar(x + width / 2, transfer_vals, width, yerr=transfer_errs, capsize=3,
           label="Transfer (Families 7-9, held-out)")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    for i, n in enumerate(names):
        gap = results[n].get("generalization_gap_pct", results[n].get("generalization_gap_pct_mean"))
        # gap = (train - transfer) / train * 100, so a positive gap is a real
        # drop from train to transfer and a negative gap means transfer beat
        # train. Label as a signed "change" so it reads correctly either way
        # instead of always showing a leading minus sign.
        ax.text(i, max(train_vals[i], transfer_vals[i]) + 0.05, f"{-gap:+.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.0, 1.1)
    ax.set_title("Cross-Family Transfer Gap (Train on 1-6, Test on 7-9)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    figures_dir = Path("paper/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    board = load_leaderboard(Path("leaderboard/v5_leaderboard.json"))
    plot_heatmap(board, figures_dir / "v5_heatmap.png")
    plot_access_summary(board, figures_dir / "v5_access_summary.png")

    transfer_path = Path("leaderboard/cross_family_transfer.json")
    if transfer_path.exists():
        plot_transfer(transfer_path, figures_dir / "v5_transfer.png")
    else:
        print(f"skipping transfer plot: {transfer_path} not found yet "
              f"(run scripts/cross_family_transfer.py first)")


if __name__ == "__main__":
    main()
