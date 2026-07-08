"""Generate the 4 result plots from ``results/detector_evaluation/results_table.csv``.

Plot 1: Heatmap (6 detectors x 4 pairs, AUROC)
Plot 2: Access-level summary (best-L0/L1/L2 per pair)
Plot 3: Difficulty curve (AUROC vs. pair difficulty, one line per detector)
Plot 4: Onset timing accuracy (normalized MAE, grouped bars)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAIRS = ["tier1/pair_01_gridworld", "tier2/pair_02_easy", "tier2/pair_03_medium", "tier2/pair_04_hard"]
PAIR_LABELS = ["Pair 01\n(gridworld)", "Pair 02\n(easy)", "Pair 03\n(medium)", "Pair 04\n(hard)"]
# Reference L2-AUROC from each pair's admission certificate, used as the x-axis
# difficulty ordinate for the difficulty-curve plot.
ADMISSION_L2 = {"tier1/pair_01_gridworld": 1.000, "tier2/pair_02_easy": 0.977, "tier2/pair_03_medium": 0.891, "tier2/pair_04_hard": 0.821}
DETECTOR_ORDER = ["Reward Threshold", "Reward CUSUM", "Reward MLP", "State Divergence", "Behavioral Threshold", "Trajectory MLP"]
ACCESS_LEVEL = {"Reward Threshold": "L0", "Reward CUSUM": "L0", "Reward MLP": "L0", "State Divergence": "L1", "Behavioral Threshold": "L2", "Trajectory MLP": "L2"}


def load_results(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_heatmap(rows: list[dict], out_path: Path) -> None:
    matrix = np.full((len(DETECTOR_ORDER), len(PAIRS)), np.nan)
    for row in rows:
        if row["detector"] in DETECTOR_ORDER and row["pair_id"] in PAIRS:
            i = DETECTOR_ORDER.index(row["detector"])
            j = PAIRS.index(row["pair_id"])
            matrix[i, j] = float(row["discrimination_auroc"])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(PAIRS)))
    ax.set_xticklabels(PAIR_LABELS)
    ax.set_yticks(range(len(DETECTOR_ORDER)))
    ax.set_yticklabels(DETECTOR_ORDER)
    for i in range(len(DETECTOR_ORDER)):
        for j in range(len(PAIRS)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="black", fontsize=10)
    ax.set_title("Detection Performance Across Access Levels and Difficulty")
    fig.colorbar(im, ax=ax, label="Discrimination AUROC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_access_levels(rows: list[dict], out_path: Path) -> None:
    best_by_level = {pid: {"L0": 0.0, "L1": 0.0, "L2": 0.0} for pid in PAIRS}
    for row in rows:
        pid, det = row["pair_id"], row["detector"]
        if pid not in PAIRS or det not in ACCESS_LEVEL:
            continue
        level = ACCESS_LEVEL[det]
        auroc = float(row["discrimination_auroc"])
        best_by_level[pid][level] = max(best_by_level[pid][level], auroc)

    x = np.arange(len(PAIRS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, level in enumerate(["L0", "L1", "L2"]):
        vals = [best_by_level[pid][level] for pid in PAIRS]
        ax.bar(x + (k - 1) * width, vals, width, label=level)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(PAIR_LABELS)
    ax.set_ylabel("Best discrimination AUROC")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Best Detector at Each Access Level")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_difficulty_curve(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    difficulty_x = [ADMISSION_L2[pid] for pid in PAIRS]
    for det in DETECTOR_ORDER:
        vals = []
        for pid in PAIRS:
            match = [float(r["discrimination_auroc"]) for r in rows if r["detector"] == det and r["pair_id"] == pid]
            vals.append(match[0] if match else np.nan)
        ax.plot(difficulty_x, vals, marker="o", label=f"{det} ({ACCESS_LEVEL[det]})")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("Pair difficulty (admission L2-AUROC)")
    ax.set_ylabel("Discrimination AUROC")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Detector Performance vs. Benchmark Difficulty")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_onset_timing(rows: list[dict], out_path: Path) -> None:
    x = np.arange(len(PAIRS))
    width = 0.13
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, det in enumerate(DETECTOR_ORDER):
        vals = []
        for pid in PAIRS:
            match = [float(r["onset_mae"]) for r in rows if r["detector"] == det and r["pair_id"] == pid]
            vals.append(match[0] if match else np.nan)
        ax.bar(x + (k - 2.5) * width, vals, width, label=det)
    ax.set_xticks(x)
    ax.set_xticklabels(PAIR_LABELS)
    ax.set_ylabel("Onset MAE (normalized)")
    ax.set_title("Onset Localization Accuracy (lower = better)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    results_dir = Path("results/detector_evaluation")
    rows = load_results(results_dir / "results_table.csv")

    plot_heatmap(rows, results_dir / "heatmap.png")
    plot_access_levels(rows, results_dir / "access_levels.png")
    plot_difficulty_curve(rows, results_dir / "difficulty_curve.png")
    plot_onset_timing(rows, results_dir / "onset_timing.png")

    print(f"4 plots written to {results_dir}/")


if __name__ == "__main__":
    main()
