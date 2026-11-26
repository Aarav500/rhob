"""Generate full-scope v5 result plots from real experiment data.

Unlike ``scripts/plot_results.py`` (which plots the original 6-detector x
4-variant-set Family 1-2 case study from ``results/detector_evaluation/``),
this script plots the full v5 scope: 30 detectors x all registered families, from
``leaderboard/v5_leaderboard.json``, plus the cross-family transfer results
from ``leaderboard/cross_family_transfer.json`` once that experiment has run.

Outputs (written to ``docs/figures/``):
  v5_heatmap.png        30 detectors x all registered families, per-family AUROC; a
                         family that is ``null`` on the board for a detector (no
                         channel, or fewer episodes than the detector's declared
                         ``min_episodes``) is drawn grey with no number, never as 0.5
  v5_access_summary.png Mean AND best AUROC by access level (L0/L1/L2/L3), each detector
                         over the cells it is not null on, L0 without the families the
                         admission ledger marks DEGENERATE, duplicate detectors excluded.
                         The L0 note names the detectors measured on fewer families
  v5_access_summary.md  The same access-level table in Markdown, for citing in docs
  v5_transfer.png       Train vs. per-family-transfer AUROC (Reward MLP, State
                         Divergence, Trajectory MLP, Ensemble) -- only produced
                         if leaderboard/cross_family_transfer.json exists.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Run the checkout this file lives in, not whatever the editable install points at.
# In a git worktree those differ, and the failure is silent: the script imports a
# same-named module from the OTHER tree and publishes results attributed to this one.
# Enforced for every script by tests/test_scripts_run_their_own_checkout.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rhob.v3.leaderboard.access_summary import (  # noqa: E402
    ACCESS_LEVELS,
    degenerate_families_from_ledger,
    fewer_families_groups,
    render_access_summary_md,
    summarize_access_levels,
)

# Families 1-9 order matches Table tab:families-sigma in the paper; 10-14 are the
# v1.4 additions (2x REWARD_TAMPERING, 2x DECEPTIVE_ALIGNMENT, 1x RM_OVEROPTIMIZATION);
# 15-33 are numbered as in README.md's family list (15-18 MuJoCo, 19-23 RLHF-RM, 24-28
# PettingZoo, 29-33 sequence generation). The heatmap plots these in this order, then any
# other family on the board in the board's order, labelled by registry name with no
# number (see ``_heatmap_families``), so a label never carries a number README does not.
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
    "reward_channel_tampering",
    "sensor_calibration_tampering",
    "monitored_sandbagging",
    "eval_probe_sandbagging",
    "rlhf_reward_model_overopt",
    "mujoco_camping",
    "mujoco_goal_misgeneralization",
    "mujoco_joint_limit_gaming",
    "mujoco_sensor_decoupling",
    "rlhf_sparse_coverage_gaming",
    "rlhf_label_noise_exploitation",
    "rlhf_feature_blindspot_gaming",
    "rlhf_kl_penalty_gaming",
    "rlhf_preference_population_bias",
    "pettingzoo_tacit_collusion_gaming",
    "pettingzoo_communication_deception",
    "pettingzoo_free_rider_exploitation",
    "pettingzoo_fixed_opponent_exploitation",
    "pettingzoo_population_goodhart",
    "sequence_keyword_stuffing",
    "sequence_format_camping",
    "sequence_repetition_shortcut",
    "sequence_lexicon_gaming",
    "sequence_length_padding",
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
    "10. Reward Chan.\nTampering",
    "11. Sensor Calib.\nTampering",
    "12. Monitored\nSandbagging",
    "13. Eval-Probe\nSandbagging",
    "14. RLHF RM\nOveropt.",
    "15. MuJoCo\nCamping",
    "16. MuJoCo Goal\nMisgen.",
    "17. MuJoCo\nJoint-Limit Gaming",
    "18. MuJoCo Sensor-Channel\nDecoupling",
    "19. RM Sparse-Coverage\nGaming",
    "20. RM Label-Noise\nExploit.",
    "21. RM Feature-Blindspot\nGaming",
    "22. KL-Penalty\nGaming",
    "23. Preference-Population\nBias",
    "24. Tacit Collusion\nGaming",
    "25. Comm.-Channel\nDeception",
    "26. Free-Rider\nExploit.",
    "27. Fixed-Opponent\nExploit.",
    "28. Population-Level\nGoodhart",
    "29. Keyword-Stuffing\nGaming",
    "30. Format-Compliance\nCamping",
    "31. Repetition-Blind-Spot\nShortcut",
    "32. Lexicon-Sentiment\nGaming",
    "33. Length-Padding\nExploit",
]

# Detector order matches Section 5 (Detector Suite) of the paper: L0 (13), L1 (8), L2 (7),
# L3 (2 rows on the board, but only 1 independent detector -- Perfect Feature Oracle is a
# duplicate of the L2 Behavioral Threshold baseline and is excluded from the access-level
# aggregates by ``plot_access_summary``; see ``rhob.detectors.redundancy``). It stays in the
# heatmap, which is a per-detector view and not an aggregate.
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


def _heatmap_families(board: dict) -> tuple[list[str], list[str]]:
    """Every family on the board: :data:`FAMILY_ORDER` first, then the rest in board order.

    Returns the families and their axis labels. A family in :data:`FAMILY_ORDER` carries
    its README number; any other family is labelled by its registry name alone, since a
    number from its position here would not match README's. A board without
    ``families_evaluated`` (written before that key existed) falls back to the families
    its detectors report.
    """
    on_board = list(board.get("families_evaluated") or [])
    if not on_board:
        for info in board["results"].values():
            for fam in info.get("per_family") or {}:
                if fam not in on_board:
                    on_board.append(fam)
    families = [f for f in FAMILY_ORDER if f in on_board]
    labels = [
        lab.replace("\n", " ") for f, lab in zip(FAMILY_ORDER, FAMILY_LABELS) if f in on_board
    ]
    for fam in on_board:
        if fam not in families:
            families.append(fam)
            labels.append(fam)
    return families, labels


def plot_heatmap(board: dict, out_path: Path) -> None:
    results = board["results"]
    detectors = [d for d in DETECTOR_ORDER if d in results]
    families, labels = _heatmap_families(board)
    matrix = np.full((len(detectors), len(families)), np.nan)
    for i, det in enumerate(detectors):
        per_family = results[det].get("per_family") or {}
        for j, fam in enumerate(families):
            # ``null`` means not applicable: the family emits no channel this
            # detector's access level can read, or too few episodes of it, so nothing
            # was measured. Leave the heatmap cell as NaN (drawn grey, no annotation)
            # rather than colouring it as if a value had been observed -- in this
            # colour map 0.5 is white, so a blank cell would read as chance.
            if per_family.get(fam) is not None:
                matrix[i, j] = per_family[fam]

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#c8c8c8")
    fig, ax = plt.subplots(figsize=(max(11.0, 0.5 * len(families) + 4.0), 11))
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(labels, fontsize=7.5, rotation=60, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(detectors)))
    ax.set_yticklabels(detectors, fontsize=8)
    for i in range(len(detectors)):
        for j in range(len(families)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="black", fontsize=6)
    ax.set_title(
        f"RHOB v5: {len(detectors)} Detectors x {len(families)} Families (AUROC; "
        "grey = not applicable, no measurement)"
    )
    fig.colorbar(im, ax=ax, label="Discrimination AUROC", shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_access_summary(board: dict, out_path: Path) -> str:
    """Plot mean *and* best AUROC per access level, and return the same table as Markdown.

    Three changes from the mean-only version this replaces, all from the audit:

    1. The best detector at each level is plotted alongside the mean. A level's mean
       measures how many weak detectors happen to have been written for it as much as it
       measures the benchmark; the max is what speaks to "is the signal present at this
       access level?". On the board regenerated 2026-11-19 they diverge sharply at L1
       (mean 0.5701, max 0.9270) and L2 (mean 0.5716, max 0.9430).
    2. Perfect Feature Oracle is excluded from L3, because it is a relabelled copy of
       the L2 Behavioral Threshold baseline and agrees with it on 33/33 families. See
       :mod:`rhob.detectors.redundancy`. Including it made L3 look like a two-detector
       ceiling when it is one detector plus an echo of L2.
    3. The families the admission ledger marks DEGENERATE are held out of L0, and only
       L0. Their proxy is constant, nearly constant, or identical in every run, so an L0
       detector reads ~0.5 on them by construction rather than by matching, and
       averaging them into a negative control pads it with cells that could not have
       come out any other way. The set is read from ``admission/admission_ledger.json``
       at plot time rather than hardcoded here, so the figure tracks the ledger. Cells
       the ledger measures NOT ADMITTED are not held out; whether they should be is open
       (``ROADMAP.md``). See :mod:`rhob.v3.leaderboard.access_summary` for why the
       exclusion stops at L0 and what it costs in re-weighting.

    The spread bars are the SD across the detectors at a level -- suite heterogeneity,
    not measurement error (all detectors share a single evaluation draw), so they are
    drawn as a light band and labelled as such rather than as capped error bars, which
    read as confidence intervals.
    """
    summaries = summarize_access_levels(
        board["results"], degenerate_families=degenerate_families_from_ledger()
    )
    levels = list(ACCESS_LEVELS)

    def _stat(level: str, attr: str) -> float:
        value = getattr(summaries[level], attr)
        return np.nan if value is None else value

    means = [_stat(lv, "mean_auroc") for lv in levels]
    stds = [_stat(lv, "std_auroc") for lv in levels]
    maxes = [_stat(lv, "max_auroc") for lv in levels]
    ns = [summaries[lv].n_detectors for lv in levels]

    fig, ax = plt.subplots(figsize=(7.6, 5))
    x = np.arange(len(levels))
    ax.bar(x, means, width=0.62, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"],
           label="Mean across detectors at this level")
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="#33373f", elinewidth=8, alpha=0.28,
                label="+-1 SD across detectors (suite spread, not a CI)")
    ax.scatter(x, maxes, marker="_", s=900, linewidths=2.6, color="#111318", zorder=3,
               label="Best single detector at this level")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
    for i, (m, mx, n) in enumerate(zip(means, maxes, ns)):
        if not np.isnan(mx):
            ax.text(i, mx + 0.025, f"max {mx:.3f}", ha="center", fontsize=8.5)
        if not np.isnan(m):
            ax.text(i, 0.02, f"n={n}", ha="center", fontsize=9, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("AUROC (not-applicable cells left out)")
    # Headroom for a two-column legend above the bars: at L1 the best detector sits near
    # 0.93, where a corner legend would cover its marker.
    ax.set_ylim(0.0, 1.32)
    ax.set_title("AUROC by Access Level: Mean and Best Detector")
    # markerscale keeps the best-detector bar inside its legend handle: drawn at the
    # plot's s=900 it is wider than the handle and runs into its own label.
    ax.legend(fontsize=7.5, loc="upper center", ncol=2, framealpha=0.92, markerscale=0.45)

    # Both exclusions are stated on the figure itself. A plot that silently drops
    # detectors or families is the thing this whole section of the audit was about.
    notes = []
    excluded = [
        f"{name} (duplicate of {source})"
        for lv in levels
        for name, source in summaries[lv].excluded_duplicates
    ]
    if excluded:
        notes.append("Excluded from the level aggregates: " + "; ".join(excluded))
    proxy = summaries["L0"]
    if proxy.excluded_families:
        including = proxy.mean_auroc_including_degenerate
        fewer = [f"{', '.join(names)} over {n}" for n, names in fewer_families_groups(proxy)]
        notes.append(
            f"L0 is the negative control and is computed over {proxy.n_families_scored} "
            "families"
            + (
                f" ({'; '.join(fewer)}: shorter families are below the horizon each declares)"
                if fewer
                else ""
            )
            + f": {len(proxy.excluded_families)} whose proxy the admission ledger "
            "marks DEGENERATE are held out of L0 only ("
            + ", ".join(proxy.excluded_families)
            + f"). With them it reads {including:.4f} instead of {proxy.mean_auroc:.4f}. "
            "NOT ADMITTED cells stay in."
        )
    # Wrapped explicitly rather than with matplotlib's ``wrap=True``: that wraps at draw
    # time against the figure width, so the caller cannot know how many lines a note
    # became and the notes overprint each other.
    # The axes are laid out before the notes go in: laid out after, tight_layout shrinks
    # the axes to make room for every note line, and the lines, placed in axes units,
    # close up onto the tick labels. bbox_inches="tight" extends the saved figure instead.
    fig.tight_layout()
    lines = [line for note in notes for line in textwrap.wrap(note, width=105)]
    for i, line in enumerate(lines):
        ax.text(
            0.5, -0.125 - 0.052 * i, line, transform=ax.transAxes, ha="center",
            fontsize=7.5, color="#4a4f59",
        )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return render_access_summary_md(summaries)


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
           label="Transfer (Families 7-14, held-out)")
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
    ax.set_title("Cross-Family Transfer Gap (Train on 1-6, Test on 7-14)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    figures_dir = Path("docs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    board = load_leaderboard(Path("leaderboard/v5_leaderboard.json"))
    plot_heatmap(board, figures_dir / "v5_heatmap.png")
    summary_md = plot_access_summary(board, figures_dir / "v5_access_summary.png")
    summary_path = figures_dir / "v5_access_summary.md"
    summary_path.write_text(summary_md + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    print(summary_md)

    transfer_path = Path("leaderboard/cross_family_transfer.json")
    if transfer_path.exists():
        plot_transfer(transfer_path, figures_dir / "v5_transfer.png")
    else:
        print(f"skipping transfer plot: {transfer_path} not found yet "
              f"(run scripts/cross_family_transfer.py first)")


if __name__ == "__main__":
    main()
