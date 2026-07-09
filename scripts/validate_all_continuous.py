"""Validate all continuous pairs (02, 03, 04) and render the difficulty-spectrum plot.

Trains the shared camping policy once, admits each pair at ``--n-seeds`` seeds, and
plots L2-AUROC across the three difficulty levels (the paper's difficulty spectrum).
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rhob.agents.dqn import train_camper  # noqa: E402
from rhob.environments import pair_02, pair_03, pair_04  # noqa: E402
from rhob.evaluation.continuous_admission import compute_continuous_admission  # noqa: E402

PAIRS = [pair_04, pair_03, pair_02]  # hard -> easy


@click.command()
@click.option("--n-seeds", default=30, show_default=True)
@click.option("--camper-seed", default=0, show_default=True)
@click.option("--plot/--no-plot", default=True, show_default=True)
def main(n_seeds: int, camper_seed: int, plot: bool) -> None:
    click.echo("Training shared camping policy (DQN) once for all pairs ...")
    camper = train_camper(seed=camper_seed)

    reports = []
    for pair in PAIRS:
        a = [pair.generate_hacking_run(camper, s) for s in range(n_seeds)]
        b = [pair.generate_legitimate_run(camper, 1000 + s) for s in range(n_seeds)]
        rep = compute_continuous_admission(a, b, pair.CONFIG, pair.PAIR_ID, pair.L2_MIN)
        reports.append((pair, rep))
        click.echo("\n" + str(rep))

    click.echo("\n" + "=" * 64)
    click.echo("CONTINUOUS TIER SUMMARY")
    click.echo("=" * 64)
    click.echo(f"{'pair':<24}{'d':>5}{'L2':>7}{'TV':>7}{'true(B-A)':>11}  admit")
    for pair, rep in reports:
        click.echo(
            f"{pair.PAIR_ID:<24}{pair.CONFIG.separation:>5.2f}{rep.l2_auroc:>7.3f}"
            f"{rep.tv:>7.3f}{rep.true_diff:>11.3f}  {'YES' if rep.admitted else 'no'}"
        )
    n_admit = sum(r.admitted for _, r in reports)
    click.echo(f"\nAdmitted: {n_admit}/{len(reports)}")

    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ds = [p.CONFIG.separation for p, _ in reports]
        l2 = [r.l2_auroc for _, r in reports]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ds, l2, marker="o")
        for p, r in reports:
            ax.annotate(
                p.PAIR_ID.split("/")[-1],
                (p.CONFIG.separation, r.l2_auroc),
                textcoords="offset points",
                xytext=(0, 8),
                fontsize=8,
            )
        ax.axhspan(0.80, 0.95, color="green", alpha=0.1, label="medium band")
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_xlabel("attractor separation d")
        ax.set_ylabel("L2-AUROC")
        ax.set_title("Continuous-tier difficulty spectrum (learned DQN camper)")
        ax.set_ylim(0.45, 1.02)
        ax.legend()
        fig.tight_layout()
        out = Path(__file__).resolve().parents[1] / "docs" / "figures" / "difficulty_spectrum.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        click.echo(f"Saved difficulty-spectrum plot to {out}")

    sys.exit(0 if n_admit == len(reports) else 1)


if __name__ == "__main__":
    main()
