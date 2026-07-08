"""Validate CR1 Pair 01 against the admission gate.

Generates ``--n-seeds`` runs per variant, computes the admission report, and
prints PASS/FAIL for every criterion. Exit code 0 iff the pair is ADMITTED.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rhob.environments.pair_01 import (  # noqa: E402
    DEFAULT,
    generate_run,
    label_onset,
    make_hacking_env,
    make_legitimate_env,
)
from rhob.evaluation.admission_gate import compute_admission  # noqa: E402


@click.command()
@click.option("--n-seeds", default=30, show_default=True, help="Runs per variant.")
def main(n_seeds: int) -> None:
    cfg = DEFAULT
    click.echo(f"Generating {n_seeds} runs per variant (Pair 01) ...")
    # Disjoint seed ranges so the two variants never share a random stream.
    runs_hack = [generate_run(make_hacking_env(cfg), seed=s, config=cfg) for s in range(n_seeds)]
    runs_legit = [
        generate_run(make_legitimate_env(cfg), seed=1000 + s, config=cfg) for s in range(n_seeds)
    ]

    report = compute_admission(runs_hack, runs_legit, cfg)

    # Supporting diagnostics.
    w = cfg.steady_window

    def mean_last(runs, attr):
        return float(np.mean([getattr(r, attr)[-w:].mean() for r in runs]))

    onsets_h = [
        label_onset(r.frac_tile2, cfg.onset_threshold, cfg.onset_sustain) for r in runs_hack
    ]
    onsets_l = [
        label_onset(r.frac_tile2, cfg.onset_threshold, cfg.onset_sustain) for r in runs_legit
    ]
    det = lambda xs: [x for x in xs if x >= 0]  # noqa: E731

    click.echo("\nSupporting statistics (last-100-episode means):")
    click.echo(
        f"  proxy   A={mean_last(runs_hack, 'proxy'):7.2f}  B={mean_last(runs_legit, 'proxy'):7.2f}"
    )
    click.echo(
        f"  true    A={mean_last(runs_hack, 'true'):7.2f}  B={mean_last(runs_legit, 'true'):7.2f}"
    )
    click.echo(
        f"  behav   A={mean_last(runs_hack, 'behav'):7.2f}  B={mean_last(runs_legit, 'behav'):7.2f}"
    )
    if det(onsets_h):
        click.echo(
            f"  onset A: mean={np.mean(det(onsets_h)):.1f} std={np.std(det(onsets_h)):.1f} "
            f"({len(det(onsets_h))}/{n_seeds})"
        )
    if det(onsets_l):
        click.echo(
            f"  onset B: mean={np.mean(det(onsets_l)):.1f} std={np.std(det(onsets_l)):.1f} "
            f"({len(det(onsets_l))}/{n_seeds})"
        )

    click.echo("\n" + str(report))
    sys.exit(0 if report.admitted else 1)


if __name__ == "__main__":
    main()
