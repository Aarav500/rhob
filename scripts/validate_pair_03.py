"""Validate continuous Pair 03 against the admission gate (30 seeds/variant)."""

from __future__ import annotations

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rhob.agents.dqn import train_camper  # noqa: E402
from rhob.environments.pair_03 import (
    CONFIG,
    L2_MIN,
    PAIR_ID,
    generate_hacking_run,
    generate_legitimate_run,
)  # noqa: E402
from rhob.evaluation.continuous_admission import compute_continuous_admission  # noqa: E402


@click.command()
@click.option("--n-seeds", default=30, show_default=True, help="Runs per variant.")
@click.option("--camper-seed", default=0, show_default=True, help="Seed for the shared camper.")
def main(n_seeds: int, camper_seed: int) -> None:
    click.echo("Training shared camping policy (DQN) ...")
    camper = train_camper(seed=camper_seed, box=CONFIG.box)
    click.echo(f"Generating {n_seeds} hacking + {n_seeds} legitimate runs for {PAIR_ID} ...")
    runs_a = [generate_hacking_run(camper, s) for s in range(n_seeds)]
    runs_b = [generate_legitimate_run(camper, 1000 + s) for s in range(n_seeds)]
    report = compute_continuous_admission(runs_a, runs_b, CONFIG, PAIR_ID, L2_MIN)
    click.echo("\n" + str(report))
    sys.exit(0 if report.admitted else 1)


if __name__ == "__main__":
    main()
