"""Generate the GridWorld-Wireheading trajectory dataset (Milestone 1).

Trains tabular Q-learning agents on the GridWorld-Wireheading environment and
writes the resulting trajectories (with oracle onset labels) to an HDF5 file.

By default it produces a 70/30 hacking/clean split of 10 runs, matching the
Milestone 1 dataset specification. Detectors are then evaluated on the
pre-recorded file -- training is fully decoupled from detection.

Usage::

    python scripts/generate_gridworld_data.py --output data/gridworld_wireheading.h5
    python scripts/generate_gridworld_data.py --n-hacking 35 --n-clean 15   # 50 seeds
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

# Allow running directly from the repo without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rhob.data.storage import save_dataset  # noqa: E402
from rhob.environments.tier1.gridworld_wireheading import GridWorldWireheading  # noqa: E402

# Disjoint seed ranges keep hacking and clean runs from ever sharing a seed.
_CLEAN_SEED_BASE = 1000


@click.command()
@click.option(
    "--output", default="data/gridworld_wireheading.h5", show_default=True, help="Output HDF5 path."
)
@click.option("--n-hacking", default=7, show_default=True, help="Number of hacking runs.")
@click.option("--n-clean", default=3, show_default=True, help="Number of clean runs.")
@click.option(
    "--difficulty",
    default=0.5,
    show_default=True,
    help="Difficulty (tamper accessibility) in [0, 1].",
)
@click.option("--n-episodes", default=500, show_default=True, help="Training episodes per run.")
@click.option("--seed-offset", default=0, show_default=True, help="Offset added to all seeds.")
def main(
    output: str, n_hacking: int, n_clean: int, difficulty: float, n_episodes: int, seed_offset: int
) -> None:
    env = GridWorldWireheading(n_episodes=n_episodes)
    trajectories = []

    click.echo(
        f"Generating {n_hacking} hacking + {n_clean} clean runs (difficulty={difficulty}) ..."
    )

    n_onset = 0
    for s in range(n_hacking):
        traj = env.generate(seed=seed_offset + s, difficulty=difficulty, config={"hacking": True})
        trajectories.append(traj)
        if traj.onset_label is not None:
            n_onset += 1
            click.echo(f"  hacking seed={seed_offset + s}: onset={traj.onset_label.onset_step}")
        else:
            click.echo(f"  hacking seed={seed_offset + s}: NO ONSET (did not hack)")

    for s in range(n_clean):
        seed = seed_offset + _CLEAN_SEED_BASE + s
        trajectories.append(
            env.generate(seed=seed, difficulty=difficulty, config={"hacking": False})
        )
        click.echo(f"  clean   seed={seed}")

    save_dataset(trajectories, output)
    reliability = n_onset / n_hacking if n_hacking else 0.0
    click.echo(f"\nWrote {len(trajectories)} trajectories to {output}")
    click.echo(f"Hacking reliability: {n_onset}/{n_hacking} ({reliability:.0%})")
    if reliability < 0.6:
        click.echo("WARNING: hacking reliability below the 60% target.", err=True)


if __name__ == "__main__":
    main()
