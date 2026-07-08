"""Generate evaluation data from all 4 admitted pairs.

Produces, per pair: 2*n_runs ``RunData`` objects (n_runs Variant A + n_runs
Variant B) plus the oracle onset labels for Variant A, ready for the detector
evaluation protocol. Importable as a module (used by
``scripts/run_full_evaluation.py``) and runnable standalone to cache results
to ``.npz`` files under ``results/detector_evaluation/eval_data/``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rhob.agents.dqn import DQNCamper, train_camper
from rhob.detectors.posthoc import RunData
from rhob.environments import pair_02, pair_03, pair_04
from rhob.environments.pair_01.config import DEFAULT as PAIR01_DEFAULT
from rhob.environments.pair_01.env_hacking import make_hacking_env
from rhob.environments.pair_01.env_legitimate import make_legitimate_env
from rhob.evaluation.eval_rollout import generate_continuous_rundata, generate_gridworld_rundata

GRIDWORLD_PAIR_ID = "tier1/pair_01_gridworld"
CONTINUOUS_PAIRS = [pair_04, pair_03, pair_02]  # hard -> easy, matches validate_all_continuous.py

# Measured L2-AUROC from each pair's admission certificate (docs/pair_0X.md), used only
# as a reference point for the evaluation's L2-consistency check -- not re-derived here.
ADMISSION_L2 = {
    GRIDWORLD_PAIR_ID: 1.000,
    "tier2/pair_02_easy": 0.977,
    "tier2/pair_03_medium": 0.891,
    "tier2/pair_04_hard": 0.821,
}


@dataclass
class PairEvalData:
    """Evaluation data for one pair: both variants plus Variant-A onset labels."""

    pair_id: str
    n_episodes: int
    runs_a: list[RunData]
    runs_b: list[RunData]
    onsets_a: list[int]
    admission_l2: float  # reference value from the pair's admission certificate


def generate_pair01_eval_data(n_runs: int, seed_base: int = 10_000) -> PairEvalData:
    """Generate evaluation data for Pair 01 (gridworld, tabular Q-learning)."""
    config = PAIR01_DEFAULT
    runs_a, runs_b, onsets_a = [], [], []

    for i in range(n_runs):
        env_a = make_hacking_env(config)
        run_a, onset_a = generate_gridworld_rundata(env_a, seed=seed_base + i, config=config)
        runs_a.append(run_a)
        onsets_a.append(onset_a)

        env_b = make_legitimate_env(config)
        run_b, _onset_b = generate_gridworld_rundata(env_b, seed=seed_base + 5000 + i, config=config)
        runs_b.append(run_b)

    return PairEvalData(
        pair_id=GRIDWORLD_PAIR_ID,
        n_episodes=config.n_episodes,
        runs_a=runs_a,
        runs_b=runs_b,
        onsets_a=onsets_a,
        admission_l2=ADMISSION_L2[GRIDWORLD_PAIR_ID],
    )


def generate_continuous_pair_eval_data(
    pair_module, camper: DQNCamper, n_runs: int, seed_base: int
) -> PairEvalData:
    """Generate evaluation data for one continuous pair using a shared trained camper."""
    config = pair_module.CONFIG
    runs_a, runs_b, onsets_a = [], [], []

    for i in range(n_runs):
        run_a, onset_a = generate_continuous_rundata(camper, "A", config, seed=seed_base + i)
        runs_a.append(run_a)
        onsets_a.append(onset_a)

        run_b, _onset_b = generate_continuous_rundata(
            camper, "B", config, seed=seed_base + 5000 + i
        )
        runs_b.append(run_b)

    return PairEvalData(
        pair_id=pair_module.PAIR_ID,
        n_episodes=config.n_episodes,
        runs_a=runs_a,
        runs_b=runs_b,
        onsets_a=onsets_a,
        admission_l2=ADMISSION_L2[pair_module.PAIR_ID],
    )


def generate_all_pairs(
    n_runs: int = 20, camper_seed: int = 0, verbose: bool = True
) -> dict[str, PairEvalData]:
    """Generate evaluation data for all 4 admitted pairs.

    Args:
        n_runs: Runs per variant per pair (total runs per pair = 2*n_runs).
        camper_seed: Seed for the shared continuous-tier DQN camper.
        verbose: Print progress.

    Returns:
        ``{pair_id: PairEvalData}``.
    """
    results: dict[str, PairEvalData] = {}

    if verbose:
        print("Generating Pair 01 (gridworld) evaluation data ...")
    t0 = time.time()
    results[GRIDWORLD_PAIR_ID] = generate_pair01_eval_data(n_runs)
    if verbose:
        print(f"  done in {time.time() - t0:.1f}s ({2 * n_runs} runs)")

    if verbose:
        print("Training shared continuous-tier DQN camper ...")
    t0 = time.time()
    camper = train_camper(seed=camper_seed)
    if verbose:
        print(f"  done in {time.time() - t0:.1f}s")

    for pair_module in CONTINUOUS_PAIRS:
        if verbose:
            print(f"Generating {pair_module.PAIR_ID} evaluation data ...")
        t0 = time.time()
        data = generate_continuous_pair_eval_data(pair_module, camper, n_runs, seed_base=20_000)
        results[pair_module.PAIR_ID] = data
        if verbose:
            print(f"  done in {time.time() - t0:.1f}s ({2 * n_runs} runs)")

    return results


def _save_run(path: Path, run: RunData, onset: int) -> None:
    np.savez_compressed(
        path,
        proxy_rewards=run.proxy_rewards,
        true_rewards=run.true_rewards,
        state_counts=run.state_counts if run.state_counts is not None else np.array([]),
        behav_trace=run.behav_trace if run.behav_trace is not None else np.array([]),
        onset=onset,
    )


def save_eval_data(data: dict[str, PairEvalData], output_dir: Path) -> None:
    """Cache all pairs' evaluation data to ``.npz`` files, one per run."""
    for pair_id, pair_data in data.items():
        pair_dir = output_dir / pair_id.replace("/", "_")
        pair_dir.mkdir(parents=True, exist_ok=True)
        for i, (run, onset) in enumerate(zip(pair_data.runs_a, pair_data.onsets_a)):
            _save_run(pair_dir / f"A_seed{i:02d}.npz", run, onset)
        for i, run in enumerate(pair_data.runs_b):
            _save_run(pair_dir / f"B_seed{i:02d}.npz", run, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation data for detectors")
    parser.add_argument("--n-runs", type=int, default=20, help="Runs per variant per pair")
    parser.add_argument("--camper-seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/detector_evaluation/eval_data")
    )
    args = parser.parse_args()

    data = generate_all_pairs(n_runs=args.n_runs, camper_seed=args.camper_seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_eval_data(data, args.output_dir)

    print(f"\nEvaluation data cached to {args.output_dir}")
    for pair_id, pair_data in data.items():
        print(f"  {pair_id}: {len(pair_data.runs_a)} A + {len(pair_data.runs_b)} B runs")


if __name__ == "__main__":
    main()
