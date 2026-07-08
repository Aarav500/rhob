"""Minimal end-to-end RHOB evaluation.

Generates a small GridWorld-Wireheading dataset in memory, evaluates the Random
and CUSUM baselines, and prints the leaderboard-style comparison table.

Run from the repo root::

    python examples/minimal_evaluation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rhob  # noqa: E402


def main() -> None:
    env = rhob.GridWorldWireheading()

    # 7 hacking + 3 clean runs (70/30 split).
    trajectories = [env.generate(seed=s, config={"hacking": True}) for s in range(7)]
    trajectories += [env.generate(seed=1000 + s, config={"hacking": False}) for s in range(3)]

    reports = rhob.compare(
        [rhob.RandomDetector(), rhob.CUSUMDetector()],
        trajectories,
    )

    print(rhob.results_table(reports))
    print()
    for report in reports:
        env_metrics = report.per_environment["tier1/gridworld_wireheading"]
        print(
            f"{report.detector_name:8s} "
            f"AUROC={env_metrics.auroc:.3f} "
            f"miss={env_metrics.miss_rate:.2f} "
            f"onset labelled for {env_metrics.n_hacking_runs} hacking runs"
        )


if __name__ == "__main__":
    main()
