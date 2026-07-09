"""Difficulty-spectrum figure with the theoretical sigmoid overlaid.

Plots the three admitted continuous pairs (measured admission L2-AUROC) against the
prediction L2 = Phi(d / (sqrt(2) * sigma_a)), showing that the three operating points
fall on a single continuous curve. Uses the recorded admission numbers (no retraining);
writes to docs/figures/.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIGMA_A = 0.5
# (d, measured admission L2-AUROC) for the three admitted pairs.
POINTS = [(0.55, 0.821, "pair_04_hard"), (0.75, 0.891, "pair_03_medium"), (1.25, 0.977, "pair_02_easy")]


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def theory(d: float, sigma_a: float = SIGMA_A) -> float:
    return phi(d / (math.sqrt(2.0) * sigma_a))


def main() -> None:
    ds = np.linspace(0.0, 1.5, 200)
    curve = [theory(d) for d in ds]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ds, curve, color="tab:blue", lw=2,
            label=r"theory: $\Phi(d/\sqrt{2}\sigma_a)$, $\sigma_a=0.5$")
    for d, l2, name in POINTS:
        ax.plot(d, l2, "o", color="tab:red", ms=9)
        ax.annotate(name, (d, l2), textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.plot([], [], "o", color="tab:red", label="admitted pair (measured L2)")

    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("attractor separation $d$")
    ax.set_ylabel("L2-AUROC (behavioral separability)")
    ax.set_title("Continuous-tier difficulty spectrum: measured points vs. theory")
    ax.set_ylim(0.45, 1.02)
    ax.set_xlim(0.0, 1.5)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out = Path(__file__).resolve().parents[1] / "docs" / "figures" / "difficulty_spectrum.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
