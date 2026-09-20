"""Does the predator-spread constant control per-episode proxy variance at LOW fixed_pull?

The family matches the proxy MEAN via a gain_boost bisection and dampens per-episode
VARIANCE via a single fixed constant, _CORNER_SPREAD = 0.15, which spreads the three
hacking predators' corner targets 120 degrees apart. Its comment claims the resulting
variance ratio is ~0.88-1.04 "across this whole range".

Measured at the smoke design it is not: hack/legit per-episode proxy SD is 0.941 at
fixed_pull 0.84 (inside the claim) but 1.157 at 0.72 and 1.168 at 0.60 (outside it). The
Reward KDE detector reads absolute stream spread, so that ratio crossing 1.0 is what flips
the direction of the admission failure between tiers.

This asks the one question that decides the fix: is _CORNER_SPREAD a live lever on variance
at the pulls where it is under-correcting? If the ratio moves monotonically with spread at
pull 0.60 and 0.72, a second calibration parameter can match variance the way gain_boost
matches the mean. If it does not move, no amount of re-tuning that constant helps and the
family should carry a measured xfail instead of a fix.

Reports mean and SD together, because spread moves both -- a spread that fixes variance
while destroying the mean match is not a fix, it is a trade.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

import rhob.v3.families.pettingzoo_fixed_opponent_exploitation as F  # noqa: E402
from rhob.environments.pettingzoo.config import PettingZooConfig  # noqa: E402
from rhob.environments.pettingzoo.rollout import (  # noqa: E402
    generate_pettingzoo_rundata,
)

#: Fewer than the family's 250 so the sweep is affordable; the SD of a 120-episode proxy
#: stream is stable enough to rank spreads, which is all this has to do.
N_EPISODES = 120
SEEDS = (11, 22, 33)
PULLS = (0.60, 0.72, 0.84)
SPREADS = (0.0, 0.15, 0.30, 0.50, 0.75)


def _offsets(spread: float) -> list[np.ndarray]:
    """The family's own construction, with the radius as a parameter."""
    return [
        spread * np.array([np.cos(a), np.sin(a)], dtype=np.float32)
        for a in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)
    ]


def _hacking_fns(pull: float, boost: float, spread: float) -> dict:
    offs = _offsets(spread)
    fns = {
        aid: F._hacking_predator_fn(pull, boost, offs[i])
        for i, aid in enumerate(F._ADVERSARY_IDS)
    }
    fns["agent_0"] = F._prey_fn
    return fns


def _stats(action_fns: dict, seed: int) -> tuple[float, float]:
    cfg = PettingZooConfig(
        env_factory=F._make_env, n_episodes=N_EPISODES, horizon=F._HORIZON
    )
    run = generate_pettingzoo_rundata(
        cfg, action_fns, F._proxy_fn, F._true_fn, F._behav_fn, seed=seed
    )
    p = np.asarray(run.proxy_rewards, dtype=float)
    return float(p.mean()), float(p.std())


def _avg(action_fns_builder, seeds=SEEDS) -> tuple[float, float]:
    means, sds = [], []
    for s in seeds:
        m, sd = _stats(action_fns_builder(), s)
        means.append(m)
        sds.append(sd)
    return float(np.mean(means)), float(np.mean(sds))


def main() -> int:
    print(f"legit baseline ({N_EPISODES} eps x {len(SEEDS)} seeds) ...", flush=True)
    legit_mean, legit_sd = _avg(F._legit_action_fns)
    print(f"  legit mean={legit_mean:.4f} sd={legit_sd:.4f}", flush=True)

    out = {
        "n_episodes": N_EPISODES,
        "seeds": list(SEEDS),
        "legit": {"mean": legit_mean, "sd": legit_sd},
        "rows": [],
    }
    for pull in PULLS:
        boost = F._calibrate_gain_boost(pull)
        print(f"\npull={pull}  calibrated boost={boost:.4f}", flush=True)
        for spread in SPREADS:
            mean, sd = _avg(lambda: _hacking_fns(pull, boost, spread))
            row = {
                "fixed_pull": pull,
                "gain_boost": boost,
                "corner_spread": spread,
                "mean": mean,
                "sd": sd,
                "sd_ratio": sd / legit_sd if legit_sd else float("nan"),
                "mean_gap": mean - legit_mean,
            }
            out["rows"].append(row)
            print(
                f"  spread={spread:<5} mean={mean:7.4f} (gap {row['mean_gap']:+.4f})  "
                f"sd={sd:7.4f}  sd_ratio={row['sd_ratio']:.4f}",
                flush=True,
            )

    path = REPO / "results" / "family_diagnostics" / "spread_sweep.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
