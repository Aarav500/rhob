# Installation Guide

## Requirements

- Python ≥ 3.10
- pip (or [uv](https://github.com/astral-sh/uv), which is faster)

## Standard Install

```bash
git clone https://github.com/Aarav500/rhob.git
cd rhob
pip install -e ".[dev]"
```

This installs RHOB plus its test/lint tooling. Core runtime dependencies
(`numpy`, `scipy`, `scikit-learn`, `h5py`, `pydantic`, `click`) are enough to
run every detector except the Reward MLP and `TrajectoryMLPDetector`, which
need PyTorch (see below), and every family except `continuous_camping`
(PyTorch, the `continuous` extra), the four MuJoCo families (the `mujoco`
extra) and the five PettingZoo families (the `pettingzoo` extra).

## Verify the Install

```bash
python -m pytest tests/ -q
```

Expect 207+ passing tests in under 5 minutes on a laptop CPU.

## Optional: PyTorch-based detectors

`TrajectoryMLPDetector` (L2) and the Reward MLP (L0) use PyTorch:

```bash
pip install -e ".[continuous]"
```

## Optional: Continuous-control environment generation

If you need to regenerate the continuous-control Tier-2 datasets (DQN camping
agent) from scratch rather than using the committed data:

```bash
pip install -e ".[environments]"
```

The legacy scripts for the v1 continuous pairs (`scripts/generate_eval_data.py`, which
builds those datasets, and `scripts/validate_all_continuous.py` and
`scripts/validate_pair_02.py` to `scripts/validate_pair_04.py`, which validate the
pairs) train their own camper. The v3 benchmark's `continuous_camping` family does not:
since 2026-11-27 it loads frozen weights,
`src/rhob/environments/continuous/frozen_camper.py`, because training the camper under
the numerics pins gave different weights on Intel and AMD runners. Loading them still
needs PyTorch, so the family needs the `continuous` extra.
`scripts/freeze_camper.py` rewrites them, under the pins on an Intel CPU.

## Docker

If you'd rather not manage a local Python environment:

```bash
docker build -t rhob .
docker run --rm rhob pytest tests/ -q
```

See the root [`Dockerfile`](../Dockerfile) for what's included.

## Google Colab

For a zero-install walkthrough, open
[`notebooks/rhob_quickstart.ipynb`](../notebooks/rhob_quickstart.ipynb) in
Colab. It clones the repo, installs dependencies, and evaluates a detector
end-to-end in a single runtime.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: rhob` | Run `pip install -e .` from the repo root, not a subdirectory. |
| `ImportError: torch required` | Install the `continuous` extra, or use a non-PyTorch detector. |
| Tests hang or take >10 min | You're likely running the full leaderboard or replication scripts, not `pytest`. Those take hours by design — see [REPRODUCIBILITY.md](../REPRODUCIBILITY.md). The cross-family transfer script takes minutes with torch on one thread (`OMP_NUM_THREADS=1`; the committed run's four rows took 201 s together on a hosted runner under the numerics pins, `provenance.wall_clock_s` in `leaderboard/cross_family_transfer_v5.json`), and can take far longer with torch's default thread count on a busy machine. |
| `pydantic` version conflicts | RHOB requires pydantic ≥ 2.0; check `pip show pydantic`. |

## Next Steps

- [Detector Tutorial](TUTORIAL_DETECTOR.md) — evaluate or add a detector in under 30 minutes
- [Environment Tutorial](TUTORIAL_ENVIRONMENT.md) — add a new hacking-mechanism family
- [CONTRIBUTING.md](../CONTRIBUTING.md) — submission process and admission-gate requirements
