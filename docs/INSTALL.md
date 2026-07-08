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
run every family and every detector except `TrajectoryMLPDetector`, which
needs PyTorch (see below).

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
| Tests hang or take >10 min | You're likely running the full leaderboard/transfer scripts, not `pytest`. Those take hours by design — see [REPRODUCIBILITY.md](../REPRODUCIBILITY.md). |
| `pydantic` version conflicts | RHOB requires pydantic ≥ 2.0; check `pip show pydantic`. |

## Next Steps

- [Detector Tutorial](TUTORIAL_DETECTOR.md) — evaluate or add a detector in under 30 minutes
- [Environment Tutorial](TUTORIAL_ENVIRONMENT.md) — add a new hacking-mechanism family
- [CONTRIBUTING.md](../CONTRIBUTING.md) — submission process and admission-gate requirements
