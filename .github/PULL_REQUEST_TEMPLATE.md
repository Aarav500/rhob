## Summary

<!-- What does this PR do, and why? Link any related issue. -->

## Type of change

- [ ] New detector
- [ ] New environment family
- [ ] Bug fix
- [ ] Documentation
- [ ] Other (describe below)

## If this adds a detector

- [ ] Implements `PosthocDetector` with the correct `access_level`
- [ ] Added to `src/rhob/detectors/__init__.py` exports
- [ ] Tests added under `tests/test_detectors/`

## If this adds a family

- [ ] States the proxy-preserving symmetry σ, the proxy, and the discriminating
      feature (see [Environment Tutorial](docs/TUTORIAL_ENVIRONMENT.md))
- [ ] Passes the admission gate (matched proxy, behavioral separation,
      true-reward divergence, onset localizability, camping quality)
- [ ] Tests added under `tests/test_v3/`

## Checklist

- [ ] `pytest tests/ -q` passes locally
- [ ] `ruff check src tests scripts` passes locally
- [ ] No hardcoded seeds outside the `seed` parameter; rollouts are deterministic
- [ ] Docs updated if this changes documented behavior (README, REPRODUCIBILITY.md, tutorials)

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full submission workflow and
admission-gate requirements.
