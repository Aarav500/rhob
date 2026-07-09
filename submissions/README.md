# Community Submissions

Drop a detector-result submission JSON here (one file per submission) to have it
validated automatically and considered for inclusion in the leaderboard.

## How to submit

```bash
# 1. Evaluate your detector across the benchmark suite.
python -m rhob evaluate --detector your_detector.py --output submissions/your_detector_name.json

# 2. Validate it locally before opening a PR.
python -m rhob validate submissions/your_detector_name.json

# 3. Open a PR adding the file. .github/workflows/leaderboard_validate.yml
#    automatically re-validates any changed file under submissions/.
```

A submission JSON must contain at minimum:

```json
{
  "detector_name": "My Detector",
  "access_level": "L2",
  "overall_auroc": 0.87,
  "n_cells": 40
}
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full detector-submission workflow, and
the [Detector Tutorial](../docs/TUTORIAL_DETECTOR.md) for how to build and evaluate a
detector in the first place.

Once validated, a maintainer merges the submission into the tracked leaderboard via
`python -m rhob submit submissions/your_detector_name.json`, which updates
`leaderboard/leaderboard.json` and regenerates the standings tables.
