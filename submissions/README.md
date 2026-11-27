# Community Submissions

Drop a detector-result submission JSON here (one file per submission) to have it
validated automatically and considered for the community board (`leaderboard/community/`).

## How to submit

```bash
# 1. Evaluate your detector across the benchmark suite.
python -m rhob evaluate --detector your_detector.py --output submissions/your_detector_name.json

# 2. Validate it locally before opening a PR.
python -m rhob validate submissions/your_detector_name.json

# 3. Open a PR adding the file. .github/workflows/leaderboard_validate.yml
#    automatically re-validates any changed file under submissions/.
```

A submission JSON must contain at minimum the four keys below (`rhob submit` fills in
`author` as `anonymous` and `timestamp` as the time of submission when they are absent):

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

Once validated, a maintainer records the submission on the community board with
`python -m rhob submit submissions/your_detector_name.json`. That adds an entry to
`leaderboard/community/board.json` and re-renders the tables next to it
(`README.md`, `by_mechanism.md`, `by_difficulty.md`); `--board-dir DIR` writes to another
directory instead. The community board holds self-reported, single-draw results and is
kept apart from the benchmark's own boards in `leaderboard/` (see
[leaderboard/README.md](../leaderboard/README.md)), which `submit` never writes. It
refuses, and writes nothing, when the `board.json` it finds is not a community board or a
Markdown file there was not written by `submit`.

`n_cells` is the number of cells the overall AUROC averages. `rhob evaluate` writes the
measured cells only: a cell where your detector is not applicable (the family does not
emit the channel it reads, or the horizon is shorter than its `min_episodes`) is neither
averaged nor counted.
