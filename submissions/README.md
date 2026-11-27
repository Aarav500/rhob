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
refuses, and writes nothing, when the submission does not validate, when the `board.json`
it finds is not a valid community board, or when `README.md`, `by_mechanism.md` or
`by_difficulty.md` in that directory does not start with the community header
(`# RHOB community submissions (self-reported, single draw)`). Other files in the
directory are left alone.

`validate` checks the values as well as the keys:

- `detector_name` must be a non-empty string.
- `access_level` must be one of `L0`, `L1`, `L2` or `L3`.
- `overall_auroc` must be a number in [0, 1], or `null` for a run with no measured cell.
- `n_cells` must be a non-negative integer.
- `author` and `timestamp`, when given, must be strings.
- `family_auroc`, `mechanism_auroc` and `tier_auroc`, when given, must map names to
  numbers in [0, 1]. The `tier_auroc` names must be difficulty tiers (`TRIVIAL`, `EASY`,
  `MEDIUM`, `HARD`, `EXTREME`).

`NaN` and `Infinity` are refused: they are not JSON.

`n_cells` is the number of cells the overall AUROC averages. `rhob evaluate` writes the
measured cells only: a cell where your detector is not applicable (the family does not
emit the channel it reads, or the horizon is shorter than its `min_episodes`) is neither
averaged nor counted.
