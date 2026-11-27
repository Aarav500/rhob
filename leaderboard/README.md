# leaderboard/

The benchmark's committed result boards. This page is an index: it quotes no figures, so
it cannot go stale when a board is regenerated. Read the numbers from the files, or from
the Space (`space/app.py`), which renders them.

| File | What it is | Written by |
|---|---|---|
| [`v5_replicated.json`](v5_replicated.json) | **The headline board.** Every detector's mean AUROC overall and per family, each with a bootstrap interval over independent draws of the whole suite (`n_replicates`; the draws are `results/replication/replicate_*.json`). Not-applicable cells are excluded, never imputed. This is the board `space/app.py` serves. | `scripts/replicate_leaderboard.py` (one draw per run), then `scripts/aggregate_replication.py`; on CI, `.github/workflows/regenerate.yml` |
| [`v5_leaderboard.json`](v5_leaderboard.json) | One draw of the suite, with its `provenance` (commit, package versions, `argv`), its `sampling` block and its `cell_semantics`. A `null` per-family value means not applicable, not chance. Use it for what one run measured; use the replicated board for how much that varies. | `scripts/v5_leaderboard_and_transfer.py`; on CI, `regenerate.yml` |
| `cross_family_transfer_v5.json` | Cross-family transfer measured under the current rules (behavioural-sign randomization on, not-applicable cells excluded from both fitting and scoring), with a `configs` block naming the run. When this file is not in the directory, the current transfer figures are those in [`docs/figures/sign_randomization_impact.json`](../docs/figures/sign_randomization_impact.json). | `scripts/cross_family_transfer.py --out leaderboard/cross_family_transfer_v5.json`; on CI, `regenerate.yml` |
| [`cross_family_transfer.json`](cross_family_transfer.json) | **Historical record, not a transfer result.** The transfer run of 2026-07-09, made before the behavioural sign was randomized, when the sign of the feature was the label. Kept as the record of what was measured; superseded by `cross_family_transfer_v5.json`. See [`docs/figures/sign_randomization_impact.md`](../docs/figures/sign_randomization_impact.md) for what the randomization changed. | an earlier `scripts/cross_family_transfer.py`, whose exact version was not recorded; the current script refuses to overwrite this file |
| `community/` | Self-reported, single-draw submissions recorded with `python -m rhob submit` (`board.json` plus rendered tables). Kept apart from the boards above. Created by the first `rhob submit`. | `python -m rhob submit` (see [`submissions/README.md`](../submissions/README.md)) |

Python readers: `rhob.v3.leaderboard.replicated.load_replicated_leaderboard` for
`v5_replicated.json` (keeps the intervals);
`rhob.v3.leaderboard.adapters.load_any_leaderboard_json` for `v5_leaderboard.json`,
`v5_replicated.json` or `community/board.json` when only the standings order is needed;
and `rhob.v3.leaderboard.transfer.load_transfer_results` for the two transfer files.

## Retired: the v3.2 tables

Until 2026-11, this directory also held the v3.2 tables: `leaderboard.json` and the
standings in this `README.md`, `by_mechanism.md` and `by_difficulty.md`, written by
`scripts/populate_leaderboard_v32.py` and `scripts/populate_v3_leaderboard.py`. They were
removed because nothing read them, they predated the 2026-08-03 audit and the
not-applicable rule and disagreed with both v5 boards, and every detector they ranked
except `Ensemble(BehavioralThreshold x2)` is on both v5 boards. The tables were last
committed on 2026-07-07, and `populate_v3_leaderboard.py` could not run after 2026-09-02,
when `CellResult` lost its onset column.

They remain in the history:

```bash
git show b7324ec:leaderboard/leaderboard.json
git show b7324ec:leaderboard/README.md
git show b7324ec:leaderboard/by_mechanism.md
git show b7324ec:leaderboard/by_difficulty.md
git show 4728596:scripts/populate_leaderboard_v32.py   # last version before removal
git show 4728596:scripts/populate_v3_leaderboard.py    # last version before removal
```
