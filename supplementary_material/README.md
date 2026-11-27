# RHOB v1.0 Supplementary Material

> **Note (2026-11-19).** This file describes the material for the v1.0 release
> (2026-07-08) and is kept as that record. The `paper_appendices/` PDFs it lists are not in
> this repository. The current
> [`leaderboard/v5_leaderboard.json`](../leaderboard/v5_leaderboard.json) scores 30
> detectors on 33 families (123 cells); it was regenerated on 2026-11-19 in `52656cb` and
> writes N/A cells as `null`. [`admission/ADMISSION_LEDGER.md`](../admission/ADMISSION_LEDGER.md)
> covers all 33 families since `d4b700d`: of 123 cells, 42 ADMITTED, 30 DEGENERATE and 51
> NOT ADMITTED. `leaderboard/cross_family_transfer.json` was last changed in v1.4
> (`d872916`), with 6 training and 8 held-out families.
>
> *(2026-11-27: `scripts/cross_family_transfer.py` no longer writes
> `cross_family_transfer.json`. It refuses to overwrite that file, which is kept as the
> 2026-07-09 record, and writes `leaderboard/cross_family_transfer_v5.json` under the
> current rules: sign randomization on, and a cell the detector cannot score not
> applicable rather than 0.5 (`827aec9`). CI first wrote it on 2026-11-27, at the
> published design: 10 training and 20 held-out seeds per variant, five trials (run
> 45987934132, committed in `9c038f6`). Run with torch on one thread
> (`OMP_NUM_THREADS=1`), its published design took about five minutes on a 4-core
> machine, not the 1.5–2 hours given below.)*

This directory contains supporting material for the RHOB v1.0 paper.

## Paper Appendices

- `paper_appendices/admission_certificates.pdf` — Full admission gate certificates for all 9 families
- `paper_appendices/hyperparameters.pdf` — Complete hyperparameter listings for all 30 detectors
- `paper_appendices/additional_results.pdf` — Extended results tables and per-family breakdown

## Datasets & Leaderboards

### v5_leaderboard.json
Full 30×9 in-distribution discrimination AUROC results (Section 5.2, Figure 6).
- 30 detectors × 9 families
- Per-family default-difficulty pairs
- 5-fold stratified cross-validation scores
- Runtime: ~2–3 hours on CPU

See `../leaderboard/v5_leaderboard.json`

### cross_family_transfer.json
Cross-family transfer results: train on Families 1–6, test on Families 7–9 (Section 6.3, Table 2).
- Train AUROC (in-distribution 5-fold CV) vs transfer (held-out, no retraining)
- Per-family transfer scores
- Generalization gap (%)
- Runtime: ~1.5–2 hours on CPU

See `../leaderboard/cross_family_transfer.json`

## How to Regenerate

See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for step-by-step instructions:
- Figures 6–8 (v5 results): ~30 seconds each
- Table 2 (transfer results): ~1.5–2 hours
- Full leaderboard: ~2–3 hours

## Citation

If you use RHOB supplementary material, please cite the main paper:

```bibtex
@misc{shah2026rhob,
  title={Checks That Cannot Fail: Unfalsifiable Construction Claims in Benchmarks, and Three Diagnostics That Find Them},
  author={Shah, Aarav},
  year={2026},
  howpublished={Zenodo},
  doi={10.5281/zenodo.22005218},
  url={https://doi.org/10.5281/zenodo.22005218}
}
```
