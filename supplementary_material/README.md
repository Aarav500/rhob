# RHOB v1.0 Supplementary Material

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

See `../../leaderboard/v5_leaderboard.json`

### cross_family_transfer.json
Cross-family transfer results: train on Families 1–6, test on Families 7–9 (Section 6.3, Table 2).
- Train AUROC (in-distribution 5-fold CV) vs transfer (held-out, no retraining)
- Per-family transfer scores
- Generalization gap (%)
- Runtime: ~1.5–2 hours on CPU

See `../../leaderboard/cross_family_transfer.json`

## How to Regenerate

See [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md) for step-by-step instructions:
- Figures 6–8 (v5 results): ~30 seconds each
- Table 2 (transfer results): ~1.5–2 hours
- Full leaderboard: ~2–3 hours

## Citation

If you use RHOB supplementary material, please cite the main paper:

```bibtex
@article{shah2026rhob,
  title={RHOB v1.0: Generalizable Reward Hacking Detection Through Matched-Proxy Benchmarking},
  author={Shah, Aarav},
  journal={TMLR},
  year={2026}
}
```
