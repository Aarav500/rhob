# RHOB Leaderboard Specification

**Document:** LEADERBOARD_SPEC · **Spec version:** 0.1 · **Status:** Design (all **[Planned]**)

Designs the future public leaderboard. Nothing here is implemented in M1; this
freezes the target so that the evaluation runner, dataset format, and detector API
are built to support it. Grounded in the engineering spec §10 and the current
`EvaluationReport` structure.

Guiding principle: **submitting must be low-friction and verifying must be
automatic.** A submission is a small JSON artifact plus a reproducible code
reference; CI validates it and appends to the results.

---

## 1. Submission format

A submission is a single JSON object with three blocks: `metadata`, `results`,
`verification`.

```json
{
  "metadata": {
    "detector_name": "Flight Recorder",
    "detector_id": "flight_recorder",
    "detector_version": "1.2.0",
    "authors": ["..."],
    "affiliation": "...",
    "paper_url": "https://arxiv.org/abs/...",
    "code_url": "https://github.com/.../flight_recorder",   // REQUIRED, reproducible
    "access_level": "L2",
    "is_oracle_free": true,
    "causal": true,                       // online/deployable vs offline/analysis-only
    "training_data_used": "none",         // none | train_split | external
    "compute_used": "1x A100, 2h",
    "date": "2027-01-15"
  },
  "results": {
    "per_environment": {
      "tier1/gridworld_wireheading": {"auroc": 0.99, "latency": 0.01, "miss_rate": 0.0, "fpr_at_k": 0.0}
    },
    "rhob_score": 0.823,
    "total_auroc": 0.85,
    "mean_latency": 0.12,
    "miss_rate": 0.05
  },
  "verification": {
    "config_hash": "…",
    "data_version": "v1.0",
    "rhob_package_version": "0.1.0",
    "random_seed": 42,
    "reproducibility_hash": "…"          // hash of the raw per-step score arrays
  }
}
```

The runner should emit this artifact directly (`rhob submit`), derived from an
`EvaluationReport` — submitters do not hand-author it.

## 2. Required metadata

| Field | Why required |
|---|---|
| `code_url` | Reproducibility — must resolve to a working implementation |
| `access_level` | Determines which sub-leaderboards the entry appears on |
| `is_oracle_free` | Only oracle-free methods appear on the main board |
| `causal` | Separates deployable (online) from analysis-only (offline) methods |
| `training_data_used` | Checked against the code — no test-set peeking |
| `compute_used`, `date` | Cost accounting and tie-breaking |

## 3. Required outputs

- **Per-environment** AUROC, detection latency, miss rate, FPR@k.
- **Aggregate** RHOB-Score (with CI), total AUROC, mean latency, miss rate.
- **Raw scores hash** (`reproducibility_hash`) over the per-step score arrays,
  enabling exact re-verification without shipping the full arrays.
- **Cost:** runtime and memory on declared hardware (once those metrics land).

## 4. Required metrics

The headline set from `METRICS_SPEC` §7: RHOB-Score (+CI), per-tier AUROC, miss
rate, TFD, FPR@k, and cost (runtime/memory). Precision/Recall/F1 optional with a
declared threshold.

## 5. Version compatibility

- An entry is bound to a `(data_version, rhob_package_version)` pair.
- A **dataset MAJOR** version change opens a new **leaderboard epoch**; prior
  entries are preserved but marked "prior epoch" and not ranked against new ones.
- The package declares which data versions it supports; incompatible pairs are
  rejected at submission.

## 6. Reproducibility requirements

A submission is accepted only if:

1. **Format valid** — passes JSON-schema validation.
2. **Reproducible** — re-running the detector (from `code_url`) on ≥5 random
   test-split trajectories reproduces the submitted scores within floating-point
   tolerance (`reproducibility_hash` matches).
3. **Code available** — `code_url` resolves and contains a runnable detector
   conforming to `DETECTOR_API`.
4. **No test peeking** — `training_data_used` is consistent with the code (no test
   labels imported); learning-based detectors trained on the train split only.
5. **Version-compatible** — `data_version`/`rhob_package_version` supported.

## 7. Evaluation protocol

1. Detectors are scored on the **withheld test split** (labels not public).
2. A **fixed evaluation config** (the canonical `paper.yaml`) is used for all
   entries — identical data streams, thresholds, and bootstrap settings.
3. All methods receive identical, access-filtered observations; none see
   `reward_true` (except a non-submittable oracle ceiling shown for reference).
4. Cost is measured on declared, comparable hardware.
5. Results are deterministic given the config, data version, and seed.

## 8. Ranking & sub-leaderboards

- **Primary sort:** RHOB-Score (descending).
- **Tie-breaks (in order):** mean detection latency → miss rate → computational
  overhead → submission date (earlier wins). *(M1's report currently sorts on 2
  keys; the full chain is REFACTOR_PLAN #11.)*
- **Sub-leaderboards:** by access level (L1-only, L2-only, …), by tier, by hacking
  type, oracle-free-only (the **main** board), online-only (deployable), and a
  latency-constrained board (must detect within a horizon fraction).
- **Reference rows:** an oracle ceiling (perfect) and a random floor (0.5) frame
  every board.

## 9. Infrastructure

- **Storage:** submission JSONs committed to the repository (version-controlled,
  forkable) under `leaderboard/results/`, with a generated `leaderboard.json`.
- **Submission flow:** GitHub issue/PR with the JSON → CI validates (§6) → merge
  appends to results and regenerates the board.
- **Refresh:** the adversarial tier is refreshed when methods saturate; the test
  split rotates per dataset version, triggering a new epoch.
- **Website:** a static site generated from the committed JSON.

## 10. Governance **[Planned]**

Clear contribution process for environments, detectors, and metrics; a documented
3-year maintenance horizon; and eventual community governance. The leaderboard's
credibility depends on the authors' own method being one entry among many,
evaluated identically, with code released.
