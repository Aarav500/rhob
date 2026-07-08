# RHOB Dataset Specification

**Document:** DATASET_SPEC · **Spec version:** 0.1 · **Format version:** 1.0 ·
**Status:** Draft for freeze
Source of truth: `src/rhob/data/storage.py`, `docs/data_schema.md`.

Formalizes the on-disk dataset so that **external researchers can generate
compatible datasets** and have them evaluated by the standard tooling.

Design principles: exact round-trips (determinism), self-describing files
(portability), and a stable format contract independent of the package version.

---

## 1. Container & file-level layout **[Stable — format 1.0]**

One dataset = one HDF5 file.

```
<dataset>.h5
├── attrs
│   ├── rhob_version    : str    # package version that wrote the file
│   ├── format_version  : str    # dataset format, "1.0"
│   └── n_runs          : int    # number of run groups
├── run_00000/
├── run_00001/
└── ...
```

Run groups are named `run_{i:05d}`, ordered, zero-padded to 5 digits.

## 2. Run (trajectory) group schema **[Stable]**

Each `run_NNNNN` group:

```
run_NNNNN/
├── attrs
│   ├── environment_id        : str
│   ├── seed                  : int
│   ├── algorithm             : str    # e.g. "tabular_q_learning"
│   ├── is_hacking_run        : bool
│   ├── hacking_type          : str    # HackingType value, or ""
│   ├── generation_timestamp  : str    # ISO 8601
│   ├── config_hash           : str
│   ├── onset_label           : str    # JSON object, or "null"
│   └── metadata              : str    # JSON object
├── rewards_proxy    : float64[T]              # required
├── rewards_true     : float64[T]              # required (oracle-only)
└── policy_features  : float64[T, F]           # optional (L2 signal)
```

- **Datasets** are gzip-compressed `float64`.
- `rewards_proxy` and `rewards_true` MUST share length `T`.
- `policy_features`, when present, MUST have leading dimension `T`.

## 3. Trajectory schema (in-memory) **[Stable]**

The `Trajectory` object mirrors the group:

| Field | Type | Required | Meaning |
|---|---|:---:|---|
| `environment_id` | str | ✓ | producing environment |
| `seed` | int | ✓ | generation seed |
| `algorithm` | str | ✓ | training algorithm label |
| `is_hacking_run` | bool | ✓ | hacking manifested (see note) |
| `reward_proxy` | `float64[T]` | ✓ | detector-visible (L1) |
| `reward_true` | `float64[T]` | ✓ | oracle-only |
| `policy_features` | `float64[T,F]` | ○ | detector-visible (L2) |
| `onset_label` | `OnsetLabel \| None` | ✓ | ground truth |
| `hacking_type` | `HackingType \| None` | ○ | category |
| `metadata` | dict | ○ | generation provenance |
| `config_hash` | str | ○ | reproducibility key |
| `generation_timestamp` | str | ○ | ISO 8601 |

**Note (integrity):** `is_hacking_run` currently means "configured hacking AND
onset detected"; a configured-hacking run without a manifested onset is stored as
clean. A future format will add an explicit run-outcome status to distinguish
*clean* from *missed-hack* (ARCHITECTURE_REVIEW W16 / REFACTOR_PLAN #6). External
generators should be aware this field conflates the two today.

## 4. Labels — `OnsetLabel` JSON schema **[Stable]**

Stored in the `onset_label` attribute as a JSON object (or the string `"null"`):

```json
{
  "onset_step": 237,
  "confidence": 0.99,
  "hacking_type": "reward_tampering",
  "detection_method": "two_sample",
  "confidence_interval": [217, 257],
  "severity": 0.03
}
```

| Field | Type | Meaning |
|---|---|---|
| `onset_step` | int ≥ 0 | labelled onset `t*` |
| `confidence` | float ∈ [0,1] | oracle confidence |
| `hacking_type` | str | `HackingType` value |
| `detection_method` | str | how the onset was found |
| `confidence_interval` | [int, int] | uncertainty in `t*` |
| `severity` | float ≥ 0 | post-onset true-reward degradation rate |

`onset_step` MUST satisfy `0 ≤ onset_step < T`. Clean runs have `null`.

## 5. Versioning **[Stable policy]**

Two independent version axes:

- **Format version** (this document): `MAJOR.MINOR`. MAJOR = breaking schema
  change (new required fields, renamed datasets); MINOR = additive, backward
  compatible. Readers MUST reject a MAJOR they don't support and MAY read a higher
  MINOR by ignoring unknown fields.
- **Dataset version** (content): `vMAJOR.MINOR`, e.g. `v1.0`. MAJOR = new tiers,
  schema fields, or split reassignment; MINOR = added seeds/algorithms/adversarial
  refresh. The package declares which dataset versions it supports.

Package version and dataset version are decoupled.

## 6. Compression & determinism **[Stable]**

- Numerical datasets use **gzip** compression and **`float64`** dtype.
- `float64` (not `float32`) is mandated so that save→load is **exact**; this
  underpins the determinism guarantee. External generators MUST NOT down-cast
  reward/feature data before writing.

## 7. Train / test split **[Stable policy]**

Split assignment is a deterministic hash of `(environment_id, seed)` so the same
seed always lands in the same split:

```
split = "train" if hash(environment_id + str(seed)) % 5 < 3 else "test"   # 60/40
```

- **Train split:** labels public — detector development and tuning.
- **Test split:** labels withheld in the public release — leaderboard evaluation.
- The split is fixed per dataset version. *(Withheld-label packaging and the split
  utility are **[Planned]**; M1 ships the full labelled set.)*

## 8. Compatibility requirements for external datasets

An externally-generated dataset is **RHOB-compatible** if:

1. It is a single HDF5 file with the file-level attrs of §1 (`format_version` a
   supported value).
2. Every run group follows §2: required attrs, `rewards_proxy`/`rewards_true` as
   equal-length `float64`, optional `policy_features` with leading dim `T`.
3. `onset_label` is valid JSON per §4 (or `"null"`), with `0 ≤ onset_step < T`.
4. Labels are self-consistent: for hacking runs, mean `reward_true` after
   `onset_step` is lower than before (the oracle's `validate_label` contract).
5. `reward_true` is present (needed for re-labelling and verification) but is
   never surfaced to detectors by the tooling.
6. Datasets are exactly round-trippable (`float64`, no lossy transform).

A `rhob validate-dataset` conformance checker is **[Planned]** to automate this.

## 9. Provenance & reproducibility

Each run records `algorithm`, `seed`, `config_hash`, and generation `metadata`, so
a dataset is regenerable. **Caveat:** `generation_timestamp` is wall-clock, so two
regenerations produce content-identical but byte-different files; file-level
hashing should exclude the timestamp (ARCHITECTURE_REVIEW W20).

## 10. Sizes & scaling **[Planned]**

Target scales (from the vision): v1.0 ≈ 1k trajectories (~10 GB compressed),
v2.0 ≈ 3k (~50 GB), streamed per-tier. Lazy / memory-mapped loading is required at
these scales (REFACTOR_PLAN #9); the current loader is eager and suits M1's small
datasets.
