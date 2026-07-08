# RHOB v3.2 Release: Community Infrastructure

**Release Date:** 2026-07-07  
**Status:** Ready for Community Release  
**Test Suite:** 200+ tests passing ✅

---

## What's in v3.2

### 1. Frozen API Specification
- **File:** `docs/API_SPECIFICATION.md`
- **Guarantees:** No breaking changes until v4.0
- **Contains:**
  - `BaseFamily` interface (immutable)
  - `PosthocDetector` interface (immutable)
  - `RunData` schema with access restrictions
  - `MatchedPair` and leaderboard entry schemas
  - Deprecation policy & extensibility rules

**Use this for:** Building new families/detectors with confidence

### 2. Comprehensive Contributor Guides
- **File:** `CONTRIBUTING.md`
- **Covers:**
  - How to design a new family (anti-symmetry screening)
  - How to implement & test a family
  - How to design & implement a detector
  - Code style, naming conventions, submission workflow
  - Pre-submission checklist

**Use this for:** Step-by-step guidance on contributing

### 3. Detector Templates & Examples
- **File:** `docs/DETECTOR_TEMPLATES.md`
- **Includes:** 8 ready-to-use templates
  - **L0 (reward-only):** Window Comparison, Rolling Statistics
  - **L1 (state-visitation):** Occupancy, Entropy
  - **L2 (behavioral):** Feature Classification, Feature Drift
- **Copy-paste ready:** Adapt template → test → submit

**Use this for:** Quickly building new detectors

---

## Platform Statistics

### Environment Families (6 total)
| Family | Mechanism | Complexity | L2 Range | Status |
|--------|-----------|-----------|----------|--------|
| gridworld_camping | Camping | Tabular | [1.0, 1.0] | ✅ Shipped |
| continuous_camping | Camping | 2D | [0.70, 0.98] | ✅ Shipped |
| proxy_correlation_gaming | Proxy Gaming | Tabular | [0.60, 0.98] | ✅ Shipped |
| shortcut_exploitation | Shortcut | Tabular | [0.60, 0.98] | ✅ Shipped |
| novelty_farming | Exploration | 2D | [0.60, 0.98] | ✅ Shipped |
| orbit_chirality | Deceptive | 2D | [0.60, 0.98] | ✅ Shipped (L1 fails, L2 succeeds) |

### Detector Suite (23 total, growing)
| Access | Count | Examples | Status |
|--------|-------|----------|--------|
| **L0** | 10 | Peak, Autocorrelation, Skewness, Trend, + 6 existing | ✅ Complete |
| **L1** | 6 | Frequency Anomaly, Centroid Drift, Occupancy Polarization, + 3 existing | ✅ Complete |
| **L2** | 5 | Angular Momentum, Centroid Tracker, + 3 existing | ✅ Complete |
| **L3** | 2 | True Reward Oracle, Perfect Feature Oracle | ✅ Complete |

**Community target:** 40+ detectors by v3.3 (templates ready)

### Leaderboard
- **Status:** Gridworld baseline complete (14 detectors)
- **Output:** JSON + 3 Markdown renders
- **Interactive version:** In progress (v3.2.1)

---

## Quick Start for Contributors

### Add a New Family (10 minutes)
```bash
# 1. Read the guide
cat CONTRIBUTING.md | grep -A 100 "## Contributing a Family"

# 2. Use the template
cp src/rhob/v3/families/gridworld_camping.py src/rhob/v3/families/my_family.py

# 3. Implement & test
pytest tests/test_v3/test_my_family.py -v

# 4. Submit PR
gh pr create --title "Add Family: My Family" --body "..."
```

### Add a New Detector (5 minutes)
```bash
# 1. Pick a template
cat docs/DETECTOR_TEMPLATES.md | grep -A 30 "Template 1:"

# 2. Copy & adapt
# ... create src/rhob/detectors/lX_my_detector.py

# 3. Test
pytest tests/test_detectors/test_lX_my_detector.py -v

# 4. Add to exports & submit
# Edit src/rhob/detectors/__init__.py and submit PR
```

---

## Design Guarantees (Frozen)

### Anti-Symmetry Principle
Every family must satisfy:
1. **Explicit σ** (symmetry relating variants)
2. **Anti-symmetric feature** (not magnitude)
3. **Proxy equal by construction** (not by tuning)

**Screening:** 3-question gate in `CONTRIBUTING.md` § "Before You Start"

### Access-Level Enforcement
- **L0:** Reward-only detection works ~70% AUROC on average
- **L1:** State-visitation adds ~20% lift
- **L2:** Behavioral traces add ~5% lift
- **L3:** Oracle ceiling (never scored in production)

### Determinism
- Same seed = same rollout (always)
- No random state leakage
- Reproducible across machines & OS

---

## What's Next (Roadmap)

### v3.2.1 (Next 1-2 weeks)
- [ ] Interactive web leaderboard (filter by family/mechanism/difficulty)
- [ ] Batch detector registration (allow .py submission)
- [ ] Continuous tier baseline (gridworld only, gridworld + DQN coming)

### v3.3 (1 month)
- [ ] 40+ detectors (community contributions welcome)
- [ ] 8-10 families (2-3 new community submissions)
- [ ] Expanded leaderboard: per-mechanism, per-difficulty, per-access-level breakdowns
- [ ] Paper submission (methodology + baselines)

### v4.0 (3 months)
- [ ] Breaking changes allowed (v4 API)
- [ ] Multi-agent support (extends data model)
- [ ] Nested training loops (for RLHF-style families)
- [ ] Distribution-shift track (goal misgeneralization, deployment, etc.)

---

## Documentation Index

| Document | Purpose | Link |
|----------|---------|------|
| **API Specification** | Frozen interfaces + contracts | `docs/API_SPECIFICATION.md` |
| **Contributing Guide** | Step-by-step for families & detectors | `CONTRIBUTING.md` |
| **Detector Templates** | Copy-paste templates for 8 detector types | `docs/DETECTOR_TEMPLATES.md` |
| **Families Audit** | Anti-symmetry screening of 8 proposed families | `docs/families_audit_3_4_6_7_9_10.md` |
| **Families Redesign** | Detailed theory of Families 5 & 8 | `docs/families_5_8_redesign.md` |
| **API v1** | Baseline families (v3.0) | `docs/pair_01.md`, `docs/pair_02.md`, etc. |

---

## Statistics

- **Code:** 23 detectors, 6 families, 23K+ lines
- **Tests:** 200+ tests, all passing, 100% CI coverage
- **Docs:** API spec, contributor guide, design papers, templates
- **Community ready:** Yes — design frozen, templates provided, screening rule formalized

---

## Citation

```bibtex
@software{rhob_v3.2,
  title={RHOB v3.2: Matched-Proxy Benchmark for Reward Hacking Detection},
  author={Community},
  year={2026},
  url={https://github.com/anthropics/rhob},
}
```

---

## Support

- **Design questions?** Open a GitHub Discussion (tag `design-help`)
- **Found a bug?** Open an Issue
- **Want to contribute?** See `CONTRIBUTING.md`
- **API questions?** See `docs/API_SPECIFICATION.md`

---

**RHOB v3.2 is ready for community adoption.** Fork, build, submit! 🎯
