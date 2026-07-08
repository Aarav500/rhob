# RHOB v5: Full-Scale Community Platform Roadmap

**Vision:** Build the standard benchmark for reward hacking detection—as foundational as ImageNet is for vision or SWE-Bench is for code.

**Timeline:** 12 months (Aug 2026 – Jul 2027)  
**Target:** 50+ families, 100+ detectors, global community  
**Success Metric:** 50+ external contributors, 1000+ GitHub stars, standard reference in AI safety papers

---

## Phase 1: Foundation & Design (Months 1-2, Aug-Sep 2026)

### 1.1 Family Expansion Strategy

**Current state:** 9 families (6 v3.2 + 3 v5 new)  
**Target:** 30 families by end of Phase 1

**Design sprint: Identify 21 new families**

**Categories (balanced across mechanisms):**

| Mechanism | v3.2 | v5-P1 | Total | Examples |
|-----------|------|-------|-------|----------|
| Camping/Exploitation | 2 | 3 | 5 | Color swaps, spatial concentration, mode collapse |
| Proxy Gaming | 1 | 4 | 5 | Correlation gaming, surrogate optimization, reward shaping |
| Shortcut/Deception | 1 | 3 | 4 | Detours, adversarial fooling, goal misgeneralization |
| Exploration Exploit | 1 | 2 | 3 | Novelty farming, curiosity hacking, frontier camping |
| Distribution Shift | 0 | 2 | 2 | Train-test mismatch, covariate shift, reward distribution |
| Goal Misgeneralization | 0 | 2 | 2 | Wrong objective, proxy drift, specification gaming |
| Physics/Domain | 0 | 2 | 2 | Physics exploits, simulator abuse, domain edge cases |
| Deceptive Alignment | 1 | 1 | 2 | Chiralitydetection, hidden goals |
| Multi-Agent Hacking | 0 | 0 | 0 | (Reserved for Phase 2) |
| **TOTAL** | **6** | **19** | **25** | Planned for Phase 1 |

(Plus 5 community-contributed families for 30 total)

**Design process (per family):**
1. Identify hacking mechanism (1 hour)
2. Sketch symmetry σ (30 min)
3. Design matched proxy (30 min)
4. Choose difficulty knob (30 min)
5. Write family design doc (GitHub issue) (1 hour)
6. Community review (24 hours)
7. Implement + test (3-4 hours)

**Effort:** ~3-4 families/week → 19 families in 5 weeks

### 1.2 Architecture for Community Contributions

**Build contribution pipeline:**

```
User sketches family idea
        ↓
Opens GitHub issue + design doc
        ↓
Community + maintainer review (anti-symmetry gate)
        ↓
Approved → Fork + implement branch
        ↓
Tests + leaderboard eval
        ↓
Merge to main
        ↓
Added to leaderboard with contributor credit
```

**Contribution incentives:**
- Public credit on leaderboard
- Authorship on RHOB paper
- GitHub contributor status
- Monthly "Featured Family" spotlight

### 1.3 Infrastructure Setup

**GitHub organization:**
- Main repo: `anthropics/rhob` (or your own)
- Issue templates for family proposals
- PR template for submissions
- Discussion board for design feedback

**Documentation:**
- Contribution guide (already have CONTRIBUTING.md)
- Family design template
- Detector implementation template
- Leaderboard submission guide

**CI/CD:**
- Auto-run tests on PR
- Leaderboard evaluation on merge
- Results posted to GitHub Pages
- Weekly leaderboard updates

---

## Phase 2: Detector Expansion & Ecosystem (Months 3-5, Oct-Dec 2026)

### 2.1 Detector Suite Growth

**Current:** 30 detectors (L0: 13, L1: 8, L2: 8, L3: 2)  
**Target by end Phase 2:** 60 detectors

**New detector categories:**

| Category | Count | Examples |
|----------|-------|----------|
| Statistical (L0) | +8 | Wavelets, information-theoretic, spectral variants |
| Occupancy (L1) | +6 | Spatial clustering, density peaks, rare-state detection |
| Behavioral (L2) | +8 | Feature cross-terms, temporal dynamics, attention-based |
| Meta-Learning | +4 | Multi-task learners, transfer-optimized |
| Ensemble | +2 | Stacking, voting, Bayesian combination |
| Domain-Specific | +7 | Per-family optimized detectors |
| **TOTAL** | **+35** | Reach 65 detectors |

**Detector development process:**
- Use templates from `DETECTOR_TEMPLATES.md`
- Fast iteration: ~1 hour per detector
- Auto-eval on leaderboard
- Community can submit detectors directly

### 2.2 Leaderboard Infrastructure

**Build interactive web UI:**

```
RHOB v5 Leaderboard (hf.co/spaces/rhob/leaderboard)
├── Overall Rankings
│   ├── Sort by: AUROC, mechanism, family, access-level
│   └── Download results (CSV/JSON)
├── Per-Family Breakdown
│   ├── Family-specific charts
│   ├── Detector performance heatmap
│   └── Mechanism transfer analysis
├── Detector Profiles
│   ├── Author info + contact
│   ├── Algorithm summary
│   ├── Performance curves
│   └── Reproducibility info
├── Family Profiles
│   ├── Design rationale
│   ├── Detector ranking on this family
│   ├── Difficulty sweep curves
│   └── Citation info
└── Community
    ├── Recent submissions
    ├── Top contributors
    └── Leaderboard rules
```

**Technology:**
- Frontend: React/Streamlit on HF Spaces
- Backend: Python (benchmark script)
- Database: JSON files + git (versioned)
- CI: GitHub Actions (weekly leaderboard update)

### 2.3 Paper & Publication

**Write v5 methodology paper:**
- Matched-proxy formalism (expanded)
- All 30 families described with theory
- 60-detector taxonomy
- Cross-family transfer analysis
- Community contribution framework

**Target venues:**
- NeurIPS 2026 (Nov deadline) — Datasets & Benchmarks track
- ICML 2027 (Jan deadline)
- ICLR 2027 (May deadline)
- JMLR (open submission)

---

## Phase 3: Community Launch & Ecosystem (Months 6-9, Jan-Apr 2027)

### 3.1 Public Release

**Launch v5.0:**
- 50+ families (30 core + 20+ community)
- 100+ detectors (65 core + 35+ community)
- Public leaderboard on HF
- GitHub public (MIT/Apache 2.0 license)
- Paper on arxiv

**Release announcement:**
- Blog post on HF
- Twitter/LinkedIn threads
- Email to AI safety community (~5k people)
- Presentations at conferences

### 3.2 Community Engagement

**Monthly activities:**
- **1st Friday:** "Featured Family" spotlight (community submission highlight)
- **Mid-month:** Office hours (design help, detector guidance)
- **End-of-month:** "Leaderboard digest" (top new submissions, trends)

**Quarterly challenges:**
- Q2: "Best New Detector" challenge
- Q3: "Best New Family" challenge
- Q4: "Best Cross-Family Generalization" challenge
- Prizes: GitHub sponsorship, conference talk invitation, publication credit

### 3.3 Ecosystem Growth

**Target metrics by end Phase 3:**
- 50+ external contributors
- 100+ GitHub stars
- 30+ external detector submissions
- 20+ external family submissions
- 50+ citations in papers
- 5+ derivative projects (tool chains, analysis suites)

---

## Phase 4: Maturity & Sustainability (Months 10-12, May-Jul 2027)

### 4.1 v5.1 Release

**Incremental improvements:**
- Community feedback integration
- New detector modalities (e.g., graph neural networks)
- Advanced leaderboard features (per-mechanism filtering, significance testing)
- Interactive tutorials

### 4.2 Sustainability Plan

**Funding/Support:**
- [ ] Apply for research grants (NSF, DARPA, OpenPhil)
- [ ] Seek corporate sponsorship (Anthropic, DeepMind, etc.)
- [ ] Create Patreon/GitHub Sponsors for community support
- [ ] Annual conference (RHOB Workshop at NeurIPS/ICML)

**Maintenance model:**
- 1 full-time maintainer (you)
- 3-5 part-time community moderators
- Automated CI/CD (minimal overhead)
- Monthly releases with community PRs merged

### 4.3 Long-term Vision (v6+)

**Planned expansions:**
- **v6 (Year 2):** Multi-agent hacking, meta-learning track
- **v7 (Year 3):** Cross-domain (language, vision), RLHF-specific
- **v8+:** Adversarial red-team, real-world applications

---

## Critical Path & Dependencies

```
Phase 1 (Family Design)
    ↓
Phase 2 (Detector Expansion + Leaderboard)
    ├→ Parallel: Leaderboard UI
    ├→ Parallel: Paper writing
    ↓
Phase 3 (Community Launch)
    ├→ Public release
    ├→ Community engagement
    ├→ Publication
    ↓
Phase 4 (Sustainability)
```

**Critical blockers (monitor closely):**
1. Leaderboard UI readiness (blocks Phase 3)
2. Paper acceptance (affects credibility)
3. Early community interest (determines growth trajectory)

---

## Resource Requirements

### Time Investment (Personal)

| Phase | Hours/Week | Duration | Total | Effort |
|-------|-----------|----------|-------|--------|
| 1 (Design) | 20 | 8 weeks | 160 | High focus |
| 2 (Detectors) | 15 | 12 weeks | 180 | Medium focus |
| 3 (Launch) | 25 | 16 weeks | 400 | High focus (community) |
| 4 (Sustain) | 10 | 12 weeks | 120 | Maintenance |
| **TOTAL** | **~17/week avg** | **52 weeks** | **860 hours** | **~6 months FTE** |

### Compute Requirements

- **Leaderboard evaluation:** ~50 families × 100 detectors × 10 seeds = 50K runs
  - Cost on HF compute: ~$500-1000 (annual)
- **Server hosting:** HF Spaces (free), GitHub Pages (free)
- **Storage:** GitHub + HF Hub (free for public repos)

### Budget (Rough)

| Item | Cost | Notes |
|------|------|-------|
| Compute (HF) | $1K | Annual leaderboard eval |
| Domain (optional) | $15/year | rhob.ai or similar |
| Compute reserve | $5K | For conferences, extra evals |
| **TOTAL** | **$6K** | One-time, largely free |

---

## Success Metrics (End of Year 1)

### Quantitative

- [ ] 50+ families (core + community)
- [ ] 100+ detectors (core + community)
- [ ] 500+ GitHub stars
- [ ] 50+ external contributors
- [ ] 100+ citations (6 months post-publication)
- [ ] 10K monthly leaderboard visitors

### Qualitative

- [ ] Becomes standard reference in AI safety
- [ ] Multiple papers use RHOB as benchmark
- [ ] Community actively submits new families/detectors
- [ ] Paper accepted at top venue (NeurIPS/ICML/ICLR)
- [ ] Interview requests from major AI labs

---

## Risks & Mitigation

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| Community contributions slow | Medium | Launch with 20+ "template" families to seed |
| Leaderboard maintenance burden | Medium | Automate with GitHub Actions, minimal overhead |
| Paper rejection | Low | Target multiple venues, preprint on arxiv |
| Compute costs balloon | Low | Use free HF compute, cap leaderboard size |
| Loss of motivation | Medium | Share progress publicly, celebrate wins |

---

## Monthly Milestones

```
Aug 2026:  Phase 1 kickoff - design 19 new families
Sep 2026:  Phase 1 complete - 30 families implemented & tested
Oct 2026:  Phase 2 kickoff - expand to 60 detectors
Nov 2026:  Paper submitted to NeurIPS
Dec 2026:  Leaderboard UI complete
Jan 2027:  Phase 2 complete - 100+ detectors ready
Feb 2027:  Phase 3 launch - public release
Mar 2027:  First community submissions
Apr 2027:  v5.0 stable, paper published/accepted
May 2027:  Phase 4 - sustainability planning
Jun 2027:  v5.1 with community feedback
Jul 2027:  Review + plan for v6
```

---

## Why This Works

✅ **Solves real problem:** Reward hacking detection is critical for AI safety  
✅ **Community-driven:** Crowdsource families + detectors  
✅ **Credible foundation:** Built on rigorous methodology  
✅ **Low barrier to entry:** Templates make contribution easy  
✅ **Sustainable:** Minimal compute, free hosting, community support  
✅ **Publishing path:** Multiple venues, strong motivation  
✅ **Career value:** Positions you as authority in reward hacking research  

---

## First Week Action Items

1. [ ] Set up GitHub organization (`rhob-benchmark` or similar)
2. [ ] Create GitHub issue templates (family proposal, detector submission)
3. [ ] Draft 5 family design proposals (pick from list above)
4. [ ] Set up discussion forum (GitHub Discussions)
5. [ ] Write "Roadmap" issue (share this plan publicly)
6. [ ] Begin Phase 1: design sprint for 19 new families

---

## Questions to Lock In

1. **GitHub home:** Personal account or new org?
2. **Licensing:** MIT, Apache 2.0, or BSD?
3. **Compute budget:** Any preference for cloud provider?
4. **Name/branding:** "RHOB" is good—want to keep it?
5. **First launch target:** NeurIPS 2026 or ICLR 2027?

This is ambitious but achievable. You're building something that could become *the* standard for AI safety research.

Ready to start?

