# Week 1 Launch Checklist

**Goal:** Ship v3.2 paper + launch v5 community platform infrastructure  
**Timeline:** Aug 5-11, 2026 (This Week!)  
**Status:** Ready to execute

---

## Day 1 (Monday): GitHub Setup

### Morning
- [ ] Create GitHub repo: `github.com/your-username/rhob`
  - License: MIT
  - Description: "RHOB: The standard benchmark for reward hacking detection"
  - Make PUBLIC
  - Enable Discussions
  
- [ ] Copy files to repo
  - [ ] README_GITHUB.md → README.md
  - [ ] CONTRIBUTING.md (already have)
  - [ ] docs/API_SPECIFICATION.md
  - [ ] docs/DETECTOR_TEMPLATES.md
  - [ ] RHOB_V5_PLATFORM_ROADMAP.md
  - [ ] FAMILY_DESIGN_TEMPLATE.md
  - [ ] src/rhob/ (all code)
  - [ ] tests/ (all tests)

- [ ] Create GitHub issue templates
  - `.github/ISSUE_TEMPLATE/family_proposal.md`
  - `.github/ISSUE_TEMPLATE/detector_submission.md`
  - `.github/ISSUE_TEMPLATE/bug_report.md`

- [ ] Set up GitHub Discussions
  - [ ] Create "General" category
  - [ ] Create "Family Design" category
  - [ ] Create "Detector Development" category
  - [ ] Pin announcement: "Welcome to RHOB! Here's how to contribute"

- [ ] Initial commit
  ```bash
  git init
  git add .
  git commit -m "RHOB v3.2: Initial release

  - 9 environment families (v3.2 baseline + 3 v5 new)
  - 30 baseline detectors across 4 access levels
  - Matched-proxy methodology with anti-symmetry screening
  - Cross-family transfer analysis
  - Community contribution infrastructure

  Ready for v5 Phase 1 expansion.
  
  Co-Authored-By: Claude <noreply@anthropic.com>"
  
  git branch -M main
  git remote add origin https://github.com/your-username/rhob.git
  git push -u origin main
  ```

### Afternoon
- [ ] Add GitHub topics
  - `benchmark`, `reward-hacking`, `ai-safety`, `reinforcement-learning`, `detection`

- [ ] Update GitHub profile
  - Add link to RHOB repo

- [ ] Star this repo yourself (to kick off counter)

**Deliverable:** Live GitHub repo with 100% code + docs

---

## Day 2 (Tuesday): Paper Finalization

### Morning
- [ ] Polish v3.2 paper (`docs/RHOB_V5_PAPER.md`)
  - [ ] Add real leaderboard results (use v5 results)
  - [ ] Add Figure 1: Matched-proxy principle diagram
  - [ ] Add Table 2: 30-detector breakdown
  - [ ] Add Table 3: Cross-family transfer AUROC
  - [ ] Write Results section with findings
  - [ ] Proofread + fix typos
  - [ ] Target: 4,500-5,000 words

- [ ] Create arXiv-ready PDF
  - [ ] Use arxiv LaTeX template
  - [ ] Keep version as v3.2 (not v5 - separate papers)
  - [ ] Include GitHub link in abstract/conclusion
  - [ ] Save as `rhob_v3.2.pdf`

### Afternoon
- [ ] Prepare submission packages for venues
  - [ ] TMLR: Copy paper as `.pdf`, prepare short abstract, test submission
  - [ ] NeurIPS Benchmarks: Check Nov deadline, format requirements
  - [ ] ICLR Benchmarks: Check Jan deadline, format requirements

**Deliverable:** v3.2 paper ready for submission (4,500+ words, final PDF)

---

## Day 3 (Wednesday): Venue Submission + Family Issues

### Morning
- [ ] Submit paper to TMLR
  - TMLR accepts rolling submissions
  - Expected decision: 4-8 weeks
  - Get decision early → informs other submissions
  - Submission link: https://openreview.net/group?id=TMLR

- [ ] Start writing NeurIPS Benchmarks submission (if targeting Nov deadline)
  - Otherwise defer to ICLR Jan deadline

### Afternoon
- [ ] Create 20 GitHub issues for family proposals
  - [ ] Copy from PHASE_1_FAMILY_PROPOSALS.md
  - [ ] Create 20 separate issues (one per family)
  - [ ] Format: `[Family Name] - [Mechanism]`
  - [ ] Label: `family-proposal`, `phase-1`
  - [ ] Example: `#1 Mode Collapse - Camping Exploit`
  
- [ ] Pin family proposal thread
  - Open GitHub Discussion: "Phase 1 Family Design Sprint: 20 proposals now open!"
  - Link to issues
  - Explain: "Help us refine these designs or propose your own"
  - Invite feedback: "Deadline: Friday for community comments"

**Deliverable:** Paper submitted to TMLR, 20 family issues open, community invited

---

## Day 4 (Thursday): Family Implementation Kickoff

### Morning
- [ ] Respond to community feedback on family proposals
  - Answer questions
  - Refine designs based on suggestions
  - Mark 5-10 as "approved for implementation"

- [ ] Select first 5 families to implement
  - Pick: Mode Collapse, Reward Leakage, Sparse Reward Camping, Curriculum Gaming, Generalization Failure
  - Reason: highest confidence + diverse mechanisms
  - Create branches: `family/mode_collapse`, etc.

### Afternoon
- [ ] Implement Family 1: Mode Collapse
  - Create: `src/rhob/v3/families/mode_collapse.py` (170 lines)
  - Create: `tests/test_v3/test_family_mode_collapse.py` (7 tests)
  - Target: Complete by EOD

- [ ] Start Family 2 implementation

**Deliverable:** 1-2 families implemented + tested

---

## Day 5 (Friday): Leaderboard Infrastructure + Announcement

### Morning
- [ ] Update family registry
  - [ ] Import new families in `src/rhob/v3/families/__init__.py`
  - [ ] Verify all 14 families (9 existing + 5 new) register
  - [ ] Test: `FamilyRegistry.list_families()` returns 14

- [ ] Continue family implementations
  - [ ] Target: 5 families complete by EOD
  - [ ] Run full test suite: `pytest tests/ -v`
  - [ ] All 242+ tests passing

### Afternoon
- [ ] Prepare leaderboard infrastructure
  - [ ] Update `scripts/v5_leaderboard_and_transfer.py`
  - [ ] Extend to handle 25+ families
  - [ ] Test with 14 families (current + 5 new)
  - [ ] Generate sample results JSON

- [ ] Make first weekly announcement
  - GitHub Discussions: "Week 1 Complete: 5 new families shipped, 20 proposals open"
  - Share metrics: 14 families, 30 detectors, 250+ tests
  - Invite participation: "First community family implementations starting next week"
  - Show leaderboard preview (static image)

**Deliverable:** 5 families implemented, tests passing, community announcement

---

## End-of-Week Metrics

✅ GitHub repo live (public, 100+ stars goal)  
✅ v3.2 paper submitted to TMLR  
✅ 20 family proposals open (community feedback)  
✅ 5 new families implemented & tested  
✅ 14 total families, 30 detectors  
✅ 250+ tests all passing  
✅ Infrastructure ready for Phase 1 scale-up  
✅ Weekly communication established  

---

## What You Need RIGHT NOW

### GitHub Setup
- [ ] GitHub account (if not already)
- [ ] GitHub Discussions enabled on repo
- [ ] Basic Git knowledge (init, commit, push)

### Paper Submission
- [ ] arXiv account (free): https://arxiv.org/
- [ ] TMLR account (free): https://openreview.net/group?id=TMLR
- [ ] LaTeX (optional, can write in Markdown)

### Family Implementation
- [ ] Already have templates + examples
- [ ] Python environment running (already set up)
- [ ] Pytest working (already tested)

### Community
- [ ] Email list to invite (optional, can start organic)
- [ ] Twitter handle (optional, for announcements)

---

## Contingency Plans

**If GitHub setup takes longer:**
- Skip fancy issue templates, just create raw issues
- Can add templates later (doesn't block launch)
- Focus on repo + content

**If paper submission is complicated:**
- Submit to TMLR first (simplest process)
- Can submit to other venues later
- arXiv preprint can happen in parallel

**If family implementation is slow:**
- Start with 3 families instead of 5
- Quality > quantity
- Community can help implement remaining ones

**If no community response:**
- That's fine! Still ship infrastructure
- Momentum builds over weeks/months
- Focus on your own 20 families
- Community will follow once critical mass is reached

---

## Success Criteria (Week 1)

🟢 **Must-have:**
- [ ] GitHub repo created and public
- [ ] v3.2 paper submitted to venue
- [ ] 3+ new families implemented
- [ ] Family design proposals open

🟡 **Should-have:**
- [ ] 5 new families implemented
- [ ] Community feedback on proposals
- [ ] Weekly announcement posted
- [ ] 100+ GitHub stars

🔵 **Nice-to-have:**
- [ ] All 20 families designed
- [ ] Leaderboard live
- [ ] First community family submission

---

## Templates & Files Ready to Go

✅ README_GITHUB.md (copy → README.md)  
✅ FAMILY_DESIGN_TEMPLATE.md (for issues)  
✅ PHASE_1_FAMILY_PROPOSALS.md (copy to issues)  
✅ RHOB_V5_PLATFORM_ROADMAP.md (link in README)  
✅ All code + tests (just push)  

**Everything is ready. You just need to execute.**

---

## Quick Links

- **GitHub New Repo:** https://github.com/new
- **TMLR Submission:** https://openreview.net/group?id=TMLR
- **NeurIPS Benchmarks:** https://neurips.cc/ (Track announcement)
- **ICLR Benchmarks:** https://iclr.cc/ (Track announcement)

---

## Let's Go!

**This is the week everything changes.** After this week:
- v3.2 is published (credibility ✓)
- v5 infrastructure is live (scale ✓)
- Community is engaged (momentum ✓)
- You're 20% to v5 at scale ✓

**One week. You've got this. 🚀**

