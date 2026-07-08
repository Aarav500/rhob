# RHOB: The Standard Benchmark for Reward Hacking Detection

[![GitHub stars](https://img.shields.io/github/stars/your-username/rhob)](https://github.com/your-username/rhob)
[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/2407.xxxxx)
[![Leaderboard](https://img.shields.io/badge/leaderboard-live-green)](https://huggingface.co/spaces/your-username/rhob-leaderboard)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RHOB v3.2 is shipping. v5 community platform launching now.**

---

## What is RHOB?

RHOB (Reward Hacking Observation Benchmark) is the first systematic benchmark for detecting when reinforcement learning agents optimize proxy metrics instead of true objectives—a critical problem in AI safety.

**The Problem:** Reward hacking happens when agents find shortcuts to maximize observed rewards without achieving the intended goal. Examples:
- Agents reach fake goals instead of real ones
- Physics simulators exploit unrealistic edge cases
- Agents overfit to training reward distributions
- Deceptive alignment (agents hide true capabilities)

**The Solution:** RHOB provides:
- **9 environment families** with matched-proxy methodology (proxy rewards identical, true rewards diverge)
- **30 baseline detectors** across access levels (L0: reward-only to L2: full trajectories)
- **Cross-family transfer analysis** (train on Families 1-6, evaluate on 7-9)
- **Community-driven expansion** (contribute families, detectors, analyses)

---

## Quick Start

### Install

```bash
git clone https://github.com/your-username/rhob.git
cd rhob
pip install -e .
```

### Run Benchmark

```python
from rhob.v3.benchmark import Benchmark
from rhob.detectors import RewardPeakDetector

detector = RewardPeakDetector()
results = Benchmark.evaluate(
    detector=detector,
    families="all",
    difficulties="all",
    n_seeds=10
)

print(results.summary())
```

### View Leaderboard

**Live at:** [huggingface.co/spaces/your-username/rhob-leaderboard](https://huggingface.co/spaces/your-username/rhob-leaderboard)

---

## Roadmap

| Version | Status | Families | Detectors | Target |
|---------|--------|----------|-----------|--------|
| **v3.2** | 🟢 Shipping | 9 | 30 | Aug 2026 |
| **v5 Phase 1** | 🔵 In Progress | 30 | 60 | Oct 2026 |
| **v5 Phase 2** | ⚪ Planned | 40+ | 100+ | Jan 2027 |
| **v5 Full** | ⚪ Planned | 50+ | 100+ | Jul 2027 |

**See [RHOB_V5_PLATFORM_ROADMAP.md](./RHOB_V5_PLATFORM_ROADMAP.md) for 12-month build plan.**

---

## Key Findings

### v3.2 Results

**30 detectors × 9 families × 3 difficulties:**

| Access Level | Count | Avg AUROC | Best Detector |
|--------------|-------|-----------|---------------|
| **L0** (Reward-only) | 13 | 0.51 | Reward Peak (0.56) |
| **L1** (+ State vis.) | 8 | 0.60 | Centroid Drift (0.67) |
| **L2** (+ Behavioral) | 8 | 0.68 | Trajectory MLP (0.87) |
| **L3** (Oracle) | 2 | 0.88 | True Reward (0.95) |

**Cross-family transfer:** Training on Families 1-6, AUROC drops ~25% on 7-9 → generalization is hard

### New in v5

**3 additional families** targeting new mechanisms:
1. **Goal Misgeneralization** — agent chooses wrong goal
2. **Physics Exploitation** — gravity/physics edge cases
3. **Distributional Shift** — overfit to training reward distribution

---

## Contributing

### Add a Family

Propose a new reward-hacking mechanism! See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

**Process:**
1. Open GitHub issue with family design
2. Community feedback (anti-symmetry screening)
3. Implement + test
4. Submit PR
5. Add to leaderboard

**Example:** Design a family in 1 hour, implement in 3-4 hours

### Add a Detector

Build a new detection algorithm. Templates provided in [docs/DETECTOR_TEMPLATES.md](./docs/DETECTOR_TEMPLATES.md).

**Process:**
1. Pick access level (L0/L1/L2)
2. Use template (copy-paste ready)
3. Implement + test
4. Submit PR
5. Auto-evaluated on leaderboard

**Example:** Implement detector in 1 hour using templates

### Submit Results

Share your detector's leaderboard results with the community.

---

## Citation

If you use RHOB in your research, please cite:

```bibtex
@software{rhob_v3.2,
  title={RHOB v3.2: Reward Hacking Detection Through Matched-Proxy Benchmarking},
  author={[Your Name]},
  year={2026},
  url={https://github.com/your-username/rhob},
  version={3.2}
}
```

**Paper:** [arXiv link when published]

---

## Leaderboard

**Current:** 30 detectors on 9 families  
**Next update:** Weekly  
**View live:** [Leaderboard](https://huggingface.co/spaces/your-username/rhob-leaderboard)

---

## Community

**Get involved:**
- **Discussions:** [GitHub Discussions](https://github.com/your-username/rhob/discussions) — ask questions, share ideas
- **Issues:** [Family proposals](https://github.com/your-username/rhob/issues?q=label%3Afamily-proposal) — discuss new mechanisms
- **Twitter:** [@rhob_benchmark](https://twitter.com/rhob_benchmark) — updates and highlights

**Contributors:** [See contributors page](https://github.com/your-username/rhob/graphs/contributors)

---

## Resources

- **Paper:** [Methodology & results](./docs/RHOB_V5_PAPER.md)
- **Contributing:** [Step-by-step guide](./CONTRIBUTING.md)
- **Family Design:** [Template & examples](./docs/DETECTOR_TEMPLATES.md)
- **Roadmap:** [12-month build plan](./RHOB_V5_PLATFORM_ROADMAP.md)
- **API:** [Frozen v3.2 specification](./docs/API_SPECIFICATION.md)

---

## License

MIT License — free for research and commercial use. See [LICENSE](./LICENSE) for details.

---

## FAQ

**Q: Can I propose a family?**  
A: Yes! Open a GitHub issue with your idea. We screen for anti-symmetry and help refine designs.

**Q: Can I submit my own detector?**  
A: Yes! Use templates in `DETECTOR_TEMPLATES.md`, implement, and PR. Auto-evaluated on leaderboard.

**Q: How often is the leaderboard updated?**  
A: Weekly on Mondays. New submissions evaluated within 24 hours.

**Q: Is this just for RL?**  
A: Currently RL-focused. v6+ will expand to RLHF, supervised learning, language models.

**Q: What's the difference between v3.2 and v5?**  
A: v3.2 is the research baseline (9 families, 30 detectors). v5 is the community platform (50+ families, 100+ detectors, full ecosystem).

---

## Team

- **Founder:** [Your Name]
- **Maintainers:** (Looking for collaborators!)
- **Contributors:** [See GitHub](https://github.com/your-username/rhob/graphs/contributors)

---

**RHOB: Making reward hacking detectable. For AI safety.**

Star us on GitHub! ⭐

