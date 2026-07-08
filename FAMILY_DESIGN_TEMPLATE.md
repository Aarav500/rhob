# Family Design Template

Use this template to propose a new reward-hacking family. Open a GitHub issue and fill this out!

---

## Basic Info

**Family Name:** [Name of the family, snake_case, max 32 chars]  
**Proposed Mechanism:** [From HackingMechanism enum or new]  
**Proposed Complexity:** [TABULAR, CONTINUOUS_SIMPLE, CONTINUOUS_COMPLEX, SEQUENTIAL, MULTI_AGENT]  
**Author:** @[your-github-username]

---

## Design Description

### 1. The Problem (What hacking looks like)

*Describe what an agent does when it hacks the reward. What's the shortcut?*

Example: "Agent learns to jump very high but crashes on landing, optimizing only for max height."

### 2. Symmetry σ (The core insight)

*Describe an explicit transformation relating the two variants.*

Example: "Flip gravity direction: up ↔ down. In both variants, proxy reward (height) is identical. True reward (safe landing) differs."

**Questions:**
- Is σ explicit and formal? (not vague)
- Is σ easy to describe in 1-2 sentences?
- Can you implement it in code?

### 3. Proxy Reward (Observable)

*How is proxy reward computed? Why is it σ-invariant?*

Example: "Proxy reward = max height reached. Under gravity flip σ, agent still reaches same max height in both variants."

**Validation:**
- [ ] Proxy totals match between variants (±20%)
- [ ] Proxy is identical by construction, not by tuning

### 4. True Reward (Hidden)

*How is true reward computed? Why does it diverge?*

Example: "True reward = max height - crash penalty. Hacking variant crashes (penalty), legitimate doesn't."

**Validation:**
- [ ] True rewards diverge: hacking < legitimate
- [ ] Divergence is ~30%+ (noticeable)

### 5. Discriminating Feature (Anti-symmetric)

*What distinguishes hacking from legitimate? Must be anti-symmetric under σ.*

Example: "Brake commitment: how much agent brakes before landing. Positive = hacking (no brake), negative = legitimate (brakes)."

**Validation:**
- [ ] Feature is anti-symmetric under σ (flips sign)
- [ ] Feature is NOT a magnitude (not count, rate, convergence)
- [ ] Feature separates hacking from legitimate ~0.5-1.0 range

### 6. Difficulty Knob (L2-AUROC control)

*Parameter that smoothly controls detection difficulty from 0.60 (hard) to 0.98 (easy).*

Example: "Crash penalty: 0.2 (easy, small penalty → easy to hack) to 0.8 (hard, severe penalty → risky to hack)"

**Validation:**
- [ ] Knob parameter is continuous (not discrete)
- [ ] L2-AUROC varies smoothly: AUROC ranges from ~0.60-0.98
- [ ] Parameter has intuitive meaning

---

## Anti-Symmetry Screening (3-Question Gate)

**Answer YES to all three questions:**

1. **Is symmetry σ explicit?**  
   - [ ] YES — I can write the transformation mathematically
   - [ ] NO — It's vague or implicit

2. **Is discriminating feature anti-symmetric under σ?**  
   - [ ] YES — Feature flips sign: f(hacking) = -f(legitimate under σ)
   - [ ] NO — Feature just has different magnitude

3. **Is proxy equal by construction?**  
   - [ ] YES — Proxy is σ-invariant by design, not by tuning
   - [ ] NO — We had to tune parameters to match proxy

**If all YES, proceed to implementation. If any NO, needs revision.**

---

## Implementation Sketch

*How would you implement this? Outline the key parts.*

```python
class MyFamily(BaseFamily):
    def generate_pair(self, difficulty: float, seed: int = 0):
        # Map difficulty to knob parameter
        knob = 0.2 + (0.8 - 0.2) * (1.0 - (difficulty - 0.60) / 0.38)
        
        def rollout_hacking(seed: int):
            # Agent exploits proxy reward
            # Returns: RunData with proxy_r, true_r, state_counts, behav_trace
            pass
        
        def rollout_legit(seed: int):
            # Agent optimizes true reward
            # Returns: RunData with proxy_r, true_r, state_counts, behav_trace
            pass
        
        return MatchedPair(rollout_hacking, rollout_legit, ...)
```

---

## Related Families

*Is this similar to existing families? How does it differ?*

Example: "Similar to Novelty_Farming (exploration exploit), but novelty_farming uses spatial frontiers while this uses temporal reward distributions."

---

## Expected Detector Performance

*Rough estimate: how hard is this family to detect?*

- **L0 (reward-only):** ~0.50 AUROC (guess?)
- **L1 (+ state-vis):** ~0.60 AUROC (guess?)
- **L2 (+ behavioral):** ~0.75 AUROC (guess?)

*(These are guesses; actual performance determined after implementation)*

---

## Questions for Reviewers

*Ask the community for feedback.*

Example: 
- "Is the brake commitment feature the right signal, or should I use something else?"
- "Does the difficulty knob make sense, or should I use a different parameter?"

---

## References

*Any papers or ideas this is based on?*

Example: "Inspired by [Russell & Norvig IRL paper], explores specification gaming in physics simulators."

---

## Checklist

Before submitting PR:

- [ ] Symmetry σ is explicit and clear
- [ ] Anti-symmetry screening: all 3 questions YES
- [ ] Proxy rewards match (±20%)
- [ ] True rewards diverge (hacking < legitimate)
- [ ] Feature is anti-symmetric
- [ ] Difficulty knob is continuous and intuitive
- [ ] Implementation sketch is coherent
- [ ] 3+ test cases written
- [ ] All tests pass
- [ ] Ready for code review

---

## Next Steps

1. Open GitHub issue with this filled out
2. Community feedback (24-48 hours)
3. Refine based on feedback
4. Implement (3-4 hours)
5. Test (1-2 hours)
6. Submit PR
7. Merged → added to leaderboard

**Questions?** Ask in [GitHub Discussions](https://github.com/your-username/rhob/discussions)

