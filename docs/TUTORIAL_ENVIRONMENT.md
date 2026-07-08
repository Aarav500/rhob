# Environment (Family) Tutorial: Add a New Hacking Mechanism

A "family" is a matched-proxy pair generator for one hacking mechanism. This
walks through the minimum path to a working, admitted family.

## 0. Read one example first (10 min)

Don't design from scratch. Read the simplest shipped family end-to-end:
[`src/rhob/v3/families/gridworld_camping.py`](../src/rhob/v3/families/gridworld_camping.py)
(tabular, ~60 lines) or
[`src/rhob/v3/families/novelty_farming.py`](../src/rhob/v3/families/novelty_farming.py)
(continuous, roaming behavior).

## 1. Design on paper before writing code (15-20 min)

Answer these three questions — they are exactly what the admission gate
checks, so getting them right up front saves iteration:

1. **What is the proxy-preserving symmetry σ?** The transformation that maps
   the hacking variant to the legitimate variant while leaving the *proxy*
   reward invariant. E.g. "swap grid rows and columns," "reflect the goal
   position," "relabel which state cluster counts as `class_a`."
2. **What is the proxy reward, and why is it σ-invariant?** If your proxy
   depends on which state/action was taken rather than only on
   σ-symmetric aggregate quantities, it likely leaks information L0
   detectors shouldn't have — this was [a real bug we shipped and fixed](../REPRODUCIBILITY.md#3-distributional-shift-proxy-matching-leak).
3. **What is the discriminating behavioral feature?** A scalar, computable
   per-episode, that is ~0 pre-onset and takes *opposite signs* between the
   hacking and legitimate variant post-onset. This anti-symmetry is what
   makes L2 detection possible without reading true reward.

If you can't answer all three cleanly, the family will likely fail the
admission gate — revisit the design before coding.

## 2. Implement `BaseFamily` (30-60 min depending on environment complexity)

```python
from rhob.v3.base_family import BaseFamily, MatchedPair
from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism

@FamilyRegistry.register("my_family")
class MyFamily(BaseFamily):
    @property
    def name(self) -> str:
        return "my_family"

    @property
    def mechanism(self) -> HackingMechanism:
        return HackingMechanism.CAMPING_EXPLOIT  # or add a new enum value

    @property
    def complexity(self) -> EnvironmentComplexity:
        return EnvironmentComplexity.TABULAR  # or CONTINUOUS

    def difficulty_range(self) -> tuple[float, float]:
        return (0.60, 0.98)  # target-L2-AUROC range this family can hit

    def generate_pair(self, difficulty: float, seed: int = 0) -> MatchedPair:
        def rollout_hacking(s: int):
            ...  # returns RunData with proxy_rewards, true_rewards, state_counts, behav_trace

        def rollout_legit(s: int):
            ...

        return MatchedPair(
            family=self.name,
            mechanism=self.mechanism,
            complexity=self.complexity,
            difficulty=difficulty,
            n_episodes=200,
            rollout_hacking=rollout_hacking,
            rollout_legit=rollout_legit,
        )
```

## 3. Run the admission gate (2 min)

```python
from rhob.v3.admission_gate import AdmissionGate
from rhob.v3.registry import FamilyRegistry

gate = AdmissionGate()
family = FamilyRegistry.get("my_family")
certificate = gate.certify(family, n_seeds_per_variant=30)
print(certificate)
```

All five checks must pass:

| Check | What it verifies |
|---|---|
| `proxy_matched` | L0 detector 95% CI on proxy reward contains 0.5 (no proxy leak) |
| `behavioral_separated` | L2 AUROC exceeds the floor (feature actually discriminates) |
| `true_reward_diverges` | Legitimate variant's true reward > hacking variant's |
| `onset_localizable` | Onset timing std < 10% of episode horizon |
| `camping_quality` | Post-onset behavior sustains rather than reverting |

If a check fails, it almost always means one of the three design questions
in Step 1 wasn't actually satisfied by the implementation — the checks are
diagnostic, not just gatekeeping. See
[REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for three real examples of gate
violations pinpointing implementation bugs.

## 4. Write tests

Every family needs an admission test. Copy the pattern from
`tests/test_v3/test_family_*.py` for an existing family and swap in yours.

## 5. Submit

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the PR checklist.
