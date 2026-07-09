---
name: Environment family proposal
about: Propose a new hacking mechanism, environment, or benchmark extension (RLHF, multi-agent, higher-dimensional control, etc.)
title: "[Family] "
labels: family-proposal
assignees: ''
---

## The hacking mechanism

What behavior does this family model? What does the proxy reward, and what
does the true objective?

## The three design questions (see [Environment Tutorial](https://github.com/Aarav500/rhob/blob/main/docs/TUTORIAL_ENVIRONMENT.md))

1. **Proxy-preserving symmetry σ:** the transformation mapping the hacking
   variant to the legitimate variant while leaving the proxy reward invariant.
2. **Why is the proxy σ-invariant?** (i.e. why won't an L0 detector leak here)
3. **Discriminating behavioral feature:** the scalar, per-episode signal that's
   ~0 pre-onset and flips sign between variants post-onset.

## Complexity / scope

- [ ] Tabular / gridworld
- [ ] Continuous control
- [ ] Something else (RLHF, multi-agent, higher-dimensional): describe below

## Have you run a pilot?

If you have preliminary numbers (even a hand-coded controller, not a trained
agent) showing the proxy is roughly matched and the feature separates, share
them here — it's the fastest way to tell if the design is admission-gate-viable
before investing in a full implementation.
