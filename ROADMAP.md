# RHOB Roadmap

RHOB's long-term goal is to become the shared, default benchmark for reward-hacking
detection — the way ImageNet became the default benchmark for image classification. That
requires two things a single benchmark release can't provide on its own: a metric the
whole field rallies around, and a visible reason to keep coming back. This roadmap covers
both.

## The headline metric: RTS

The **RHOB Transfer Score (RTS)** — mean AUROC on 8 held-out, mechanistically-unseen test
families after training on 6 — is now the benchmark's primary reported number (see
[README.md](README.md) and the [live leaderboard](https://rhob.aarav-shah.com)). Every
new detector submission is expected to report RTS alongside in-distribution AUROC, the
same way new vision models report top-5/top-1 accuracy on ImageNet rather than an
in-house metric. Overall-AUROC-only submissions are still accepted, but won't rank on the
RTS leaderboard.

## Versioned milestones

- **v1.4 (current)**: 14 families, 3 REWARD_TAMPERING/DECEPTIVE_ALIGNMENT/RM_OVEROPT
  mechanisms, 35 detectors, RTS established as the headline metric.
- **v1.5**: extend RTS evaluation to the full 35-detector suite (currently only 4
  representative detector classes have been transfer-tested). Add 2-3 community-submitted
  families if admission-gate-valid submissions arrive in the meantime.
- **v1.6**: multi-agent extension — matched-proxy environments where hacking degrades a
  *shared* reward, or agents collude around a flawed joint reward. Directly follows up on
  the interest-check issues opened with PettingZoo/Farama.
  ([tracking issue](https://github.com/Aarav500/rhob/issues) — open once scoped)
- **v2.0**: target 25+ families and a real RLHF-scale setting (beyond the current toy
  preference-bandit), contingent on community contributions and/or a lab partnership.

## The RHOB Challenge: a recurring cadence

A benchmark that never changes doesn't generate a "who's ahead this year" story —
ImageNet had ILSVRC for exactly this reason. Proposed cadence:

- **Twice yearly** (aligned loosely with major ML conference deadlines), publish a
  snapshot of the RTS leaderboard: who's on top, what moved since last time, which
  mechanism is still hardest (currently: reward-channel/sensor-calibration tampering are
  the two hardest held-out families for every non-ensemble detector).
- Publicize each snapshot the same way a version bump is publicized — a short write-up,
  not just a commit.
- Once there are 3+ independent (non-maintainer) detector submissions, propose a
  **workshop shared task** at an ICML/NeurIPS safety/alignment workshop, built directly
  around the RTS leaderboard as the shared evaluation. A short pitch draft for this is in
  [docs/WORKSHOP_PITCH.md](docs/WORKSHOP_PITCH.md).

## What would accelerate this

- A citable paper (arXiv, currently blocked on an endorsement in progress).
- At least one detector submission from a researcher outside the maintainer's own
  network — the first external submission is qualitatively different from every
  maintainer-authored one, the same way ImageNet only mattered once outside labs adopted
  it.
- A co-author or named collaborator from an established lab, which changes the project's
  perceived provenance from "one researcher's repo" to "a field resource."
