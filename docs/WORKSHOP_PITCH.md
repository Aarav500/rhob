# RHOB Shared Task — Workshop Pitch (Draft)

A short pitch for proposing an RTS-based shared task at an ICML/NeurIPS
safety/alignment workshop, once there's enough external submission activity to justify
it (see [ROADMAP.md](../ROADMAP.md)). Not ready to submit yet — this is the draft to
adapt once a specific workshop CFP is targeted.

## Working title

**"The RHOB Shared Task: Does Your Reward-Hacking Detector Generalize?"**

## One-paragraph pitch

Reward-hacking detectors are usually evaluated in-distribution, against the same
mechanism they were designed to catch. RHOB's matched-proxy construction makes
reward-only detection *provably* impossible by design, and its cross-family transfer
protocol asks a harder, previously under-tested question: does a detector trained on
one set of hacking mechanisms generalize to mechanistically unseen ones? We propose a
shared task built directly on RHOB's existing 14-family suite and RTS metric
(mean AUROC on 8 held-out mechanisms after training on 6), inviting submissions of new
detectors, new families, or both.

## Why this fits a workshop (not just a benchmark release)

- **A concrete, reproducible task** with an existing submission format
  (`python -m rhob validate`), an existing CI-validated leaderboard, and a clear
  single ranking metric (RTS) — everything a shared task needs already exists.
- **A genuinely open question**, not a solved one: the current best result (an
  ensemble of deterministic behavioral detectors) reaches near-perfect RTS, but a
  *single* learned detector's transfer performance is highly seed- and
  test-set-composition-dependent — exactly the kind of open methodological question
  a workshop audience would want to dig into together.
- **Low barrier to entry**: `pip install -e ".[dev]"` and a documented 30-minute
  detector-evaluation tutorial already exist.

## What we'd need before submitting this

- At least 2-3 non-maintainer detector or family submissions, to show the task has
  organic pull rather than being purely maintainer-run.
- A named co-organizer from an established lab or workshop-organizing community,
  since a shared task proposed by a single first-time submitter carries less weight
  than one with a recognizable co-organizer.
- A specific target venue and CFP deadline to adapt this draft against.

## Draft submission tracks

1. **Detector track**: submit a detector, report RTS + in-distribution AUROC on the
   existing 14 families.
2. **Family track**: submit a new matched-proxy family passing the admission gate
   (5 criteria — see [CONTRIBUTING.md](../CONTRIBUTING.md)); accepted families get
   added to the next held-out test split.
