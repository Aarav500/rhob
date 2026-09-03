# RHOB roadmap

This file used to plan an ImageNet-style benchmark with a headline transfer score and a
twice-yearly challenge. That plan is withdrawn. The headline score did not survive a sign
control (0.994 to 0.508; see the correction in [README.md](README.md)), the onset column
the acronym refers to is unvalidated, and the audit that found both became the paper this
repository now serves as the case study for.

What the repository is: a 33-family matched-pair benchmark with an admission gate, a
20-draw replicated leaderboard, and the audit record of the checks in it that could not
have failed. The benchmark is the evidence. The paper is the claim.

## Retired

- The RHOB Transfer Score as a headline metric, and any leaderboard ranked on it.
- The challenge cadence and the workshop shared task built around it.
- The internal onset column. Every admitted cell records an onset label with zero
  dispersion across seeds, so a constant prediction equal to the label scores a perfect
  error of 0.000. The column is no longer reported; the code path is kept only for the
  external HVTA evaluation, where the label was not written by us.
- Any claim that RHOB detects reward hacking. The replicated result is that the three
  sub-oracle rungs sit within 0.09 of chance for unsupervised detectors.

## Active

1. **Regenerate every published number under one convention.** A cell a detector could
   not have scored is N/A, never a 0.5. The L1 row was fixed in the 2026-08 correction.
   The L0 Reward Skewness row had the same defect from a different cause (its windows
   need 100 episodes; 19 families run 40 or 60) and is being re-scored on this branch.
2. **Extend the admission ledger to the short-horizon families.** The nightly smoke tier
   reported most of them DEGENERATE, and most of that was the Skewness horizon defect
   rather than the families. With the detector declaring its horizon, they can be
   measured.
3. **The survey, axis A.** The paper's external evidence rests on an enumeration of
   preconditions whose search patterns select against equivalence-test vocabulary. The
   pre-registered remediation (revised patterns, the three unaudited members, the power
   re-run) is the one item that changes what the paper is evidence *about*.
4. **HVTB.** The onset-timestamp patch for the 89 hack-verifiable Terminal Bench tasks is
   going upstream. Step-indexed onset from real agents is future work, and a different
   paper.

## Not planned

Learned families as certified benchmark members, a multi-agent hacking class as a
contribution, or new trace collection before the paper deadline. Each was designed and
argued against; the arguments are in the paper's limitations section.

## What would change this

An external detector submission, or an external corpus in which hacking is decidable by
construction and caused by the agent rather than scheduled by the authors. The second
exists (HVTB) and lacks a per-step label; supplying one is the next thing that would move
the project.
