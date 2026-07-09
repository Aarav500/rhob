---
name: Detector submission / result report
about: Share a detector you've run against RHOB, submit one for inclusion, or report results (positive or negative)
title: "[Detector] "
labels: detector
assignees: ''
---

## What did you test?

- **Detector name / approach:**
- **Access level (L0/L1/L2/L3):**
- **Families tested against:**
- **Link to code (if applicable):**

## Results

Paste your AUROC numbers, or describe what you observed. Both "this works well"
and "this doesn't work / behaves inconsistently" are useful reports.

| Family | AUROC | Notes |
|---|---|---|
|  |  |  |

## Is this a submission for inclusion in RHOB's detector suite?

- [ ] Yes — I'd like this added as a baseline detector
- [ ] No — this is just a result report / feedback

If yes, please confirm you've read [CONTRIBUTING.md](https://github.com/Aarav500/rhob/blob/main/CONTRIBUTING.md#contributing-a-detector)
and the detector implements the `PosthocDetector` interface with the correct
`access_level`.

## Anything surprising?

Did the benchmark itself behave in a way you didn't expect (e.g. a family that
didn't discriminate the way its access level implied)? Tell us — several real
bugs in RHOB were found exactly this way (see [REPRODUCIBILITY.md](https://github.com/Aarav500/rhob/blob/main/REPRODUCIBILITY.md)).
