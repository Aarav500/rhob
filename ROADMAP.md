# Roadmap

## What changed (2026-10-31)

This roadmap used to aim RHOB at becoming "the shared, default benchmark for
reward-hacking detection — the way ImageNet became the default benchmark for image
classification", with the RHOB Transfer Score (RTS) as its headline number and a
twice-yearly "RHOB Challenge" to give the field a reason to keep coming back. All three
are withdrawn.

- **RTS does not measure transfer.** It was 0.994 on held-out families. Once each
  family's behavioural feature stopped carrying the label in its sign, it fell to 0.508,
  chance. Even before that, the held-out families were written by one author under one
  construction recipe, so the most RTS could show was transfer within a single authored
  convention.
- **RHOB cannot be the default measure of detection.** In 27 of its 33 families the hack
  itself is authored, and in all 33 so is the flaw it exploits. A benchmark built that way
  can say what a signal contains. It cannot say how well a detector catches hacking an
  optimizer found, which is the question the ImageNet framing promised to answer.

The withdrawn version is in git history, last at `06df9f3`, rather than here. A plan whose
headline metric is retracted should not sit beside one that is current.
[docs/WORKSHOP_PITCH.md](docs/WORKSHOP_PITCH.md), which proposed a shared task built on
RTS, is withdrawn with it.

## The goal now

Make the failure class findable in other people's benchmarks. The contribution is the
class — checks that cannot return a negative verdict under the design that ships — and the
diagnostics that find it. RHOB is the case study where seven instances were found, and the
main evidence that the diagnostics find real defects.

## Milestones

- **`cannotfail` 0.1 — done (2026-10-31).** D1 and D2 as a standalone package in
  [`tools/cannotfail`](tools/cannotfail), installable without RHOB and held to RHOB's
  admission gate by [`tests/test_cannotfail_parity.py`](tests/test_cannotfail_parity.py).
- **One maintainer outside the author's network runs it on their own score vectors — by
  2026-12-15.** Not a detector submission to RHOB, which was the bar this roadmap used to
  set: a run of the diagnostics on a benchmark the author did not build.
- **The paper at a venue.** A submission to TMLR, after the paper's claims are made to
  match the repository's.
- **The survey port.** The paper's 77 preconditions as rows, reconciled against its
  Appendix F by `rhob.audit.frame.reconcile`, coded twice for claim shape per
  [docs/SURVEY_PROTOCOL.md](docs/SURVEY_PROTOCOL.md). Authors are contacted per
  [docs/AUTHOR_OUTREACH.md](docs/AUTHOR_OUTREACH.md) before any table naming them is
  published.
- **After 2026-12-15: test the no-learner objection on someone else's learner.** Wong,
  Engels and Nanda's RL reward-hacking environment is open source, with a run that has a
  loophole and a baseline that does not. The question is whether hacking that an optimizer
  found is invisible to the proxy, as RHOB's construction assumes. The outcomes are fixed
  before any data exists, and one of them is stated now: **if the proxy separates the
  hacking and legitimate runs, that is the headline**, and it means RHOB's matched-proxy
  regime may be an artifact of scripting.
- **D3 in `cannotfail`** — only once a second benchmark's harness shows what a channel
  digest would need to be told.

## What RHOB itself is for now

- **A fixture for the diagnostics.** Its admission ledger is the calibration set: the 30
  degenerate `proxy_matched` cells and `onset_localizable`, whose statistic is exactly 0.0
  in all 50 cells, are known positives that `cannotfail` must find.
- **A worked example of a check built so it can fail.** The matched-pair construction and
  the admission gate, whose nightly smoke screen rejected 12 of the 13 families it screens.
- **Not a leaderboard to rank detectors on.** The old v1.5, v1.6 and v2.0 milestones —
  RTS for every detector, a multi-agent extension, 25 or more families — are dropped. Each
  would enlarge a benchmark whose central construction is now the thing in question.

## Discoverability: the Inspect Evals Register

HVTB — the Terminal Bench 2.1 hack-rate eval on this repository's `inspect-eval` branch —
is listed in the [Inspect Evals
Register](https://ukgovernmentbeis.github.io/inspect_evals/evals/hvtb/), maintained by
Generality Labs with contributions from the UK AI Security Institute, Arcadia Impact and
the Vector Institute.

That is a distribution channel, not a milestone. What is listed measures agents' hack
rates; it is neither the diagnostics nor RHOB's detector suite, the register publishes no
numbers for it, and its checks are structural — the Contributor Guide's scoping, validity
and scoring standards are "not mandatory for registry acceptance." It does not count as an
outside user. What it plausibly buys is that someone searching for reward-hacking evals in
Inspect finds this repository without having been told about it, and nothing here
substitutes for asking people directly.

## What would accelerate this

- **The paper at a venue.** The arXiv route has produced no endorsement; a journal needs
  none.
- **One outside maintainer running the diagnostics on their own benchmark.** Every run so
  far is the author's own, on a benchmark the author built. The first run by someone who
  can contest the result is qualitatively different.
- **A co-author or named collaborator from an established lab**, which changes the
  project's provenance from one researcher's repository to something others have checked.
