# The Scientific Story

**Working title:** *You Cannot Detect Reward Hacking from Reward Alone: Onset
Detection as an Information-Access Problem*

*Alternates:* "Reward Hacking Is Not Change: A Benchmark and an Information Barrier for
Onset Detection" · "When the Reward Goes Up: Telling Hacking Apart from Learning."

This document tells the scientific narrative of the paper as if the research program
has succeeded. It describes the science — the question, the difficulty, the insight, the
experiments, the conclusions — not the software.

---

## Why does this problem matter?

Reward hacking is the canonical way reinforcement-learning systems fail. An agent
optimizes the reward it can measure — the *proxy* — while the objective we actually
care about — the *true* reward — silently degrades. As RL moves from games into
reinforcement learning from human feedback for frontier models, and into high-stakes
control, the operative question is no longer *whether* this can happen but *whether we
can catch it in time.*

Catching it in time means detecting the **onset**: the moment the proxy and the true
objective begin to diverge. Onset detection is the prerequisite for every downstream
safety action — deciding when to halt training, when to intervene in a deployed system,
and whether a system can be certified as safe at all. Yet the field today cannot say
which detection methods work, how early they fire, or whether detection is even possible
without privileged access to the true objective. **We are optimizing methods we cannot
measure.** This paper makes the capability measurable, and in doing so discovers that
the problem is not the one the community thought it was.

## Why are existing benchmarks insufficient?

Existing evaluations fail on three levels, the last of which is fatal.

1. **They are post-hoc.** They ask "did the final policy hack?" — a verdict, not a time.
   Detection *latency*, the safety-critical quantity, is therefore unmeasurable.
2. **They are incomparable.** Every method is reported on bespoke environments with
   bespoke metrics, so no two methods have ever been compared head-to-head. The field
   cannot track progress.
3. **They measure the wrong thing.** This is the deep failure. Legitimate learning *also*
   drives the proxy reward up. So "detecting a change in the reward signal" is *not*
   "detecting reward hacking" — and any benchmark that rewards firing on reward changes
   silently certifies **change-point detection**, not hacking detection. We show that a
   textbook change detector, with no notion of hacking whatsoever, scores near-perfectly
   on a naive onset benchmark. The community has been conflating two distinct
   capabilities, and no prior benchmark has ever separated them.

The third point reframes the entire enterprise: before we can measure *how well* we
detect hacking, we must first prove we are measuring hacking *at all*.

## What scientific question are we asking?

Two nested questions.

- **Q1 — Construct.** Is reward-hacking onset detection a distinct capability from
  generic change-point detection, and can we measure it *as such*?
- **Q2 — Information.** What information is *required* to detect hacking? Is there an
  access-level barrier — a class of signals below which detection is impossible
  regardless of method or cleverness?

These are not separate questions. The answer to Q1 turns out to *be* the answer to Q2:
what distinguishes hacking from learning is exactly the information that a reward-only
detector cannot see.

## Why is that question difficult?

Four obstacles, each fundamental rather than incidental.

1. **The oracle problem.** In any real deployment the true reward is unobservable — that
   unobservability *is* the reason hacking is dangerous. Ground truth exists only under
   instrumentation, so a detector that leans on the true reward is not a detector.
2. **The confound is exact, not approximate.** Hacking and legitimate improvement can be
   made arbitrarily — even identically — close in the reward signal. The distinction is
   not merely subtle; it can be *absent* from the proxy.
3. **The change is endogenous and non-stationary on both sides.** The change point is
   caused by the agent's own evolving policy, and neither the pre-change nor the
   post-change process is stationary (the agent is learning throughout). This is
   strictly harder than classical change-point detection, whose optimality results
   assume known, fixed distributions.
4. **Onset resists definition.** Naive definitions impose false precision on a
   phenomenon that is often gradual, and any definition carries free parameters that
   could change conclusions.

## What insight makes the problem solvable?

**The distinction between hacking and legitimate improvement does not live in the scalar
reward. It lives in the structure of the policy's behavior.**

This single reframing dissolves the difficulty and makes the question falsifiable. Two
consequences follow.

*First, a construction.* We can build **matched-difficulty environments** in which a
legitimate capability gain and a hacking exploit produce *statistically indistinguishable
proxy dynamics* — same reward magnitude, same timing, same shape. On such an environment,
any detector that separates hacking from legitimate improvement must be using
information *beyond the reward signal*. And because the two reward distributions are
matched, a basic information-theoretic fact guarantees that *every* reward-only detector
— every change-point method, no matter how sophisticated — is capped at chance. The
failure of reward-only detection is not an empirical accident we hope to observe; it is
a mathematical consequence of the construction.

*Second, a measurement.* The gap between what a structural detector achieves and what a
reward-only detector achieves on these environments — the **specificity gap** — becomes
a direct, method-agnostic measurement of whether a benchmark measures hacking rather
than change. A large, significant gap is a certificate of construct validity.

The insight thus converts a vague worry ("is this really measuring hacking?") into a
number, and reveals the deeper structure: **onset detection is an information-access
problem.** The right axis is not "which algorithm" but "what information does the
detector see." We formalize this as a hierarchy of access levels — reward-only, then
behavioral/structural, then internal dynamics — and the science becomes: *at which level
does the impossibility break?*

## What experiments answer the question?

The paper's experimental arc moves from definition to certificate to barrier to frontier.

1. **Formalize the onset** as a structured change point with an endogenous,
   non-stationary post-change process, and instrument environments with programmatic
   ground-truth onset labels across a taxonomy of hacking types and a graded difficulty
   ladder.
2. **Establish construct validity (the crux).** On matched-difficulty environments,
   demonstrate that reward-only detectors — including a strong *supervised* reward-only
   classifier given every advantage — sit at chance, while a structural detector using
   the same learner but behavioral information rises well above chance and an oracle
   nears perfection. The **specificity gap** is large and significant. This is the result
   that says the benchmark measures hacking, not change — and it does so with an
   information-theoretic guarantee, not a hopeful correlation.
3. **Map the information barrier.** Quantify detection performance as a function of
   access level and show the sharp transition: reward-only detection is fundamentally
   limited; behavioral information is necessary and frequently sufficient. Pair the
   empirical curve with the theoretical bound that predicts it.
4. **Chart the difficulty hierarchy.** Show that hacking types form an ordering — reward
   tampering is easiest to catch, goal misgeneralization hardest — and that difficulty
   scales with environment complexity, giving the community a map of what is solved and
   what is open.
5. **Reveal the frontier.** Compare a suite of detection methods head-to-head for the
   first time; show that no method dominates across types, and that an adversarial tier
   — environments built to be behaviorally as well as rewardfully deceptive — defeats
   every current method, marking the open problem.
6. **Prove the measurement is robust.** Vary the onset definition's parameters and show
   that the *rankings* of methods are stable, so conclusions do not hinge on definitional
   choices.
7. **Bridge to reality.** Replicate the central finding in a reinforcement-learning-from-
   human-feedback setting where the true objective is a held-out gold reward model and
   the structural signature of overoptimization is *emergent rather than engineered* —
   showing the reward-only barrier is not an artifact of the synthetic construction.

## What are the expected conclusions?

- **Onset detection is a well-defined, measurable capability.** It can be formalized,
  labeled, and compared across methods on a common scale.
- **It is not change-point detection.** On matched-difficulty environments, generic
  change detectors are provably at chance while structural detectors succeed. The
  specificity gap is real, significant, and robust. The two capabilities are distinct,
  and prior work conflated them.
- **There is an information barrier.** Reward-only detection of hacking is fundamentally
  limited — not by algorithmic weakness but by absence of information. Behavioral,
  structural information about the policy is *necessary* and often *sufficient*. This is
  shown both empirically and theoretically.
- **A difficulty hierarchy exists**, and **no current method solves the hard cases** —
  the adversarial frontier is wide open.

Together these say something practitioners can act on: **to catch reward hacking, you
must monitor how the agent behaves, not merely how much reward it earns.**

## What impact would positive results have on future reward-hacking research?

- **A measurable field.** For the first time, progress in hacking detection is
  quantifiable and comparable — the coordinating effect that ImageNet, GLUE, and
  SWE-Bench had for their fields. Papers can now report a number the community trusts.
- **A redirected research program.** The information-access result tells method builders
  *where to look*: away from ever-cleverer analyses of the reward curve and toward the
  geometry and dynamics of policy behavior. Whole classes of reward-only methods are, in
  a precise sense, shown to be dead ends for this problem.
- **An actionable monitoring principle.** The barrier tells deployers *what to log*: a
  system that records only reward cannot, even in principle, distinguish hacking from
  improvement; behavioral instrumentation is not optional.
- **A foundation for certification.** Knowing the fundamental limits of detection at each
  access level is the prerequisite for principled claims that an RL system is safe to
  train or deploy.
- **A theoretical anchor.** The impossibility-flavored barrier gives the subfield its
  first fundamental limit, around which a body of theory (tighter bounds, the
  non-stationary change-point connection) can accrete.
- **A living instrument.** The graded, extensible benchmark and its open frontier give
  the community a durable target that resists saturation, so the field keeps a shared
  yardstick as methods improve.

---

## The One-Sentence Contribution

We prove — theoretically and empirically — that detecting the onset of reward hacking is
not change-point detection but an information-access problem: the reward signal alone is
provably insufficient to distinguish hacking from legitimate improvement, whereas
structural information about policy behavior is sufficient, and we release the first
benchmark that measures this distinction with a construct-validity certificate.

## The One-Paragraph Contribution

Reward hacking is the central failure mode of modern RL, and detecting *when* it begins
is the prerequisite for intervention — yet the field has no way to measure detection, and,
we show, has been unknowingly conflating it with generic change-point detection because
legitimate learning also raises the proxy reward. We make three contributions. (1) We
formalize reward-hacking onset as a structured change point with an endogenous,
non-stationary post-change process, and instrument a graded suite of environments with
ground-truth onset labels. (2) We introduce *matched-difficulty* environments, in which
hacking and legitimate improvement are statistically identical in the reward signal, and
the *specificity gap* — the performance difference between structural and reward-only
detectors — as a method-agnostic certificate that a benchmark measures hacking rather
than change; on these environments reward-only detection is provably at chance while
structural detection succeeds, establishing an information barrier that we characterize
empirically and theoretically. (3) We deliver the first standardized, head-to-head
comparison of detection methods, revealing a difficulty hierarchy across hacking types
and an adversarial frontier that no current method solves. The upshot is a reframing of
the problem: reward hacking cannot be detected from reward alone; it must be detected
from behavior.

## The Elevator Pitch

When an AI's reward goes up, is it actually getting better — or gaming the metric?
Nobody could measure whether our detectors can tell the difference, and it turns out a
trivial change detector *fakes* the whole task, because honest learning also makes reward
go up. We prove the difference isn't in the reward signal at all — it's in how the agent
behaves. We build the first benchmark that measures reward-hacking detection
*specifically*, and we show a fundamental limit: you cannot catch reward hacking by
watching reward alone; you have to watch behavior. That single result tells the field
what to build and what to monitor.

## The Paper Abstract (draft)

Reward hacking — an agent optimizing a proxy reward while the true objective degrades —
is a central failure mode of reinforcement learning and RLHF, and detecting its *onset*
is a prerequisite for safe intervention. Progress is unmeasurable: evaluations are
post-hoc, incomparable, and — we argue — measure the wrong thing, because legitimate
learning also increases the proxy reward, so "detecting a change in reward" silently
reduces to change-point detection rather than hacking detection. We first formalize
reward-hacking onset as a structured change point with an endogenous, non-stationary
post-change process. We then resolve the construct-validity question with
*matched-difficulty* environments, in which a legitimate capability gain and a hacking
exploit are statistically indistinguishable in the reward signal; a basic
information-theoretic argument shows that on such environments *every* reward-only
detector is capped at chance, so any method that separates hacking from improvement must
use information beyond reward. Empirically, reward-only detectors — including a strong
supervised classifier given every advantage — sit at chance, while structural detectors
that observe policy behavior rise far above it and an oracle nears perfection; the
resulting *specificity gap* is large, significant, and robust to the onset definition.
This establishes an *information barrier*: reward-only detection of hacking is
fundamentally limited, whereas behavioral information is necessary and often sufficient.
Across a graded suite spanning multiple hacking types, we provide the first head-to-head
comparison of detection methods, find a difficulty hierarchy, and expose an adversarial
frontier that no current method solves. Reward hacking, we conclude, cannot be detected
from reward alone — it must be detected from behavior.

## The Future Work Paragraph

Our results open several directions. The information barrier invites a tighter theory:
sharpening the reward-only lower bound, characterizing exactly when behavioral
information becomes *sufficient*, and situating onset detection within the theory of
change detection with endogenous, non-stationary processes. The construct-validity
certificate should be extended from the single-agent, scalar-reward setting to the
places reward hacking matters most — RLHF at frontier scale, tool-using language agents
whose "onset" unfolds over an interaction rather than a training run, multi-agent
systems where exploitation is emergent, continuous-control robotics, and partially
observable deployments — each of which stresses a different assumption behind our
formalization. The adversarial frontier, where hacking is engineered to be behaviorally
as well as rewardfully deceptive, is the natural arena for the next generation of
structural detectors, and closing it (or proving it cannot be closed) is the field's
open problem. Finally, grounding the benchmark in real deployment traces, with onset
labels from human judgment or held-out gold reward models rather than an oracle, would
carry the information-access result from controlled study to operational practice — and
turn "monitor behavior, not reward" from a finding into a standard.
