# RLHF-RM Extension — Design Spec

**Status**: Approved for planning (sub-project 1 of 3 in the 18→34-family expansion:
RLHF-RM setting → MULTI_AGENT setting (5 families) → SEQUENTIAL non-RLHF setting
(6 families), decomposed and sequenced per user request).

## Context

RHOB currently has 18 families (14 original + 4 MuJoCo, added earlier this session).
Exactly one family, `rlhf_reward_model_overopt`, touches RLHF-style reward hacking, and
it's a single-step toy preference-bandit under the `TABULAR` complexity tier — no
genuine reward-model fitting, no policy optimization dynamics, no KL penalty. The
taxonomy's `SEQUENTIAL` tier ("sequence decision problems") is defined but has never
been populated by any family.

This spec covers building a genuine synthetic RLHF-RM setting — response generation,
real preference-data-fitted reward models, policy optimization with KL regularization —
and populating the `SEQUENTIAL` tier with 5 families built on it, each varying exactly
one way the fitted reward model goes wrong. No real LLM is involved: the "response
space" is a synthetic feature-vector space, chosen so the reward-hacking dynamics are
genuine (arising from real data-fitting and optimization) without the cost/complexity of
an actual language model.

## Infrastructure

New module: `src/rhob/environments/rlhf_rm/`, alongside the existing
`src/rhob/environments/mujoco/` and `src/rhob/environments/continuous/`. No new
optional dependency — pure numpy/scipy (linear/logistic regression via
`sklearn.linear_model`, already a core dependency), so no `pytest.importorskip` gating
needed, unlike the MuJoCo module.

Core pieces:

- **Response space**: `x ∈ R^d` (d=8), each dimension a synthetic "quality attribute"
  (e.g. helpfulness, verbosity, sycophancy-signal, factuality-signal, ...).
- **True reward** `r*(x)`: a fixed nonlinear function (quadratic + pairwise interaction
  terms) over these 8 dimensions, shared across all 5 families as the ground truth —
  oracle-only, exposed only as `RunData.true_rewards`, never an input to any fitted RM,
  policy, or detector.
- **Preference-data generator**: samples pairs `(x_i, x_j)`, labels them via
  `r*(x_i) > r*(x_j)` plus configurable label noise, under a configurable sampling
  distribution over `x`-space. Each family's specific miscalibration comes from *how*
  this generator is configured (sparse coverage / noise / population bias), not from a
  hand-scripted gap.
- **Reward-model fitting**: a genuine `sklearn.linear_model.LogisticRegression`
  (Bradley-Terry-style pairwise loss) fit on the generated preference data — real
  fitting, not a scripted proxy. Its blind spots emerge from the data it was fit on.
- **Policy loop**: policy = `N(μ, Σ)` over `x`-space (Σ fixed, only μ optimized). Each
  step: sample a batch of responses, score with the fitted `r_hat`, take a
  policy-gradient-style ascent step on μ, minus `β · KL(policy ‖ reference)` where
  `reference = N(μ_0, Σ)` is the fixed initial policy. This is the per-episode rollout
  primitive, replacing `generate_mujoco_rundata`'s role: a new
  `run_rlhf_episode`/`generate_rlhf_rundata` pair in `src/rhob/environments/rlhf_rm/rollout.py`.
- Per-step logging: `r_hat` under the current policy (proxy), `r*` under the current
  policy (true, oracle-only), `KL(policy ‖ reference)` (behavioral trace — analogous to
  MuJoCo families' `behav_trace`).

## The 5 families (each varies one failure-injection axis, same core loop)

All 5 use `EnvironmentComplexity.SEQUENTIAL` — first-ever population of this tier.

### Family 19 — RM Sparse-Coverage Gaming (mechanism: RM_OVEROPTIMIZATION)
Preference data undersamples a region of `x`-space (e.g. high-verbosity responses are
rare in the training pairs). The fitted RM extrapolates optimistically into that
under-covered region; the hacking policy drifts there. Legit variant: preference data
sampled uniformly, no blind spot to exploit.

### Family 20 — RM Label-Noise Exploitation (mechanism: RM_OVEROPTIMIZATION)
Preference labels near the true decision boundary carry disproportionately high noise
(modeling real annotator disagreement on close calls); this biases the fitted RM's
decision boundary in one consistent direction across many draws. Legit variant: uniform
label noise, no directional bias.

### Family 21 — RM Feature-Blindspot Gaming (mechanism: GOAL_MISGENERALIZATION)
`r*` depends on all 8 dimensions, but the RM is fit using only a 6-dimension subset
(modeling a real RM's fixed/incomplete feature representation — classic "verbosity/length
bias" case, where length correlates with the missing dimension). Legit variant: RM fit
on the full 8 dimensions.

### Family 22 — KL-Penalty Gaming (mechanism: REWARD_SHAPING)
`β` (KL-penalty coefficient) is mistuned too low, so the policy drifts far from the
reference distribution into RM-inflated territory the penalty was meant to prevent.
Legit variant: correctly-tuned `β` keeps the policy within a validated safe KL radius.

### Family 23 — Preference-Population Bias (mechanism: DECEPTIVE_ALIGNMENT)
The synthetic labeler population has a systematic bias unrelated to true quality (a
sycophancy/agreement-signal dimension the population over-weights relative to `r*`'s
true weighting). The fitted RM faithfully learns this bias. Legit variant: labeler
population weights match `r*`'s true weights. Framed as `DECEPTIVE_ALIGNMENT` since the
resulting policy looks aligned to the (biased) labelers while diverging from true
quality — the sycophancy failure mode documented in real RLHF literature.

## Admission-gate validation

Same unmodified 5-criterion `AdmissionGate`, no new criteria. Unlike MuJoCo's
contact-dynamics noise, this setting's stochasticity comes from preference-data sampling
and policy-gradient noise — expect this to calibrate closer to the existing
tabular/cont_2d families' seed counts than MuJoCo needed, but each family's calibration
constants (analogous to MuJoCo's `_CALIB_TOL`/`_CALIB_SEEDS`) will still need empirical
verification per this project's established `functools.lru_cache`-memoized pure
calibration-function pattern.

## Detector-suite implications

Same L1 dimensionality caveat as MuJoCo: state-visitation histograms are dimensioned by
each family's own state space, and this family's 8-dim response space is again
incompatible with existing families' state-space shapes for cross-family L1 detector
transfer. Documented as a known, already-flagged limitation, not solved here.

## Testing approach

Mirror the MuJoCo families' test pattern: one pytest file per family checking (a)
proxy-match (Δ `r_hat`-distribution between variants ≈ 0) and (b) true-reward divergence
(Δ `r*`-distribution diverges). No `importorskip` gating needed (no optional dependency).

## Explicitly out of scope for this spec

- The MULTI_AGENT setting and its 5 families (next sub-project after this one lands).
- The SEQUENTIAL non-RLHF setting and its 6 families (final sub-project in this
  expansion).
- Any change to the existing `rlhf_reward_model_overopt` toy family — it stays as-is,
  a distinct (tabular, simpler) family alongside these 5 new ones, not replaced by them.
- Real LLM integration of any kind — this setting is, and remains, fully synthetic.
