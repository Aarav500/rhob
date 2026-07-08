# Feasibility Audit: Families 3, 4, 6, 7, 9, 10

Applies the screening rule from `families_5_8_redesign.md` to the six remaining
proposed families **before any implementation work**, exactly as that rule is meant
to be used. This is desk analysis (no pilots run); verdicts below are "should this be
attempted, and how" -- not admission certificates.

## The screening rule, restated

A pair separates at L2 **iff its discriminating feature is anti-symmetric under a
symmetry `sigma` that also matches the proxy** (`FINDINGS.md`). Three questions,
asked in order, catch almost every failure mode we've hit so far:

1. **Is there an explicit `sigma`** relating the hacking and legitimate variants (a
   reflection, transpose, or discrete swap), or does the proposal just assert "these
   two behaviors should look similar"? No `sigma` -> proxy-matching is not guaranteed
   by construction and will need the kind of blind parameter search that already
   failed for the original Pair 02 (3+ configurations tried, all failed).
2. **Is the discriminating feature a magnitude** (a count, a rate, "how much,"
   "whether X converges/decreases")? Magnitudes are symmetric under `sigma` and
   vanish exactly when the proxy is matched -- this was the exact defect in the
   original Families 5 and 8.
3. **Is the total proxy actually equal in expectation** between variants, or does the
   narrative just assume it ("similar profile," "roughly the same total")? If hacking
   and legitimate spend different amounts of time in the reward-bearing regime (e.g.
   one loops indefinitely, the other reaches a terminal state), the proxy totals
   differ by construction and no amount of tuning fixes it.

| Family | (1) explicit `sigma`? | (2) feature type | (3) proxy equal by construction? | Verdict |
|---|---|---|---|---|
| 3 Reward Shaping | no | magnitude ("progress toward goal") | no (unbounded looping vs. bounded reach) | **inadmissible as specified** |
| 4 Shortcut | no | discrete/spatial (room visited) | asserted, not constructed | **needs redesign** (promising feature type) |
| 6 Proxy Correlation | yes (color swap) | directional (blue/red balance) | yes, by construction | **admissible as specified** |
| 7 Goal Misgeneralization | n/a -- wrong paradigm | n/a | n/a | **out of scope for v3** (not a matched-proxy pair at all) |
| 9 Multi-Agent | no | directional but unbalanced | no (blocking's net effect on the sum is asserted, not constructed) | **inadmissible as specified**, plus infra gap |
| 10 Sequential/RLHF | no | magnitude ("edit distance decreasing") | no | **inadmissible as specified** (same defect as original Family 8), plus infra gap |

Two families pass or nearly pass; two fail the same way Families 5/8 originally
failed; one is a structurally different problem, not a family; one has both a
construction defect and a missing subsystem. Detail below.

---

## Family 6, Proxy Correlation Gaming -- ADMISSIBLE AS SPECIFIED

This is the one family that already satisfies the rule as written, because it is,
underneath the "blue/red items" narrative, the same construction as `FINDINGS.md`
Result 3 (the validated two-tile positive control): two reward sources of **equal
proxy value**, one aligned with the true objective, one not.

- **`sigma`**: swap the labels blue <-> red. The proxy ("total items, weighted
  equally") is invariant under this swap by construction -- an agent farming red
  and an agent farming blue earn identical proxy in distribution, with no tuning
  required. This is the strongest kind of `sigma`: it holds by definition, not by
  approximate balancing.
- **Feature**: blue-fraction minus one-half (or blue-count minus red-count). This is
  *directional* (signed around a neutral point), not a magnitude -- passes rule (2).
- **Proxy equality**: automatic from equal-weighting, not asserted -- passes rule (3).
- **Difficulty knob**: spatial correlation between blue/red placement is a
  plausible knob (high correlation -> the two item types are hard to tell apart
  spatially -> harder for the agent's location alone to reveal which it's farming ->
  the L2 feature itself, being about *inventory contents* rather than location,
  should be fairly robust to this knob; the knob more directly controls how easy the
  *task* is, which is a different axis from L2 difficulty). This needs a pilot to
  confirm the knob actually moves L2 smoothly rather than being binary like the
  original gridworld.

**Recommendation**: build this one first among the six. It requires the least new
machinery (extends the two-tile/camping pattern to two *item types* instead of two
*locations*) and has the cleanest `sigma`.

---

## Family 4, Shortcut Exploitation -- NEEDS REDESIGN (promising)

**Why it fails as specified.** "Multi-room navigation, agent A takes a glitch
shortcut, agent B takes a genuinely efficient path with a *similar* proxy-distance
profile" has no `sigma` -- "similar profile" is asserted, not constructed. This is
exactly the shape of Pair 02's original three failed attempts: without an explicit
symmetry forcing the proxy processes to match, tuning a residual parameter (here,
"how efficient is B's path") produces a continuum of TV/L2 values that don't
decouple, not a clean matched point.

**The fix.** The discriminating feature ("which rooms visited") is spatial/discrete
-- the same feature *type* as Pair 01's tile occupancy, which we know works. Rebuild
the environment as an explicit mirror pair rather than a single asymmetric layout:
two symmetric detour rooms `D_A`, `D_B` placed as reflections about the direct A->C
path, each shaving the *same* Euclidean distance off the route. Variant A's shortcut
routes through `D_A` (an out-of-bounds glitch region with no real task content);
variant B's route through `D_B` **is** the intended path (e.g., a real room with a
sub-objective). Both variants shave identical distance, so the proxy
(distance-reduction) is matched by the reflection, not by hand-tuning. The
discriminating feature is then "which detour room was entered" -- anti-symmetric
under the reflection, structurally identical to Pair 01's `tile2_hacking` vs
`tile2_legit`.

**Honest caveat.** Once corrected this way, Family 4 is mechanistically very close to
gridworld camping (occupy one of two symmetric locations); its claim to being a
distinct "mechanism" rests on the *multi-room/sequential* structure (must pass
through rooms in order) rather than on a new discriminating-feature type. That is a
legitimate difference (it stresses sequential-dependency detection, e.g. can a
detector tell "skipped room 2" from "visited room 2 briefly"?) but should be
described honestly as a structural variant of camping, not an unrelated mechanism.

---

## Family 3, Reward Shaping Exploitation -- INADMISSIBLE AS SPECIFIED

**Why it fails.** This is a rerun of the *original* Family 5 failure
(`FINDINGS.md` Result 1), relocated from "exploration bonus" to "waypoint shaping
bonus." The legitimate agent's efficient route **also** collects the shaping
bonuses en route to the goal -- so "loop at waypoints" (hacking) and "pass through
waypoints efficiently" (legitimate) are graded shades of the *same* behavior, not two
different behaviors, exactly the trap Result 1 diagnosed: *"the foil relabels the
same behavior; it does not produce a different one."* Worse, the proxy totals are not
naturally equal: hacking loops indefinitely and keeps earning shaping reward for the
whole episode, while legitimate reaches the goal and (presumably) the episode ends or
shaping stops -- rule (3) fails outright, before feature type is even relevant. The
proposed feature ("progress toward terminal goal") is also a magnitude, failing rule
(2) as well. Two independent disqualifiers.

**The fix, if pursued.** Apply the same move as the Family 5 -> Novelty-Farming
redesign: replace "efficient vs. inefficient traversal of the same waypoints" with
two **mirror-image** waypoint circuits (reflection about the direct path), where
hacking commits to indefinitely farming circuit `A` (never reaching the terminal
room) and legitimate commits to circuit `B`, which **is** the route to the terminal
room. Note that this collapses Family 3 into essentially the same construction as
the corrected Family 5 (a decaying/roaming reward over a mirror-image region pair)
-- it does not appear to add a mechanistically distinct family. **Recommendation**:
do not build Family 3 separately; if waypoint-shaping-specific dynamics are wanted,
fold them into the Novelty-Farming family as a parameter (e.g. bonus decay rate)
rather than shipping a second near-duplicate family.

---

## Family 7, Goal Misgeneralization -- OUT OF SCOPE FOR v3 (wrong paradigm)

This is not a rule-2/rule-3 failure; it fails at rule (1) in a deeper way: **there is
no matched pair to construct.** The proposal trains *one* policy under a
goal-landmark correlation and evaluates it after the correlation breaks at deployment
-- a train/test distribution-shift experiment, not two co-trained variants with a
proxy-preserving symmetry. Consequences:

- There is no "Variant A vs Variant B" trained in parallel; "onset" would mean *the
  moment of deployment*, not an emergent mid-training behavioral switch, so the
  entire onset-labeling and admission-gate machinery (built around a training-time
  switch detected from a behavioral trace) does not apply.
- The proxy is explicitly "same as true during training" per the spec -- i.e. there
  is no proxy/true divergence to hide during the phase a detector would observe. The
  divergence appears only after deployment, by which point the policy is fixed.
- This maps onto a *different* evaluation question ("does a policy generalize its
  proxy-true alignment out of distribution?") that would need its own admission
  criteria (a generalization-gap gate, not a matched-proxy gate).

**Recommendation**: do not attempt as a v3 family. If goal misgeneralization is
wanted in the benchmark, it needs a separate methodology track (plausibly a v4
direction, alongside the RLHF extension already in the paper's Future Work), not a
`BaseFamily` subclass slotted into the existing registry.

---

## Family 9, Multi-Agent Reward Hacking -- INADMISSIBLE AS SPECIFIED + infrastructure gap

**Construction problem.** "Total proxy goes up because A's gain exceeds B's loss in
the accounting" is exactly a rule-(3) violation: it *asserts* the hacking and
cooperative totals end up close, without a `sigma` that forces it. Whether blocking
nets out to the same joint proxy as efficient co-travel depends on the specific
environment geometry and would need the same kind of delicate, likely-fragile
parameter search that failed repeatedly for the original Pair 02. No reflection or
swap symmetry is proposed to force the equality structurally.

**Infrastructure problem, independent of the construction issue.** `BaseFamily`,
`MatchedPair`, and `RunData` are all built around a single agent's rollout
(`rollout_hacking(seed) -> (RunData, onset)`); there is no joint/multi-agent rollout
type, no per-agent access-level splitting, and `agents/multi_agent.py` does not
exist. Building this family would mean extending the core v3 data model, not just
authoring a new `BaseFamily` subclass -- a materially larger scope than any other
family on this list.

**Recommendation**: defer. If pursued later, first design the `sigma` on paper (a
literal mirror-image two-agent layout, analogous to Family 4's corrected design,
where blocking-in-region-`X` is the reflection of cooperating-in-region-`Y`) *and*
scope the multi-agent data-model extension as its own infrastructure task, before
writing any family code.

---

## Family 10, Sequential/RLHF-Proxy -- INADMISSIBLE AS SPECIFIED + infrastructure gap

**Construction problem.** "Edit distance to target: decreasing (legitimate) vs.
stable/increasing despite high proxy (hacking)" is a magnitude/convergence feature --
the *exact* defect that broke the original Family 8 ("whether velocity decreases"),
relocated to sequence space. There is also only one target pattern, so there is no
`sigma` relating a hacking sequence-distribution to a legitimate one; rules (1) and
(2) both fail.

**The fix, sketched.** Following the Orbit Chirality pattern: introduce **two**
target patterns, `T_A` and `T_B`, related by some sequence-level symmetry (e.g. a
fixed token permutation or reversal) under which the reward model's *score* is
invariant (its blind spot is symmetric across the two targets) while true
quality (closeness to the *actual* held-out target) is not. The discriminating
feature would be a signed quantity such as `edit_distance(T_A) - edit_distance(T_B)`
(which target the sequence is drifting toward), not raw convergence. This is a
real design sketch, not a validated one -- whether a small reward model actually
develops a *symmetric* blind spot across two token-level-symmetric targets is an
open empirical question that would need a pilot before any of the rest is built.

**Infrastructure problem.** The proxy itself is a reward model that must be trained
(on limited preference data) before the policy is trained against it -- a nested
two-level training loop with its own instability risks, unlike every other family
where the proxy is a fixed, hand-specified function. This is materially more
implementation surface than any family built so far.

**Recommendation**: lowest priority of the six. Matches the original spec's own
framing ("Families 8-10 are harder, most novel") -- correct here, probably an
understatement given the added RM-training complexity.

---

## Summary and recommended order

| Rank | Family | Status | Effort to next milestone |
|---|---|---|---|
| 1 | 6 Proxy Correlation | admissible as specified | pilot only (numpy, cheap) |
| 2 | 4 Shortcut | needs redesign (sketched above) | redesign is done here; then pilot |
| 3 | 3 Reward Shaping | fold into Novelty-Farming, don't build separately | none (scope cut) |
| 4 | 7 Goal Misgeneralization | out of scope for v3 | defer to a separate track |
| 5 | 9 Multi-Agent | needs redesign + new data model | design `sigma` + scope infra, then pilot |
| 6 | 10 Sequential/RLHF | needs redesign + nested-training infra | design + pilot the RM-symmetry question first |

Combined with the two already-redesigned families (5 -> Novelty-Farming, 8 -> Orbit
Chirality, both pilot-validated for mechanism plausibility) and the two shipped
families (gridworld camping, continuous camping), a realistic v3 mechanism count is
**five** genuinely distinct constructions (camping, novelty-farming, orbit-chirality,
shortcut/room-skip, proxy-correlation) rather than eight -- Family 3 collapses into
novelty-farming, and Families 7/9/10 need work substantially beyond a family
generator (a different paradigm, a new data model, and a nested training loop,
respectively) before they're tractable.
