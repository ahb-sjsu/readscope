# The five principles

> **Observation Theory v0.2 (2026-08-15).** v0.1 was frozen at commit
> `a490e4e`, put through the seven-test Crucible campaign, and **did not
> graduate** (three of the five core predictions survived; the campaign
> verdict, scorecard, and rules are in
> [geometric-observation/crucible](https://github.com/ahb-sjsu/geometric-observation/tree/master/crucible),
> `OT-CAMPAIGN-VERDICT.md`). Per the freeze rules, revision happened
> only after closure: **P1–P3 are unchanged** (their owed predictions
> were tested and survived; results are now instances), **P4 and P5 are
> revised** exactly as the refuting measurements demand (full
> before/after diff in `OT-V0.2-REVISION.md`), and each principle owes
> one **new** untested prediction. Still five principles — the
> no-sixth-principle rule held and holds.

The object throughout: a consumer `C` reading a representation `x`, with
read operator

    P_C = E_D[ Jᵀ G J ],   J = ∂C/∂x,   G the consumer's output metric,
                            D the distribution x is probed under.

---

## P1 — Consumer relativity *(unchanged; prediction survived)*

**The geometry that matters on a representation is induced by what reads
it, not by the representation itself.**

*Forbids:* any consumer-independent notion of "important directions,"
and any expectation that reconstruction error tracks downstream damage
in general.

*Instances:* the codec experiment (16/16 vs 2/16); the closed-form read
subspaces (queries; decay-attenuated readouts) at resolution 1.000;
**OT-1 (prospective):** damage ratio `cos²θ` to four decimals with zero
free parameters, codec-preference disagreement switching on at the
derived 45°, and the sign of real head-pair disagreement predicted 8/8
by blind-probed operators with no refit. **OT-6 (prospective,
cross-domain):** the same trace picked the ranking-destroyer among
equal-Euclidean-energy perturbations on 400/400 queries over real book
embeddings, machinery verbatim.

*Owed prediction (new, untested):* **composition.** For a weighted
ensemble of consumers, the ensemble's codec preference is predicted by
the weight-averaged traces `Σᵢ wᵢ tr(P_i Σ_δ)` computed from component
operators alone — no probe of the ensemble. Runnable on multi-head
attention today.

## P2 — Measure dependence *(unchanged; prediction survived)*

**`P_C` is an expectation over a probing distribution, so every reading
is a reading *somewhere*.**

*Forbids:* "the" sensitivity of a consumer, unqualified; scalar
corrections for probe loading.

*Instances:* probe loading as the known error term and the alignment
result; the 0.647 reference-dependence affair; **OT-2 (prospective):**
the first-order law `dP_C/dε|₀ = E_D[h·A]` measured — the entire
reading-error curve predicted from under the unshifted measure (shape
deviation 0.003), a full-magnitude functionally-orthogonal shift
producing machine-zero, and linear convergence in ε. Loading is a
covariance, not a distance; three failed scalar corrections now have
their closing statement.

*Owed prediction (new, untested):* **forward transfer of readings.**
Given the operator probed under a synthetic measure and the measured
alignment functional between that measure and the activation measure,
the law predicts the real-activation operator without probing under
it — graded on real heads against direct activation-measure probes.

## P3 — Observation complexity *(unchanged; prediction survived, and became a theorem)*

**Identifying `P_C` blind costs the ambient dimension, not the
effective rank; rank prices description and allocation after
identification.** Blindness is expensive; structure only pays once
found — in every basis (rank is GL-invariant; the spectrum is not).

*Forbids:* cheap discovery of "just the top direction"; graceful
degradation below `k = d`; buying the cliff down by reparameterization.

*Instances:* the budget cliff (predicted, then measured,
rank-independent); **OT-3 (prospective):** the confinement lower-bound
*theorem* (adaptive to d−2, oblivious to d−1; adaptive d−1 recorded
open) and its side-information consequence — a known k₀-dimensional
exclusion moves the cliff to exactly d−k₀ and never softens it —
measured in six cells at the predicted location, with sub-cliff "lucky
recoveries" matching the Haar chance-alignment rate analytically.

*Owed prediction (new, untested):* **the noisy cliff.** With
observation noise of scale σ on each scalar reading, identification at
`k ≥ d` floors at an error derivable from σ and the spectrum gap, while
the cliff's location does not move — noise prices accuracy, never
admission. Theorem extension first, then measurement.

## P4 — Temporal nonstationarity *(REVISED after OT-4's refutation)*

**`P_C` is a process, not a constant, and staleness has a measured
price — but drift explains a system's degradation only where the
staleness channel dominates, and demonstrating that dominance is part
of the claim.** A feedback-severing control (teacher forcing or its
domain equivalent) is not an optional robustness check; it is the
discriminator the principle must *predict it will pass* before drift
may be named as mechanism. Where feedback compounds a constant error,
drift is not entitled to the wreckage.

*Forbids:* compress-once-as-correctness (unchanged); drift-based
explanations of production degradations without a feedback-severed
control (new — this is what OT-4 punished); gradings of drift against
1.0 instead of a paired null (unchanged).

*Instances:* operator drift along the sequence, null-corrected, with
the 225% stale-allocation penalty (C-11c — stands as measurement);
**OT-4 (prospective, REFUTING the old owed prediction):** the real
long-generation collapse (+13.4 ROUGE under the symmetric codebook)
survives A0, then teacher forcing removes the consistent growth (sign
p = 0.42) and orientation does no work against the rotated null — the
mechanism is feedback compounding of a ~4.3-nat constant error, and
C-11c's claim to explain it is dead by the declaration's own sentence.

*Owed prediction (new, untested):* **a feedback-free staleness
system.** Streaming retrieval has no autoregression: an embedding
index quantized against the day-0 query operator, served under a
drifting query stream, has damage that grows with measured
`d(P_C(t₀), P_C(t))` and is removed by re-allocation at a cadence
derived from the drift rate — and the feedback-severing control passes
trivially because the channel does not exist. Runnable on the OT-6
substrate with query drift by book/language strata.

## P5 — Metric consequence *(REVISED after OT-5's refutation)*

**`δxᵀP_Cδx` predicts consumer damage at full fidelity wherever the
consumer emits any differential signal, and at the response floor it
fails closed — silence, never confident error.** The boundary is a
floor, not a slope: OT-5 measured ceiling accuracy down to a
differential fraction of 0.094 and then zero informative comparisons,
with not one wrong-while-differential cell. The applicability
measurement locates the floor; it does not meter a gradual decay.

*Forbids:* gating codecs on reconstruction error where a differential
consumer is known (unchanged); applying the quadratic form to
selection consumers (unchanged); and now also *discounting* the metric
in low-but-nonzero differential regimes — partial smoothness is full
writ, per measurement.

*Instances:* 16/16 vs 2/16 (the founding one); the regime-gate refusal
behavior; **OT-5 (prospective, REFUTING the old monotone shape):** the
step — accuracy 0.93–1.00 from DF 1.0 down to 0.094, then 0/30
informative pairs; **OT-6:** ceiling prediction of a *discrete*
ranking metric (top-10 overlap), consistent with writ-until-the-floor.

*Owed prediction (new, untested):* **the two curves at the floor.** In
a consumer family approaching its floor gradually (output quantized to
g levels, g decreasing), the *informative fraction* of codec
comparisons decays with g while accuracy *on informative pairs* stays
at ceiling — both curves predicted, jointly falsifiable: accuracy
sagging while informative pairs remain would kill the revision.

---

## Status

v0.1's campaign: OT-1 ✅ OT-2 ✅ OT-3 ✅ (theorem) OT-5 ❌ OT-6 ✅
(cross-domain, 400/400) OT-7 ✅ (invariance) OT-4 ❌ — graduation
denied under the pre-registered rule, and the two refutations are the
sources of the two revisions above. v0.2 carries five new owed
predictions; they are the theory's live exposure, and none has been
run. The research question is unchanged:

> **What can be known about an internal representation by observing how
> another system consumes it; what geometry does that observation
> induce; and what are the fundamental limits of measuring that
> geometry?**
