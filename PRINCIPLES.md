# The five principles

> **Observation Theory v0.2 (2026-08-15; verdicts updated
> 2026-08-16).** v0.1 was frozen at commit `a490e4e`, put through the
> seven-test Crucible campaign, and **did not graduate** (three of the
> five core predictions survived; the campaign verdict, scorecard, and
> rules are in
> [geometric-observation/crucible](https://github.com/ahb-sjsu/geometric-observation/tree/master/crucible),
> `OT-CAMPAIGN-VERDICT.md`). Per the freeze rules, revision happened
> only after closure: **P1–P3 are unchanged** (their owed predictions
> were tested and survived; results are now instances), **P4 and P5 are
> revised** exactly as the refuting measurements demand (full
> before/after diff in `OT-V0.2-REVISION.md`), and each principle owes
> one **new** untested prediction. Still five principles — the
> no-sixth-principle rule held and holds.
>
> **Two further campaigns have since consumed three of the five owed
> predictions** (`OT-CRUCIBLE-2-VERDICT.md`, `OT-CRUCIBLE-3-VERDICT.md`):
> P1's composition (OT-8 ✅), P4's feedback-free staleness (OT-14 ✅),
> and P2's forward transfer (OT-15 ✅, via the corrected estimator after
> OT-9's death). P5's floor curves remain **untested in substance after
> two instrument/authoring deaths** (OT-12, OT-13 — the latter leaving a
> perfect but ungraded exploratory record). P3's noisy cliff was
> discharged 2026-08-17: theorem (OT-3N), then measurement (OT-16,
> PASS on a fresh seed). Consumed predictions are recorded as instances below; **new
> owed predictions are not minted here** — that is a revision act with
> its own seal, and the program's precedent says not to rush it.

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

*Owed prediction — CONSUMED, survived (OT-8, Second Crucible):*
**composition.** The ensemble's codec preference predicted by
weight-averaged component traces `Σᵢ wᵢ tr(P_i Σ_δ)` with no probe of
the ensemble: **29/30 cells**, with live cross-terms costing exactly
the one predicted cell. Now an instance; P1 currently carries no open
owed prediction.

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

*Owed prediction — CONSUMED, survived in corrected form (OT-9 †,
OT-15 ✅, Third Crucible):* **forward transfer of readings.** The
prediction as first worded — importance-reweighting via an alignment
functional — died as an estimator (OT-9: ESS ≈ 1.0 in 128-d; the
"correction" was one point, not the law). The substance survived via
the estimator that design should have been: **direct moment-matched
probing** — draw operating points from N(μ̂, Σ̂) fitted to ≤ 64 real
key samples and probe there, no weights anywhere. On the 12 real
Llama-3.2-3B heads: beats iso-Gaussian probing **11/12** at median
error ratio **8.27×** (iso is catastrophic on real heads, median
relative error 6.4). The recorded residual is the fitted family's
inadequacy — one head where even full-sample Gaussian moments cannot
beat 1.0 — which is the honest content of any transfer claim. Now an
instance; P2 currently carries no open owed prediction.

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

*Owed prediction — CONSUMED, survived (OT-3N theorem + OT-16
measurement, 2026-08-17):* **the noisy cliff.** Proved first
(`OT3-NOISY-THEOREM.md`): oblivious lower bounds survive every σ by
total variation zero — noise cannot un-confine a design — and the
k = d floor is Θ(σ√d/γ) by Davis–Kahan with a matching two-point KL
lower bound; the adaptive quantifier-order cell recorded open,
matching OT-3's own. Then measured on a fresh seed (OT-16, PASS, no
revisions spent): decade linearity 0.84–1.18 with √d/γ slope collapse
1.33 across six (d, γ) cells, the confined face pinned at 1/√2, the
step resolved at 22–31×, and the side-information shift to d − k₀
holding in all four cells. **Noise prices accuracy, never
admission** — theorem, then measurement, in that order. Now an
instance; P3 currently carries no open owed prediction.

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

*Owed prediction — CONSUMED, survived (OT-14 v2, Third Crucible):*
**a feedback-free staleness system.** On the OT-6 substrate with a
qualified mixture-drift dial (Spearman 1.000, range 17×): staleness
excess tracks measured drift at **ρ = 0.857** across seven strata,
refresh at the stratum cadence removes **77%** of stale damage at
full drift (lever 15× eval noise, exact pool-blend operators), and
the severing control was declared trivial-by-construction *in
advance*, exactly as the dominance obligation demands. One edge is
recorded with the pass: refresh slightly *hurts* below drift ≈ 0.1 —
the derived cadence has a floor, and re-allocating beneath it costs.
Now an instance; P4 currently carries no open owed prediction.

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

*Owed prediction (still untested — three times, by instrument):* **the
two curves at the floor.** In a consumer family approaching its floor
gradually, the *informative fraction* of codec comparisons decays
while accuracy *on informative pairs* stays at ceiling — both curves
predicted, jointly falsifiable: accuracy sagging while informative
pairs remain would kill the revision. Three campaigns have failed to
grade it: OT-12 (family without interior, VOID); OT-13 (v1: probe
structurally silent; v2: accuracy 1.000 at every graded step, VOID on
a seed-fragile per-step check); and OT-17 (Fourth Crucible, 2026-08-17:
the band-level check fixed, and the family's own interior proved
seed-fragile — VOID with no escape hatch, campaign closed unresolved).
Three perfect ungraded accuracy curves now sit in the record with zero
verdict weight. The prediction remains the theory's sole live
exposure; the deepened lesson — **family interior must be demonstrated
across seeds, not at one** — is recorded in `OT-CRUCIBLE-4.md`, and
the next attempt requires a multi-seed-qualified family in a later
session.

---

## Status

v0.1's campaign: OT-1 ✅ OT-2 ✅ OT-3 ✅ (theorem) OT-5 ❌ OT-6 ✅
(cross-domain, 400/400) OT-7 ✅ (invariance) OT-4 ❌ — graduation
denied under the pre-registered rule, and the two refutations are the
sources of the two revisions above.

The Second Crucible (OT-8..OT-12) closed 1/5 with four instrument
deaths and zero clean refutations; its durable product is the
rate-limit rule (no appendix sealed the day its family is first
constructed). The Third Crucible (OT-13..OT-15), run on
pre-qualified families under that rule, closed 2/3: **P4's and P2's
owed predictions are earned** (OT-14, OT-15), P1's was earned earlier
(OT-8), and OT-13 closed unresolved. The theory's live exposure is
now exactly **one** owed prediction: **P5's floor curves** (never
graded — three campaigns, all instrument/family deaths). P3's
noisy cliff was discharged 2026-08-17 (OT-3N + OT-16). No v1.0
declaration is available while P5 stands open, and new owed
predictions for P1/P2/P3/P4 await a sealed revision act. The research question is
unchanged:

> **What can be known about an internal representation by observing how
> another system consumes it; what geometry does that observation
> induce; and what are the fundamental limits of measuring that
> geometry?**
