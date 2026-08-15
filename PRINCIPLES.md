# The five principles

> **FROZEN as Observation Theory v0.1 (2026-08-15).** The five principle
> statements below are sealed at commit `a490e4e` — no edits to them, and
> no sixth principle, until the Crucible campaign resolves. The campaign
> (seven preregistered tests, graduation rule, kill conditions) lives in
> the program repo:
> [geometric-observation/crucible](https://github.com/ahb-sjsu/geometric-observation/tree/master/crucible).
> This banner is the only permitted kind of post-freeze edit.

*An external reviewer (2026-08-15) observed that the results in this
repository read as excellent-but-separate sensitivity findings, and would
read as a theory exactly when a small set of general principles makes them
instances. This document adopts that frame, with one correction the
measurements force (P3), and with the honesty flag the frame requires: the
principles below were articulated **after** most of the results existed.
A principle earns the word "theory" by what it forbids and by the untested
predictions it makes — so each one here carries both. When one of the
predictions fails, the principle changes or dies; this file is where that
gets recorded.*

The object throughout: a consumer `C` reading a representation `x`, with
read operator

    P_C = E_D[ Jᵀ G J ],   J = ∂C/∂x,   G the consumer's output metric,
                            D the distribution x is probed under.

---

## P1 — Consumer relativity

**The geometry that matters on a representation is induced by what reads
it, not by the representation itself.** A substrate has no privileged
metric; every consumer equips it with one (`P_C`), and different consumers
equip it with different ones.

*Forbids:* any consumer-independent notion of "important directions," and
any expectation that reconstruction error (the substrate's own metric)
tracks downstream damage in general.

*Instances:* the codec experiment — two codecs tied on reconstruction at
7.5e-9 differing downstream by 1.85 median relative-KL (postdiction: the
result motivated the principle). The closed-form ground truths — an
attention head's read subspace is spanned by its queries; a recurrent
state's by its decay-attenuated readouts (derived, then measured at
resolution 1.000 across 124 cells). The `regimes` refusal: selection
consumers induce a geometry finite differences cannot see, so the
instrument declines rather than reporting a confident zero.

*Owed prediction (untested):* two consumers of the same representation
disagree about a codec in proportion to the principal angles between
their read subspaces. Measurable today with two heads reading one KV
cache; nobody has run it.

## P2 — Measure dependence

**`P_C` is an expectation over a probing distribution, so every reading
is a reading *somewhere*.** Change where you probe and you change what
you measure — not as noise but as a different, equally well-defined
operator.

*Forbids:* "the" sensitivity of a consumer, unqualified; and scalar
corrections for probe loading (a correction would have to be a function
of the probing shift alone, and degradation measurably is not).

*Instances:* probe loading as the known error term — three failed
attempts at a correction, resolved by the alignment result: loading
degrades a reading only when the probing shift aligns with the read
subspace, which no function of loading alone can capture (postdiction,
and the sharpest evidence for P2). The 0.647 affair: a published recovery
number turned out to measure the distance between two *references*
(softmax-weighted vs unweighted query covariance, ~0.3 apart), not probe
error — reference choice is measure choice (postdiction).

*Owed prediction (untested):* loading damage is predictable from the
measured probing-shift/read-subspace angle, quantitatively, across
families — the alignment result run forward instead of forensically.

## P3 — Observation complexity (stated as the cliff measured it, not as first intuition says it)

**Identifying `P_C` blind costs the ambient dimension, not the effective
rank.** First intuition — and the reviewer's draft of this principle —
says a low-rank operator should be cheap to find (`rank_eff ⇒ cost`). The
budget-cliff measurement says the opposite, and the theory adopts the
measurement: below `k = d` probe directions the estimate is a projection
onto a random subspace, and a projected operator's leading eigenvector is
not the operator's. **Rank sets the cost of *describing and allocating
against* `P_C` once found; ambient dimension sets the cost of *finding*
it.** Blindness is expensive; structure only pays after identification.

*Forbids:* cheap discovery of "just the top direction"; any budget story
in which recovery degrades gracefully below `k = d`.

*Instances:* the cliff itself — 16 ranks, one cliff at `k = d`,
rank-independent (this one ran as a *prediction* of the projection
argument and confirmed it). The vector-consumer discount: `m` outputs per
probe direction reach output-width resolution at half budget — the cost
is per-scalar-observation, which is the same principle counted correctly.

*Owed prediction (untested):* side information changes the constant, not
the cliff — a probe seeded with an approximate subspace (e.g., yesterday's
`P_C`) should identify at `k ≈ d − k₀` but still fail catastrophically
below it, not degrade smoothly.

## P4 — Temporal nonstationarity

**`P_C` is a process, not a constant: `P_C(t₁) ≠ P_C(t₂)` in general,
and the difference is operationally priced.** A reading is stamped with
its moment; acting on a stale operator has a computable cost.

*Forbids:* compress-once-against-one-operator as a correctness argument;
gradings of drift against 1.0 rather than against a paired null (two
samples of the *same* moment do not give identical operators either —
C-11b overstated drift by >2× exactly this way).

*Instances:* operator drift along the sequence — the dominant read
direction moves even where the null is nearly perfect; allocating against
the early operator costs the late consumer 225% (of a uniform split's
cost) more than allocating against the late one (postdiction, null-
corrected). The turboquant-pro long-generation degradation is the named
candidate consequence.

*Owed prediction (untested, already flagged in the README):* the drift
curve, run against a *real* degradation curve — if drift is the
mechanism, degradation onset should track the measured 225% allocation
penalty, and a drift-aware refresh cadence should flatten it.

## P5 — Metric consequence

**`δxᵀ P_C δx` predicts consumer damage better than any substrate metric,
exactly where the consumer is differential.** The applicability clause is
part of the principle, and it is enforced in code (`regimes`): where a
consumer's differential fraction is low — selection, argmax, top-k — the
quadratic form is undefined as a damage model and the claim is *not made*.

*Forbids:* gating codecs on reconstruction error where a differential
consumer is known; and, symmetrically, applying `tr(P_C Σ_δ)` to
selection consumers as if it meant something.

*Instances:* 16/16 vs 2/16 on exactly-tied codecs (postdiction — the
founding one). The refusal behavior as the boundary of the claim,
inherited from the turboquant-pro consumer-regime analysis.

*Owed prediction (untested):* the quality of `tr(P_C Σ_δ)` as a damage
ranking degrades monotonically with a consumer's measured differential
fraction — a dose-response curve across the regime boundary, which would
turn the applicability clause from a fence into a measurement.

---

## What this document is, status-wise

Four of five principles are currently organized *around* existing results
(postdictive); one (P3) ran as a genuine prediction and survived, and one
reviewer-proposed form of it was corrected by the data before adoption.
The five owed predictions above are the theory's live exposure. The
research question they jointly serve, stated once:

> **What can be known about an internal representation by observing how
> another system consumes it; what geometry does that observation induce;
> and what are the fundamental limits of measuring that geometry?**

KV-cache compression is the first paying customer of that question, not
its content.
