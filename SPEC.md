# readscope specification

**This instrument is not yet fully specified.** What follows is the form of
a specification, with every field named and most of them still empty. The
empty cells are the point of this document. An instrument is trusted because
it is specified, not because it worked once.

Every number here names the record it came from, in
`calibration/records/` for calibrations run in this repository, or the sealed
preregistration in the `geometric-observation` evidence repository for the
accuracy points inherited from that program. A number with no record does not
appear. A CI census check (`tools/check_spec_census.py`) now fails the
build when a record exists that this document never cites, when a section
heading duplicates, or when a known-stale phrase returns; full automatic
regeneration from the records is still not
done, so for now the binding is by citation and not by script.

---

## What the instrument measures

The blind probe recovers a consumer's read operator `P_C = J^T G J` from the
consumer's outputs alone, by finite-differencing around operating points and
accumulating `S = E[g g^T]`. The top eigenvectors of `S` span the read
subspace. No labels enter and no oracle direction enters.

The reading is a **response spectrum**: where a consumer's sensitivity sits
across directions. Bits are then allocated against that spectrum by reverse
water-filling, which is the same optimization as power allocation across
frequency bins.

---

## Specification fields, and what each one means here

| Field | Scope equivalent | Meaning for this instrument | Status |
|---|---|---|---|
| Sample rate | samples/second | Consumer evaluations spent per operating point. `2d` exact, `2k` sketched. | **Measured, exactly known** |
| Minimum usable budget | minimum input signal | The direction budget below which the probe resolves almost nothing. | **Measured twice over: a cliff at `k = d` at matched point counts (C-2e), and the cliff survives equal-total-budget reallocation (C-15) — sub-dimensional budgets do not catch up within 8× the flagship spend** |
| Bandwidth | −3 dB frequency | The rank range over which recovery stays above the noise floor. How many eigendirections can be resolved before the reading is chance. | **Measured for this package's estimators, and it is bad** |
| Noise floor | volts RMS | Chance overlap for the shape being read, `rank / dim`. Reported with every reading. | **Measured, exactly known** |
| Accuracy over range | percent of reading | Recovered-subspace overlap as a function of rank, dimension, probe budget, and loading. | **108 real-model cells with closed-form references (48 attention heads / four families, 12 Mamba channels, 48-cell Qwen scale ladder) plus the three sealed GO runs; one full loading curve on a synthetic consumer** |
| Input impedance | ohms | Probe loading, on a dimensionless axis. | **Axis works. A scalar correction is the wrong shape; effect depends on alignment** |
| Linearity | percent | Whether the recovered magnitude tracks the true magnitude across scale and across domain. | **Partial. Direction transfers, magnitude does not** |
| Temperature drift | ppm/°C | Stability of a reading across architectures and scales at matched geometry. | **Four families, spread 1e-15. Four scales, spread 7e-16** |
| Applicability | probe coupling | Which consumer regimes this probe can be attached to at all. | **Bounded, and enforced in code** |
| Backend equivalence | inter-lab calibration | Whether a reading depends on where it was computed: CPU vs GPU, one machine vs another, one silicon generation vs another. | **Measured (C-13/C-14). Same-seed spectra agree to ≤ 4e-11 across five devices, two architectures, two sites; cross-BLAS operator floor ~1e-8 (a derived tolerance, not an aspiration). Serial probing is enforcement-incompatible on utilization-policed commons GPUs (median 0% util); the batched variant (C-14) is compatible and identical to the serial reading at the arithmetic floor** |

---

## Bandwidth, measured, and the worst news in this document

C-2b, record `calibration/records/c2b-bandwidth.json`, PASS on all six bars.
Planted subspace with a graded spectrum, ambient dimension 64, three seeds,
192 operating points. The statistic is **resolution**,
`(overlap - chance) / (1 - chance)`, which is zero at the noise floor and one
at perfect recovery whatever the shape. Raw overlap cannot be compared across
ranks because the floor moves with them.

Bandwidth is the largest read rank whose entire prefix holds resolution at or
above 0.5.

| Estimator | Calls per point | res @1 | @2 | @4 | @8 | @16 | @32 | Bandwidth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exact | `2d` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **32** |
| sketch, k=64 | `2k` | 0.995 | 0.503 | 0.283 | 0.139 | 0.060 | 0.041 | **2** |
| sketch, k=16 | `2k` | 0.973 | 0.478 | 0.219 | 0.116 | 0.032 | 0.013 | **1** |

**The exact estimator resolves the whole sweep. This package's random-sketch
estimator resolves one or two directions.** `CALIBRATION.md` listed exactly
this outcome in advance as one of the three findings that would mean the
instrument is not worth specifying, on the grounds that it would then report
the dominant direction and nothing more.

The cause is structural. The two-point sketch is unbiased for the gradient
but noisy, and squaring a noisy gradient adds a roughly isotropic term of
order `||g||^2 / k` to the recovered operator. That term does not rotate the
leading eigenvector, which is why rank one survives, and it buries every
weaker planted direction, which is why nothing else does.

At fixed rank the sketch's resolution is close to flat in ambient dimension,
0.21 to 0.25 for `k=16` across dimensions 16 through 128, so the limit is the
sketch budget rather than the size of the space.

### This is not the probe that produced the accuracy numbers below

Converting the published figures to the same statistic:

| Run | Overlap | Chance | **Resolution** |
|---|---:|---:|---:|
| GO-P-2026-011, planted | 0.936 | 0.059 | **0.932** |
| GO-P-2026-020, Llama, the miss | 0.567 | 0.126 | **0.505** |
| GO-P-2026-021, Llama, the pass | 0.647 | 0.126 | **0.596** |

The chance value of 0.126 is consistent with a rank-16 read subspace in a
128-dimensional head space. **At rank 16 this package's sketch scores 0.03 to
0.06 where the published probe scored 0.60.** That is roughly an order of
magnitude, and the cause is now known: the source probe runs at `k/d = 1.25`
and the sketch measurements were all sub-dimensional. `mode="lstsq"` and
`jacobian_probe` port the source design, and the budget law above says what
it costs.

The residual gap is the interesting one. At `k/d >= 1` this package recovers
a planted subspace at resolution 1.000, while the source program recovered a
**real attention head** at 0.596. That difference is not the estimator. It is
the difference between a clean planted subspace and a real one, and measuring
it is exactly what C-3 exists for.

So the accuracy table in the next section describes a probe design this
package does not yet implement. Until it does, those numbers are provenance
rather than specification, and this document will keep saying so.

### Correction: the fix this document proposed does not work

An earlier version of this page said the gap would most obviously be closed
by subtracting the sketch's isotropic bias. **That was wrong, and C-2c and
C-2d were declared to establish it rather than to quietly replace it.**

The bias is real and now exact. For iid Gaussian directions,

    E[ghat ghat^T] = (1 + 1/k) g g^T + (||g||^2 / k) I

by Isserlis, confirmed numerically. `readscope.probe.debias_sketch` inverts it
in closed form. But **it is a multiple of the identity**, so it shifts every
eigenvalue equally and moves no eigenvector at all. Debiasing cannot buy a
single direction of bandwidth. E1 tested that prediction directly and it held
to 4e-16.

What debiasing does fix is the spectrum, and that is worth having on its own:
mean trace error against the exact estimator falls from **6.21 to 0.068** at
`k=8` and from **3.05 to 0.050** at `k=16`. A bit allocation computed by
water-filling on the raw sketch spectrum is allocating against a flattened
operator. Run `debias_sketch` before `water_fill`, and do not expect it to
help subspace recovery.

The actual cause of the bandwidth limit is the sketch's **variance**, not its
bias.

### The orthonormal estimator, and where it does and does not help

`mode="ortho"` draws an orthonormal frame and recombines as `U^T y`, which is
exactly the orthogonal projection of the gradient onto the drawn subspace. It
costs the same `2k` calls. At `k = d` the projector is the identity, so the
estimate is exact, and E5 confirmed that to 1e-9.

C-2d, ten seeds, ambient dimension 32:

| Estimator | res @1 | @2 | @4 | @8 | @16 | Bandwidth |
|---|---:|---:|---:|---:|---:|---:|
| exact, k=32 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **16** |
| sketch, k=16 | 0.985 | 0.490 | 0.240 | 0.128 | 0.033 | **1** |
| ortho, k=16 | 0.995 | 0.539 | 0.269 | 0.145 | 0.065 | **2** |
| sketch, k=8 | 0.966 | 0.481 | 0.221 | 0.118 | 0.056 | **1** |
| ortho, k=8 | 0.981 | 0.473 | 0.237 | 0.113 | 0.034 | **1** |

**Ortho buys bandwidth at `k/d = 0.5` and does not at `k/d = 0.25`.** At the
larger budget it dominates the sketch at every rank. At the smaller one it
wins at ranks 1 and 4 and loses at 2, 8 and 16. E3, which asked that ortho
never score worse anywhere, **failed at both three seeds and ten**, so this is
a real effect and not sampling noise. E4, which asked only that ortho buy
bandwidth at some budget, passed.

**The recommendation is therefore conditional, which is what a specification
is for.** Use `ortho` when the direction budget is a substantial fraction of
the ambient dimension. Below roughly a quarter, the two estimators are
interchangeable within this sweep and neither resolves more than the dominant
direction. Where that crossover sits has been measured at exactly two points
and nowhere else.

---

## Applicability: consumers this probe must not be pointed at

Inherited from `turboquant_pro.operator_sensitivity` and
`turboquant_pro.operator_trace`, which established these regimes before this
package existed. A scope that reads 0 V on a circuit it cannot couple to is
not a working scope, and neither is a probe that returns a confident zero.

The blind probe assumes the consumer is a **differentiable scalar margin**.
Every calibration in this document is on that regime and nothing else. Two
regimes that matter in practice break the assumption.

**Selection.** Top-k routers and argmax gates read the *order* of their
logits, not the values. They are invariant to a common-mode shift, and their
derivative is zero almost everywhere and undefined on the decision boundary.
Finite differencing returns exactly zero, which an unguarded probe would
report as a consumer that reads nothing at all. The correct instruments are
the **routing margin**, the gap between rank `k` and `k+1`, and the
**differential fraction**, the share of a perturbation that is not
common-mode. Both are in `readscope.regimes`.

**Recurrence.** For a per-channel linear recurrence `h_t = a h_{t-1} + b_t` a
pointwise Jacobian is well defined but misleading, because decay error
compounds along the sequence. Sensitivity of the accumulated state goes as
`1 / (1 - a)^2`, so slow channels over long sequences dominate and a
single-step probe sees none of that. `readscope.regimes.decay_sensitivity`
gives the right coefficient.

**This is enforced, not just documented.** `blind_probe` runs an
applicability check by default and raises rather than returning a reading it
cannot justify. The check costs up to 128 extra consumer calls and can be
turned off with `check_regime=False` once a regime is established. The
verdict is recorded in the result's metadata so a reading always carries the
evidence that the probe was entitled to take it.

The check's own threshold, the share of zero responses above which a consumer
is called a selection regime, is declared at 0.9 and **has not been swept by
any calibration here**. It is a specification field with an unmeasured value,
which is the honest status.

---

## What else was inherited

`readscope.quotient` ports the tangential and radial displacement split from
`turboquant_pro.a2_probe`. It answers a question the read operator does not:
of the variation actually present in the data, how much survives
normalization. A quotient that discards scale is safe exactly when the
consumer's metric is carried by the tangential part. Reading a spectrum
without checking that is how a quantizer scores well on reconstruction and
destroys the ranking anyway.

### Backported the other way

`turboquant_pro.calibration_coverage`, landed at commit f526400. That
package's Lloyd-Max path fits a codebook from "a representative set of real
key activations" and had no way to check the word representative. A
calibration sample shifted or differently shaped from serving traffic fits
the codebook to the wrong measure and fails quietly, because the codebook
still reconstructs its own calibration set beautifully. That is probe
loading with a different name, so `readscope.loading` went back the other
way as a coverage guard with a warn and a strict CI gate.

Deliberately **not** ported, and better used from their own package:
`turboquant_pro.rank_certificate`, which supplies distribution-free floors on
rank agreement and belongs with the retrieval path that consumes it, and
`turboquant_pro.operator_trace`, which infers a consumer's regime from a
torch graph and would drag a heavy dependency into a numpy-only package. Both
compose with this one rather than needing to live inside it.

---

## The budget law: the most actionable number here

Reading the source program's own probe settled where the accuracy gap came
from. `gateB_llama_rematch.py` runs `N_PROBE = 160` directions in a `d = 128`
head space, so `k/d = 1.25`. **It is not a cheaper estimator, it is an
overdetermined one.** Everything C-2b and C-2d measured was the
sub-dimensional regime, which nothing in the source program ever relied on.
The premise that a sketch was "the affordable estimator" was mine, not the
program's.

C-2e, record `calibration/records/c2e-budget-law.json`, five of six bars.
Least-squares recovery, ambient dimension 32, five seeds, 96 operating
points.

| Budget `k/d` | res @1 | @2 | @4 | @8 | @16 | Bandwidth |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.967 | 0.460 | 0.214 | 0.114 | 0.003 | **1** |
| 0.50 | 0.988 | 0.549 | 0.246 | 0.161 | 0.077 | **2** |
| 0.75 | 0.996 | 0.746 | 0.359 | 0.166 | 0.115 | **2** |
| 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **16** |
| 1.25 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **16** |
| 1.50 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **16** |

**Bandwidth is a cliff at `k = d`, not a slope — at matched operating-point
counts.** It goes 1, 2, 2, then straight to the exact estimator's 16. Three
quarters of the directions buys two of sixteen; the last quarter buys the
other fourteen. Within this design — every row measured at the same 96
points — there is no graceful degradation to trade against. **Scope note
(2026-08-17):** these rows compare equal `n`, not equal total call budget
`2kn`; the OT-3 theorem behind the cliff covers subspace-confined designs
and explicitly not generic-position allocation across points, and the
sketch expectation retains `S`'s eigenspaces at every `k`. C-15 has
now decided it (`records/c15-budget-surface.json`, sealed decision rule):
it does not catch up. See the C-15 section below — "no partial budget
worth spending" is licensed at equal total budget too, within the
measured range.

**Read as a specification: for a scalar-margin consumer, pay `2d` consumer
calls per operating point or expect the dominant direction and nothing
else.** That is the honest cost of this instrument, and it is why the source
program pays it.

### C-15: the cliff survives equal-budget reallocation

C-15, record `calibration/records/c15-budget-surface.json`, sealed
decision rule applied as frozen (`DECLARATION-C15.md`; sealed by recorded
owner override, which the declaration itself records). The reviewer-posed
question: C-2e's rows compare equal point counts, and the sketch
expectation `(1+1/k)S + tr(S)/k·I` retains `S`'s eigenspaces at every
`k` — so can many cheap points buy back what few directions lose, at
equal **total** consumer calls?

Surface arm: total directional observations fixed at the C-2e flagship's
`kn = 3072`; `k/d` from 1/8 to 1.25 with `n = 3072/k`. Scaling arm:
`k = d/4`, `n` up to 3072 — 8× the flagship's total budget. Five seeds,
ranks {4, 16}, lstsq, the C-2e planted family.

| `k/d` (surface, rank 16) | 0.125 | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 |
|---|---:|---:|---:|---:|---:|---:|
| `n` at equal budget | 768 | 384 | 192 | 128 | 96 | 76 |
| median res@16 | 0.02 | 0.11 | 0.05 | 0.04 | **1.0** | **1.0** |

- **D1, dominance: holds.** Per-seed margin of `k = d` over the best
  sub-dimensional equal-budget cell: median **0.883**, minimum 0.841,
  against a sealed 0.3 rule.
- **D2, convergence within 8×: none visible.** At `k = d/4`, rank 16,
  median res@16 is 0.062 at `n = 384` and **0.057 at `n = 3072`** —
  flat to declining. Rank 4 medians (0.285, 0.267, 0.256, 0.338) show
  no clean climb either.
- AV: 100/100 cells, declared calls exact, `k = d` sanity at 1.0.

**The law, stated once:** the population algebra permits sub-dimensional
convergence; the measured sample complexity walls it off — within 8× the
full-dimension budget, reallocating directions into operating points buys
nothing at any graded rank. The cliff is a property of *total* consumer
calls in the measured range, which is a stronger statement than C-2e
alone licensed and is exactly the second budget law the reviewer's
algebra pointed at. The asymptotic question (would 100× converge?)
remains open and is priced accordingly: nobody spends 100× to avoid 1×.

### The discount, which is real and does not go all the way

A vector-valued consumer returns `m` numbers per direction instead of one, so
a direction carries `m` times the information at identical call cost. Whether
that substitutes for directions was declared as an open question with both
answers useful.

At `k/d = 0.5`, holding everything else fixed:

| Consumer output `m` | res @1 | @2 | @4 | @8 | @16 | Bandwidth |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.988 | 0.510 | 0.236 | 0.134 | 0.049 | **2** |
| 2 | 0.988 | 0.883 | 0.429 | 0.208 | 0.048 | **2** |
| 4 | 0.989 | 0.977 | 0.696 | 0.382 | 0.124 | **4** |
| 8 | 0.990 | 0.989 | 0.909 | 0.556 | 0.243 | **8** |

**Vector output buys bandwidth and does not buy the cliff away.** At half the
direction budget, eight outputs reach bandwidth 8 where a scalar margin
reaches 2, a fourfold gain for free. It still falls short of the exact
estimator's 16, so S5 failed while S4 passed, which is exactly the middle
outcome the declaration named as informative.

The pattern in these four rows is that bandwidth tracks `m` once `m` is at
least 4. **That is a suggestion, not a law.** It rests on one ambient
dimension, one budget ratio, and four values of `m`, and nothing here has
tested whether it survives any of those changing.

**Practical reading.** If your consumer emits a vector, use
`jacobian_probe` and you can work at half the direction budget for a
bandwidth of roughly its output width. If it emits a scalar, pay `k >= d` or
accept a one-direction reading.

---

## Real attention heads, three families

C-3b, record `calibration/records/c3b-architecture-spread.json`, **PASS on
all seven bars.** 36 head-cells from Llama-3.2-3B, Qwen2.5-1.5B and
Mistral-7B, three depths each, four heads each, on the models' own post-RoPE
keys and queries.

The ground truth is exact rather than planted. For a softmax head reading one
key, `∂pᵢ/∂k_s = pᵢ(e_s − p_{i,s})·qᵢ/√d`, so the read operator is
`Σ_s Σ_i a_{i,s} qᵢqᵢᵀ/d` in closed form. **The read subspace of an attention
head with respect to a key is spanned by its queries.** That is a fact about
softmax attention, not a claim of this program, and it is what makes real
weights gradeable at all.

| Family | cells | head_dim | resolution @ k/d = 0.5 | @ k/d = 1.25 |
|---|---:|---:|---:|---:|
| llama | 12 | 128 | 0.322 | **1.0000** |
| qwen | 12 | 128 | 0.302 | **1.0000** |
| mistral | 12 | 128 | 0.377 | **1.0000** |
| gemma | 12 | 256 | 0.462 | **1.0000** |
| all | 48 | — | **0.366** | **1.0000** |

**Across-family spread is 1e-15.** At the source program's budget ratio the
probe recovers the analytic operator exactly on every cell of every family,
and the budget cliff reproduces on real activations at 0.366 against 1.000.
C-2e's headline was not an artifact of planted subspaces.

Gemma-3 is the only family here at head_dim 256, so it varies dimension as
well as architecture. Its rotary embedding keeps separate tables for sliding
and full attention and looks them up by a layer type that lives on the config
rather than on the layer, which is why the first two extraction attempts
failed on it.

Median attention row entropy across the 48 cells is **4.30 bits**, against
0.152 for the failed first attempt. That single number is what separates a
real measurement here from an artifact.

### What C-3 got wrong first

The first attempt drew its query set from the key stream, because a KV cache
stores keys and values and not queries. The self-match term saturated the
softmax: median attention row entropy came out at **0.152 bits** across 42
cells where 192 positions allow 7.6, with eleven analytic operators below the
graded rank and some at rank zero.

Reported naively that would have read as "real attention heads are
degenerate." It is not a fact about any model, it is a fact about a shortcut
this document declared in advance and then measured. With real queries hooked
from `q_proj` and the model's own rotary embedding, median entropy is above
one bit by declared bar and every analytic operator reaches full rank.

---

## What the published 0.647 measures

C-4, record `calibration/records/c4-reference-choice.json`, four of five bars.

C-3b recovers the analytic operator at resolution 1.000 on real heads at the
same budget ratio where the source program reports overlap 0.647. Both cannot
be statements about probe fidelity.

Reading `gateB_llama_rematch.py`, the difference is the **reference**. That
script grades against `Qsetᵀ Qset / n_q`, the unweighted query covariance. A
finite-difference probe recovers the Jacobian Gram, which is the same queries
weighted by how much the softmax actually responds along each. Both are
spanned by the queries; they are not the same operator.

Measured on the 36 real cells, using only probe-free closed forms:

| Quantity | median | range |
|---|---:|---:|
| weighted against unweighted reference | **0.796** | 0.678 to 0.985 |
| probe against weighted reference | **1.000** | 1.000 to 1.000 |

**R2 passed**: the reference disagreement lands inside the band set by the
two published figures, whose resolutions are 0.505 and 0.596. **R3 passed**:
the probe is exact against its own target, so the disagreement is not a
degraded instrument.

**R1 failed**, and it mattered. The bar asked that the two references
disagree on *every* cell, and one cell agreed at 0.985. So the reference
choice was a large and real contributor and not the whole of it, with a
residual left for the query capture, the grouped-query grouping and the model
and layer set. **C-10 closed that residual.**

### C-10 closes it, and the query set was the cause

C-10 matches all three unmatched factors to `gateB_llama_rematch.py`:
Llama-3.2-3B, layers {8, 16}, float32, and `Qset` as **every query in a
key-value head's group across the whole sequence**, 576 vectors rather than
the 24 C-4 sampled. Probe settings are the source's: 32 probe keys, 160
unit-norm directions, step 1e-3, pseudoinverse, graded at rank 16. **PASS on
all six bars.**

| | queries per cell | unweighted reference rank | median overlap |
|---|---:|---:|---:|
| C-4 | 24 | 24 | 0.821 |
| **C-10** | **576** | **128** | **0.703** |
| published GO-P-2026-021 | 576 | 128 | **0.647** |

**Matching the query set closed 68 percent of the distance to the published
figure**, from 0.174 away to 0.056. The mechanism is the one the extraction
predicted: 24 query vectors give a rank-24 covariance whose top-16 is well
separated, while 576 give a full-rank one whose top-16 is far less
determined, so the two references disagree more.

**The probe recovers the weighted operator at resolution 1.000000 on all 16
cells** under the source's own settings, so nothing about the instrument is
implicated. M3, that the probe-against-unweighted and
weighted-against-unweighted overlaps agree, came out at 1.6e-9 and is
**arithmetic given M1 rather than an independent finding**, which the sweep
said in advance: if the probe *is* the weighted operator then its overlap
with any third object is that operator's overlap.

**So the published figure is the reference choice, entirely.** A
finite-difference probe recovers the softmax-weighted Jacobian Gram exactly;
the source grades it against the unweighted query covariance; and the number
reported is how much those two differ.

**What is not controlled**, and is the honest size of what remains: the input
text and the probe-key draw, neither matched to the source. The residual 0.056
is the scale that explains, and this sweep does not claim to reproduce 0.647
to the digit. It claims the magnitude and the mechanism, which is what the
remainder was about.

None of this is a correction of anyone. An unweighted query covariance is a
defensible account of what a head reads, precisely because it does not depend
on which key is being perturbed. The datasheet lesson stands and is now
quantified: **two reasonable references for the same head differ by about
0.3 in overlap, so a recovery number without its reference named is not
interpretable.**

**This audits a definition, it does not correct anyone.** An unweighted query
covariance is a defensible account of what a head reads, precisely because it
does not depend on which key is being perturbed. The finding is that a
datasheet has to say which reference a number is against, because two
reasonable choices differ by roughly a fifth of the available range.

---

## A non-transformer consumer

C-3c, record `calibration/records/c3c-state-space.json`, four of six bars.
Twelve channel-cells from a real Mamba-790m at three depths, state dimension
16.

`CALIBRATION.md` asked for a consumer that is not attention, so that nothing
in this specification is secretly a statement about softmax. A selective SSM
supplies one, and it comes with its own exact ground truth. Writing
`g_t[n] = prod_{u} exp(A[n] dt[u])` for the accumulated decay,

    d y_t / d h_s = C_t * g_t,
    M_true = sum_{t >= s} (C_t * g_t)(C_t * g_t)^T

**The read subspace of a recurrent state is spanned by its readout vectors,
attenuated by how much of each has already decayed.** That is the same shape
as attention, where a head's read subspace with respect to a key is spanned
by its queries.

**Two results, both from bars that passed.**

The probe recovers the analytic operator at **resolution 1.000 on every
cell** at `k/d = 1.25`, on a consumer it was never designed around. The
instrument is not an attention instrument.

**The budget cliff is a property of the probe, not of attention.** At
`k/d = 0.5` resolution runs from **−0.32 to 0.10**, meaning at or below
chance, against 1.000 at `k/d = 1.25`. On a 16-dimensional state, half the
directions buys nothing at all. That is a sharper cliff than either the
synthetic sweep or the attention sweep showed, and it settles that C-2e's
headline was never about softmax.

**Measured, and worth having on its own:** Mamba's channels differ enormously
in memory. Effective memory runs from **1.8 to 82 steps**, median 7.9. Most
channels read almost entirely their immediate past, median near-quarter trace
fraction 0.993, while a few spread across the whole horizon. That is the
compounding `readscope.regimes` warns about, measured on a real model rather
than asserted.

### Two bars I declared wrongly

**N3 asked that every cell take more than half its trace from the nearest
quarter of the horizon.** Eleven of twelve do, at a median of 0.993. One
channel with 82-step memory sits at 0.470. The failure is not the model
disagreeing with the measurement, it is my having declared a universal about
a quantity that is heterogeneous by construction: a per-channel decay rate is
exactly the thing an SSM is free to vary. The correct treatment is the one
C-3b uses for family spread, report the distribution and place no bar on its
size.

**N5 required the accumulated decay at the horizon to sit inside a band**,
meant to catch cells graded on a state that has already emptied. Two channels
fall below the floor at 6e-22 and 5e-18. But those cells are not degenerate:
their analytic operators have rank 10 to 16 against a graded rank of 8, and
the probe recovers them at resolution 1.000. A channel that forgets in two
steps is a real short-memory channel, not a broken measurement. Anti-vacuity
belongs on the operator, which N4 already checks, and not on the decay.

**Evaluated on the same record, both corrected bars hold**: every analytic
operator has rank at least 8, and the near-quarter fraction is reported as a
distribution running 0.470 to 1.000. **No sweep was re-run**, because no
measured number would change; only the labels on two bars would. Saying that
is preferable to producing a PASS by restating the same numbers.

---

## The scale ladder

C-5, record `calibration/records/c5-scale-ladder.json`, **PASS on all six
bars.** 48 head-cells across Qwen2.5 at 1.5B, 7B, 14B and 32B, three depths
and four heads each, all loaded in bfloat16 so precision is not confounded
with size.

Qwen2.5 holds `head_dim` at 128 across the whole ladder while layer count
goes 28 to 64, head count 12 to 40 and grouping 2 to 8 key-value heads.
**The geometry the probe works in is constant and only the substrate grows**,
which is the only way to ask the scale question without confounding it with
dimension.

| scale | res @ k/d = 0.5 | @ k/d = 1.25 | effective rank | entropy, bits |
|---:|---:|---:|---:|---:|
| 1.5B | 0.299 | **1.0000** | 1.66 | 4.10 |
| 7B | 0.318 | **1.0000** | 1.76 | 4.00 |
| 14B | 0.349 | **1.0000** | 2.19 | 2.78 |
| 32B | 0.341 | **1.0000** | 1.92 | 3.65 |

**Across-scale spread at `k/d = 1.25` is 6.7e-16.** A twentyfold increase in
parameters changes nothing about what the probe needs. The budget law has no
scale term, so the specification does not need one.

### The unbarred column that matters most

Effective rank was reported without a bar, because no prior said what it
should do. It turned out to be the most useful number in the sweep.

**Every cell's analytic read operator has an effective rank between 1.20 and
3.29, median 1.82**, while its exact rank is 24 at every single cell. So the
read operator spans two dozen directions and its sensitivity is concentrated
in roughly two of them, at every scale from 1.5B to 32B.

That looked like it might reframe the bandwidth story. C-2e's cliff is about
recovering a **rank-16 subspace**, and a real head does not put much
sensitivity in most of those sixteen, so a sub-dimensional probe might have
been adequate for the one or two directions that carry the mass.

**C-6 tested that and it is false.** The observation earned an experiment and
the experiment killed it. See the next section.

---

## The rank-budget surface: the cliff does not care about rank

C-6, record `calibration/records/c6-rank-budget-surface.json`, four of six
bars, and the bar that failed is the one that mattered.

C-5's effective-rank observation suggested the budget cliff might soften if
you only asked for the directions that carry the mass. C-2e had swept budget
at a fixed graded rank of 16 and never swept the other axis. C-6 sweeps both,
on the four-family real-activation set, running the probe once per budget and
grading the same recovered operator at every rank.

Mean resolution, 48 cells, four families:

| graded rank | k/d=0.125 | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 | required k/d |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.158 | 0.363 | 0.646 | 0.856 | 1.000 | 1.000 | **1.0** |
| 2 | 0.149 | 0.292 | 0.524 | 0.735 | 1.000 | 1.000 | **1.0** |
| 4 | 0.130 | 0.234 | 0.448 | 0.686 | 1.000 | 1.000 | **1.0** |
| 8 | 0.109 | 0.197 | 0.393 | 0.633 | 1.000 | 1.000 | **1.0** |
| 16 | 0.093 | 0.181 | 0.366 | 0.583 | 1.000 | 1.000 | **1.0** |

**S2 failed. The required budget is 1.0 at every rank, including rank one.**
Asking for a single direction instead of sixteen buys real improvement at
intermediate budgets, 0.646 against 0.366 at half the dimension, and never
enough to clear the bar below `k = d`.

**So the cliff is rank-independent and the specification hardens rather than
softens.** Pay `k >= d` regardless of how many directions you need. The
effective-rank observation is true and operationally worth nothing.

The mechanism is visible once stated. Below full dimension the estimate is a
projection onto a random `k`-subspace, and the leading eigenvector of a
projected operator is not the leading eigenvector of the operator unless the
random subspace happens to contain it. The projection distorts every
direction, not just the tail, so there is no low-rank shortcut to be had.

**S5 also failed**, on my own declaration rather than the data. It required
median attention row entropy above 1.0 bits on *every* cell, carrying over a
threshold C-3b applied to the median *across* cells. One cell of 48 sits at
0.339 bits. Recomputed on the 47 cells that clear the bar, the surface is
unchanged to three decimals and every required budget is still 1.0, so the
S2 conclusion does not depend on it. That is the fourth time in this
programme I have declared a universal where a distribution was called for.

---

## The loading correction does not work yet, and the sweep that said it did was wrong

`CALIBRATION.md` has said since C-1b that turning the loading curve into a
correction is the point of having it. C-7 and C-7b tried. Neither produced a
usable correction and the second one produced a **false pass**, which is worth
more space than a success would be.

### C-7 was invalid, not failed

It estimated loading from the 16 operating points it probed at, in a
128-dimensional head. A covariance fitted from 16 points in 128 dimensions has
rank 15, the ridge dominates its log determinant, and the divergence came back
around **1e11 nats**. No bar was tested. The run was stopped once the
readings were seen to be degenerate, so it wrote no record; the script stays
in `calibration/` as the design that was wrong.

The instrument now refuses that case. `probe_loading` raises when the sample
count does not exceed the dimension and warns below five samples per
dimension. **An instrument that reports 1e11 without complaint is worse than
one that stops.**

The conceptual error underneath was mine: loading is a property of two
*distributions*, not of the sample a probe happens to visit. Conflating them
tied the loading estimate's quality to the probe budget, which are unrelated.

### C-7b passed all six bars and the pass is an artifact

With loading estimated from 1024 independent draws per distribution, C-7b
reported a **92.1 percent reduction in mean absolute error** and PASS on
every bar including T2, the falsifiable one.

It is not a result. The correction was fitted over 0.89 to 91.64 nats. In
C-7b:

- **only 15 percent of readings fell inside that range**, and the maximum
  loading was 1.9e12 nats
- outside it the correction clamps to its endpoint attenuation of 0.437, so
  every reading was divided by the same constant
- **202 of 240 corrected values came out at exactly 1.0**, pinned by the
  output clip
- the truth being compared against is also 1.0

So the correction did not predict anything. It divided by a constant and the
clip did the rest, and the error against a target of 1.0 went to zero because
the answer was clipped onto the target. **T2 passed for a reason that has
nothing to do with the correction working.**

`LoadingCorrection.correct` now refuses to extrapolate by default. That is the
bar C-7b was missing: a correction must be evaluated inside its fitted domain,
and a reading that needs extrapolation must be recorded as out of range rather
than clamped into an answer.

### The axis was the obstacle, and it is now fixed

Raw Jeffreys fails as a datasheet quantity in two ways at once. It grows with
dimension for a mismatch identical in every direction, and it is positive when
there is no mismatch at all, because two finite samples of one law have
different fitted Gaussians.

`readscope.loading` now reports

    loading = max(0, jeffreys - null_floor(n_p, n_a, d)) / d

The null floor is simulated per shape and cached. It is **distribution free**,
because Jeffreys between fitted Gaussians depends on the moments only through
products invariant to a shared affine map, which is why it can be tabulated
once rather than measured per use.

C-8, record `calibration/records/c8-dimensionless-loading.json`, measures
whether that works. A fixed per-direction mismatch, constructed identically at
five dimensions:

| dimension | 16 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|
| normalised loading | 1.140 | 1.135 | 1.125 | 1.129 | 1.131 |
| raw Jeffreys, nats | 18.3 | 36.5 | 72.4 | 145.1 | 290.7 |

**Relative spread falls from 242 percent to 1.3 percent, a factor of 187**,
across a sixteenfold range of dimension. Two independent samples of one law
read exactly 0.000 at every dimension after the null correction, where raw
Jeffreys reads 0.1 to 1.3. The axis is monotone in mismatch at every
dimension.

So the axis is fixed and comparable. **The correction is still not
established.**

### C-8's correction half passed vacuously, and F-1 predicted it

D5 asked whether a correction fitted at dimension 32 transfers to 16, 64, 128
and 256. It reported 95 percent of readings in domain and a mean absolute
error of 0.0000.

Every reading in that half was exactly 1.000. The fitted attenuation curve was
therefore the identity, and an identity correction cannot have error. **The
probe never degraded, so there was nothing for a correction to predict.**

The cause is the degeneracy this programme already found and wrote down. F-1,
from C-1: **probe loading cannot degrade subspace recovery when the
consumer's read subspace does not vary across the input space.** The gated
consumer's gradient always lies in the span of its three defining vectors, so
graded at rank three, which is that whole span, the recovered subspace is the
span wherever you probe. Using the exact least-squares estimator at
`k/d = 1.25` removed the last source of variation.

C-1b avoided this by accident, through a noisier estimator. C-8 did not, and
the missing bar is one C-1b had and C-8 dropped: **before testing whether a
correction predicts degradation, bar that degradation exists in the fit
family.** C-1b's L3 separation bar was exactly that.

### C-9, the third attempt, is the first valid one and the answer is no

C-9 carries the separation bar the previous two lacked, on a consumer built so
that loading can bite: a rank-12 planted subspace graded at rank 4, where the
gradient weighting ``sech^2(b_j . x)`` moves with position so which four
directions dominate is a property of where one probes.

**E0 passed**, with a fit-family separation of **+0.237**, so unlike C-7b and
C-8 there really was degradation for a correction to predict. **The dimension-
less axis also did its job: 100 percent of readings landed inside the fitted
domain, against 15 percent for C-7b.** Both repairs worked.

The correction still does not transfer.

| dimension | 16 | 32 (fit) | 64 | 128 |
|---|---:|---:|---:|---:|
| separation | +0.146 | **+0.237** | **−0.005** | **−0.076** |

**E2 failed. Loading degrades recovery at low dimension and stops doing so at
high dimension**, and at 128 the reading is very slightly *better* under
loading. E4 failed too: the correction cut mean absolute error from 0.147 to
0.095, a 36 percent reduction against a 50 percent bar, with 67 percent of its
outputs still pinned at 1.0.

The mechanism is visible and worth stating, because it says when to worry
about probe loading at all. The planted basis is random, so in high dimension
every read direction sees nearly the same variance from an anisotropic
probing distribution. **Concentration of measure removes the differential
saturation that makes loading bite.** Loading damages a reading when the shift
in the probing distribution is *aligned* with the read directions, and a
random shift in high dimension is not.

So loading is not a function of loading alone. Degradation depends on the
alignment between the probing shift and the read subspace, which a scalar axis
cannot carry no matter how well normalised it is.

**Loading remains a warning, not a correction, after three attempts.** The two
repairs those attempts produced are real and are kept: the axis is
dimensionless and the estimator refuses to extrapolate or to read a
rank-deficient sample. What is now known, and was not before, is that a
scalar correction is the wrong shape for this effect.

---

## Read operators drift along the sequence

C-11c, record `calibration/records/c11c-operator-drift.json`, **PASS on all
six bars**, on the sixteen source-matched Llama-3.2-3B head-cells. (Two
predecessors stay on the record: C-11's first cut and C-11b's null-free
intermediate, `records/c11-operator-drift.json` and
`records/c11b-operator-drift.json` — C-11c's paired null is what they
lacked, and the sequence is kept rather than collapsed.)

A head's read operator is spanned by its queries, so if the query
distribution moves with position then a key compressed against an early
operator is later read by a different one. That is probe loading along the
time axis, and it is a mechanism candidate for the long-generation
degradation turboquant-pro reports and does not explain. (The C-12
long-generation follow-up under the corrected symmetric codebook,
`records/c12-longgen-drift-sym.json`, is where that mechanism claim went
to die: OT-4 established the collapse is feedback compounding of a
constant codebook error, not drift — the record is cited here because a
census that omits its own refutations is not a census.)

**Every quantity is paired with a null**, because two disjoint samples of one
distribution do not give identical operators either. Windows cut by position
are compared against windows of identical size drawn at random, five draws
per cell.

| graded rank | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| positional | 0.667 | 0.385 | 0.252 | 0.193 | 0.181 |
| random null | 0.933 | 0.572 | 0.470 | 0.405 | 0.414 |
| **gap** | **+0.266** | **+0.187** | **+0.219** | **+0.213** | **+0.233** |

**The null is more than half the naive effect.** Read against 1.0, the drift
at rank 2 looks like 0.615; against the null it is 0.224. The earlier attempt
made exactly that error and this table is why the sweep was rebuilt.

The cleanest reading is at rank one, where the tail cannot contribute and the
null is nearly perfect: **the dominant read direction itself moves with
position**, 0.667 against a null of 0.933. Fourteen of sixteen cells show a
positive gap.

**Cost.** Allocating against the early operator costs the late consumer
**225 percent of a uniform split's cost more** than allocating against the
late one, above what resampling alone costs. Allocating against the
whole-sequence operator is better on **every cell**.

**Scope.** Sixteen cells, two layers, one 3B model, 192 positions. A
mechanism candidate on a short sequence, not a demonstration on the
long-generation regime, and never run against a real degradation curve.

---

## Accuracy: everything measured so far

Three runs, on two substrates, under sealed preregistration in the
`geometric-observation` evidence repository. Two of the three are partial or
negative and both stay on the record.

| Run | Substrate | Consumer | Overlap | Chance | Ratio | Cells | Verdict |
|---|---|---|---|---|---|---|---|
| GO-P-2026-011 | Planted low-rank read subspace | tanh margin | **0.936** | 0.059 | ≈16× | synthetic | PASS 5/5 |
| GO-P-2026-020 | Llama-3.2-3B post-RoPE keys | softmax attention | **0.567** | ≈0.126 | ≈4.5× | 16 | **MISSED** the sealed 0.60 bar (NEG-12) |
| GO-P-2026-021 | Llama-3.2-3B post-RoPE keys | softmax attention | **0.647** | 0.126 | ≈5.1× | 16 | PASS, 32-key probe |

The 16 cells are layers {8, 16} × 8 KV heads of one 3B model. They were
the instrument's entire real-model evidence base when this section was
first written; they no longer are. The census as of C-15's drafting:
**48 attention head-cells across four families** (Llama, Qwen, Mistral,
Gemma) at resolution 1.0000 against closed-form references
(`records/c3-architecture-spread.json`, `c3b`), **12 state-space
channel-cells** from Mamba-790m with its own analytic operator
(`records/c3c-state-space.json`), and a **48-cell Qwen2.5 scale ladder**
from 1.5B to 32B (`records/c5-scale-ladder.json`) — alongside the three
sealed GO runs above, which stay on the record with their bracket. The
sections below carry each cell; `tools/check_spec_census.py` fails CI if
this census drifts from the records again.

Two facts that a spec sheet must carry and a demo would omit.

**The two Llama numbers bracket the bar.** 0.567 and 0.647 are the same
instrument on the same model, differing by probe design. A single accuracy
figure that does not say which probe produced it is not a specification.
NEG-12 stands as its own negative rather than being absorbed into the later
success.

**0.647 is conditional, not flat.** Review pass VI-6 established that the
figure holds as a certificate under omission and equal weights, and it is
written that way in the source. A datasheet number stated without its
conditions is the thing this document exists to prevent.

**Median, not typical.** 0.647 is a median across the 16 cells, with strong
heads near 0.96 in the earlier run. The spread across cells is part of the
specification and is not yet reported per-cell here.

---

## Linearity, measured and partly negative

GO-P-2026-041, on 20 Newsgroups, tested the strongest form: recover `P_C`
non-oracle on a fresh untouched domain, and commit both the winning code and
its magnitude before opening the test split.

The blind winning-code prediction passed, AUROC 0.975 against 0.910, so the
framework chose the better code before any label. The two-metric flip tied on
held-out and the point magnitude overshot its sealed band.

**Blind direction transfers across domains. Blind magnitude does not.**
Across rates the magnitude tracks, Pearson 0.995. Across domains the scale
constant does not yet transfer. Until it does, any magnitude this instrument
reports is a within-domain reading.

---

## Probe loading, the known error term

The probe estimates `E[g g^T]` over the operating points it is handed, and
those points are drawn from a probing distribution that is not the activation
distribution the consumer meets in service. The recovered operator is the read
operator averaged over the wrong measure.

This is probe loading. In instrumentation it is characterized, specified, and
corrected for, and there is no reason it cannot be here.

`readscope.loading` supplies the axis: Jeffreys divergence, Bhattacharyya
distance, mean shift in the activation metric, and the worst-direction
variance ratio, all between the fitted Gaussians of the two distributions.
`interpolate_distribution` sweeps a probing distribution toward an activation
distribution along a path that keeps every intermediate covariance positive
definite, so that recovery quality can be plotted against loading with
everything else held fixed.

### The first calibration curve

C-1b, record `calibration/records/c1b-loading-curve.json`, PASS on all five
declared bars. Gated synthetic consumer, ambient dimension 24, graded rank 3,
chance overlap 0.125, five seeds, 48-direction sketch, 384 operating points.
Truth is the read operator recovered by the exact estimator on the activation
distribution, which is what a user cares about rather than any planted
subspace.

| Probe loading, Jeffreys nats | Overlap | SD across seeds | Worst seed |
|---:|---:|---:|---:|
| 0.89 | 0.9919 | 0.0029 | 0.9863 |
| 3.23 | 0.9889 | 0.0019 | 0.9867 |
| 10.73 | 0.9748 | 0.0059 | 0.9687 |
| 25.25 | 0.8521 | 0.0280 | 0.8089 |
| 50.50 | 0.4914 | 0.0354 | 0.4361 |
| 91.64 | 0.4365 | 0.0445 | 0.3741 |

**The curve has a knee between 25 and 50 nats.** Below about 11 nats the
reading is essentially exact and the seed spread is under a percent. By 25
nats it has lost fifteen points. By 50 it has halved, and past that it flattens
onto a shelf around 0.44, still well above the 0.125 floor, so a badly loaded
probe returns a degraded reading rather than noise.

**Read as a specification:** overlap at or above 0.97 for probe loading up to
roughly 11 nats, falling to about 0.44 by 92 nats, on this consumer family.
That is one configuration and not yet a correction factor. Extending it across
architectures and rank profiles is C-3, and turning the curve into a
correction is the point of having it.

### What C-1 taught before C-1b measured anything

The first attempt failed four of five bars and one of its three defects was
not a bug.

**A consumer whose read subspace does not vary across the input space cannot
exhibit probe loading at all.** If the Jacobian's row space is the same
everywhere, the recovered subspace is that row space regardless of where the
probe points come from. Probe loading acts only on consumers whose local
sensitivity direction changes over the input space, and what goes wrong is
then the average over the wrong measure.

This bounds where the error term applies, and it is a claim about the
instrument that the instrument's own calibration produced.

---

## What this instrument does not claim

It does not claim that recovered sensitivity is causal.

It does not claim any magnitude transfers across domains. GO-P-2026-041 says
it does not.

It does not claim a specification. It claims the measurements in the
census above — sealed GO runs (one synthetic, one negative, one positive)
and the calibration cells recorded in `calibration/records/` — and an
argument that the missing measurements (population/sample uncertainty,
the equal-budget `(k, n)` surface) are the ones worth making next.

It does not require anyone to accept an account of what a consumer is. The
reading is a spectrum of output sensitivity, and it means the same thing
whatever you believe about the vocabulary it was derived in.
