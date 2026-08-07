# readscope specification

**This instrument is not yet fully specified.** What follows is the form of
a specification, with every field named and most of them still empty. The
empty cells are the point of this document. An instrument is trusted because
it is specified, not because it worked once.

Every number here names the record it came from, in
`calibration/records/` for calibrations run in this repository, or the sealed
preregistration in the `geometric-observation` evidence repository for the
accuracy points inherited from that program. A number with no record does not
appear. Regenerating this document from the records automatically is not yet
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
| Minimum usable budget | minimum input signal | The direction budget below which the probe resolves almost nothing. | **Measured. It is a cliff at `k = d`** |
| Bandwidth | −3 dB frequency | The rank range over which recovery stays above the noise floor. How many eigendirections can be resolved before the reading is chance. | **Measured for this package's estimators, and it is bad** |
| Noise floor | volts RMS | Chance overlap for the shape being read, `rank / dim`. Reported with every reading. | **Measured, exactly known** |
| Accuracy over range | percent of reading | Recovered-subspace overlap as a function of rank, dimension, probe budget, and loading. | **Three real-model points; one full loading curve on a synthetic consumer** |
| Input impedance | ohms | Probe loading. Divergence between the probing distribution and the activation distribution. | **One curve measured, synthetic consumer** |
| Linearity | percent | Whether the recovered magnitude tracks the true magnitude across scale and across domain. | **Partial. Direction transfers, magnitude does not** |
| Temperature drift | ppm/°C | Stability of a reading across architectures at matched rank profile. | **Measured on three families. Spread 1e-15** |
| Applicability | probe coupling | Which consumer regimes this probe can be attached to at all. | **Bounded, and enforced in code** |

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

**Bandwidth is a cliff at `k = d`, not a slope.** It goes 1, 2, 2, then
straight to the exact estimator's 16. Three quarters of the directions buys
two of sixteen; the last quarter buys the other fourteen. There is no
graceful degradation to trade against, and no partial budget worth spending.

**Read as a specification: for a scalar-margin consumer, pay `2d` consumer
calls per operating point or expect the dominant direction and nothing
else.** That is the honest cost of this instrument, and it is why the source
program pays it.

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

**R1 failed**, and it matters. The bar asked that the two references disagree
on *every* cell, and one cell agrees at 0.985. So the reference choice is a
large and real contributor to the gap and it is **not uniform, and not the
whole of it**: median reference disagreement sits at 0.796 where the
published figures sit near 0.55, so something else contributes too. The
remaining candidates are the query capture, the GQA grouping and the model
and layer set, none of which this sweep matched to the source.

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

## Accuracy: everything measured so far

Three runs, on two substrates, under sealed preregistration in the
`geometric-observation` evidence repository. Two of the three are partial or
negative and both stay on the record.

| Run | Substrate | Consumer | Overlap | Chance | Ratio | Cells | Verdict |
|---|---|---|---|---|---|---|---|
| GO-P-2026-011 | Planted low-rank read subspace | tanh margin | **0.936** | 0.059 | ≈16× | synthetic | PASS 5/5 |
| GO-P-2026-020 | Llama-3.2-3B post-RoPE keys | softmax attention | **0.567** | ≈0.126 | ≈4.5× | 16 | **MISSED** the sealed 0.60 bar (NEG-12) |
| GO-P-2026-021 | Llama-3.2-3B post-RoPE keys | softmax attention | **0.647** | 0.126 | ≈5.1× | 16 | PASS, 32-key probe |

The 16 cells are layers {8, 16} × 8 KV heads of one 3B model. That is the
entire real-model evidence base for this instrument's accuracy.

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

It does not claim a specification. It claims three measurements, one
synthetic, one negative, one positive, on one real model, and an argument
that the missing measurements are the ones worth making next.

It does not require anyone to accept an account of what a consumer is. The
reading is a spectrum of output sensitivity, and it means the same thing
whatever you believe about the vocabulary it was derived in.
