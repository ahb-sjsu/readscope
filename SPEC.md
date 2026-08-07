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
| Sample rate | samples/second | Consumer evaluations spent per operating point. `2d` for the exact estimator, `2k` for the `k`-direction sketch. | **Measured, exactly known** |
| Bandwidth | −3 dB frequency | The rank range over which recovery stays above the noise floor. How many eigendirections can be resolved before the reading is chance. | **Measured for this package's estimators, and it is bad** |
| Noise floor | volts RMS | Chance overlap for the shape being read, `rank / dim`. Reported with every reading. | **Measured, exactly known** |
| Accuracy over range | percent of reading | Recovered-subspace overlap as a function of rank, dimension, probe budget, and loading. | **Three real-model points; one full loading curve on a synthetic consumer** |
| Input impedance | ohms | Probe loading. Divergence between the probing distribution and the activation distribution. | **One curve measured, synthetic consumer** |
| Linearity | percent | Whether the recovered magnitude tracks the true magnitude across scale and across domain. | **Partial. Direction transfers, magnitude does not** |
| Temperature drift | ppm/°C | Stability of a reading across architectures at matched rank profile. | **Not characterized** |
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
magnitude, and it means the 32-key probe used in `geometric-observation` and
the Gaussian sketch shipped here are not the same instrument.

So the accuracy table in the next section describes a probe design this
package does not yet implement. Until it does, those numbers are provenance
rather than specification, and this document will keep saying so. Closing the
gap, most obviously by subtracting the isotropic bias the sketch introduces,
is the top engineering item in `CALIBRATION.md`.

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

Deliberately **not** ported, and better used from their own package:
`turboquant_pro.rank_certificate`, which supplies distribution-free floors on
rank agreement and belongs with the retrieval path that consumes it, and
`turboquant_pro.operator_trace`, which infers a consumer's regime from a
torch graph and would drag a heavy dependency into a numpy-only package. Both
compose with this one rather than needing to live inside it.

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
