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
| Bandwidth | −3 dB frequency | The rank range over which recovery stays above the noise floor. How many eigendirections can be resolved before the reading is chance. | **Not characterized** |
| Noise floor | volts RMS | Chance overlap for the shape being read, `rank / dim`. Reported with every reading. | **Measured, exactly known** |
| Accuracy over range | percent of reading | Recovered-subspace overlap as a function of rank, dimension, probe budget, and loading. | **Three real-model points; one full loading curve on a synthetic consumer** |
| Input impedance | ohms | Probe loading. Divergence between the probing distribution and the activation distribution. | **One curve measured, synthetic consumer** |
| Linearity | percent | Whether the recovered magnitude tracks the true magnitude across scale and across domain. | **Partial. Direction transfers, magnitude does not** |
| Temperature drift | ppm/°C | Stability of a reading across architectures at matched rank profile. | **Not characterized** |

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
