# readscope

**An oscilloscope and spectrum analyzer for model consumers.** Point it at a
consumer, get back what that consumer actually reads.

[![Spec](https://img.shields.io/badge/spec-partial-orange)](SPEC.md)
[![Calibration](https://img.shields.io/badge/calibration-C--0_C--1b_pass-yellow)](CALIBRATION.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

```python
from readscope import blind_probe, spectrum_of, water_fill

probe = blind_probe(consumer, operating_points, mode="sketch", sketch_dim=32)
spec = spectrum_of(probe.S)

print(spec.effective_rank)      # how many directions this consumer reads
print(spec.energy_rank(0.9))    # how many carry 90% of its sensitivity

alloc = water_fill(spec.eigenvalues, budget=4.0 * spec.dim)
print(alloc.n_starved)          # directions that earn no bits
```

`consumer` is any callable from a vector to a **scalar margin**. A logit, a
ranking score, an attention weight. The probe never sees a label, a Jacobian,
or a hint about which directions matter.

It will refuse a consumer it cannot couple to. Point it at a top-k router and
it raises rather than reporting the confident zero that finite differencing
would produce, because selection reads the *order* of its logits and its
derivative vanishes almost everywhere. `readscope.regimes` carries the right
instruments for that case, the routing margin and the differential fraction,
inherited from turboquant-pro.

---

## What it does

The read operator `P_C = J^T G J` is built from a derivative, and derivatives
are measurable. Perturb the input, watch the output move, accumulate

```
S = E[ g g^T ],   g = grad_x C(x)
```

The top eigenvectors of `S` span the read subspace. That is the trace.

The spectrum of `S` is where a consumer's sensitivity sits across directions,
the same way a spectrum analyzer shows where a signal's energy sits across
frequency. Allocating a bit budget against that spectrum by reverse
water-filling is not an analogy to power allocation across frequency bins. It
is the same optimization, with a downstream task's sensitivity in place of a
signal's power.

## What it is for

Compression and quantization decide what to keep, and they almost always
decide by reconstruction error. On Llama-3.2-3B attention heads, two key
codecs tied on reconstruction to 7.5e-9 by construction still differed
downstream by a median relative-KL gap of 1.85. The consumer-relative
distortion `tr(P_C Σ_δ)` picked the worse arm on 16 of 16 heads.
Reconstruction, exactly tied, picked it on 2 of 16.

So: measure what the consumer reads, spend bits against that, and stop
accepting compression on a metric the consumer is blind to.

## You do not have to buy anything to use it

Nobody adopted the oscilloscope's ontology. You do not need a theory of what
voltage is to point a probe at a node and read a trace.

This is the same. The reading is a spectrum of output sensitivity. It means
what it means whether or not you ever use the word consumer, and whether or
not you find the theory it came out of interesting.

---

## Status: read SPEC.md first

**This instrument does not have a specification yet, and the calibration so
far has been unkind to it.** Read [SPEC.md](SPEC.md) before relying on
anything here.

**Bandwidth, measured.** The exact estimator resolves all 32 read directions
swept. The random-sketch estimator, which is the affordable one, resolves
**one or two**. The cause is identified, an isotropic bias of order
`||g||^2 / k` from squaring a noisy gradient, and a correction is the top
engineering item rather than a fundamental limit.

**The published accuracy figures came from a different probe.** Expressed on
a common statistic, the Llama result scores 0.596 at rank 16 where this
package's sketch scores 0.03 to 0.06. The 32-key probe in the source program
and the Gaussian sketch shipped here are not the same instrument, and until
that is closed the table below is provenance rather than specification.

Those three inherited measurements:

| Run | Substrate | Overlap | Chance | Verdict |
|---|---|---|---|---|
| GO-P-2026-011 | planted low-rank subspace | 0.936 | 0.059 | PASS 5/5 |
| GO-P-2026-020 | Llama-3.2-3B, 16 heads | 0.567 | 0.126 | **missed the 0.60 bar** |
| GO-P-2026-021 | Llama-3.2-3B, 16 heads | 0.647 | 0.126 | PASS, 32-key probe |

The 16 cells are layers {8, 16} × 8 KV heads of one 3B model. The two Llama
numbers bracket the sealed bar and differ only by probe design, which is
exactly why a single accuracy figure would be misleading. The earlier miss
stays on the record as its own negative.

An instrument is trusted because it is specified, not because it worked once.
A scope ships with bandwidth, sample rate, noise floor, and accuracy over
range. This one currently ships with a sample rate, a noise floor, and three
points on one accuracy axis. [SPEC.md](SPEC.md) names every field and marks
the empty ones.

### The known error term

Probe loading. A scope's input impedance draws current from the node, so what
you read is not quite what was there. Here, the probing distribution is not
the activation distribution the consumer meets in service, so the recovered
operator is the read operator averaged over the wrong measure.

That is not a flaw that invalidates the instrument. It is a characterizable
effect, and the first curve is measured.

| Probe loading, Jeffreys nats | Overlap |
|---:|---:|
| 0.89 | 0.992 |
| 10.73 | 0.975 |
| 25.25 | 0.852 |
| 50.50 | 0.491 |
| 91.64 | 0.437 |

Near-exact below about 11 nats, a knee between 25 and 50, then a shelf around
0.44 that stays well clear of the 0.125 chance floor, so a badly loaded probe
degrades rather than dissolving into noise. One synthetic consumer family,
five seeds, record `calibration/records/c1b-loading-curve.json`.

The first attempt at this curve failed four of five declared bars, and one of
its three defects was not a bug: **probe loading cannot degrade recovery at
all for a consumer whose read subspace is the same everywhere in the input
space.** That bounds where the error term applies, and it came out of the
instrument's own calibration rather than from thinking about it.

---

## Install

```bash
pip install -e ".[dev]"
```

Pure numpy. Torch is optional and only needed for the model-facing
calibration harnesses.

## Layout

```
readscope/       the instrument
  probe.py       blind recovery of S = E[g g^T]
  spectrum.py    the response spectrum, effective rank, energy rank
  allocate.py    reverse water-filling against the spectrum
  loading.py     probe loading, the calibration axis
  metrics.py     subspace overlap, always with its chance floor
  regimes.py     which consumers the probe may be attached to
  quotient.py    tangential/radial displacement split
calibration/     the program that has to produce a spec sheet
tests/           exact controls with closed forms
SPEC.md          the specification, mostly empty on purpose
CALIBRATION.md   what has to be measured, and what would sink it
```

## Provenance

The theory and the three measurements come from the observation-theory
program: [geometric-observation](https://github.com/ahb-sjsu/geometric-observation)
(the blind probe, the sealed preregistrations, the claims ledger) and
[turboquant-pro](https://github.com/ahb-sjsu/turboquant-pro) (the production
compression path that consumes a read operator).

This repository is the instrument pulled out of that program and specified on
its own terms, so it can be used by people with no interest in the program.

## License

MIT.
