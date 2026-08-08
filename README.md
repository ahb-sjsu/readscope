# readscope

**An oscilloscope and spectrum analyzer for model consumers.** Point it at a
consumer, get back what that consumer actually reads.

[![PyPI](https://img.shields.io/pypi/v/readscope)](https://pypi.org/project/readscope/)
[![Spec](https://img.shields.io/badge/spec-partial-orange)](SPEC.md)
[![Calibration](https://img.shields.io/badge/calibrations-C--0_to_C--10-blue)](CALIBRATION.md)
[![Tests](https://img.shields.io/badge/tests-73-green)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

```python
from readscope import blind_probe, spectrum_of, water_fill

# consumer: a callable from a vector to a scalar margin
probe = blind_probe(consumer, operating_points)
spec = spectrum_of(probe.S)

spec.effective_rank      # how many directions this consumer really reads
spec.energy_rank(0.9)    # how many carry 90% of its sensitivity

alloc = water_fill(spec.eigenvalues, budget=4.0 * spec.dim)
alloc.n_starved          # directions that earn no bits
```

The probe never sees a label, a Jacobian, or a hint about which directions
matter. It sees `consumer(x)` and reconstructs what `consumer` is sensitive
to.

---

## The one thing to know before using it

**Budget `2d` consumer calls per operating point, or expect the dominant
direction and nothing else.**

Recovery quality against the direction budget `k/d` is a cliff, not a slope:

| `k/d` | 0.25 | 0.5 | 0.75 | **1.0** | 1.25 | 1.5 |
|---|---:|---:|---:|---:|---:|---:|
| directions resolved | 1 | 2 | 2 | **16** | 16 | 16 |

Three quarters of the directions buys two of sixteen; the last quarter buys
the other fourteen. **This does not soften if you only want the top
directions** — the required budget is the same whether you grade at rank 16
or rank 1, because below full dimension the estimate is a projection onto a
random subspace and a projected operator's leading eigenvector is not the
operator's.

The default `mode="exact"` spends `2d` calls and is correct. `mode="lstsq"`
with `sketch_dim >= dim` is equivalent and is what the source program uses.
The cheaper `"sketch"` and `"ortho"` modes exist, are honestly characterized
in [SPEC.md](SPEC.md), and resolve one or two directions — reach for them
only if that is genuinely all you need.

If your consumer returns a **vector** rather than a scalar, use
`jacobian_probe`: each direction then returns `m` numbers instead of one, and
at half the direction budget it reaches roughly its output width in resolved
directions.

## It refuses consumers it cannot couple to

Point it at a top-k router and it raises rather than reporting the confident
zero that finite differencing would produce, because selection reads the
*order* of its logits and its derivative vanishes almost everywhere.
`readscope.regimes` carries the right instruments for that case — the routing
margin and the differential fraction — inherited from
[turboquant-pro](https://github.com/ahb-sjsu/turboquant-pro).

Recurrences are admitted but flagged: a pointwise Jacobian is well defined
and misleading, because decay error compounds along the sequence as
`1/(1-a)^2`.

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

## Status

An instrument is trusted because it is specified, not because it worked once.
A scope ships with bandwidth, sample rate, noise floor and accuracy over
range. Here is what this one ships with. [SPEC.md](SPEC.md) has every number
and every empty cell.

| Field | Status |
|---|---|
| Sample rate | Exact. `2d` calls per point, or `2k` sketched |
| Noise floor | Exact. Chance overlap `rank/dim`, reported with every reading |
| Minimum usable budget | **Measured. A cliff at `k = d`, rank-independent** |
| Accuracy | **124 real model cells at resolution 1.000 at `k/d = 1.25`** |
| Applicability | Bounded, and enforced in code |
| Temperature drift | Four families spread 1e-15; four scales spread 7e-16 |
| Input impedance | Axis measured and dimensionless. **Correction: no** |
| Linearity | Partial. Direction transfers across domains, magnitude does not |

**Accuracy, on real weights.** 48 attention head-cells across Llama-3.2-3B,
Qwen2.5-1.5B, Mistral-7B and Gemma-3-4B; 48 more across the Qwen2.5 ladder at
1.5B, 7B, 14B and 32B; 12 channel-cells of a Mamba-790m; 16 more matched to
the source program's protocol. Every one graded against an **exact closed-form
ground truth**, since the read subspace of an attention head with respect to a
key is spanned by its queries, and that of a recurrent state by its
decay-attenuated readout vectors. Resolution is 1.000 at `k/d = 1.25`
throughout, with across-family spread 1e-15 and across-scale spread 7e-16.

**Probe loading is the known error term** — the probing distribution is not
the activation distribution, so the recovered operator is averaged over the
wrong measure. The axis for it is now dimensionless and a fixed mismatch
reads within 1.3% across a sixteenfold range of dimension. **It is still not a
correction**, and after three attempts there is a reason rather than a gap:
loading degrades a reading only when the probing shift is *aligned* with the
read subspace, and in high dimension a random shift is not. Degradation is not
a function of loading alone, so a scalar correction is the wrong shape for it.

**What the source program's published 0.647 measures.** Under its own
settings this probe recovers the softmax-weighted Jacobian Gram at resolution
1.000000. That program grades against the *unweighted* query covariance, and
the number it reports is how much those two references differ. **Two
reasonable references for one head differ by about 0.3 in overlap**, so a
recovery number without its reference named is not interpretable.

### Honest negatives, kept

**Ten of seventeen calibrations failed**, and the failures moved the
specification further than the successes did. `CALIBRATION.md` keeps all of them
with the wrong claims left standing for the corrections to point at. Two are
worth reading before trusting any number here:

- **Two sweeps passed every bar for spurious reasons** — one because a
  correction saturated against its output clip, one because the quantity it
  measured was pinned by construction. Both are written up as failures. They
  produced the rule the later sweeps obey: *before believing a bar of the
  form "X predicts Y", require a prior bar that Y varies.*
- **One sweep nearly reported that real attention heads are degenerate.**
  They are not; the query set had been drawn from the key stream, which
  saturates the softmax. The shortcut was declared before the run and an
  anti-vacuity bar measured attention entropy, so the artifact announced
  itself instead of becoming a result.

---

## Install

```bash
pip install readscope
```

Pure numpy. Torch is optional and only needed by the model-facing extraction
scripts under `calibration/`, which ship in the repository rather than the
wheel. For development, `pip install -e ".[dev]"`.

**Using it with turboquant-pro?** `pip install tqp-readscope` registers these
probes as read-operator providers, so `tr(P_C Σ_δ)` becomes a number you can
gate a codec on.

## Layout

```
readscope/       the instrument
  probe.py       blind recovery of S = E[g g^T]; exact, lstsq, sketch, ortho
  spectrum.py    the response spectrum, effective rank, energy rank
  allocate.py    reverse water-filling against the spectrum
  metrics.py     subspace overlap and resolution, always with a chance floor
  loading.py     probe loading on a dimensionless axis
  regimes.py     which consumers the probe may be attached to
  quotient.py    tangential/radial displacement split
calibration/     the sweeps, their declared bars, and their records
tests/           exact controls with closed-form answers
SPEC.md          the specification, including the empty fields
CALIBRATION.md   what has been measured, what failed, and what would sink it
```

Calibration records are append-only JSON with the declaration, every cell,
every bar and a verdict computed from the data rather than written in
advance. A sweep that is superseded keeps its record and names what replaced
it.

## Provenance

The theory and the first measurements come from the observation-theory
program: [geometric-observation](https://github.com/ahb-sjsu/geometric-observation)
(the blind probe, the sealed preregistrations, the claims ledger) and
[turboquant-pro](https://github.com/ahb-sjsu/turboquant-pro) (the production
compression path that consumes a read operator, and the consumer-regime
analysis this package inherits).

This repository is the instrument pulled out of that program and specified on
its own terms, so it can be used by people with no interest in the program.

## License

MIT.
