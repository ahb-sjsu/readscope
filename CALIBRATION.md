# The calibration program

The instrument's accuracy specification is currently three real-model
points and one loading curve on a synthetic consumer. This document says what
has to be measured to turn that into a spec
sheet, in the order that makes each step falsifiable before the next one
depends on it.

The discipline is the campaign's: every sweep is declared with its bars
before it runs, records are append-only and hashed, honest negatives stay in
the record, and a verdict is computed from data rather than written in
advance. Calibrating your own instrument against known ground truth is the
same move as a sealed ledger, pointed at the measuring device instead of the
claim.

---

## C-0, the instrument layer

Exact controls with closed forms, all of which must pass before any sweep is
believed.

A consumer whose read operator is known exactly, where the probe must recover
it to machine precision at full budget. A consumer that reads a single
direction, whose recovered effective rank must be one. A consumer that reads
isotropically, whose recovered effective rank must be the ambient dimension.
The cosine-ranking consumer, whose margin gradient has the closed form in
`readscope.probe.retrieval_margin_gradient`, so the probe can be graded
without a planted subspace. Water-filling against a flat spectrum, which must
reproduce uniform allocation. Water-filling against a rank-one spectrum,
which must starve every other direction.

Nothing here is a claim. It is the check that the instrument is an instrument.

---

## C-1, the loading curve

**Status: run, failed, corrected, measured.** C-1 failed four of five
bars and C-1b passes all five. The curve is in `SPEC.md` and what the failure
was worth is in the findings at the bottom of this file.

Hold the consumer, the rank, the dimension and the probe budget fixed. Sweep
the probing distribution from itself toward the activation distribution with
`interpolate_distribution`, and at each step record the loading reading and
the recovered-subspace overlap.

The result is overlap as a function of loading, which is a calibration curve.
It answers the question a user of the instrument will actually ask, which is
whether a reading taken with convenient probe points means anything about
behavior on real traffic.

Declared in advance, because the outcome is not known. Overlap should fall
monotonically as loading rises. If it does not, either the loading measure is
not capturing what degrades recovery, or recovery is insensitive to loading
over the swept range, and both would be results. If overlap falls to chance
at a loading value that real deployments routinely exceed, the instrument
needs a correction factor before it is usable, and that too is a result worth
having early.

**A correction is the goal, not just a warning.** A scope specifies input
impedance so the measurement can be corrected for it. If the curve is
well-behaved, the same is available here: given a measured loading, deflate
the reported overlap by the curve. That is what would let the instrument be
trusted off its calibration points.

---

## C-2, the bandwidth sweep

**Status: run, failed on a declaration error, corrected, measured.** C-2
barred monotonicity on raw overlap while the chance floor moves with rank,
and its bandwidth criterion was unsatisfiable past rank `dim / 2` by
construction. C-2b bars the resolution instead and defines bandwidth as a
prefix. See F-3 below for what it found.

Overlap against rank at fixed dimension, and against dimension at fixed rank,
until recovery falls to the chance floor. The rank at which it crosses is the
instrument's bandwidth, and it is meaningless to quote an accuracy without
one.

This has to be run per estimator, since the exact estimator and the sketch
have different noise, and the sketch's `k` is the sample-rate knob that trades
cost against bandwidth.

---

## C-3, the architecture spread

Repeat C-1 and C-2 across architectures and rank profiles. Attention heads
across depth in more than one model family, at more than one scale, plus at
least one non-transformer consumer so the specification is not secretly a
statement about attention.

The current evidence base is 16 cells of one 3B model. Until this exists,
every number in `SPEC.md` is a reading and not a specification, and the
document says so.

---

## C-4, the magnitude question

GO-P-2026-041 found that blind direction transfers across domains and blind
magnitude does not. Across rates the magnitude tracks at Pearson 0.995 and
across domains the scale constant does not.

Until that constant is understood, the instrument reports direction across
domains and magnitude only within one. C-4 is the attempt to find the
constant, and it is the one part of this program that may simply fail. If it
does, the specification says so and the instrument is still useful, in the
way a scope with a specified but uncorrectable limitation is still useful.

---

## What would make this instrument not worth specifying

Recorded now so the program can end honestly.

If C-1 shows overlap collapsing to chance at loading values typical of any
realistic gap between probe points and live traffic, and no correction
recovers it, then the instrument only works when you already have the
activation distribution, which is most of what you were trying to learn.

If C-3 shows the recovered spectrum is essentially the same across consumers
that behave very differently, the reading is measuring the substrate rather
than the consumer and the instrument is a dressed-up covariance estimator.

If C-2 shows bandwidth of one or two directions at realistic budgets, the
instrument reports the dominant direction and nothing else, which is worth
knowing and is much less than what is currently implied.

C-1b rules out the first one for its own consumer family, which is one
configuration and not a class. The other two are open. That is why most of
the specification is still empty.


---

## Findings so far

### F-1, from C-1's failure: loading needs a moving read subspace

C-1 failed four of five bars. Two causes were harness bugs, a consumer that
collapsed its rank-four subspace to rank one and a saturating nonlinearity
whose derivative vanished, both caught by the anti-vacuity bar reporting
operator ranks of 0, 5 and 32 against a declared 4.

The third was not a bug. **Probe loading cannot degrade subspace recovery for
a consumer whose read subspace is the same everywhere in the input space.**
The recovered subspace is the Jacobian's row space no matter where the probe
points are drawn from. The sweep would have measured nothing even with the
first two defects repaired.

That bounds the error term. Loading matters for consumers whose local
sensitivity rotates with position, and the amount it matters is what the
curve measures. Any future calibration that reports no loading effect must
first show its consumer could have exhibited one.

### F-2, from C-1b: the curve has a knee and a shelf

Recovery is flat and near-exact below about 11 nats of Jeffreys divergence,
loses fifteen points by 25 nats, halves by 50, and then flattens onto a shelf
around 0.44 rather than falling to the 0.125 chance floor. A badly loaded
probe returns a degraded reading, not noise, which is a more useful failure
mode than the alternative and should not be assumed to hold on real models.

The seed spread tightens as loading falls, from 0.045 at the worst point to
0.003 at the best. Loading costs precision as well as accuracy.


### F-3, from C-2b: the shipped sketch has a bandwidth of one or two

The exact estimator resolves the full rank sweep. The random-sketch
estimator, which is the affordable one and therefore the one that would ever
be pointed at a frontier model, holds resolution above 0.5 only at rank one
at `k=16` and rank two at `k=64`.

This is one of the three outcomes listed above as grounds for concluding the
instrument is not worth specifying. It is not fatal, because the cause is
identified and is correctable rather than fundamental: the two-point sketch
is unbiased for the gradient but noisy, and squaring a noisy gradient adds an
approximately isotropic term of order `||g||^2 / k` to the recovered
operator. It leaves the leading eigenvector alone and buries everything else.

**That proposed fix was wrong, and F-5 records why.** It is left standing
here as written so the correction has something to point at.

### F-4, the gap between this package and its own provenance

Expressed as resolution, the published Llama figure is 0.596 at what the
chance value implies is rank 16. This package's sketch scores 0.03 to 0.06 at
rank 16. The 32-key probe that produced the published number and the Gaussian
sketch shipped here are not the same instrument, and the accuracy table in
`SPEC.md` therefore describes a probe design this package does not yet
implement.

That gap was invisible until the bandwidth sweep put both on one axis, which
is the entire argument for building the datasheet before building the
adoption story.


### F-5, from C-2c and C-2d: debiasing cannot buy bandwidth, and I said it would

F-3 named the sketch's isotropic inflation as the cause of its one-to-two
direction bandwidth and proposed subtracting it as the immediate fix. The
first half was right and the second half was wrong.

The bias is exact, `E[ghat ghat^T] = (1 + 1/k) g g^T + (||g||^2 / k) I`, and
`debias_sketch` inverts it in closed form. Being a multiple of the identity,
it shifts every eigenvalue equally and rotates nothing, so it cannot change a
recovered subspace by construction. E1 was declared to test exactly that and
it held to 4e-16.

What it does fix is the spectrum, mean trace error falling from 6.21 to 0.068
at `k=8`. That is real value for bit allocation and none at all for
bandwidth.

The cause of the bandwidth limit is variance, not bias.

### F-6, the orthonormal estimator helps conditionally, and the bar that said otherwise failed

An orthonormal frame removes the magnitude noise of an iid Gaussian one and
is exact at `k = d`. It raises bandwidth from one to two at `k/d = 0.5` and
does not help at `k/d = 0.25`, where it wins at some ranks and loses at
others.

E3 asked that it never be worse anywhere. It failed at three seeds. Rather
than weaken the bar, C-2d re-ran the identical sweep at ten seeds, generated
mechanically from C-2c so that only the seed list, the docstring and the
output path could differ. **E3 failed again**, which settles it: ortho is
genuinely worse at some ranks at low sampling ratio, and the estimator
recommendation has to be conditional rather than a blanket preference.

Generating the replication rather than editing the original is the only way
to make "no bar was moved after seeing the numbers" checkable by a reader
instead of merely asserted.


### F-7, from C-2e: the budget law is a cliff, and my "affordable estimator" premise was wrong

The source program's probe runs 160 directions in a 128-dimensional head,
`k/d = 1.25`. It never relied on a sub-dimensional estimator, and the idea
that a cheap sketch was the operating point was mine rather than the
program's.

Measured, bandwidth against budget goes 1, 2, 2, 16, 16, 16 as `k/d` runs
from 0.25 to 1.5. **Three quarters of the directions buys two directions of
sixteen and the last quarter buys the rest.** There is no graceful
degradation, so there is no partial budget worth spending, and the
specification says plainly that a scalar-margin probe costs `2d` calls per
operating point.

### F-8, the vector discount is real, partial, and the shape of it is only a hint

A vector-valued consumer returns `m` numbers per direction at the same call
cost. At half the direction budget, bandwidth goes 2, 2, 4, 8 for `m` of 1,
2, 4, 8. Real, useful, and short of the exact estimator's 16, so S5 failed
while S4 passed.

Bandwidth tracking `m` in those four rows is suggestive and is not
established. One ambient dimension, one ratio, four values of `m`. Writing it
down as a law would be the kind of overreach this file exists to prevent.

### What C-3 has to answer now

At `k/d >= 1` this package recovers a **planted** subspace at resolution
1.000. The source program recovered a **real attention head** at 0.596 at
`k/d = 1.25`. The estimator is no longer the explanation for that gap.

So the remaining question is what a real read subspace does that a planted
one does not, and that is C-3's whole job. Until it runs, the instrument's
accuracy on anything real rests on sixteen head-cells of one 3B model.
