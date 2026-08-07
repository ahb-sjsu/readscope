# The calibration program

The instrument's accuracy specification is currently three points on one
axis. This document says what has to be measured to turn that into a spec
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

**The headline calibration. Run once, failed, corrected, and now measured. See SPEC.md for the curve and the findings below for what the failure was worth.**

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

None of these has been ruled out. That is why the specification is empty.


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
