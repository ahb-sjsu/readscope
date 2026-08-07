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

**The headline calibration, and the one the framing demands.**

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
