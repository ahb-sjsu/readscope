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
impedance so the measurement can be corrected for it.

**Attempted in C-7 and C-7b, and it does not work yet.** See F-17 and F-18.
Loading remains a warning.

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

**Status: run, failed on a declared shortcut, corrected, passed. Extended by
C-5 across scale.** C-3 drew its
queries from the key stream and saturated the softmax; C-3b hooks `q_proj`
and passes all seven bars on three families. See F-9 and F-10.

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


### F-9, from C-3 and C-3b: the shortcut nearly produced a fake finding

C-3's query set came from the key stream, because a KV cache stores keys and
values and not queries. The self-match term saturated the softmax and median
attention entropy came out at 0.152 bits across 42 cells, with eleven
analytic operators below the graded rank and some at rank zero.

Written up carelessly that is "real attention heads are degenerate", which is
a striking claim and completely false. What saved it was that the shortcut
was declared in the sweep before it ran and the anti-vacuity bar measured
entropy explicitly, so the artifact announced itself instead of becoming a
result.

C-3b hooks `q_proj` and re-applies the model's own rotary embedding. Median
entropy clears one bit by declared bar, every operator reaches full rank, and
the probe recovers the analytic operator at resolution 1.000 on all 36 cells
of all three families with an across-family spread of 1e-15.

### F-10, from C-4: the published number is against a different reference

The source program grades against the unweighted query covariance
`Qset^T Qset / n_q`. The probe recovers the Jacobian Gram, the same queries
weighted by the softmax response. On real cells those two references agree at
median resolution 0.796, range 0.678 to 0.985, while the probe agrees with
its own target at 1.000 everywhere.

That lands inside the band set by the published figures, so R2 passed. R1
failed, because one cell has the references agreeing at 0.985 and the bar
asked for disagreement everywhere. **So reference choice is a large part of
the gap and demonstrably not all of it**, since the median disagreement sits
well above the published figures. The unmatched remainder is the query
capture, the GQA grouping, and the model and layer set.

An unweighted query covariance is a defensible definition. The lesson for a
datasheet is only that a recovery number is meaningless without saying what
it was graded against, because two reasonable references differ by about a
fifth of the range.


### F-11, the NRP question, answered by measurement rather than preference

The extraction was going to be bursted to NRP, on the reasoning that Atlas
cannot use its GPUs (torch 2.11.0+cu130 against a CUDA 12.8 driver). Before
submitting anything, `calibration/rightsize_extraction.py` profiled the
workload against the cluster's own bands. Record
`calibration/records/rightsize-extraction.json`.

| model | peak RSS | trough RSS | peak/trough | best wall | verdict |
|---|---:|---:|---:|---:|---|
| qwen2.5-1.5B | 8701 MiB | 276 MiB | 31.6 | 11.0 s | **IMPOSSIBLE** |
| gemma-3-4B | 24768 MiB | 272 MiB | 91.0 | 17.5 s | **IMPOSSIBLE** |
| mamba-790m | 3930 MiB | 273 MiB | 14.4 | 11.3 s | **IMPOSSIBLE** |

Requests equal limits on NRP, so the request must be at least the peak or the
pod OOMs, and at most the trough over 0.20 or the memory floor is violated.
Every one of these exceeds five times its trough, so **none of them has a
legal request**. The trough is the interpreter and torch import before any
weights load; the peak is the weights resident.

The wall times settle it independently. **These jobs run for 11 to 18
seconds.** A pod that spends minutes pulling an image and downloading weights
to do fifteen seconds of work is exactly the shape the enforcer averages over
five minutes and kills, and no ballast trick changes the fact that the job is
too small and too bursty for the platform.

**So the extraction stays on Atlas, and the GPU was never the constraint.**
The job is model-loading bound: a single 192-token forward pass is
milliseconds of compute against tens of seconds of load. A GPU pod would have
sat below the 40 percent utilisation floor by construction. batch-probe's
thermal probe put the safe thread count on this box at 7, and the wall-time
sweep showed the knee at 8 threads, so nothing here wanted more machine than
it had.

What would suit NRP is a job that holds high memory and high CPU for many
minutes. This is not that, and the right-sizer said so before a single pod
was submitted, which is the entire point of running it first.


### F-12, from C-3c: the cliff is the probe's, not attention's

A selective SSM gives a consumer that is not attention and still has an exact
closed form, because the read subspace of a recurrent state is spanned by its
readout vectors attenuated by the accumulated decay. On twelve channel-cells
of a real Mamba-790m the probe recovers that operator at resolution 1.000 at
`k/d = 1.25`.

At `k/d = 0.5` resolution runs from -0.32 to 0.10, at or below chance. On a
16-dimensional state, half the directions buys nothing. **The budget cliff
therefore belongs to the estimator and not to softmax**, which is what a
non-attention consumer was added to decide.

Measured alongside: Mamba channel memory runs 1.8 to 82 steps, median 7.9,
with most channels taking essentially all of their read operator's trace from
the nearest quarter of the horizon and a few spreading across all of it.

### F-13, two more of my own bars declared wrongly

N3 asserted a universal about the near-quarter trace fraction. Per-channel
decay is precisely the quantity an SSM varies, so a universal was the wrong
form; the distribution should be reported as C-3b reports family spread.

N5 put anti-vacuity on the decay rather than on the operator. Two channels
have decayed to 1e-18 or below by the horizon and are nonetheless perfectly
measurable, with operator ranks of 10 to 16 against a graded 8 and resolution
1.000. A two-step-memory channel is a real channel.

Both corrected bars hold when evaluated against the record C-3c already
produced. **No re-run was performed**, because nothing measured would change
and only two labels would move. A PASS obtained that way would be worth less
than saying so.


### F-14, from C-5: recovery is scale-invariant, and heads read about two directions

Qwen2.5 holds head_dim at 128 from 1.5B to 32B while depth, head count and
grouping all change, so the ladder varies substrate with geometry fixed.
Across 48 cells the mean resolution at `k/d = 1.25` is 1.0000 at every scale,
with an across-scale spread of **6.7e-16**. Twenty times the parameters,
no change in what the probe needs. The budget law has no scale term.

The reported column with no bar on it turned out to matter more. **Every
cell's analytic read operator has exact rank 24 and effective rank between
1.20 and 3.29, median 1.82.** A real attention head spans two dozen
directions and concentrates its sensitivity in about two, at every scale
measured.

That is a reframing of the bandwidth result and not a contradiction of it.
The cliff is about recovering a rank-16 subspace. This says most of those
sixteen carry little. **The next experiment is obvious and not yet run:
grade a rank-2 subspace at sub-dimensional budget.** If a cheap probe
recovers the directions that actually carry the mass, the pessimistic
specification softens for the common case, and if it does not, the cliff
stands unqualified.

Recording it as an observation rather than folding it into the spec is the
point. It was measured without a bar, so it has not been tested, and the
specification does not move until it has been.


### F-15, from C-6: the cliff is rank-independent, and my hypothesis was wrong

C-5's unbarred effective-rank observation suggested the budget cliff might
only apply to recovering a full rank-16 subspace, and that a cheap probe
might suffice for the one or two directions carrying the sensitivity. That
hypothesis was post hoc, which is why C-6 was declared with a bar that could
kill it.

It killed it. Required budget is 1.0 at every graded rank from 1 to 16. Rank
one recovers better than rank sixteen at intermediate budgets, 0.646 against
0.366 at half the dimension, and never enough to clear 0.90 below `k = d`.

**The specification hardens: pay `k >= d` regardless of how many directions
you need.** Below full dimension the estimate is a projection onto a random
subspace, and a projected operator's leading eigenvector is not the
operator's unless that subspace contains it. There is no low-rank shortcut.

The effective-rank observation remains true and is now known to be
operationally worthless, which is a better place to be than believing it
might help.

### F-16, the fourth universal declared where a distribution belonged

S5 required per-cell entropy above 1.0 bits, carrying over a threshold C-3b
applied to the median across cells. One cell of 48 sits at 0.339. Recomputed
on the 47 that clear it, the surface is unchanged to three decimals, so
nothing about S2 depends on it.

Four times now: C-3c's N3 and N5, and this. The pattern is specific enough to
state as a rule. **A bar on every cell is only appropriate for a quantity the
substrate has no freedom to vary.** Entropy, decay rate and trace
concentration are all properties real models differ in by design, and they
belong in reported distributions with bars on their aggregates.


### F-17, from C-7: the instrument reported 1e11 nats without complaining

C-7 estimated loading from the 16 points it probed at, in a 128-dimensional
head. A covariance fitted from n <= d is rank deficient, the ridge dominates
its log determinant, and the divergence came back around 1e11 nats. The sweep
was **invalid rather than failed** and no bar was tested. It was stopped once
the readings were seen to be degenerate, so it produced no record; the script
remains in `calibration/` as the design that was wrong.

`probe_loading` now raises when n <= d and warns below five samples per
dimension. The underlying mistake was conceptual: loading is a property of
two distributions, not of the sample a probe happens to visit, and tying its
estimate to the probe budget confused two unrelated concerns.

### F-18, from C-7b: a bar passed because the correction saturated

C-7b reported a 92.1 percent error reduction and PASS on all six bars,
including the falsifiable T2. It is an artifact.

The correction was fitted over 0.89 to 91.64 nats. Only 15 percent of C-7b's
readings fell inside that range and the largest was 1.9e12 nats. Outside the
range the correction clamps to its endpoint attenuation of 0.437, so every
reading was divided by one constant, and **202 of 240 corrected values came
out at exactly 1.0** because the output clip pinned them onto the target the
error was measured against.

**This is the most dangerous failure in the programme so far, because it
passed.** Every other declaration error produced a FAIL that demanded
attention. This one produced a success that would have gone into the
specification.

Two repairs. `LoadingCorrection.correct` refuses to extrapolate by default,
so a reading outside the fitted domain is an error rather than a clamped
guess. And the missing bar is now named: **a correction sweep must require
that its readings lie inside the correction's fitted domain**, and report the
fraction that do.

### F-19, why the correction cannot transfer as currently defined

The loading axis is not dimensionless. Jeffreys divergence between fitted
Gaussians grows with dimension, so the same qualitative mismatch reads 0.89
to 92 nats at the 24-dimensional synthetic consumer, thousands at head_dim
128, and billions at head_dim 256.

The consumer family was never the obstacle, which is what C-7 was designed to
test. **The units are.** Normalising loading so the same physical mismatch
reads the same number at any dimension is a change to the axis rather than to
the curve, and it has to happen before any correction fitted at one dimension
can be applied at another.


### F-20, from C-8: the axis is fixed and it is decisive

Normalising loading as ``max(0, jeffreys - null_floor(n_p, n_a, d)) / d``
makes it comparable across dimensions. A fixed per-direction mismatch reads
1.140, 1.135, 1.125, 1.129, 1.131 at dimensions 16 through 256, a relative
spread of **1.3 percent** where raw Jeffreys spreads **242 percent** over the
same configurations. Independent samples of one law read exactly zero after
the null correction.

The null floor is distribution free, so it is a property of
``(n_probe, n_activation, dim)`` alone and is simulated once and cached. It is
also large where it matters: at dimension 256 with two samples per dimension
it is 260 nats, which is the number C-7 was unknowingly reading as signal.

### F-21, the second vacuous pass in two sweeps, and the rule that catches both

C-8's D5 reported a correction transferring with mean absolute error 0.0000.
Every reading in that half was exactly 1.000, so the fitted attenuation was
the identity and no error was possible. The probe never degraded, so there was
nothing to predict.

F-1 already said why. **Probe loading cannot degrade subspace recovery when
the consumer's read subspace does not vary across the input space**, and the
gated consumer's gradients always lie in the span of its three defining
vectors, which is exactly the rank C-8 graded at. Using the exact
least-squares estimator removed the last source of variation. C-1b escaped
this only through a noisier estimator.

Two sweeps in a row have now passed a bar for a reason unrelated to the claim:
C-7b through saturation, C-8 through a pinned quantity. Both were caught the
same way and only that way, so it is now a standing rule rather than a habit:

**Before believing a bar of the form "X predicts Y", require a prior bar that
Y varies.** C-1b had one, as its L3 separation bar. C-7b and C-8 dropped it
and both produced a confident number out of nothing.


### F-22, from C-9: the first valid correction test, and it says no

C-9 is the first of three attempts that measured anything. E0, the separation
bar the rule from F-21 demands, passed with +0.237 in the fit family, so there
was real degradation to predict. The dimensionless axis put **100 percent** of
readings inside the fitted domain against C-7b's 15 percent.

Both repairs worked and the correction still failed. E2: separation is +0.146
at dimension 16, +0.237 at 32, **-0.005 at 64 and -0.076 at 128**. E4: error
fell from 0.147 to 0.095, a 36 percent reduction against a 50 percent bar,
with 67 percent of outputs still pinned.

**The mechanism tells you when probe loading matters at all.** With a random
planted basis, every read direction sees nearly the same variance from an
anisotropic probing distribution once the dimension is large, so concentration
of measure removes the differential saturation that makes loading bite.
Loading damages a reading when the probing shift is *aligned* with the read
subspace. A random shift in high dimension is not aligned with anything.

So degradation is not a function of loading alone, and no scalar axis can
carry it however well normalised. **A scalar correction is the wrong shape for
this effect.** That is a more useful conclusion than a correction would have
been, and it took three attempts and two vacuous passes to reach honestly.

Kept from the attempts: the dimensionless axis, the refusal to extrapolate,
the refusal to read a rank-deficient sample, and the separation bar as a
standing requirement. Loading stays a warning.


### F-23, from C-10: C-4's remainder is closed, and the query set was it

Matching all three unmatched factors to `gateB_llama_rematch.py` moved the
median weighted-against-unweighted overlap from **0.821 to 0.703** against a
published 0.647, closing **68 percent** of the distance. The cause is the
query set: the source uses every query in a key-value head's group across the
sequence, 576 vectors, where C-4 sampled 24. That takes the unweighted
reference from rank 24 with a well separated top-16 to full rank 128 whose
top-16 is far less determined, so the two references disagree more.

Under the source's own probe settings the instrument recovers the weighted
operator at resolution **1.000000 on all 16 cells**, so nothing about the
probe is implicated in the published number.

M3, the closure identity, is **arithmetic given M1** and was declared as such
before the run rather than presented as a discovery. If the probe is the
weighted operator, its overlap with any third object is that operator's
overlap. The bar could have failed only by the probe not being exact, which is
what made it worth stating.

**Left uncontrolled and stated as such:** the input text and the probe-key
draw. The residual 0.056 is the scale those explain. This sweep claims the
magnitude and the mechanism, not the digit.

The datasheet lesson is now quantified rather than qualitative: **two
reasonable references for one attention head differ by about 0.3 in overlap.**
A recovery number without its reference named is not interpretable.
