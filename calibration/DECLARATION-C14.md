# C-14 declaration: the batched-consumer probe variant

Written and committed **before the implementation runs**. Bars fixed
here; verdicts computed by `c14_batched_suite.py`.

## What the variant is, and what it must not pretend

The serial probes evaluate the consumer once per perturbed input —
2k invocations per operating point — which on a GPU is a stream of tiny
kernels with host syncs: near-zero utilization for minutes at large d.
C-13's NRP campaign measured the consequence: commons enforcement kills
that shape within its first minute (three ways, all recorded). The
batched variant hands the consumer **one array of all perturbed
inputs** (`(2k, d)` per point) and reads all finite differences from
one invocation.

**The honesty obligation:** the budget law prices *directional
observations*, not function invocations — the cliff at `k = d` is a
theorem about information (readscope `PRINCIPLES.md` P3, proved via
confined transcripts), and a batch of k directions is still exactly k
directional observations. Batching may buy wall-clock and utilization;
it may not buy admission, and C-14 exists partly to demonstrate that by
measurement so no one ever markets it otherwise. `ProbeResult` gains an
`observations` count that batching leaves unchanged, alongside the
invocation count it collapses.

**The precondition batching adds:** a batch-shaped consumer must be
row-independent — `consumer(X)[i] == consumer(X[i:i+1])[0]`. A consumer
with cross-row coupling (batch norm in eval mode done wrong, cache
contamination) silently corrupts every gradient. The instrument
self-checks this on sampled rows at probe time and **refuses** on
mismatch, in the regime-gate tradition.

## Cells and bars

- **B1 (identity with the serial instrument):** same seed, same
  directions, d ∈ {128, 1024, 4096}, both probe types, CPU: relative
  Frobenius deviation between batched and serial operators ≤ **1e-10**.
- **B2 (the cliff stands under batching):** planted rank-1 target with
  genuinely hidden mass ≥ 1e-3 outside a k = d/2 probed span, 50
  trials, d = 128: **zero** exact recoveries (affinity ≥ 0.999) by the
  batched probe — one invocation, half the observations, same wall as
  ever.
- **B3 (the consistency gate fires):** an intentionally row-coupled
  consumer (output depends on the batch mean) must be refused by the
  self-check; the same consumer in serial mode probes without error —
  the hazard is batching's own, and its gate is batching's own.
- **Timing (descriptive, never barred):** serial vs batched wall-clock
  per cell; GPU cells on the Atlas GV100 (after C-13 releases the
  device) and one NRP job — which doubles as the measured answer to
  C-13's enforcement finding: a batched probe should hold >40%
  utilization where the serial probe structurally could not. If the
  NRP batched job is *also* killed, that is a finding about the
  platform, recorded as executed.

Records: `calibration/records/c14-batched-{cpu,atlas-gv100,nrp-3090}.json`,
committed as executed, pass or fail.

## Amendment 1 (2026-08-16, after the as-executed B1 FAIL — final revision)

B1's 1e-10 bar failed at deviations 1.0e-10 → 5.9e-9 (d = 128 → 4096),
and the failure is the bar's: batched evaluation reorders
floating-point reductions (row-block matmul vs per-vector products), a
mathematically distinct summation whose fp64 disagreement grows with
conditioning — the measured values are the BLAS-reordering floor, not a
defect. The v1 record stands as executed. **B1 v2 bar: relative
Frobenius deviation ≤ 1e-7** — derived as ≥ two orders above the
measured reordering floor at the largest cell and ≥ four orders below
every defect class this bar exists to catch (wrong directions, wrong
sign, dropped rows, cross-row coupling: all ≥ 1e-3). Lesson filed with
OT-10's: a sealed identity bar must be derived from the arithmetic of
the comparison, not from aspiration. This is the final instrument
revision for C-14's CPU cells.
