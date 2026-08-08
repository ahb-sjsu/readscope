# C-12 declaration: does read-operator drift explain the long-generation gap?

Written and committed **before the run**. Bars are numerical and fixed here.
The verdict is computed from the data by `c12_longgen_drift.py` and is not
written in this file.

## What is being tested

turboquant-pro records a negative it does not explain: 4-bit KV
quantization degrades on very-long-generation tasks, and the gap grows with
generation length. The controlled sweep is Qwen2.5-7B-Instruct on LongBench
gov_report, n=200, varying only `max_new_tokens`:

| max_new_tokens | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|
| fp16 ROUGE-L | 14.81 | 20.57 | 27.31 | 31.8 |
| asym-NF4 | 14.56 | 16.38 | 17.71 | 18.1 |
| **gap** | **0.25** | **4.19** | **9.60** | **13.7** |

C-11c found that a head's read operator moves along the sequence and that
allocating against an early operator misprices a late consumer. That was
sixteen cells over 192 positions of one 3B model and was never run against a
degradation curve. This is that run.

## The structural fact that makes the test possible

In `tq_paper_lb_shard.py` the cache is quantized **once per layer**, on the
first update, over the settled prefill region `[0 : T-HOT]`. The guard is a
`_qdone` set, so every later decode step concatenates **fp16** tokens onto a
frozen quantized prefix. No new quantization error is injected at any point
during generation.

So the perturbation is a **single fixed** `Σ_δ` on the prefill keys, and the
degradation still grows with generation length. That is the whole experiment:
a constant error whose damage increases.

## Two hypotheses, and they are not the same

**H_compound** — the mechanism already recorded in
`benchmarks/kvquant_matrix/results_longgen.json`: each decode step reads the
quantized prefill with a small residual error, the model conditions on its own
drifting output, and the error compounds through the token stream.

**H_drift** — C-11c's mechanism: the query distribution moves with decode
position, so the read operator that a key is read by at step `t` is not the
one the codebook was fitted against. The same fixed `Σ_δ` becomes more visible
as `t` grows.

They are not exclusive. Both can contribute and the design apportions rather
than picking a winner.

## Stage A. Teacher forcing severs the feedback loop

For each document, generate `y*` greedily under fp16, then run **both** fp16
and asym-NF4 teacher-forced on `y*` and record per-position NLL. In the
teacher-forced condition both arms read an identical token prefix at every
step, so H_compound's channel is closed by construction. The only remaining
difference is the frozen key error.

`D_tf(t) = NLL_nf4a(t) - NLL_fp16(t)`

- **A0 — anti-vacuity, the thing to be predicted must vary.** Free-running
  ROUGE-L gap at 512 generated tokens must be **>= 5.0 points** on this
  sample. The recorded value is 13.7 at n=200; this sample is smaller, so the
  bar sits well below it. **If A0 fails the setup does not reproduce the
  phenomenon and every downstream number here is void.**
- **A1 — some of the effect survives feedback removal.** Median over
  documents of `mean D_tf over t in (384,512]` minus `mean D_tf over
  t in (0,128]` must be **>= 0.01 nats**, and a two-sided sign test across
  documents must give **p < 0.05**.
- **A2 — apportionment.** Report teacher-forced rise as a fraction of
  free-running rise. **No bar.** There is no principled threshold and
  inventing one would be theatre. It is reported as a measured quantity.

If A1 fails, H_drift contributes essentially nothing to this curve, C-11c is
demoted to a real effect that is not this mechanism, and that is the result.

## Stage B. The geometry, measured on the same forward passes

`Σ_δ` per (layer, kv-head) is the actual frozen prefill key error covariance
over settled positions. It is constant for the whole generation.

`M(t)` is the C-11c operator built from the **real decode queries** in window
`t`, attention-weighted against the settled prefill keys.

`d(t) = tr(M(t) Σ_δ) / tr(M(t))`, normalised so that growth in operator
magnitude alone cancels.

**The null is a random rotation.** `Σ_rot = Q Σ_δ Qᵀ` for random orthogonal
`Q`, five draws, identical spectrum and arbitrary orientation. It answers the
only question that matters here: does the *orientation* of the error relative
to the moving operator do any work, or would an error of the same shape
pointed anywhere have grown the same way? An isotropic null is not used
because it is constant by construction and therefore uninformative.

Relative growth `g = (d(late) - d(early)) / d(early)`.

- **B1 — the orientation does work.** Median over cells of `g_positional`
  minus median over cells of `g_rotated` must be **>= 0.05**.
- **B2 — the geometry predicts the damage.** Across documents, Spearman
  correlation between per-document `g` and per-document teacher-forced
  degradation rise must be **positive with p < 0.05**, and must exceed the
  rotated null's correlation.

B2 is the claim the user asked for and it is the one most likely to fail. At
n=40 documents Spearman needs |rho| > 0.31 for p < 0.05, so a real but modest
correlation may not clear it. **Underpowered-and-null will be reported as
underpowered, not as refuted.**

## Stage C. Exposure control

As generation grows, attention mass on the settled prefill region may fall,
which would predict the fixed error mattering **less** over time. Report
prefill attention mass against `t`. If it declines while `D_tf` grows, that
is evidence against a trivial exposure explanation and strengthens H_drift.
Descriptive, no bar.

## Stage D. Intervention, gated

Run **only if A0, A1 and B1 all pass.** Re-quantize the prefill allocating
bits against an operator estimated over prefill *and* decode queries, at
matched total bits, and re-run teacher-forced. C-11c predicts `D_tf` falls.

- **D1** — median over documents of the late-window `D_tf` must fall by
  **>= 10%** relative to the early-operator allocation, at matched bits.

## Fixed configuration

Matched to the run that produced the recorded curve
(`benchmarks/kvquant_matrix/tq_abl.sh`):

```
MODEL      Qwen/Qwen2.5-7B-Instruct    MODEL_KEY qwen2.5-7b-instruct
DATASET    gov_report (LongBench)      CHAT=1    MAXLEN=31500 middle-truncated
CODEBOOK   nf4a   KEY_BITS=4  VAL_BITS=4  GROUP=32
HOT=128    SINK=4  OUTLIER_FRAC=0.02   PREROPE=0
MAXGEN     512    greedy, num_beams=1, do_sample=False
```

Declared before the run:

```
N_DOCS         40        first 40 gov_report documents by index, no selection
CELLS          layers 4, 14, 24 x 4 kv-heads = 12 cells
EARLY_WINDOW   t in (0,128]
LATE_WINDOW    t in (384,512]
N_ROT_DRAWS    5
SEED           20260808
A0_ROUGE_BAR   5.0
A1_NATS_BAR    0.01        sign test p < 0.05
B1_GROWTH_BAR  0.05
B2_ALPHA       0.05
D1_DROP_BAR    0.10
```

## Anti-vacuity, from the standing rules

The rule earned in this programme is that before believing a bar of the form
"X predicts Y", require a prior bar that **Y varies**. A0 is that bar and it
gates everything. The second rule is that a bar on every cell is only
appropriate for a quantity the substrate has no freedom to vary; attention
geometry is not such a quantity, so B1 and B2 bar medians and correlations
across cells and documents, never each cell.
