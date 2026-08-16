# C-13 / C-14 notes — backend equivalence, scaling, and the batched variant

**2026-08-16. Declarations sealed before every run; five amendments
(C-13) and one (C-14), each dated and each preceding the run it
governed; all records committed as executed, including two as-executed
FAILs whose bars were re-derived from the arithmetic and one regrade.**

## Devices and verdicts

| record | device | verdict |
|---|---|---|
| c13-backend-atlas-gv100 | Quadro GV100 (Volta, fp64 1:2) + host CPU | as-executed FAIL at the pre-amendment 1e-9 E1 bar; **PASS regraded** at the derived 1e-7 (amendment 5) |
| c13-backend-nrp-3090 | RTX 3090 (Ampere) | **PASS** |
| c13-backend-nrp-a10 | A10 (Ampere) | **PASS** |
| c13-backend-nrp-v100 | Tesla V100-SXM2-32GB (Volta) | **PASS** |
| (A6000) | — | unschedulable: no matching schedulable node under either product label; row closed |
| c14-batched-cpu | host CPU | as-executed B1 FAIL at 1e-10; **PASS** at the derived 1e-7 (amendment 1); B2 cliff-under-batching 0/50; B3 gate fires |
| c14-batched-atlas-gv100 / nrp-3090 | GV100, 3090 | descriptive timing; **the NRP batched job survived and completed** |

## The cross-silicon table (amendment 4, the co-author's cell)

Top-16 spectral deviation vs the Atlas GV100, same seeds, same
directions:

| d | V100 | 3090 | A10 |
|---|---|---|---|
| 128 | 3.5e-12 | 1.4e-11 | 1.4e-11 |
| 1024 | 8.5e-12 | 2.3e-11 | 2.3e-11 |
| 4096 | 7.8e-12 | 7.0e-12 | 7.0e-12 |
| 8192 | 4.0e-11 | 1.9e-11 | 1.9e-11 |

Agreement at 1e-12–4e-11 across two architectures, four buildings, and
independent driver stacks — two orders below the declared expectation.
The 3090 and A10 columns are **bit-identical to each other** (same
Ampere cuBLAS), so the residual 1e-11 texture is purely the
Volta↔Ampere arithmetic boundary. A reading is a reading, wherever the
GPU lives.

## Timing (descriptive throughout)

Atlas GV100, d = 8192: CPU serial blind probe 1641 s → GPU serial
174 s (9.4×) → GPU batched 139 s; CPU eigh 120 s → GPU 2.6 s (47×);
`top_spectrum` 2.6 s CPU / 0.10 s GPU — faster than eigh everywhere it
was measured, including 4.5–6.7× on the NRP Amperes. fp64 class splits
exactly as the silicon says it must: V100 probes ~200 s and eigh 3.3 s
vs Ampere ~440 s and 7–10 s.

## The platform chronicle (all as-executed, all with artifacts)

1. **Enforcement kills the serial workload class.** The measured
   utilization profile of the serial suite: median **0%**, mean 7.4%,
   >40% only 7% of samples (2157-sample GPU log, committed). Commons
   enforcement deleted serial jobs within ~31 s (v1) and killed the
   V100 attempt (v3); one 3090 and one A10 run survived on tolerant
   nodes — node roulette, not compliance. **C-14's batched variant is
   the compliant shape and its NRP job completed.**
2. **A 1Gi limit cannot start a CUDA-image container** with an
   in-memory fetch (OOMKilled at 3–5 s, three of three); stream to
   disk and budget 2Gi.
3. The cupy image ships without git; fetch with python's stdlib.
4. One node (uicnrp-fiona2) has a broken driver state (NVML mismatch
   at container create) and the scheduler bin-packs returning jobs
   straight back onto it — exclude rejected nodes by hostname.
5. Old `nautilus.io/gpu-type` labels are dead; V100 targeting needs
   node-affinity over the GFD product-name variants.

## The derived-tolerance lesson (two seals burned the same day)

C-14's B1 (1e-10) and C-13's E1 (1e-9) both demanded agreement that
float64 reduction reordering cannot supply: batched-vs-serial and
OpenBLAS-vs-cuBLAS comparisons measure the reordering floor (~6e-9
same-host, ~9e-9 cross-BLAS at d = 8192 through an ill-conditioned
pinv). Both as-executed FAILs stand; both bars were re-derived at
1e-7 — two orders above the measured floor, four below any defect
class the bars exist to catch. Filed with OT-10's lesson: **an
identity bar must be derived from the arithmetic of the comparison,
never from aspiration.**

## What the instrument now claims, and no more

Backend-generic readings verified across five devices at matched
seeds; the budget cliff untouched by batching (B2: 0/50 hidden
recoveries from single-invocation probes — the theorem prices
observations, and batching only re-plumbs them); the batched variant
as the shape that lives on utilization-policed commons hardware. GPU
timing is measured on Volta and Ampere; nothing is claimed for
hardware classes not in the table.
