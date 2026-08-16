# C-13 declaration: backend equivalence and scaling

Written and committed **before the runs**. Bars are numerical and fixed
here; verdicts are computed by `c13_backend_suite.py` per device and
never written in advance.

## What is being tested

The 2026-08-16 backend-generic core claims: (1) CuPy-backed probes and
spectra are **the same readings** as numpy's — same seed, same
directions, same operator, to linear-algebra rounding; (2) `top_spectrum`
agrees with the full decomposition on the directions it computes; (3)
the GPU path buys wall-clock at large `d`. Claims (1) and (2) carry
bars; (3) is measured and reported, never barred — a speedup is a fact
about hardware, not about the instrument's correctness.

The budget law is out of scope by construction: consumer-call counts are
identical across backends (directions are drawn once, in numpy), so no
cell here can even appear to move the cliff.

## Cells

Per device (CPU-numpy reference runs on the same host as its GPU
comparator, so BLAS-vs-BLAS is the only variable):

- d ∈ {128, 1024, 4096, 8192}; operating points n = 4; seed 20260816.
- Probes: `blind_probe` (lstsq, sketch_dim = d) on a planted rank-8
  tanh margin consumer; `jacobian_probe` (n_directions = d) on its
  8-output vector form. Consumers are written namespace-agnostically so
  the identical closure serves both backends.
- Spectra: `spectrum_of` on each recovered operator; `top_spectrum`
  (r = 16, defaults) on the same.
- CPU reference at d = 8192 runs on Atlas only (48-thread host); the
  NRP job caps its CPU reference at d ≤ 4096 to keep the pod short and
  GPU-dominated, and its d = 8192 GPU cells are graded against bars
  (2) and self-consistency, with cross-backend identity at 8192 carried
  by the Atlas record.

## Bars (sealed)

- **E1 (probe identity):** for every (d, probe) cell with both backends
  on-host: relative Frobenius deviation of recovered operators
  ≤ **1e-9**.
- **E2 (spectrum identity):** eigenvalues of `spectrum_of` across
  backends: relative deviation ≤ **1e-9** (matched cells as E1).
- **E3 (top_spectrum fidelity, per backend):** top-16 eigenvalues match
  the full decomposition to rel **1e-8**; per-direction eigenvector
  overlap ≥ 1 − 1e-8; `effective_rank` exact-match to `spectrum_of`'s
  within rel 1e-10 (it is computed from invariants and must not depend
  on r).
- **Timing:** wall-clock per cell, reported per device. No bar. The
  GV100 (strong fp64) and consumer GPUs (weak fp64) are expected to
  differ qualitatively; that expectation is written here so nobody
  reads a 3090 fp64 eigh as a defect.

## Devices declared

- Atlas workstation, Quadro GV100 (`CUDA_VISIBLE_DEVICES=1`), cupy
  13.6.x (sm_70 support), numpy reference on the same host.
- NRP Nautilus, one RTX 3090 (swarm-class batch job via the polite
  nats-bursting path, right-sized per platform policy; the job clones
  this public repo, installs nothing beyond it, computes, terminates).

Records: `calibration/records/c13-backend-{atlas,nrp3090}.json`; both
committed as executed, pass or fail, with a NOTES file and the SPEC
table updated only to what the verdicts support.

## Amendment 1 (2026-08-16, before the NRP submission)

Two instrument changes forced by platform facts, neither touching bars:

1. **Submission path.** The nats-bursting descriptor has no
   nodeSelector field (documented code gap in the ops runbook), so the
   declared RTX 3090 pinning cannot ride the controller. The NRP job
   goes as a policy-shaped manifest instead: requests == limits,
   ephemeral-storage declared, backoffLimit 0, TTL-after-finished,
   fresh name verified by creationTimestamp, deleted after collection.
2. **NRP CPU-reference cap 4096 → 1024.** A 2-CPU pod running a
   d = 4096 float64 pinv holds the GPU idle for minutes — the exact
   under-utilization pattern platform enforcement kills. The pod's CPU
   reference stops at d = 1024; E1/E2 cross-backend identity at 4096
   and 8192 is carried by the Atlas record, and the pod's large-d GPU
   cells are graded on E3 and reported for timing.

## Amendment 2 (2026-08-16, before the additional submissions)

Device set extended for GPU variety, bars unchanged: **NVIDIA A10**
(Ampere, fp64 1:64), **Tesla V100** (Volta sm_70, strong fp64 — the
same architecture class as the Atlas GV100 reference, making it the
cross-substrate same-silicon cell), and **RTX A6000** (Ampere). Same
manifest shape and pod sizing as the 3090 job; per-device records
`c13-backend-nrp-{a10,v100,a6000}.json`, each graded on E3 + the
d ≤ 1024 E1/E2 cells, timing reported. Submissions staggered to hold
at most **three** concurrent C-13 pods against the namespace's
four-heavy-pod allowance — variety is not an excuse to fill the
commons; the A6000 waits for the 3090 to finish.

## Amendment 3 (2026-08-16, after the A10 enforcement kill)

The first NRP submission was killed 31 s after container start — Job
object deleted, `Killing` the only event — the platform's resource-
floor enforcement acting exactly as the program's own policy memory
describes: during its opening seconds (tarball fetch + small CPU
cells) the pod under-used its 2-CPU/4Gi requests while holding an
idle GPU. Recorded as a finding, not fought. The NRP arm is reshaped,
bars untouched:

- **GPU-only cells on NRP** (`--gpu-only`): no CPU-reference cells in
  the pod at all — E1/E2 cross-backend identity is carried entirely by
  the Atlas record (completing amendment 2's direction); NRP grades E3
  and reports timing.
- **Descending dimension order** (`--desc`): d = 8192 runs first, so
  the heaviest kernels hit the GPU within seconds of start.
- **No host retention of operators** in GPU-only mode (nothing to
  compare on-pod), collapsing host memory.
- **Right-sized to the floors:** cpu 1, memory 1Gi, unchanged
  ephemeral. Utilization floors (20% cpu, 20% mem) are now satisfiable
  through every phase of the pod's life, not just its peak.

If the reshaped jobs are still killed, the recorded conclusion is the
workload-class one — serial consumer-call probing is kernel-launch
bound and structurally cannot hold a >40% GPU floor — and the NRP
device rows close as `enforcement-incompatible`, with Atlas carrying
C-13. Either outcome is publishable; only one of them was worth a
second submission.
