# A Reading Is a Reading

**Calibrating a blind measurement instrument across five GPUs, two
clusters, and one utilization enforcer**

*Andrew H. Bond · San José State University · 2026-08-16*

[readscope](https://github.com/ahb-sjsu/readscope) measures which
directions of a vector a downstream computation actually reads — its
*read operator* — from function calls alone. Version 0.1 of the
instrument was pure numpy. When we made its core backend-generic (GPU
linear algebra when the data lives on a GPU), we owed the claim a
calibration, not an assertion: **does a reading depend on where it was
computed?** This is the report of that calibration — C-13 and C-14 in
the instrument's public record — including the parts that failed, the
tolerances we had to re-derive from the arithmetic, and a finding about
shared-cluster GPU enforcement that the instrument's own theory
predicts.

## 1. The question, and why it needs a calibration

An oscilloscope is trusted because it is specified: bandwidth, noise
floor, drift. readscope carries the same discipline — its README leads
with a spec table, and every number in that table traces to a sealed,
pre-registered calibration whose bars were committed before the run.
The backend-generic core adds a new row-in-waiting: if the same probe
of the same consumer runs on a laptop CPU, a Volta workstation card,
and a rented Ampere across the country, are those the *same
measurement*?

The design makes the strongest version of the question askable:
**every random probe direction is drawn with numpy, in the same order,
regardless of backend**, and only then transferred to the device. Two
runs of the same seed therefore probe identical directions on any
hardware; whatever disagreement remains is arithmetic, not sampling.

## 2. Design

Sealed in `calibration/DECLARATION-C13.md` before any run (with five
dated amendments, each preceding the run it governed):

- **Cells:** d ∈ {128, 1024, 4096, 8192}; a planted rank-8 tanh
  consumer (scalar and vector forms); `blind_probe` (lstsq, k = d) and
  `jacobian_probe` (k = d); `spectrum_of` and `top_spectrum` on each
  recovered operator.
- **Devices:** Atlas workstation (Quadro GV100, Volta, strong fp64,
  plus its 48-thread host CPU as the on-host reference) and three
  NRP Nautilus commons GPUs — RTX 3090 and A10 (Ampere) and a Tesla
  V100 (Volta), the last added by co-author suggestion as the
  **same-silicon cross-host cell**: same architecture class as the
  GV100 on an entirely different host, driver, and BLAS stack, so its
  agreement isolates backend correctness from residence.
- **Bars:** E1/E2 — cross-backend operator and spectrum identity on
  matched cells; E3 — `top_spectrum` fidelity against the full
  decomposition per backend; timing measured, never barred.

C-14, sealed separately, calibrates the **batched-consumer variant**
(one invocation evaluates a whole batch of perturbed inputs): identity
with the serial instrument, a re-measurement of the budget cliff under
batching, and a self-check that refuses consumers with cross-row
coupling.

## 3. Results: the cross-silicon table

Top-16 spectral deviation of the recovered operator versus the Atlas
GV100, matched seeds, matched directions:

| d | Tesla V100 | RTX 3090 | A10 |
|---|---|---|---|
| 128 | 3.5e-12 | 1.4e-11 | 1.4e-11 |
| 1024 | 8.5e-12 | 2.3e-11 | 2.3e-11 |
| 4096 | 7.8e-12 | 7.0e-12 | 7.0e-12 |
| 8192 | 4.0e-11 | 1.9e-11 | 1.9e-11 |

Agreement at **10⁻¹²–10⁻¹¹ relative** across two GPU architectures,
four physical sites, and independent driver stacks. Two details are
worth savoring. The 3090 and A10 columns are **bit-identical to each
other** — same Ampere-generation cuBLAS — so the residual 10⁻¹¹
texture against the GV100 is purely the Volta↔Ampere arithmetic
boundary. And the V100 (same silicon class as the reference, different
everything else) is the *closest* device at three of four dimensions:
residence contributes nothing that the arithmetic does not.

A reading is a reading, wherever the GPU lives.

**Timing** (descriptive; d = 8192): the serial blind probe runs
1,641 s on the 48-thread host CPU, 174 s on the GV100, 139 s batched;
`eigh` drops 120 s → 2.6 s (47×); and `top_spectrum` — pure-numpy
block subspace iteration, no dependency added — beats the full
decomposition on every device measured, down to **0.10 s** on the
GV100. The fp64 classes split exactly as the silicon dictates: Volta
probes ~200 s and eigh ~3 s; Ampere ~440 s and 7–10 s. None of this is
a bar; a speedup is a fact about hardware, not about correctness.

## 4. Two burned seals, and the derived-tolerance law

Both calibrations failed on first execution, and both failures were
the *bars'*, kept as-executed in the record:

- C-14's identity bar (10⁻¹⁰) failed at 5.9e-9: batched evaluation
  **reorders floating-point reductions** (row-block matmul versus
  per-vector products), and through an ill-conditioned pseudoinverse
  the fp64 disagreement grows with conditioning.
- C-13's E1 bar (10⁻⁹) failed hours later at 9.4e-9 for the identical
  reason one comparison over: OpenBLAS versus cuBLAS.

The re-derived bar — **10⁻⁷**, two orders above the measured
reordering floor and four below every defect class the bars exist to
catch — passed everywhere. The law we filed, now twice-earned in one
day: **an identity bar must be derived from the arithmetic of the
comparison, never sealed at aspiration.** A bar below the reordering
floor of fp64 tests nothing but the reordering floor.

## 5. The enforcer that cannot see measurement work

The commons cluster kills GPU pods that under-utilize their devices.
Our serial probe is, by specification, a stream of tiny kernels — one
consumer call per direction, host synchronization per call — because
the instrument's budget law *prices consumer calls*. The measured GPU
utilization profile of the full serial suite: **median 0%, mean 7.4%,
above the 40% enforcement floor for 7% of samples.** The cluster
deleted our first serial job 31 seconds after start; a second died the
same way; two others completed only by landing on tolerant nodes.

The instrument's own theory names this precisely. The enforcer is a
*witness* whose certificate — "this GPU is doing work" — is computed
from FLOP throughput. Call-bound measurement work is real work that
this witness cannot see: the certificate "idle, reclaim it" is issued
while a calibration is mid-measurement. That is a **vacuous
certificate** in exactly the sense the observation-theory program
measures elsewhere (its BGP audit found operators' convergence
certificates false 35% of the time for the same structural reason: the
witness reads a proxy, not the claim).

The fix is not to fight the enforcer but to re-plumb the observations:
C-14's **batched variant** delivers all of a point's finite differences
in one large-kernel invocation. Three things had to be shown, and were:

1. **Identity** — the batched reading equals the serial reading at the
   arithmetic floor (same directions, same seed).
2. **The cliff stands** — the budget law prices *directional
   observations*, not invocations. A theorem in this program says
   identification below k = d observations is impossible; batching
   re-plumbs the observations without adding any. Measured: 0/50
   hidden-component recoveries by single-invocation sub-dimensional
   probes. Batching buys wall-clock and compliance; it cannot buy
   admission, and the record now proves nobody ever gets to claim
   otherwise.
3. **Compliance** — the batched job survived and completed on the same
   commons that killed every serial attempt.

Batching also introduces its own hazard — a consumer with cross-row
coupling (shared normalization, cache state) silently corrupts every
finite difference — so the batched path self-checks row independence
and refuses violators, in the same spirit as the instrument's existing
regime gate.

## 6. What is claimed, and no more

The spec table now carries one new row, and it says exactly this:
same-seed readings agree to ≤ 4e-11 across five devices, two
architectures, and two sites; the cross-BLAS operator floor is ~10⁻⁸
and the bars are derived from it; serial probing is
enforcement-incompatible on utilization-policed commons GPUs and the
batched variant is the compliant shape, identical at the floor. GPU
timing is measured on Volta and Ampere; nothing is claimed for
hardware not in the table. Five platform findings (a container image
without git; a 1 GiB limit that cannot start a CUDA container; a node
with a broken driver that the scheduler kept refilling; dead legacy
GPU labels; an unschedulable device class) are recorded in the notes
for whoever walks this path next.

## 7. Reproduce it

Everything is public and re-runnable from the repository:

```bash
git clone https://github.com/ahb-sjsu/readscope && cd readscope
python calibration/c13_backend_suite.py --device-label $(hostname)   # CPU cells
python calibration/c14_batched_suite.py --device-label $(hostname)   # + batched
# with CuPy + a CUDA device, the same commands add the GPU cells
```

Records (as-executed, including both failures):
`calibration/records/c13-backend-*.json`,
`calibration/records/c14-batched-*.json`; verdict logic and bars:
`calibration/DECLARATION-C13.md`, `DECLARATION-C14.md`; the full
narrative including the enforcement chronicle:
`calibration/C13-C14-NOTES.md`.

*The NRP Nautilus platform is a shared national research commons; the
runs reported here were sized and shaped to its published policies,
and the enforcement behavior we measured is the platform working as
designed — the point is that its witness, like most witnesses, has an
observation budget, and instruments that live on the wrong side of it
must adapt rather than complain.*
