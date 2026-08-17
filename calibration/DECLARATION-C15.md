# DECLARATION-C15 — the (k, n) fixed-budget surface

**STATUS: DRAFT-UNSEALED (drafted 2026-08-16 PDT, the same evening
the calibration script was written and shaken down). Per the
rate-limit rule this declaration binds nothing until a dated SEAL
line replaces this one in a later working session; the shakedown
record (`records/c15-shakedown.json`, one seed, rank 16) carries no
evidential weight and exists to show the surface has interior.**

## The question (reviewer-posed, accepted verbatim)

C-2e measured the cliff at matched operating-point counts — equal
`n`, unequal total budget. The sketch expectation
`(1 + 1/k)S + tr(S)/k·I` retains `S`'s eigenspaces at every `k`, and
per-point frames are redrawn, so `k < d` is not structurally barred
from recovering the population read subspace across many points.
**At equal total consumer calls, does reallocating directions into
operating points buy anything?** Either answer is a law worth
having: dominance of `k = d` is a measured sample-complexity floor;
catch-up is an identification–averaging tradeoff the instrument can
then specify.

## Design (constants fixed in `c15_budget_surface.py`, this commit)

Same planted family as C-2e (tanh basis consumer, d = 32, decay
0.75, input scale 0.35), ranks {4, 16}, seeds {0..4}, lstsq, eps
1e−3. **Surface arm:** total directional observations fixed at
kn = 3072 (C-2e's flagship spend); k/d ∈ {1/8, 1/4, 1/2, 3/4, 1,
1.25}, n = ⌊3072/k⌋. **Scaling arm:** k = 8 (= d/4), n ∈ {384, 768,
1536, 3072} — 1× to 8× the flagship's total budget.

## Decision rule (to bind at seal; numbers proposed now, frozen then)

- **D1 (equal-budget dominance):** if, at rank 16, the seed-median
  of [res@16 at `k = d` minus the best sub-dimensional surface
  cell] ≥ 0.3, the record states: *the cliff survives equal-budget
  reallocation at B = 6144 calls.*
- **D2 (convergence within 8×):** at `k = d/4`: seed-median res@16
  at n = 3072 ≥ 0.5 → *catch-up within 8× budget*; else if it
  exceeds the n = 384 value by ≥ 0.1 → *slow convergence, law
  unresolved at this scale*; else → *no visible convergence within
  8× budget.*
- **AV (anti-vacuity):** full cell census; calls equal to the
  declared 2kn per cell; the `k = d` surface cell at res@16 = 1.0
  (instrument sanity) — any failure voids the record.

The outcomes are **reported laws**, not graduation events: the
result updates SPEC.md's budget-law section and the scoped README
language either way. Any refinement of P3 belongs to the
observation-theory program's own sealed process, not to this
calibration.

## Shakedown note (no weight)

One seed, rank 16: surface sub-d cells 0.014–0.109 against 1.0 at
`k ≥ d`; scaling arm flat at ~0.06 through n = 3072. The surface
has interior and the machinery grades; the sealed run decides.
