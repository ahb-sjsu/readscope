# DECLARATION-C16 — the integer-codebook departure

**STATUS: DRAFT-UNSEALED (drafted 2026-08-17, the same session the
script was written and shaken down). Per the rate-limit rule nothing
here binds until a dated SEAL line replaces this one in a later
working session; the shakedown (`records/c16-shakedown.json`, one
seed, d = 32) carries no evidential weight and exists to show the
question has interior.**

## The question (from the water_fill scope note)

`water_fill` optimizes the continuous surrogate
`D = Σ w_i 2^(−2 b_i)` with real-valued per-direction bits. Real
codecs spend integer bits (or draw from a finite codebook). SPEC.md's
water_fill scoping named the departure as uncalibrated: **how much
distortion does the integer constraint cost over the continuous
optimum, and does a smart integer allocation recover it?**

## Design (`c16_integer_codebook.py`)

Planted read spectra (geometric decay, flat, low-rank), d ∈ {16, 32,
64}, five seeds, bits/dim ∈ {0.25, 0.5, 1, 2, 4}. For each cell,
compute the continuous water-fill distortion `D_cont` and the
**excess ratio** `D_int / D_cont ≥ 1` for four integer strategies:
`round`, `floor`, `greedy` (the separable-surrogate integer optimum:
add each bit where it drops distortion most), `ceil_topk` (a codec
heuristic).

## The shakedown finding, recorded (drives the sealed design)

The shakedown already exposed a confound that the sealed run must
handle: **naive `round` is not budget-feasible.** Rounding the
continuous bits up spends 1–2 bits over the integer budget (recorded
per cell in `int_budget_used` vs `int_budget`), so its excess ratio
dips *below 1* — it is not beating the optimum, it is cheating the
constraint. The fair comparison is among budget-feasible strategies
(`floor`, `greedy`, `ceil_topk` all hit the integer budget exactly).
The `feasible` flag is in every record; `round` is reported for the
record and excluded from any bar.

Shakedown medians (d = 32, geometric): the budget-feasible integer
optimum (`greedy`) sits at **1.05–1.10× the continuous floor** —
5–10% excess, roughly flat across bits/dim; `floor` (spend nothing
extra) costs ~2×; the `ceil_topk` heuristic 1.20–1.46×. The
integer-grain penalty for a *smart* codec is a small near-constant
fraction; a lazy one pays up to double.

## Decision rule (to bind at seal; numbers proposed now, frozen then)

- **B1 (greedy is the near-free integer optimum):** across the graded
  grid, the seed-median `greedy` excess ≤ **1.15** at every
  (d, spectrum, bits/dim) cell with bits/dim ≥ 0.5. If it exceeds
  1.15 somewhere, the integer constraint is not cheap and the sealed
  record says so.
- **B2 (cleverness pays):** `greedy` excess < `floor` excess at every
  cell (a budget-feasible smart allocation strictly beats dropping
  the freed budget).
- **AV (feasibility census):** every `greedy`/`floor`/`ceil_topk`
  cell budget-feasible; `round` flagged infeasible wherever it
  overspends; full cell census.

The outcome is a **reported law** (how much integer codecs pay over
the continuous ideal, and that a greedy allocation nearly closes it),
not a graduation event — it updates SPEC.md's water_fill section and
the allocate.py scope note.

## Seal

*(empty until a later working session; the seal is a dated line here,
after which the graded run — all dims, five seeds, no `--shakedown` —
executes and `records/c16-integer-codebook.json` is committed as
executed)*
