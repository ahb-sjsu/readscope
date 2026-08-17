#!/usr/bin/env python3
"""C-15, the (k, n) fixed-budget surface. Shakedown mode until the
declaration seals.

C-2e measured the cliff at matched operating-point counts: every
``k/d`` row spent 96 points, so sub-dimensional budgets were also
smaller total budgets. The reviewer's observation stands in the
sketch algebra itself: ``E[ghat ghat^T] = (1 + 1/k) S + tr(S)/k I``
shares ``S``'s eigenspaces at every ``k``, and the per-point frames
are redrawn, so nothing forbids many cheap points from averaging
their way to the population operator. Whether they do at *equal
total consumer calls* is this calibration's question.

Design, matched to C-2e everywhere it can be:

- Same planted family (tanh basis consumer, ``d = 32``, spectrum
  decay 0.75, input scale 0.35), ranks {4, 16}, seeds {0..4},
  ``lstsq`` estimator, ``eps = 1e-3``.
- **Surface arm:** total directional-observation budget fixed at
  ``kn = 32 * 96 = 3072`` (the C-2e flagship's spend, 6144 calls);
  sweep ``k/d`` in {1/8, 1/4, 1/2, 3/4, 1, 1.25} with
  ``n = floor(3072 / k)``.
- **Scaling arm:** fixed ``k = d/4 = 8``; ``n`` in {384, 768, 1536,
  3072} — 1x to 8x the flagship budget. If resolution climbs toward
  1 with ``n``, sub-dimensional budgets converge and the cliff is a
  sample-complexity statement; if it plateaus, the cliff survives
  reallocation and C-2e's law extends to total budget.

Both outcomes are a measured law. The decision rule and any bars
live in DECLARATION-C15.md and bind only once that seals; until
then this script runs with --shakedown and writes a record that
carries no evidential weight.

    python calibration/c15_budget_surface.py --shakedown
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readscope import blind_probe, subspace_overlap  # noqa: E402

DIM = 32
RANKS = [4, 16]
GRADED_RANKS = [1, 2, 4, 8, 16]
BUDGET_KN = DIM * 96  # C-2e flagship spend
SURFACE_RATIOS = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25]
SCALING_K = 8  # d/4
SCALING_N = [384, 768, 1536, 3072]
EPS = 1e-3
SEEDS = [0, 1, 2, 3, 4]
SPECTRUM_DECAY = 0.75
INPUT_SCALE = 0.35
HERE = Path(__file__).resolve().parent


def scalar_consumer(basis, weights):
    def C(x):
        return float(np.tanh(basis.T @ x) @ weights)

    return C


def setup(rank, seed, n_points):
    rng = np.random.default_rng(seed * 977 + rank * 13 + 1)
    basis = np.linalg.qr(rng.standard_normal((DIM, rank)))[0]
    weights = SPECTRUM_DECAY ** np.arange(rank)
    pts = rng.standard_normal((n_points, DIM)) * INPUT_SCALE
    return basis, weights, pts


def cell(rank, seed, k, n):
    basis, weights, pts = setup(rank, seed, n)
    cons = scalar_consumer(basis, weights)
    res = blind_probe(
        cons,
        pts,
        mode="lstsq",
        sketch_dim=k,
        eps=EPS,
        rng=np.random.default_rng(seed * 31 + k),
        check_regime=False,
    )
    curve = {
        str(r): round(
            subspace_overlap(res.read_subspace(r), basis[:, :r]).resolution,
            4,
        )
        for r in GRADED_RANKS
        if r <= rank
    }
    return {
        "rank": rank,
        "seed": seed,
        "k": k,
        "n": n,
        "kn": k * n,
        "calls": res.n_calls,
        "resolution": curve,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shakedown", action="store_true")
    args = ap.parse_args()
    seeds = [0] if args.shakedown else SEEDS
    ranks = [16] if args.shakedown else RANKS

    rows = []
    for seed in seeds:
        for rank in ranks:
            for ratio in SURFACE_RATIOS:
                k = int(round(ratio * DIM))
                n = BUDGET_KN // k
                rows.append(cell(rank, seed, k, n))
                r = rows[-1]
                print(
                    f"surface rank={rank} seed={seed} k/d={ratio:<5} "
                    f"n={n:>4} res@{rank}={r['resolution'][str(rank)]}"
                )
            for n in SCALING_N:
                rows.append(cell(rank, seed, SCALING_K, n))
                r = rows[-1]
                print(
                    f"scaling rank={rank} seed={seed} k={SCALING_K} "
                    f"n={n:>4} res@{rank}={r['resolution'][str(rank)]}"
                )

    record = {
        "calibration": "C-15" + ("-shakedown" if args.shakedown else ""),
        "sealed": not args.shakedown,
        "generated": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "constants": {
            "dim": DIM,
            "budget_kn": BUDGET_KN,
            "surface_ratios": SURFACE_RATIOS,
            "scaling_k": SCALING_K,
            "scaling_n": SCALING_N,
            "eps": EPS,
            "spectrum_decay": SPECTRUM_DECAY,
            "input_scale": INPUT_SCALE,
        },
        "rows": rows,
    }
    name = (
        "c15-shakedown.json" if args.shakedown else "c15-budget-surface.json"
    )
    out = HERE / "records" / name
    json.dump(record, open(out, "w"), indent=1)
    print(f"-> {out.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
