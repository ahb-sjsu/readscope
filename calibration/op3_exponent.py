#!/usr/bin/env python3
"""OP3 — the sample-complexity exponent. Shakedown mode until the
appendix seals.

The v1-line owed prediction OP3 (geometric-observation
``crucible/OWED-V1.md``) extends C-15's scaling arm. C-15 measured
the sub-dimensional recovery at 1x-8x the flagship budget and read a
roughly flat curve; the owed prediction is asymptotic: swept to
``m in [1, 1000]``, does the sub-dimensional excess error decay as a
**derivable power** ``error ~ m^-alpha`` (alpha predicted from the
sketch second-moment structure / spectrum), or does it **provably
plateau** (alpha = 0)? One of the two must hold and be predicted a
priori; a messy, underivable decay is the kill.

The mechanism the prediction rests on is C-15's own algebra: the
sketch second moment ``E[ghat ghat^T] = (1 + 1/k) S + tr(S)/k I``
shares ``S``'s eigenspaces at every ``k``, so averaging n cheap
sub-dimensional points is a consistent estimator of the population
operator's top eigenspaces even when ``k < r``. Consistency implies
decay, not plateau; the open question this shakedown looks at is
whether the decay is a clean power law with a derivable exponent.

This runs the SAME planted family as C-15 (tanh basis, d=32,
spectrum decay 0.75, input scale 0.35, lstsq, eps=1e-3), fixed
``k = d/4 = 8``, rank 16, sweeping ``n = 384 * m`` for
``m in {1,4,16,64,256,1000}`` across seeds {0,1,2}. For each graded
rank r it reports the resolution curve, the excess ``1 - res(r)``,
and the log-log slope of the excess vs m with its R^2 — the raw
material for the a-priori alpha derivation and the sealed bars, both
of which live in the appendix (geometric-observation
``crucible/PREREG-OP3.md``) and bind only once that seals. Until then
this writes a record that carries no evidential weight.

    python calibration/op3_exponent.py --shakedown
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

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Reused verbatim from C-15's planted family.
DIM = 32
K = 8  # d/4, the sub-dimensional sketch
RANK = 16  # the confined family: recover 16 dims from a k=8 sketch
BASE_N = 384  # C-15's m=1 (the C-2e flagship spend at k=8)
GRADED_RANKS = [1, 2, 4, 8, 16]
EPS = 1e-3
SPECTRUM_DECAY = 0.75
INPUT_SCALE = 0.35

M_GRID = [1, 4, 16, 64, 256, 1000]
SEEDS = [0, 1, 2]


def scalar_consumer(basis, weights):
    def C(x):
        return float(np.tanh(basis.T @ x) @ weights)

    return C


def setup(rank, seed, n_points):
    # Identical RNG stream to C-15 so the substrate is the same family.
    rng = np.random.default_rng(seed * 977 + rank * 13 + 1)
    basis = np.linalg.qr(rng.standard_normal((DIM, rank)))[0]
    weights = SPECTRUM_DECAY ** np.arange(rank)
    pts = rng.standard_normal((n_points, DIM)) * INPUT_SCALE
    return basis, weights, pts


def cell(seed, n):
    basis, weights, pts = setup(RANK, seed, n)
    cons = scalar_consumer(basis, weights)
    res = blind_probe(
        cons,
        pts,
        mode="lstsq",
        sketch_dim=K,
        eps=EPS,
        rng=np.random.default_rng(seed * 31 + K),
        check_regime=False,
    )
    curve = {
        r: float(subspace_overlap(res.read_subspace(r), basis[:, :r]).resolution)
        for r in GRADED_RANKS
    }
    return {"seed": seed, "n": n, "m": n // BASE_N, "calls": res.n_calls,
            "resolution": curve}


def loglog_fit(ms, ys):
    """Slope and R^2 of log(y) vs log(m). Guards non-positive y."""
    xs = np.log(np.asarray(ms, float))
    yy = np.asarray(ys, float)
    if np.any(yy <= 0):
        return None
    ly = np.log(yy)
    slope, intercept = np.polyfit(xs, ly, 1)
    pred = slope * xs + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"alpha": float(-slope), "r2": float(r2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shakedown", action="store_true",
                    help="run with no evidential weight (the only mode until seal)")
    args = ap.parse_args()

    print(f"OP3 exponent shakedown — d={DIM} k={K} rank={RANK} "
          f"m in {M_GRID} seeds {SEEDS}")
    print("(no evidential weight until crucible/PREREG-OP3.md seals)\n")

    rows = []
    # res_by_rank[r][m] = list over seeds
    res_by_rank = {r: {m: [] for m in M_GRID} for r in GRADED_RANKS}
    for m in M_GRID:
        n = BASE_N * m
        for seed in SEEDS:
            row = cell(seed, n)
            rows.append(row)
            for r in GRADED_RANKS:
                res_by_rank[r][m].append(row["resolution"][r])
            print(f"  m={m:4d}x n={n:6d} seed={seed}  "
                  + " ".join(f"r{r}={row['resolution'][r]:.3f}"
                             for r in GRADED_RANKS))
        print()

    print("mean resolution and excess (1-res) vs m, per rank:")
    fits = {}
    for r in GRADED_RANKS:
        means = [float(np.mean(res_by_rank[r][m])) for m in M_GRID]
        excess = [max(1e-6, 1.0 - v) for v in means]
        res_fit = loglog_fit(M_GRID, means)
        exc_fit = loglog_fit(M_GRID, excess)
        fits[r] = {"mean_res": means, "excess": excess,
                   "res_loglog": res_fit, "excess_loglog": exc_fit}
        tag = "in-budget" if r <= K else "CONFINED "
        print(f"  rank {r:2d} [{tag}] res: "
              + " ".join(f"{v:.3f}" for v in means))
        if exc_fit:
            print(f"           excess(1-res) alpha={exc_fit['alpha']:+.3f} "
                  f"R2={exc_fit['r2']:.3f}   "
                  f"res-climb alpha={res_fit['alpha']:+.3f} R2={res_fit['r2']:.3f}"
                  if res_fit else "")

    record = {
        "calibration": "OP3-exponent",
        "sealed": False,
        "shakedown": True,
        "note": "no evidential weight; bars live in crucible/PREREG-OP3.md",
        "generated": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "constants": {"dim": DIM, "k": K, "rank": RANK, "base_n": BASE_N,
                      "eps": EPS, "spectrum_decay": SPECTRUM_DECAY,
                      "input_scale": INPUT_SCALE, "m_grid": M_GRID,
                      "seeds": SEEDS},
        "rows": rows,
        "fits": {str(r): fits[r] for r in GRADED_RANKS},
    }
    out = ROOT / "calibration" / "records" / "op3-shakedown.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
