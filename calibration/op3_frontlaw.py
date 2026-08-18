#!/usr/bin/env python3
"""OP3 front-law validation — the corrected recovery law. Shakedown/desk.

The naive OP3 model (clean m^-1/2 subspace-angle decay to zero) was
refuted by op3_exponent.py: recovery is a spectrum-ordered front. This
script tests the *corrected* a-priori law, derived from the same C-15
sketch identity but carried one step further:

  M = E[Shat] = (1 + 1/k) S + (tr S / k) I         (C-15 identity)
  eigengap at mode i:  gap_i = (1+1/k)(lam_i - lam_{i+1}) ~ w_i^2
  fluctuation floor:   ||Shat_n - M|| ~ beta * tr(S) / (k sqrt(n))   (isotropic, common)
  Davis-Kahan:         sin th_i <~ ||Shat_n - M|| / gap_i ~ [tr S/(k sqrt n)] / w_i^2

=> sin^2 th_i ~ 1 / (n * w_i^4).  The per-mode recovery collapses onto a
single curve in the scaling variable s_i = m * w_i^p with the DERIVED
exponent **p = 4** (Davis-Kahan sin ~ 1/gap, gap ~ w_i^2). The isotropic
tr(S)/k floor -- not per-mode signal -- sets the front: the cross-mode
interference. Front position advances as i*(m) ~ i*(1) + ln(m)/ln(w^-4).

Validation: measure per-mode canonical correlations cos th_i across the
grid and seeds; for each candidate p fit the one-parameter universal
curve cos^2 th_i = s/(s+A) (s = m*w_i^p) and report the collapse RMS.
The derivation predicts p=4 wins. This is desk validation of the law
that a future sealed appendix will bar (the collapse + the front rate);
it carries no evidential weight.

    python calibration/op3_frontlaw.py
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from readscope import blind_probe  # noqa: E402
import op3_exponent as op3  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
W = op3.SPECTRUM_DECAY  # 0.75
RANK = op3.RANK  # 16
K = op3.K  # 8
BASE_N = op3.BASE_N  # 384
M_GRID = [4, 16, 64, 256, 1000]
SEEDS = [0, 1, 2]
P_CANDIDATES = [2.0, 3.0, 4.0, 5.0, 6.0]


def per_mode_cos2(seed, n):
    """cos^2 of the canonical correlations between read_subspace(16) and
    the planted top-16 basis, per mode (sorted descending)."""
    basis, weights, pts = op3.setup(RANK, seed, n)
    cons = op3.scalar_consumer(basis, weights)
    res = blind_probe(cons, pts, mode="lstsq", sketch_dim=K, eps=1e-3,
                      rng=np.random.default_rng(seed * 31 + K),
                      check_regime=False)
    A = np.linalg.qr(res.read_subspace(RANK))[0]
    B = np.linalg.qr(basis[:, :RANK])[0]
    cc = np.linalg.svd(B.T @ A, compute_uv=False)
    return (cc ** 2).tolist()  # length 16, descending


def fit_universal(s, y):
    """Fit y = s/(s+A) for scalar A>0 by 1-D search on log A; return (A, rms)."""
    s = np.asarray(s, float)
    y = np.asarray(y, float)
    best = None
    for logA in np.linspace(-8, 8, 3201):
        A = np.exp(logA)
        pred = s / (s + A)
        rms = float(np.sqrt(np.mean((pred - y) ** 2)))
        if best is None or rms < best[1]:
            best = (A, rms)
    return best


def main() -> int:
    print(f"OP3 front-law validation — d={op3.DIM} k={K} rank={RANK} "
          f"m in {M_GRID} seeds {SEEDS}")
    print("derived prediction: collapse variable s_i = m * w_i^4 (p=4)\n")

    # Measure per-mode cos^2 across the grid.
    points = []  # (mode_i, m, seed, cos2)
    for m in M_GRID:
        n = BASE_N * m
        for seed in SEEDS:
            c2 = per_mode_cos2(seed, n)
            for i, v in enumerate(c2):
                points.append({"i": i, "m": m, "seed": seed, "cos2": v})
        # progress line: seed-mean cos^2 at this m
        mean_c2 = [float(np.mean([p["cos2"] for p in points
                                  if p["i"] == i and p["m"] == m]))
                   for i in range(RANK)]
        print(f"  m={m:4d}x  mean cos^2 by mode: "
              + " ".join(f"{v:.2f}" for v in mean_c2))

    # For each candidate p, collapse and fit the universal curve.
    print("\ncollapse quality — fit cos^2 = s/(s+A), s = m*w^(p*i):")
    results = {}
    for p in P_CANDIDATES:
        s = [pt["m"] * (W ** (p * pt["i"])) for pt in points]
        y = [pt["cos2"] for pt in points]
        A, rms = fit_universal(s, y)
        results[p] = {"A": A, "rms": rms}
        flag = "  <-- derived" if p == 4.0 else ""
        print(f"  p={p:.0f}:  A={A:10.4g}   collapse RMS={rms:.4f}{flag}")

    best_p = min(results, key=lambda p: results[p]["rms"])
    print(f"\nbest-collapsing exponent: p={best_p:.0f} "
          f"(RMS {results[best_p]['rms']:.4f})   "
          f"derived p=4 {'CONFIRMED' if best_p == 4.0 else 'NOT best'}")

    # Front advance: mode where cos^2 = 0.5, i.e. s=A -> i* = ln(m/A)/ln(w^-p).
    p = 4.0
    A = results[p]["A"]
    print(f"\nfront position i*(m) at p=4 (cos^2=0.5, A={A:.4g}):")
    prev = None
    for m in M_GRID:
        istar = np.log(m / A) / np.log(W ** (-p))
        delta = "" if prev is None else f"   d_i*={istar - prev:+.2f}"
        print(f"  m={m:4d}x  i*={istar:5.2f}{delta}")
        prev = istar
    predicted_slope = 1.0 / np.log(W ** (-p))
    print(f"  predicted di*/d ln m = 1/ln(w^-4) = {predicted_slope:.3f} "
          f"(~{predicted_slope * np.log(16):.2f} modes per 16x)")

    record = {
        "calibration": "OP3-frontlaw",
        "sealed": False,
        "shakedown": True,
        "note": "desk validation of the derived front law; no evidential weight",
        "generated": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "derived_exponent_p": 4,
        "constants": {"dim": op3.DIM, "k": K, "rank": RANK, "w": W,
                      "base_n": BASE_N, "m_grid": M_GRID, "seeds": SEEDS},
        "collapse_by_p": {str(int(p)): results[p] for p in P_CANDIDATES},
        "best_p": int(best_p),
        "points": points,
    }
    out = ROOT / "calibration" / "records" / "op3-frontlaw.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
