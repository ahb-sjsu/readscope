#!/usr/bin/env python3
"""C-5, the scale ladder. Declared before it runs.

C-3b showed recovery is identical across four architectures at matched budget
ratio. It could not say anything about scale, because those four models
differ in far more than size.

Qwen2.5 fixes that. From 1.5B to 32B it keeps ``head_dim`` at 128 while layer
count goes 28 to 64, head count 12 to 40, and grouping 2 to 8 key-value
heads. **The geometry the probe works in is constant and only the substrate
grows**, which is the only way to ask the scale question without confounding
it with dimension.

Ground truth is the same closed form as C-3b, exact for softmax attention:

    M_true = sum_s sum_i a_{i,s} q_i q_i^T / d,
    a_{i,s} = || p_i * (e_s - p_{i,s} 1) ||^2

All four models are loaded in bfloat16, declared in the extraction, so
precision is not confounded with scale either.

Declared bars:

  Q1  instrument       resolution at k/d = 1.25 is at least 0.90 on every
                       cell, at every scale.
  Q2  budget cliff     resolution at k/d = 0.5 is strictly below that at
                       k/d = 1.25 on every cell.
  Q3  scale invariance the spread of mean resolution at k/d = 1.25 across the
                       four scales is below 0.05. Since head_dim is fixed and
                       the budget law is a statement about k/d, recovery
                       should not care that the model got twenty times
                       larger. **If this fails, the budget law has a scale
                       term and the specification needs one too.**
  Q4  rank floor       the analytic operator has rank at least the graded
                       rank on every cell.
  Q5  anti-vacuity     median attention row entropy exceeds 1.0 bits on every
                       scale, the same check that caught C-3's query
                       artifact.
  Q6  census           every manifest cell is attempted and skips recorded.

**Reported without bars**, because no prior says what they should do and
inventing one after seeing them is the failure this program exists to avoid:
the effective rank of the analytic operator per scale, mean resolution at
k/d = 0.5 per scale, and attention entropy per scale. If any of them trends
with size, that is a substrate finding and the next sweep can bar it.

Runs on Atlas. Numpy only; activations come from a saved artifact.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readscope import (  # noqa: E402
    jacobian_probe,
    spectrum_of,
    subspace_overlap,
)

ACTS = Path("/archive/readscope/ladder")
RATIOS = [0.5, 1.25]
GRADED_RANK = 16
PROBE_KEYS = 32
EPS = 1e-3
SEED = 0
INSTRUMENT_BAR = 0.90
SCALE_SPREAD_BAR = 0.05
ENTROPY_BAR = 1.0


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def attention(K, Q, d):
    return softmax((Q @ K.T) / np.sqrt(d))


def analytic_operator(K, Q, probe_idx, d):
    P = attention(K, Q, d)
    M = np.zeros((d, d))
    for s in probe_idx:
        e = np.zeros(P.shape[1])
        e[s] = 1.0
        W = P * (e[None, :] - P[:, s : s + 1])
        a = (W**2).sum(axis=1)
        M += (Q * a[:, None]).T @ Q / d
    return 0.5 * (M + M.T)


def top(M, r):
    _, V = np.linalg.eigh(0.5 * (M + M.T))
    return np.ascontiguousarray(V[:, ::-1][:, :r])


def row_entropy_bits(P):
    q = np.clip(P, 1e-12, 1.0)
    return float(np.median(-(q * np.log2(q)).sum(axis=1)))


def main() -> int:
    mpath = ACTS / "manifest.json"
    if not mpath.exists():
        print("no manifest; run extract_ladder.py first")
        return 2
    manifest = json.loads(mpath.read_text())
    rng = np.random.default_rng(SEED)
    rows, skipped = [], []

    for cell in manifest:
        f = ACTS / cell["file"]
        if not f.exists():
            skipped.append({**cell, "reason": "file missing"})
            continue
        z = np.load(f)
        K = z["K"].astype(np.float64)
        Q = z["Q"].astype(np.float64)
        S, d = K.shape

        probe_idx = rng.choice(S, size=min(PROBE_KEYS, S), replace=False)
        M_true = analytic_operator(K, Q, probe_idx, d)
        truth_sub = top(M_true, GRADED_RANK)
        spec = spectrum_of(M_true)

        per_ratio = {}
        for ratio in RATIOS:
            k = max(1, int(round(ratio * d)))
            acc = np.zeros((d, d))
            for s in probe_idx:

                def consumer(kv, _s=s, _K=K, _Q=Q, _d=d):
                    Kp = _K.copy()
                    Kp[_s] = kv
                    return attention(Kp, _Q, _d).ravel()

                acc += jacobian_probe(
                    consumer,
                    K[s][None, :],
                    n_directions=k,
                    eps=EPS,
                    rng=np.random.default_rng(SEED + int(s) * 7919 % 100003),
                ).S
            per_ratio[str(ratio)] = subspace_overlap(
                top(acc, GRADED_RANK), truth_sub
            ).resolution

        rows.append(
            {
                **{
                    kk: cell[kk]
                    for kk in (
                        "tag",
                        "params_b",
                        "layer",
                        "layer_fraction",
                        "head",
                        "head_dim",
                        "n_layers",
                    )
                },
                "resolution": per_ratio,
                "truth_rank": int(np.linalg.matrix_rank(M_true, tol=1e-10)),
                "effective_rank": spec.effective_rank,
                "median_row_entropy_bits": row_entropy_bits(
                    attention(K, Q, d)
                ),
            }
        )
        print(
            f"  {cell['tag']:<12} L{cell['layer']:<3} H{cell['head']} "
            f"res@0.5 {per_ratio['0.5']:.3f}  res@1.25 "
            f"{per_ratio['1.25']:.3f}  eff_rank "
            f"{spec.effective_rank:5.2f}  H "
            f"{rows[-1]['median_row_entropy_bits']:.2f}",
            flush=True,
        )

    if not rows:
        print("no cells graded")
        return 1

    scales = sorted({r["params_b"] for r in rows})

    def by_scale(key, sub=None):
        out = {}
        for p in scales:
            vals = [
                (r[key][sub] if sub else r[key])
                for r in rows
                if r["params_b"] == p
            ]
            out[str(p)] = float(np.mean(vals))
        return out

    hi = by_scale("resolution", "1.25")
    lo = by_scale("resolution", "0.5")
    eff = by_scale("effective_rank")
    ent = by_scale("median_row_entropy_bits")
    spread = max(hi.values()) - min(hi.values())

    bars = {
        "Q1_instrument": bool(
            min(r["resolution"]["1.25"] for r in rows) >= INSTRUMENT_BAR
        ),
        "Q2_budget_cliff": bool(
            all(
                r["resolution"]["0.5"] < r["resolution"]["1.25"] - 1e-9
                for r in rows
            )
        ),
        "Q3_scale_invariance": bool(spread < SCALE_SPREAD_BAR),
        "Q4_rank_floor": bool(
            all(r["truth_rank"] >= GRADED_RANK for r in rows)
        ),
        "Q5_anti_vacuity": bool(all(v > ENTROPY_BAR for v in ent.values())),
        "Q6_census": bool(len(rows) + len(skipped) == len(manifest)),
    }
    verdict = "PASS" if all(bars.values()) else "FAIL"

    record = {
        "schema": "readscope-c5-scale-ladder-v1",
        "declared": {
            "ratios": RATIOS,
            "graded_rank": GRADED_RANK,
            "probe_keys": PROBE_KEYS,
            "eps": EPS,
            "seed": SEED,
            "instrument_bar": INSTRUMENT_BAR,
            "scale_spread_bar": SCALE_SPREAD_BAR,
            "entropy_bar": ENTROPY_BAR,
            "dtype": "bfloat16, constant across the ladder",
            "geometry": "head_dim 128 at every scale",
        },
        "rows": rows,
        "skipped": skipped,
        "summary": {
            "n_cells": len(rows),
            "scales_b": scales,
            "mean_resolution_at_1.25_by_scale": hi,
            "mean_resolution_at_0.5_by_scale": lo,
            "effective_rank_by_scale": eff,
            "median_entropy_by_scale": ent,
            "across_scale_spread_at_1.25": spread,
        },
        "bars": bars,
        "verdict": {"value": verdict, "computed_from": sorted(bars)},
        "runtime": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "code_commit": os.environ.get("CODE_COMMIT", "unknown"),
        },
    }
    out = Path(__file__).resolve().parent / "records" / "c5-scale-ladder.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True))

    print("\nby scale (params B):")
    print(
        f"  {'scale':<8} {'res@0.5':>9} {'res@1.25':>9} "
        f"{'eff_rank':>9} {'entropy':>9}"
    )
    for p in scales:
        k = str(p)
        print(
            f"  {p:<8} {lo[k]:9.4f} {hi[k]:9.4f} {eff[k]:9.2f} {ent[k]:9.2f}"
        )
    print(f"across-scale spread at k/d=1.25: {spread:.2e}")
    for k in sorted(bars):
        print(f"{k:<22} {'PASS' if bars[k] else 'FAIL'}")
    print("VERDICT", verdict)
    print(out)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
