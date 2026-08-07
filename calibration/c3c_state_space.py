#!/usr/bin/env python3
"""C-3c, a non-transformer consumer. Declared before it runs.

`CALIBRATION.md` asks for at least one consumer that is not attention, so
that the specification is not secretly a statement about softmax. This is it,
on a real Mamba-790m.

**The structural claim, and it mirrors attention exactly.** A selective SSM
runs, per channel and state dimension,

    h_t[n] = exp(A[n] dt[t]) h_{t-1}[n] + ...
    y_t    = sum_n C_t[n] h_t[n] + ...

The part of ``y`` that depends on the state at position ``s`` is
``y_t = sum_n C_t[n] g_t[n] h_s[n]`` with the accumulated decay
``g_t[n] = prod_{u=s+1..t} exp(A[n] dt[u])``, so

    d y_t / d h_s = C_t * g_t
    M_true = sum_{t >= s} (C_t * g_t)(C_t * g_t)^T

**The read subspace of a recurrent state is spanned by its readout vectors,
attenuated by how much of each has already decayed.** That is the same shape
as attention, where the read subspace of a head with respect to a key is
spanned by its queries, and it is what lets this sweep have an exact ground
truth on a non-attention consumer.

It also puts the compounding that `readscope.regimes` warns about directly
under measurement. A pointwise Jacobian is well defined for a recurrence and
misleading, because error in the decay accumulates along the sequence. Here
the accumulation is in ``g_t``, and how fast it kills the contribution of
distant positions is a measurable property of a real model rather than an
assertion.

Declared bars:

  N1  instrument      resolution at k/d = 1.25 is at least 0.90 on every
                      cell. The probe must match the closed form on a
                      consumer it was never designed around.
  N2  budget cliff    resolution at k/d = 0.5 is strictly below that at
                      k/d = 1.25 on every cell. If the cliff is an attention
                      phenomenon rather than a probe phenomenon, this fails.
  N3  compounding     the accumulated decay is strictly decreasing in
                      horizon on every cell, and more than half of the read
                      operator's trace comes from the nearest quarter of the
                      horizon. This states the recurrence reads mostly its
                      near past, and it can fail: a model with slow channels
                      would spread the trace out.
  N4  rank floor      the analytic operator has rank at least the graded
                      rank on every cell.
  N5  anti-vacuity    the decay is neither saturated nor dead, meaning the
                      accumulated decay at the far end of the horizon lies
                      strictly between 1e-12 and 0.999, so no cell is graded
                      on a state that either never forgets or is already
                      empty.
  N6  census          every manifest cell is attempted and skips recorded.

Runs on Atlas. Numpy only; parameters come from a saved artifact.
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

from readscope import jacobian_probe, subspace_overlap  # noqa: E402

ACTS = Path("/archive/readscope/ssm")
RATIOS = [0.5, 1.25]
GRADED_RANK = 8
START = 32
HORIZON = 96
EPS = 1e-4
SEED = 0
INSTRUMENT_BAR = 0.90
NEAR_QUARTER_BAR = 0.5
DECAY_FLOOR, DECAY_CEIL = 1e-12, 0.999


def decay_matrix(A, dt, start, horizon):
    """``g_t[n]`` for t from start to start+horizon, accumulated from start."""
    steps = np.exp(np.outer(dt[start + 1 : start + horizon + 1], A))
    g = np.cumprod(steps, axis=0)
    return np.vstack([np.ones((1, A.size)), g])


def analytic_operator(A, dt, C, start, horizon):
    g = decay_matrix(A, dt, start, horizon)
    Cw = C[start : start + horizon + 1] * g
    return Cw.T @ Cw, g, Cw


def make_consumer(A, dt, C, start, horizon):
    """Output sequence from the state at ``start``, as a function of it."""
    g = decay_matrix(A, dt, start, horizon)
    Cw = C[start : start + horizon + 1] * g

    def consumer(h):
        return Cw @ np.asarray(h, dtype=float)

    return consumer


def top(M, r):
    _, V = np.linalg.eigh(0.5 * (M + M.T))
    return np.ascontiguousarray(V[:, ::-1][:, :r])


def main() -> int:
    mpath = ACTS / "manifest.json"
    if not mpath.exists():
        print("no manifest; run extract_ssm.py first")
        return 2
    manifest = json.loads(mpath.read_text())
    rows, skipped = [], []

    for cell in manifest:
        f = ACTS / cell["file"]
        if not f.exists():
            skipped.append({**cell, "reason": "file missing"})
            continue
        z = np.load(f)
        A = z["A"].astype(np.float64)
        dt = z["dt"].astype(np.float64)
        C = z["C"].astype(np.float64)
        d = A.size
        if START + HORIZON + 1 > dt.size:
            skipped.append({**cell, "reason": "sequence too short"})
            continue

        M_true, g, Cw = analytic_operator(A, dt, C, START, HORIZON)
        truth_rank = int(np.linalg.matrix_rank(M_true, tol=1e-12))
        truth_sub = top(M_true, GRADED_RANK)
        consumer = make_consumer(A, dt, C, START, HORIZON)

        # how much of the operator's mass sits in the nearest quarter
        contrib = (Cw**2).sum(axis=1)
        q = max(1, (HORIZON + 1) // 4)
        near_fraction = float(contrib[:q].sum() / contrib.sum())

        g_norm = np.linalg.norm(g, axis=1) / np.sqrt(d)
        monotone = bool(np.all(np.diff(g_norm) <= 1e-15))
        far = float(g_norm[-1])

        per_ratio = {}
        for ratio in RATIOS:
            k = max(1, int(round(ratio * d)))
            pr = jacobian_probe(
                consumer,
                np.zeros((1, d)),
                n_directions=k,
                eps=EPS,
                rng=np.random.default_rng(SEED + 17),
            )
            per_ratio[str(ratio)] = subspace_overlap(
                top(pr.S, GRADED_RANK), truth_sub
            ).resolution

        rows.append(
            {
                **{
                    kk: cell[kk]
                    for kk in ("tag", "family", "layer", "channel", "d_state")
                },
                "resolution": per_ratio,
                "truth_rank": truth_rank,
                "near_quarter_trace_fraction": near_fraction,
                "decay_monotone": monotone,
                "decay_at_horizon": far,
                "effective_memory_steps": float(
                    np.sum(g_norm) / max(g_norm[0], 1e-300)
                ),
            }
        )
        print(
            f"  L{cell['layer']:<3} C{cell['channel']}  "
            f"res@0.5 {per_ratio['0.5']:.3f}  res@1.25 "
            f"{per_ratio['1.25']:.3f}  near1/4 {near_fraction:.3f}  "
            f"decay@H {far:.2e}  mem {rows[-1]['effective_memory_steps']:.1f}",
            flush=True,
        )

    if not rows:
        print("no cells graded")
        return 1

    hi = [r["resolution"]["1.25"] for r in rows]
    lo = [r["resolution"]["0.5"] for r in rows]
    near = [r["near_quarter_trace_fraction"] for r in rows]

    bars = {
        "N1_instrument": bool(min(hi) >= INSTRUMENT_BAR),
        "N2_budget_cliff": bool(
            all(
                r["resolution"]["0.5"] < r["resolution"]["1.25"] - 1e-9
                for r in rows
            )
        ),
        "N3_compounding": bool(
            all(r["decay_monotone"] for r in rows)
            and min(near) > NEAR_QUARTER_BAR
        ),
        "N4_rank_floor": bool(
            all(r["truth_rank"] >= GRADED_RANK for r in rows)
        ),
        "N5_anti_vacuity": bool(
            all(DECAY_FLOOR < r["decay_at_horizon"] < DECAY_CEIL for r in rows)
        ),
        "N6_census": bool(len(rows) + len(skipped) == len(manifest)),
    }
    verdict = "PASS" if all(bars.values()) else "FAIL"

    record = {
        "schema": "readscope-c3c-state-space-v1",
        "declared": {
            "ratios": RATIOS,
            "graded_rank": GRADED_RANK,
            "start": START,
            "horizon": HORIZON,
            "eps": EPS,
            "seed": SEED,
            "instrument_bar": INSTRUMENT_BAR,
            "near_quarter_bar": NEAR_QUARTER_BAR,
            "decay_band": [DECAY_FLOOR, DECAY_CEIL],
            "truth": "closed-form sum_t (C_t * g_t)(C_t * g_t)^T for a "
            "selective SSM, g_t the accumulated decay from the start "
            "position",
        },
        "rows": rows,
        "skipped": skipped,
        "summary": {
            "n_cells": len(rows),
            "mean_resolution_at_1.25": float(np.mean(hi)),
            "mean_resolution_at_0.5": float(np.mean(lo)),
            "min_resolution_at_1.25": float(min(hi)),
            "median_near_quarter_fraction": float(np.median(near)),
            "median_effective_memory_steps": float(
                np.median([r["effective_memory_steps"] for r in rows])
            ),
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
    out = Path(__file__).resolve().parent / "records" / "c3c-state-space.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True))

    print(
        f"\nmean resolution  k/d=0.5 {np.mean(lo):.4f}   "
        f"k/d=1.25 {np.mean(hi):.4f}"
    )
    print(
        f"median near-quarter trace fraction {np.median(near):.4f}   "
        f"median effective memory "
        f"{record['summary']['median_effective_memory_steps']:.1f} steps"
    )
    for k in sorted(bars):
        print(f"{k:<22} {'PASS' if bars[k] else 'FAIL'}")
    print("VERDICT", verdict)
    print(out)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
