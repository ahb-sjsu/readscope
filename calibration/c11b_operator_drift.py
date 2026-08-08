#!/usr/bin/env python3
"""C-11b, does an attention head's read operator drift along the sequence?

Declared before it runs. Supersedes c11_operator_drift.py, which was
**invalid rather than failed** and whose record is kept.

Three defects, and P4 caught the first before anything was reported.

Windows too small for the grading.
    24 queries split four ways gives six per window, so each window's
    operator has rank at most six while the sweep graded at rank eight. The
    top-eight subspace of a rank-six matrix carries two arbitrary null
    directions, so every agreement number was part noise.

A denominator that collapses.
    Mispricing was reported as ``(cost_early - cost_late) / cost_late``. With
    a rank-six operator and a 384-bit budget, water-filling puts about
    sixty-four bits into each live direction and the late-plan cost goes to
    1e-20, so the ratio came back as 5e19 percent. That is a division by
    nothing, not a measurement.

Windowing the wrong axis, which nothing caught.
    The source-matched query set is stored head-major, ``grp * S`` rows with
    all of head zero's positions first. Slicing that flat array does not cut
    by position at all; it cuts by head for the first stretch and then mixes.
    **The sweep would have measured head-to-head variation and called it
    positional drift.** Only reading the storage layout found it.

The repairs. Use the source-matched activations, 576 queries per cell, so a
four-way split gives 144 each and rank is never the binding constraint.
Reshape to ``(grp, S, d)`` and window along ``S``, pooling across the group,
so the axis really is position. Normalise mispricing by the **uniform**
allocation cost, a denominator that is always positive and well scaled, so
the figure reads as "what fraction of an even split's cost does staleness
add".

**Provenance disclosure, unchanged.** The hypothesis is post hoc, from
turboquant-pro's unexplained long-generation negative. Two post-hoc
hypotheses in this programme have already died under their own bars, so P0
comes first and can void the sweep.

Declared bars:

  P0  the operator moves   median agreement between the first and last
                           positional window is below 0.90. **Voids the sweep
                           if it fails**, because nothing after it could be
                           about drift.
  P1  drift accumulates    agreement with the first window is non-increasing
                           across windows, allowing one inversion per cell.
  P2  early misprices late allocating against the early operator costs the
                           late consumer at least 5 percent of the uniform
                           allocation's cost more than allocating against the
                           late one. **The claim with teeth.**
  P3  the union is a fix   allocating against the whole-sequence operator
                           costs the late consumer less than allocating
                           against the early one, on a majority of cells.
  P4  anti-vacuity         every window has analytic rank at least the graded
                           rank and median row entropy above 1.0 bits.
  P5  census               every manifest cell attempted, skips recorded.

Runs on Atlas, numpy only.
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

from readscope import spectrum_of, subspace_overlap  # noqa: E402

ACTS = Path("/archive/readscope/source_match")

GRADED_RANK = 16
N_WINDOWS = 4
PROBE_KEYS = 24
BUDGET_BITS_PER_DIM = 3.0
SEED = 0

MOVES_BAR = 0.90
MISPRICE_BAR = 0.05
ENTROPY_BAR = 1.0


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def attention(K, Q, d):
    return softmax((Q @ K.T) / np.sqrt(d))


def operator(K, Q, probe_idx, d):
    P = attention(K, Q, d)
    M = np.zeros((d, d))
    for s in probe_idx:
        e = np.zeros(P.shape[1])
        e[s] = 1.0
        w = P * (e[None, :] - P[:, s : s + 1])
        a = (w**2).sum(axis=1)
        M += (Q * a[:, None]).T @ Q / d
    return 0.5 * (M + M.T)


def top(M, r):
    _, V = np.linalg.eigh(0.5 * (M + M.T))
    return np.ascontiguousarray(V[:, ::-1][:, :r])


def row_entropy_bits(P):
    q = np.clip(P, 1e-12, 1.0)
    return float(np.median(-(q * np.log2(q)).sum(axis=1)))


def water_fill(lam, var, budget):
    w = lam * var
    if not np.any(w > 0) or budget <= 0:
        return np.zeros_like(w)

    def bits_at(theta):
        b = np.zeros_like(w)
        live = w > theta
        b[live] = 0.5 * np.log2(w[live] / theta)
        return b

    hi = float(w.max())
    lo = hi
    while bits_at(lo).sum() < budget:
        lo *= 0.5
        if lo < 1e-300:
            break
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if bits_at(mid).sum() > budget:
            lo = mid
        else:
            hi = mid
    return bits_at(0.5 * (lo + hi))


def costs_under(target_M, plan_M, K, budget):
    """Distortion the target consumer feels from bits planned on plan_M.

    Returns the planned cost and the cost of an even split in the same basis,
    so mispricing can be normalised by something that cannot collapse.
    """
    d = target_M.shape[0]
    _, basis = np.linalg.eigh(0.5 * (plan_M + plan_M.T))
    basis = np.ascontiguousarray(basis[:, ::-1])
    lam_plan = np.clip(np.diag(basis.T @ plan_M @ basis), 0.0, None)
    lam_tgt = np.clip(np.diag(basis.T @ target_M @ basis), 0.0, None)
    centred = K - K.mean(axis=0, keepdims=True)
    var = np.maximum(((centred @ basis) ** 2).mean(axis=0), 0.0)

    bits = water_fill(lam_plan, var, budget * d)
    planned = float(np.sum(lam_tgt * var * np.power(2.0, -2.0 * bits)))
    flat = np.full(d, float(budget))
    uniform = float(np.sum(lam_tgt * var * np.power(2.0, -2.0 * flat)))
    return planned, uniform


def positional_windows(Q_flat, group_size, seq, n_windows):
    """Cut by position, pooling the query heads that share this key head.

    The stored array is head-major, ``(grp * S, d)``, so slicing it directly
    cuts by head rather than by position.
    """
    d = Q_flat.shape[1]
    q = Q_flat.reshape(group_size, seq, d)
    edges = np.linspace(0, seq, n_windows + 1, dtype=int)
    return [
        np.ascontiguousarray(q[:, edges[i] : edges[i + 1], :].reshape(-1, d))
        for i in range(n_windows)
    ]


def main() -> int:
    mpath = ACTS / "manifest.json"
    if not mpath.exists():
        print("no manifest; run extract_source_match.py first")
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
        grp, seq = int(cell["group_size"]), int(cell["seq"])
        if Q.shape[0] != grp * seq:
            skipped.append({**cell, "reason": "query layout unexpected"})
            continue

        probe_idx = rng.choice(S, size=min(PROBE_KEYS, S), replace=False)
        windows = positional_windows(Q, grp, seq, N_WINDOWS)
        if min(w.shape[0] for w in windows) < GRADED_RANK:
            skipped.append(
                {**cell, "reason": "window smaller than graded rank"}
            )
            continue

        ops = [operator(K, w, probe_idx, d) for w in windows]
        union = operator(K, Q, probe_idx, d)

        ranks = [int(np.linalg.matrix_rank(M, tol=1e-10)) for M in ops]
        ents = [row_entropy_bits(attention(K, w, d)) for w in windows]

        first = top(ops[0], GRADED_RANK)
        agree = [
            subspace_overlap(top(M, GRADED_RANK), first).resolution
            for M in ops
        ]

        early, late = ops[0], ops[-1]
        c_early, uni = costs_under(late, early, K, BUDGET_BITS_PER_DIM)
        c_late, _ = costs_under(late, late, K, BUDGET_BITS_PER_DIM)
        c_union, _ = costs_under(late, union, K, BUDGET_BITS_PER_DIM)
        misprice = (c_early - c_late) / uni if uni > 0 else 0.0

        rows.append(
            {
                **{
                    kk: cell[kk]
                    for kk in (
                        "tag",
                        "layer",
                        "kv_head",
                        "group_size",
                        "head_dim",
                    )
                },
                "agreement_with_first_window": agree,
                "drift": float(agree[0] - agree[-1]),
                "ranks": ranks,
                "entropies": ents,
                "effective_rank_first": spectrum_of(ops[0]).effective_rank,
                "effective_rank_last": spectrum_of(ops[-1]).effective_rank,
                "cost_early_plan": c_early,
                "cost_late_plan": c_late,
                "cost_union_plan": c_union,
                "cost_uniform": uni,
                "misprice_fraction_of_uniform": float(misprice),
                "union_helps": bool(c_union < c_early),
            }
        )
        r = rows[-1]
        print(
            f"  L{cell['layer']:<3} KV{cell['kv_head']}  agree "
            f"{' '.join(f'{a:.3f}' for a in agree)}  misprice "
            f"{misprice * 100:+7.2f}% of uniform  union_helps "
            f"{r['union_helps']}",
            flush=True,
        )

    if not rows:
        print("no cells graded")
        return 1

    last_agree = [r["agreement_with_first_window"][-1] for r in rows]
    mis = [r["misprice_fraction_of_uniform"] for r in rows]
    med_last, med_mis = float(np.median(last_agree)), float(np.median(mis))

    inversions = sum(
        sum(
            1
            for i in range(len(r["agreement_with_first_window"]) - 1)
            if r["agreement_with_first_window"][i]
            < r["agreement_with_first_window"][i + 1] - 1e-9
        )
        for r in rows
    )
    union_rate = float(np.mean([r["union_helps"] for r in rows]))

    bars = {
        "P0_operator_moves": bool(med_last < MOVES_BAR),
        "P1_drift_accumulates": bool(inversions <= len(rows)),
        "P2_early_misprices_late": bool(med_mis >= MISPRICE_BAR),
        "P3_union_is_a_fix": bool(union_rate > 0.5),
        "P4_anti_vacuity": bool(
            all(
                min(r["ranks"]) >= GRADED_RANK
                and min(r["entropies"]) > ENTROPY_BAR
                for r in rows
            )
        ),
        "P5_census": bool(len(rows) + len(skipped) == len(manifest)),
    }
    verdict = "PASS" if all(bars.values()) else "FAIL"
    if not bars["P0_operator_moves"]:
        verdict = "VOID"

    record = {
        "schema": "readscope-c11b-operator-drift-v1",
        "supersedes": "calibration/records/c11-operator-drift.json",
        "provenance": "post hoc hypothesis from turboquant-pro's unexplained "
        "long-generation negative; P0 declared first and can void the sweep",
        "declared": {
            "graded_rank": GRADED_RANK,
            "n_windows": N_WINDOWS,
            "probe_keys": PROBE_KEYS,
            "budget_bits_per_dim": BUDGET_BITS_PER_DIM,
            "seed": SEED,
            "moves_bar": MOVES_BAR,
            "misprice_bar": MISPRICE_BAR,
            "entropy_bar": ENTROPY_BAR,
            "windowing": "by position, pooling the query group; the stored "
            "array is head-major so a flat slice would cut by head",
            "misprice_denominator": "uniform allocation cost, which cannot "
            "collapse the way the late-plan cost did in C-11",
        },
        "rows": rows,
        "skipped": skipped,
        "summary": {
            "n_cells": len(rows),
            "median_agreement_last_window": med_last,
            "median_misprice_fraction_of_uniform": med_mis,
            "union_helps_rate": union_rate,
            "total_inversions": inversions,
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
    out = (
        Path(__file__).resolve().parent
        / "records"
        / "c11b-operator-drift.json"
    )
    out.write_text(json.dumps(record, indent=2, sort_keys=True))

    print(f"\nmedian agreement, first window vs last: {med_last:.4f}")
    print(f"median misprice: {med_mis * 100:+.2f}% of the uniform cost")
    print(f"union plan helps on {union_rate * 100:.0f}% of cells")
    for k in sorted(bars):
        print(f"{k:<28} {'PASS' if bars[k] else 'FAIL'}")
    print("VERDICT", verdict)
    print(out)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
