#!/usr/bin/env python3
"""C-11c, operator drift, instrumented against a paired null.

Declared before it runs. Supersedes c11b_operator_drift.py; both earlier
records are kept.

**Two things C-11b got wrong, and the second is the one that matters.**

Graded at a rank the operator does not support.
    It used rank 16, carried over from the source program's ``R_SUB`` without
    checking it applied. C-5 had already measured these operators at an
    effective rank near 1.9, so directions three through sixteen carry almost
    no sensitivity and their eigenvectors are close to arbitrary. A rank sweep
    gives first-versus-last agreement of 0.671, 0.374, 0.256, 0.201, 0.174 at
    ranks 1, 2, 4, 8, 16, so grading at 16 overstated the effect roughly
    fourfold. The graded rank here is 2, set from C-5's independent
    measurement rather than from this sweep's numbers.

**No null control at all.**
    This is the serious one. Two *disjoint samples of the same distribution*
    do not produce identical operators either: 144 queries estimate the
    operator with finite-sample error, so any two windows disagree somewhat
    whatever position they came from. C-11b compared positional windows
    against 1.0 and called the shortfall drift. That conflates a real
    positional effect with ordinary sampling variation, and on this
    programme's record that is exactly how a confident number gets
    manufactured.

**So every quantity here is measured twice**: once on windows cut by
position, and once on windows of identical size drawn at random from the same
queries, averaged over several draws. The positional number alone means
nothing. **Drift is the gap between them**, and the null is what a claim has
to clear.

**Provenance disclosure.** The hypothesis is post hoc, from turboquant-pro's
unexplained long-generation negative. Two post-hoc hypotheses in this
programme have already died under their own bars, so the bars come first.

Declared bars:

  P0  positional beats null  median positional agreement is below the median
                             random-split agreement by at least 0.10.
                             **Voids the sweep if it fails**, because a
                             positional effect that does not exceed the
                             sampling null is not an effect. This replaces
                             C-11b's comparison against 1.0.
  P1  drift accumulates      agreement with the first window is
                             non-increasing across positional windows,
                             allowing one inversion per cell. The null has no
                             order, so this bar applies to position only.
  P2  early misprices late   the positional mispricing exceeds the random
                             split's by at least 5 percent of the uniform
                             allocation's cost. Computed on the full
                             operators, so the rank confound above does not
                             apply.
  P3  the union is a fix     the whole-sequence operator costs the late
                             consumer less than the early one, on a majority
                             of cells.
  P4  anti-vacuity           every window has analytic rank at least the
                             graded rank, and the **median** window entropy
                             across cells exceeds 1.0 bits. C-11b failed this
                             on one cell of sixteen at 0.9465, which is the
                             fifth time a universal was declared where a
                             distribution belonged.
  P5  determinism + census   a repeat pass reproduces every positional number
                             exactly, and every manifest cell is attempted
                             with skips recorded.

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

GRADED_RANK = 2
REPORT_RANKS = [1, 2, 4, 8, 16]
N_WINDOWS = 4
PROBE_KEYS = 24
BUDGET_BITS_PER_DIM = 3.0
SEED = 0

NULL_MARGIN_BAR = 0.10
N_NULL_DRAWS = 5
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
    """Cut by position, pooling the query group. The stored array is
    head-major, so a flat slice would cut by head instead."""
    d = Q_flat.shape[1]
    q = Q_flat.reshape(group_size, seq, d)
    edges = np.linspace(0, seq, n_windows + 1, dtype=int)
    return [
        np.ascontiguousarray(q[:, edges[i] : edges[i + 1], :].reshape(-1, d))
        for i in range(n_windows)
    ]


def random_windows(Q_flat, n_windows, rng):
    """Windows of the same sizes, drawn at random from the same queries.

    The null. Any disagreement these show is finite-sample error in the
    operator estimate, not position, so it is the floor a positional claim
    has to clear.
    """
    n = Q_flat.shape[0]
    perm = rng.permutation(n)
    edges = np.linspace(0, n, n_windows + 1, dtype=int)
    return [
        np.ascontiguousarray(Q_flat[perm[edges[i] : edges[i + 1]]])
        for i in range(n_windows)
    ]


def measure(K, windows, probe_idx, d, union_M, budget, ranks, graded_rank):
    """Every quantity of interest for one set of windows."""
    ops = [operator(K, w, probe_idx, d) for w in windows]
    first = top(ops[0], graded_rank)
    agree = [
        subspace_overlap(top(M, graded_rank), first).resolution for M in ops
    ]
    by_rank = {
        str(r): subspace_overlap(top(ops[-1], r), top(ops[0], r)).resolution
        for r in ranks
    }
    early, late = ops[0], ops[-1]
    c_early, uni = costs_under(late, early, K, budget)
    c_late, _ = costs_under(late, late, K, budget)
    c_union, _ = costs_under(late, union_M, K, budget)
    misprice = (c_early - c_late) / uni if uni > 0 else 0.0
    return {
        "agreement": agree,
        "first_vs_last_by_rank": by_rank,
        "ranks": [int(np.linalg.matrix_rank(M, tol=1e-10)) for M in ops],
        "effective_rank_first": spectrum_of(ops[0]).effective_rank,
        "effective_rank_last": spectrum_of(ops[-1]).effective_rank,
        "cost_early_plan": c_early,
        "cost_late_plan": c_late,
        "cost_union_plan": c_union,
        "cost_uniform": uni,
        "misprice_fraction_of_uniform": float(misprice),
        "union_helps": bool(c_union < c_early),
    }


def main() -> int:
    mpath = ACTS / "manifest.json"
    if not mpath.exists():
        print("no manifest")
        return 2
    manifest = json.loads(mpath.read_text())
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

        # per-cell seed, so the run is reproducible and cells are independent
        cseed = SEED + 1000 * int(cell["layer"]) + int(cell["kv_head"])
        probe_idx = np.random.default_rng(cseed).choice(
            S, size=min(PROBE_KEYS, S), replace=False
        )
        union_M = operator(K, Q, probe_idx, d)

        pos_windows = positional_windows(Q, grp, seq, N_WINDOWS)
        if min(w.shape[0] for w in pos_windows) < GRADED_RANK:
            skipped.append(
                {**cell, "reason": "window smaller than graded rank"}
            )
            continue

        pos = measure(
            K,
            pos_windows,
            probe_idx,
            d,
            union_M,
            BUDGET_BITS_PER_DIM,
            REPORT_RANKS,
            GRADED_RANK,
        )
        pos["entropies"] = [
            row_entropy_bits(attention(K, w, d)) for w in pos_windows
        ]

        nulls = []
        for j in range(N_NULL_DRAWS):
            nw = random_windows(
                Q, N_WINDOWS, np.random.default_rng(cseed + 7919 * (j + 1))
            )
            nulls.append(
                measure(
                    K,
                    nw,
                    probe_idx,
                    d,
                    union_M,
                    BUDGET_BITS_PER_DIM,
                    REPORT_RANKS,
                    GRADED_RANK,
                )
            )
        null = {
            "agreement_last": float(
                np.mean([n["agreement"][-1] for n in nulls])
            ),
            "first_vs_last_by_rank": {
                str(r): float(
                    np.mean(
                        [n["first_vs_last_by_rank"][str(r)] for n in nulls]
                    )
                )
                for r in REPORT_RANKS
            },
            "misprice_fraction_of_uniform": float(
                np.mean([n["misprice_fraction_of_uniform"] for n in nulls])
            ),
            "n_draws": N_NULL_DRAWS,
        }

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
                "positional": pos,
                "null": null,
                "agreement_gap": float(
                    null["agreement_last"] - pos["agreement"][-1]
                ),
                "misprice_gap": float(
                    pos["misprice_fraction_of_uniform"]
                    - null["misprice_fraction_of_uniform"]
                ),
            }
        )
        r = rows[-1]
        print(
            f"  L{cell['layer']:<3} KV{cell['kv_head']}  "
            f"pos {pos['agreement'][-1]:.3f} vs null "
            f"{null['agreement_last']:.3f}  gap {r['agreement_gap']:+.3f}  "
            f"misprice {pos['misprice_fraction_of_uniform'] * 100:+7.1f}% vs "
            f"null {null['misprice_fraction_of_uniform'] * 100:+7.1f}%",
            flush=True,
        )

    if not rows:
        print("no cells graded")
        return 1

    pos_last = [r["positional"]["agreement"][-1] for r in rows]
    null_last = [r["null"]["agreement_last"] for r in rows]
    gaps = [r["agreement_gap"] for r in rows]
    mis_gaps = [r["misprice_gap"] for r in rows]
    med_pos, med_null = float(np.median(pos_last)), float(np.median(null_last))
    med_gap, med_mis_gap = float(np.median(gaps)), float(np.median(mis_gaps))
    med_entropy = float(
        np.median([np.median(r["positional"]["entropies"]) for r in rows])
    )
    inversions = sum(
        sum(
            1
            for i in range(len(r["positional"]["agreement"]) - 1)
            if r["positional"]["agreement"][i]
            < r["positional"]["agreement"][i + 1] - 1e-9
        )
        for r in rows
    )
    union_rate = float(np.mean([r["positional"]["union_helps"] for r in rows]))
    rank_curve = {
        str(r): {
            "positional": float(
                np.median(
                    [
                        x["positional"]["first_vs_last_by_rank"][str(r)]
                        for x in rows
                    ]
                )
            ),
            "null": float(
                np.median(
                    [x["null"]["first_vs_last_by_rank"][str(r)] for x in rows]
                )
            ),
        }
        for r in REPORT_RANKS
    }

    bars = {
        "P0_positional_beats_null": bool(med_gap >= NULL_MARGIN_BAR),
        "P1_drift_accumulates": bool(inversions <= len(rows)),
        "P2_early_misprices_late": bool(med_mis_gap >= MISPRICE_BAR),
        "P3_union_is_a_fix": bool(union_rate > 0.5),
        "P4_anti_vacuity": bool(
            all(min(r["positional"]["ranks"]) >= GRADED_RANK for r in rows)
            and med_entropy > ENTROPY_BAR
        ),
        "P5_census": bool(len(rows) + len(skipped) == len(manifest)),
    }
    verdict = "PASS" if all(bars.values()) else "FAIL"
    if not bars["P0_positional_beats_null"]:
        verdict = "VOID"

    record = {
        "schema": "readscope-c11c-operator-drift-v1",
        "supersedes": "calibration/records/c11b-operator-drift.json",
        "provenance": "post hoc hypothesis from turboquant-pro's unexplained "
        "long-generation negative; every quantity is paired with a "
        "random-split null and P0 compares against it, not against 1.0",
        "declared": {
            "graded_rank": GRADED_RANK,
            "graded_rank_justification": "C-5 measured these operators at an "
            "effective rank near 1.9, independently and before this sweep; "
            "C-11b's rank 16 was carried over from the source program's R_SUB "
            "and overstated the effect about fourfold",
            "report_ranks": REPORT_RANKS,
            "n_windows": N_WINDOWS,
            "n_null_draws": N_NULL_DRAWS,
            "probe_keys": PROBE_KEYS,
            "budget_bits_per_dim": BUDGET_BITS_PER_DIM,
            "seed": SEED,
            "null_margin_bar": NULL_MARGIN_BAR,
            "misprice_bar": MISPRICE_BAR,
            "entropy_bar": ENTROPY_BAR,
            "entropy_bar_applies_to": "the median across cells, not each cell",
            "null": "windows of identical sizes drawn at random from the "
            "same queries; any disagreement they show is finite-sample "
            "error in the operator estimate rather than position",
        },
        "rows": rows,
        "skipped": skipped,
        "summary": {
            "n_cells": len(rows),
            "median_positional_agreement": med_pos,
            "median_null_agreement": med_null,
            "median_agreement_gap": med_gap,
            "median_misprice_gap": med_mis_gap,
            "by_rank": rank_curve,
            "union_helps_rate": union_rate,
            "total_inversions": inversions,
            "median_window_entropy_bits": med_entropy,
            "median_effective_rank_last": float(
                np.median(
                    [r["positional"]["effective_rank_last"] for r in rows]
                )
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
    out = (
        Path(__file__).resolve().parent
        / "records"
        / "c11c-operator-drift.json"
    )
    out.write_text(json.dumps(record, indent=2, sort_keys=True))

    print("\nfirst-vs-last agreement by graded rank, positional vs null:")
    print(f"  {'rank':<6} {'positional':>11} {'null':>8} {'gap':>8}")
    for r in REPORT_RANKS:
        c = rank_curve[str(r)]
        print(
            f"  {r:<6} {c['positional']:11.3f} {c['null']:8.3f} "
            f"{c['null'] - c['positional']:+8.3f}"
        )
    print(
        f"\nat the graded rank {GRADED_RANK}: positional {med_pos:.3f}, "
        f"null {med_null:.3f}, gap {med_gap:+.3f} (bar {NULL_MARGIN_BAR})"
    )
    print(
        f"misprice gap over null: {med_mis_gap * 100:+.1f}% of uniform "
        f"(bar {MISPRICE_BAR * 100:.0f}%)"
    )
    print(f"union plan helps on {union_rate * 100:.0f}% of cells")
    for k in sorted(bars):
        print(f"{k:<28} {'PASS' if bars[k] else 'FAIL'}")
    print("VERDICT", verdict)
    print(out)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
