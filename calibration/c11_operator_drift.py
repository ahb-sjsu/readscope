#!/usr/bin/env python3
"""C-11, does an attention head's read operator drift along the sequence?

Declared before it runs.

**Provenance disclosure.** This hypothesis is post hoc. It came from noticing
that turboquant-pro's README carries an unexplained negative, that all 4-bit
KV quantization degrades on very-long-generation tasks, and asking whether the
read operator is the thing that moves. Two post-hoc hypotheses in this
programme have already died under their own bars, C-6's rank hypothesis and
C-9's loading correction, so this one is declared with a bar that can kill it
first.

**The question.** A key written at position ``s`` is read by every query at
positions ``s`` and later. Its read operator is

    P_C(t) = sum_{i <= t} a_i q_i q_i^T / d

spanned by the queries that have arrived so far. If the query distribution
drifts with position then ``P_C`` measured early is not the operator that
reads the key later, and a codebook fitted at calibration is being applied to
a consumer that has since moved. That is probe loading along the time axis,
and this programme already has the axis to measure it.

If it drifts, the long-generation degradation has a mechanism and a fix, which
is to allocate against the union operator rather than the early one. If it
does not, the cause is elsewhere and that is worth knowing too.

**The mandatory bar comes first.** C-7b and C-8 both passed bars for reasons
unrelated to their claims, so the standing rule is that before believing "X
predicts Y" one must first bar that Y varies. Here Y is the read operator
itself.

Declared bars, in the order they must be believed:

  P0  the operator moves   the early-window operator and the late-window
                           operator differ, at resolution below 0.90 on the
                           median cell. **If this fails the sweep is void**,
                           because nothing downstream can be about drift.
  P1  drift is monotone    subspace agreement with the first window is
                           non-increasing as the window moves later, allowing
                           one inversion. Drift should accumulate, not
                           oscillate.
  P2  early misprices late allocating bits against the early operator costs
                           the late consumer more than allocating against the
                           late one, by at least 5 percent in predicted
                           distortion. **This is the claim with teeth**: it
                           says a calibration-time codebook is measurably
                           wrong later, which is what would explain the
                           long-generation negative.
  P3  the union is a fix   allocating against the union operator over the
                           whole sequence costs the late consumer less than
                           allocating against the early one. If P2 holds and
                           P3 fails, drift is real and this particular repair
                           does not work.
  P4  anti-vacuity         every cell has analytic rank at least the graded
                           rank in every window, and median attention row
                           entropy above 1.0 bits, so no window is graded on a
                           degenerate head.
  P5  census               every manifest cell attempted, skips recorded.

Substrate is the four-family real activation set, graded against the exact
closed-form Jacobian Gram. Runs on Atlas, numpy only.
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

ACTS = Path("/archive/readscope/activations_v2")

GRADED_RANK = 8
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
    """Closed-form sum_s J_s^T J_s for the query set supplied."""
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
    """Reverse water-filling, the same rule turboquant-pro now ships."""
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


def cost_under(target_M, plan_M, K, budget):
    """Distortion the *target* consumer feels from bits planned on *plan*.

    Both operators are diagonalised in the plan's basis, so the allocation is
    made against the plan and charged against the target. That is exactly the
    situation a calibration-time codebook is in at serving time.
    """
    d = target_M.shape[0]
    _, basis = np.linalg.eigh(0.5 * (plan_M + plan_M.T))
    basis = np.ascontiguousarray(basis[:, ::-1])
    lam_plan = np.clip(np.diag(basis.T @ plan_M @ basis), 0.0, None)
    lam_tgt = np.clip(np.diag(basis.T @ target_M @ basis), 0.0, None)
    centred = K - K.mean(axis=0, keepdims=True)
    var = np.maximum(((centred @ basis) ** 2).mean(axis=0), 0.0)
    bits = water_fill(lam_plan, var, budget * d)
    return float(np.sum(lam_tgt * var * np.power(2.0, -2.0 * bits)))


def main() -> int:
    mpath = ACTS / "manifest.json"
    if not mpath.exists():
        print("no manifest")
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
        n_q = Q.shape[0]
        if n_q < 2 * N_WINDOWS:
            skipped.append({**cell, "reason": "too few queries to window"})
            continue

        probe_idx = rng.choice(S, size=min(PROBE_KEYS, S), replace=False)
        edges = np.linspace(0, n_q, N_WINDOWS + 1, dtype=int)
        windows = [Q[edges[i] : edges[i + 1]] for i in range(N_WINDOWS)]

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
        cost_early_plan = cost_under(late, early, K, BUDGET_BITS_PER_DIM)
        cost_late_plan = cost_under(late, late, K, BUDGET_BITS_PER_DIM)
        cost_union_plan = cost_under(late, union, K, BUDGET_BITS_PER_DIM)

        misprice = (
            (cost_early_plan - cost_late_plan) / cost_late_plan
            if cost_late_plan > 0
            else 0.0
        )
        union_helps = cost_union_plan < cost_early_plan

        rows.append(
            {
                **{
                    kk: cell[kk]
                    for kk in ("tag", "family", "layer", "head", "head_dim")
                },
                "agreement_with_first_window": agree,
                "drift": float(agree[0] - agree[-1]),
                "ranks": ranks,
                "entropies": ents,
                "effective_rank_first": spectrum_of(ops[0]).effective_rank,
                "effective_rank_last": spectrum_of(ops[-1]).effective_rank,
                "cost_early_plan": cost_early_plan,
                "cost_late_plan": cost_late_plan,
                "cost_union_plan": cost_union_plan,
                "misprice_fraction": float(misprice),
                "union_helps": bool(union_helps),
            }
        )
        r = rows[-1]
        print(
            f"  {cell['tag']:<14} L{cell['layer']:<3} H{cell['head']}  "
            f"agree {' '.join(f'{a:.3f}' for a in agree)}  "
            f"misprice {misprice * 100:+6.1f}%  union_helps "
            f"{r['union_helps']}",
            flush=True,
        )

    if not rows:
        print("no cells graded")
        return 1

    last_agree = [r["agreement_with_first_window"][-1] for r in rows]
    mis = [r["misprice_fraction"] for r in rows]
    med_last = float(np.median(last_agree))
    med_mis = float(np.median(mis))

    inversions = 0
    for r in rows:
        a = r["agreement_with_first_window"]
        inversions += sum(
            1 for i in range(len(a) - 1) if a[i] < a[i + 1] - 1e-9
        )

    p0 = bool(med_last < MOVES_BAR)
    p1 = bool(inversions <= len(rows))
    p2 = bool(med_mis >= MISPRICE_BAR)
    p3 = bool(float(np.mean([r["union_helps"] for r in rows])) > 0.5)
    p4 = bool(
        all(
            min(r["ranks"]) >= GRADED_RANK
            and min(r["entropies"]) > ENTROPY_BAR
            for r in rows
        )
    )
    p5 = bool(len(rows) + len(skipped) == len(manifest))

    bars = {
        "P0_operator_moves": p0,
        "P1_drift_monotone": p1,
        "P2_early_misprices_late": p2,
        "P3_union_is_a_fix": p3,
        "P4_anti_vacuity": p4,
        "P5_census": p5,
    }
    verdict = "PASS" if all(bars.values()) else "FAIL"
    if not p0:
        verdict = "VOID"

    record = {
        "schema": "readscope-c11-operator-drift-v1",
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
        },
        "rows": rows,
        "skipped": skipped,
        "summary": {
            "n_cells": len(rows),
            "median_agreement_last_window": med_last,
            "median_misprice_fraction": med_mis,
            "mean_union_helps": float(
                np.mean([r["union_helps"] for r in rows])
            ),
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
        Path(__file__).resolve().parent / "records" / "c11-operator-drift.json"
    )
    out.write_text(json.dumps(record, indent=2, sort_keys=True))

    print(
        f"\nmedian agreement with the first window at the last: {med_last:.4f}"
    )
    print(
        f"median misprice of the early plan on the late consumer: "
        f"{med_mis * 100:+.1f}%"
    )
    union_rate = float(np.mean([r["union_helps"] for r in rows]))
    print(f"union plan helps on {union_rate * 100:.0f}% of cells")
    for k in sorted(bars):
        print(f"{k:<28} {'PASS' if bars[k] else 'FAIL'}")
    print("VERDICT", verdict)
    print(out)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
