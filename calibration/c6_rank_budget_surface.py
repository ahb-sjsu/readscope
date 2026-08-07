#!/usr/bin/env python3
"""C-6, the rank-budget surface. Declared before it runs.

**Provenance disclosure.** C-5 reported, without a bar on it, that every real
head's analytic read operator has exact rank 24 and effective rank about 1.8.
That observation is where this hypothesis comes from, so the hypothesis is
post hoc and the bars below are the first thing that tests it. Saying so is
the only remedy available; what makes the test worth anything is that the
bars are fixed here, before the sweep, and that S2 can fail.

C-2e measured bandwidth against budget at a fixed graded rank of 16 and found
a cliff at ``k/d = 1``. It never swept the other axis. If a real head
concentrates its sensitivity in two directions, then demanding a rank-16
subspace may be demanding mostly noise, and the budget needed to recover what
actually carries the mass may be far below the cliff.

So this measures the **surface**: required budget as a function of graded
rank. For each cell and each budget the probe is run once and the recovered
operator graded at every rank, since the probe does not know what rank it
will be judged at.

Substrate is the four-family activation set from C-3b, real post-RoPE keys
and real queries, graded against the same exact closed form. Architecture
diversity matters more here than scale, which C-5 already showed invariant.

Declared bars:

  S1  cliff reproduces   at graded rank 16, the smallest budget reaching
                         resolution 0.90 is at least 1.0. This is C-2e's
                         result restated on this substrate, and it must hold
                         or the sweep is measuring something else.
  S2  the question       at graded rank 2, the smallest budget reaching
                         resolution 0.90 is strictly below 1.0. **This is the
                         falsifiable claim.** If it fails, the cliff is
                         rank-independent, the effective-rank observation
                         buys nothing operationally, and the pessimistic
                         specification stands unqualified.
  S3  monotone in rank   the required budget is non-decreasing in graded
                         rank. A higher rank cannot be cheaper to recover,
                         and if it appears to be, the statistic is wrong.
  S4  rank floor         the analytic operator has rank at least 16 on every
                         cell, so grading at 16 is never against a degenerate
                         truth.
  S5  anti-vacuity       median attention row entropy above 1.0 bits on every
                         cell, the check that caught C-3's query artifact.
  S6  census             every manifest cell attempted, skips recorded.

Runs on Atlas. Numpy only.
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

ACTS = Path("/archive/readscope/activations_v2")
BUDGETS = [0.125, 0.25, 0.5, 0.75, 1.0, 1.25]
RANKS = [1, 2, 4, 8, 16]
PROBE_KEYS = 32
EPS = 1e-3
SEED = 0
RESOLUTION_BAR = 0.90
ENTROPY_BAR = 1.0
RANK_FLOOR = 16


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


def required_budget(curve):
    """Smallest budget whose resolution clears the bar, else None."""
    for b in BUDGETS:
        if curve[str(b)] >= RESOLUTION_BAR:
            return b
    return None


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

        probe_idx = rng.choice(S, size=min(PROBE_KEYS, S), replace=False)
        M_true = analytic_operator(K, Q, probe_idx, d)
        truth_rank = int(np.linalg.matrix_rank(M_true, tol=1e-10))
        spec = spectrum_of(M_true)
        ent = row_entropy_bits(attention(K, Q, d))

        # one probe per budget, graded at every rank
        grid = {str(r): {} for r in RANKS}
        for b in BUDGETS:
            k = max(1, int(round(b * d)))
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
            for r in RANKS:
                grid[str(r)][str(b)] = subspace_overlap(
                    top(acc, r), top(M_true, r)
                ).resolution

        rows.append(
            {
                **{
                    kk: cell[kk]
                    for kk in ("tag", "family", "layer", "head", "head_dim")
                },
                "grid": grid,
                "truth_rank": truth_rank,
                "effective_rank": spec.effective_rank,
                "median_row_entropy_bits": ent,
                "required_budget": {
                    str(r): required_budget(grid[str(r)]) for r in RANKS
                },
            }
        )
        req = rows[-1]["required_budget"]
        print(
            f"  {cell['tag']:<14} L{cell['layer']:<3} H{cell['head']} "
            f"eff {spec.effective_rank:4.2f}  required k/d "
            + " ".join(f"r{r}={req[str(r)]}" for r in RANKS),
            flush=True,
        )

    if not rows:
        print("no cells graded")
        return 1

    def mean_curve(r):
        return {
            str(b): float(np.mean([x["grid"][str(r)][str(b)] for x in rows]))
            for b in BUDGETS
        }

    surface = {str(r): mean_curve(r) for r in RANKS}
    mean_required = {str(r): required_budget(surface[str(r)]) for r in RANKS}

    def as_num(v):
        return 99.0 if v is None else v

    bars = {
        "S1_cliff_reproduces": bool(as_num(mean_required["16"]) >= 1.0),
        "S2_rank2_is_subdimensional": bool(as_num(mean_required["2"]) < 1.0),
        "S3_monotone_in_rank": bool(
            all(
                as_num(mean_required[str(RANKS[i])])
                <= as_num(mean_required[str(RANKS[i + 1])])
                for i in range(len(RANKS) - 1)
            )
        ),
        "S4_rank_floor": bool(
            all(r["truth_rank"] >= RANK_FLOOR for r in rows)
        ),
        "S5_anti_vacuity": bool(
            all(r["median_row_entropy_bits"] > ENTROPY_BAR for r in rows)
        ),
        "S6_census": bool(len(rows) + len(skipped) == len(manifest)),
    }
    verdict = "PASS" if all(bars.values()) else "FAIL"

    record = {
        "schema": "readscope-c6-rank-budget-surface-v1",
        "provenance": "hypothesis is post hoc, prompted by C-5's unbarred "
        "effective-rank observation; bars fixed before this run",
        "declared": {
            "budgets": BUDGETS,
            "ranks": RANKS,
            "probe_keys": PROBE_KEYS,
            "eps": EPS,
            "seed": SEED,
            "resolution_bar": RESOLUTION_BAR,
            "entropy_bar": ENTROPY_BAR,
            "rank_floor": RANK_FLOOR,
        },
        "rows": rows,
        "skipped": skipped,
        "surface_mean_resolution": surface,
        "mean_required_budget": mean_required,
        "summary": {
            "n_cells": len(rows),
            "families": sorted({r["family"] for r in rows}),
            "median_effective_rank": float(
                np.median([r["effective_rank"] for r in rows])
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
        / "c6-rank-budget-surface.json"
    )
    out.write_text(json.dumps(record, indent=2, sort_keys=True))

    print("\nmean resolution surface, rows are graded rank")
    print("  " + f"{'rank':<6} " + " ".join(f"{b:>7}" for b in BUDGETS))
    for r in RANKS:
        print(
            f"  {r:<6} "
            + " ".join(f"{surface[str(r)][str(b)]:7.3f}" for b in BUDGETS)
            + f"   required k/d {mean_required[str(r)]}"
        )
    for k in sorted(bars):
        print(f"{k:<30} {'PASS' if bars[k] else 'FAIL'}")
    print("VERDICT", verdict)
    print(out)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
