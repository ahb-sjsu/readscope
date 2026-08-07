#!/usr/bin/env python3
"""C-4, the reference-choice audit. Declared before it runs.

C-3b found the blind probe recovers the analytic read operator at resolution
1.000 on every real head-cell at ``k/d = 1.25``. The source program reports a
median overlap of 0.647 at the same budget ratio on the same model family.
Both cannot be a statement about probe fidelity, so one of them is measuring
something else.

Reading `gateB_llama_rematch.py` shows the difference is the **reference**,
not the probe. That script grades against

    Pc_true = Qset^T Qset / n_q

the unweighted query covariance. What a finite-difference probe recovers is
the Jacobian Gram, which is the same queries weighted by how much the softmax
actually responds along each one:

    M_true = sum_s sum_i a_{i,s} q_i q_i^T / d,
    a_{i,s} = || p_i * (e_s - p_{i,s} 1) ||^2

Both are spanned by the queries. They are not the same operator, because the
softmax weights are far from uniform.

**Hypothesis under test.** The published 0.647 is the distance between those
two references, not a limitation of the probe. If so, comparing the weighted
operator against the unweighted query covariance on real cells should land in
the same range as the published figure, using only the probe-free closed
forms.

Declared bars:

  R1  references differ   on every cell, the resolution of the weighted
                          operator's top-r subspace against the unweighted
                          query covariance's top-r is below 0.90. If they
                          agree, the hypothesis is wrong and the gap needs
                          another explanation.
  R2  replicates the gap  the median of that resolution falls in [0.35, 0.85].
                          This band is set from the two published Llama
                          figures, 0.567 and 0.647, which as resolutions are
                          0.505 and 0.596. It is a replication target taken
                          from an external record, not a bar fitted to data
                          measured here, and it can fail.
  R3  probe still exact   the blind probe against the weighted operator stays
                          at resolution 0.99 or above on every cell, so R1 is
                          not being produced by a degraded probe.
  R4  anti-vacuity        every cell has median row entropy above 0.1 bits
                          and both references have rank at least the graded
                          rank.
  R5  census              every manifest cell is attempted and skips are
                          recorded.

**This audits a reference choice and does not correct anyone.** An unweighted
query covariance is a defensible definition of what a head reads, since it
does not depend on which key is being perturbed. The finding, if it holds, is
only that the published number measures agreement between two definitions
rather than the accuracy of an instrument, and that a datasheet must say
which.

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

from readscope import jacobian_probe, subspace_overlap  # noqa: E402

ACTS = Path("/archive/readscope/activations_v2")
RATIO = 1.25
GRADED_RANK = 16
PROBE_KEYS = 32
EPS = 1e-3
SEED = 0
PROBE_BAR = 0.99
DIFFER_BAR = 0.90
REPLICATION_BAND = (0.35, 0.85)
ENTROPY_FLOOR = 0.1


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def attention(K, Q, d):
    return softmax((Q @ K.T) / np.sqrt(d))


def weighted_operator(K, Q, probe_idx, d):
    P = attention(K, Q, d)
    M = np.zeros((d, d))
    for s in probe_idx:
        e = np.zeros(P.shape[1])
        e[s] = 1.0
        W = P * (e[None, :] - P[:, s : s + 1])
        a = (W**2).sum(axis=1)
        M += (Q * a[:, None]).T @ Q / d
    return 0.5 * (M + M.T)


def unweighted_operator(Q):
    return Q.T @ Q / Q.shape[0]


def top(M, r):
    _, V = np.linalg.eigh(0.5 * (M + M.T))
    return np.ascontiguousarray(V[:, ::-1][:, :r])


def row_entropy_bits(P):
    q = np.clip(P, 1e-12, 1.0)
    return float(np.median(-(q * np.log2(q)).sum(axis=1)))


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
        M_w = weighted_operator(K, Q, probe_idx, d)
        M_u = unweighted_operator(Q)
        sub_w, sub_u = top(M_w, GRADED_RANK), top(M_u, GRADED_RANK)

        k = max(1, int(round(RATIO * d)))
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

        rows.append(
            {
                **{
                    kk: cell[kk]
                    for kk in ("tag", "family", "layer", "head", "head_dim")
                },
                "probe_vs_weighted": subspace_overlap(
                    top(acc, GRADED_RANK), sub_w
                ).resolution,
                "weighted_vs_unweighted": subspace_overlap(
                    sub_w, sub_u
                ).resolution,
                "weighted_vs_unweighted_overlap": subspace_overlap(
                    sub_w, sub_u
                ).overlap,
                "rank_weighted": int(np.linalg.matrix_rank(M_w, tol=1e-10)),
                "rank_unweighted": int(np.linalg.matrix_rank(M_u, tol=1e-10)),
                "median_row_entropy_bits": row_entropy_bits(
                    attention(K, Q, d)
                ),
            }
        )
        print(
            f"  {cell['tag']:<14} L{cell['layer']:<3} H{cell['head']} "
            f"probe/weighted {rows[-1]['probe_vs_weighted']:.4f}  "
            f"weighted/unweighted {rows[-1]['weighted_vs_unweighted']:.4f}",
            flush=True,
        )

    if not rows:
        print("no cells")
        return 1

    wu = [r["weighted_vs_unweighted"] for r in rows]
    pw = [r["probe_vs_weighted"] for r in rows]
    med = float(np.median(wu))

    bars = {
        "R1_references_differ": bool(max(wu) < DIFFER_BAR),
        "R2_replicates_the_gap": bool(
            REPLICATION_BAND[0] <= med <= REPLICATION_BAND[1]
        ),
        "R3_probe_still_exact": bool(min(pw) >= PROBE_BAR),
        "R4_anti_vacuity": bool(
            all(
                r["median_row_entropy_bits"] > ENTROPY_FLOOR
                and r["rank_weighted"] >= GRADED_RANK
                and r["rank_unweighted"] >= GRADED_RANK
                for r in rows
            )
        ),
        "R5_census": bool(len(rows) + len(skipped) == len(manifest)),
    }
    verdict = "PASS" if all(bars.values()) else "FAIL"

    record = {
        "schema": "readscope-c4-reference-choice-v1",
        "declared": {
            "ratio": RATIO,
            "graded_rank": GRADED_RANK,
            "probe_keys": PROBE_KEYS,
            "eps": EPS,
            "seed": SEED,
            "probe_bar": PROBE_BAR,
            "differ_bar": DIFFER_BAR,
            "replication_band": list(REPLICATION_BAND),
            "published_reference": "geometric-observation GO-P-2026-020 "
            "and 021, overlaps 0.567 and 0.647, resolutions 0.505 and 0.596",
        },
        "rows": rows,
        "skipped": skipped,
        "summary": {
            "n_cells": len(rows),
            "median_weighted_vs_unweighted": med,
            "mean_weighted_vs_unweighted": float(np.mean(wu)),
            "min_weighted_vs_unweighted": float(min(wu)),
            "max_weighted_vs_unweighted": float(max(wu)),
            "median_weighted_vs_unweighted_raw_overlap": float(
                np.median([r["weighted_vs_unweighted_overlap"] for r in rows])
            ),
            "min_probe_vs_weighted": float(min(pw)),
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
        / "c4-reference-choice.json"
    )
    out.write_text(json.dumps(record, indent=2, sort_keys=True))

    print(
        f"\nweighted vs unweighted resolution: median {med:.4f} "
        f"range [{min(wu):.4f}, {max(wu):.4f}]"
    )
    print(f"probe vs weighted: min {min(pw):.4f}")
    for k in sorted(bars):
        print(f"{k:<26} {'PASS' if bars[k] else 'FAIL'}")
    print("VERDICT", verdict)
    print(out)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
