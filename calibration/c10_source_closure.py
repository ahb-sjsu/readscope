#!/usr/bin/env python3
"""C-10, closing C-4's unmatched remainder. Declared before it runs.

C-4 established that the published 0.647 measures agreement between two
*references* rather than the accuracy of the probe: the source program grades
against the unweighted query covariance ``Qset^T Qset / n`` while a
finite-difference probe recovers the softmax-weighted Jacobian Gram. It
measured that disagreement at median resolution 0.796, inside the band the
published figures imply but well above them, so it named a residual and left
it open. The unmatched factors were the query capture, the grouped-query
grouping and the model and layer set.

All three are now matched to `gateB_llama_rematch.py`: Llama-3.2-3B, layers
{8, 16}, float32, and crucially ``Qset`` as **every query in a key-value
head's group across the whole sequence**, 576 vectors rather than the 24 C-4
sampled. That changes the unweighted reference from a rank-24 covariance with
a well separated top-16 to a full-rank one whose top-16 is far less
determined, which is the leading candidate for the residual.

The probe settings are the source's: 32 probe keys, 160 unit-norm directions,
step 1e-3, least squares by pseudoinverse, graded at rank 16.

**The closure argument is arithmetic, not rhetorical.** If the probe recovers
the weighted operator essentially exactly, then its overlap with the
unweighted reference must equal the weighted reference's overlap with the
unweighted one. Showing those two coincide, and that their common value
reproduces the published figure, closes the remainder: nothing is left for
another cause to explain.

Declared bars:

  M1  probe is exact      probe against the weighted operator is at least
                          0.99 resolution on every cell. Everything else
                          rests on this.
  M2  replicates          the median probe-against-unweighted **overlap**
                          falls in [0.55, 0.75]. That band comes from the two
                          published Llama figures, 0.567 and 0.647, and is a
                          replication target taken from an external record
                          rather than a bar fitted here. It can fail.
  M3  the remainder closes on every cell, the probe-against-unweighted
                          overlap and the weighted-against-unweighted overlap
                          agree within 0.05. **This is the closure.** If they
                          agree and M2 holds, the whole published figure is
                          the reference choice and no residual is left.
  M4  reference matters   the median weighted-against-unweighted overlap is
                          below 0.90, so the two references really are
                          different objects and M3 is not agreement between
                          two names for one thing.
  M5  anti-vacuity        every cell's weighted operator has rank at least
                          16, its unweighted reference has rank at least 16,
                          and median attention row entropy exceeds 1.0 bits.
  M6  census              every manifest cell attempted, skips recorded.

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

ACTS = Path("/archive/readscope/source_match")

R_SUB = 16
N_PROBE = 160
PROBE_KEYS = 32
H_FD = 1e-3
SEED = 0

PROBE_EXACT_BAR = 0.99
REPLICATION_BAND = (0.55, 0.75)
CLOSURE_BAR = 0.05
REFERENCE_DIFFERS_BAR = 0.90
ENTROPY_BAR = 1.0


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def attention(K, Q, d):
    return softmax((Q @ K.T) / np.sqrt(d))


def weighted_operator(K, Q, probe_idx, d):
    """The Jacobian Gram, which is what a finite-difference probe recovers."""
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
    """What the source grades against."""
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
        print("no manifest; run extract_source_match.py first")
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

        rng = np.random.default_rng(
            SEED + 1000 * cell["layer"] + cell["kv_head"]
        )
        probe_idx = rng.choice(S, size=min(PROBE_KEYS, S), replace=False)

        M_w = weighted_operator(K, Q, probe_idx, d)
        M_u = unweighted_operator(Q)
        sub_w, sub_u = top(M_w, R_SUB), top(M_u, R_SUB)

        acc = np.zeros((d, d))
        for s in probe_idx:

            def consumer(kv, _s=s, _K=K, _Q=Q, _d=d):
                Kp = _K.copy()
                Kp[_s] = kv
                return attention(Kp, _Q, _d).ravel()

            acc += jacobian_probe(
                consumer,
                K[s][None, :],
                n_directions=N_PROBE,
                eps=H_FD,
                rng=np.random.default_rng(SEED + int(s) * 7919 % 100003),
            ).S
        sub_p = top(acc, R_SUB)

        pw = subspace_overlap(sub_p, sub_w)
        pu = subspace_overlap(sub_p, sub_u)
        wu = subspace_overlap(sub_w, sub_u)

        rows.append(
            {
                **{
                    kk: cell[kk]
                    for kk in (
                        "layer",
                        "kv_head",
                        "group_size",
                        "n_queries",
                        "head_dim",
                    )
                },
                "probe_vs_weighted_resolution": pw.resolution,
                "probe_vs_unweighted_overlap": pu.overlap,
                "weighted_vs_unweighted_overlap": wu.overlap,
                "closure_gap": abs(pu.overlap - wu.overlap),
                "rank_weighted": int(np.linalg.matrix_rank(M_w, tol=1e-10)),
                "rank_unweighted": int(np.linalg.matrix_rank(M_u, tol=1e-10)),
                "median_row_entropy_bits": row_entropy_bits(
                    attention(K, Q, d)
                ),
            }
        )
        r = rows[-1]
        print(
            f"  L{cell['layer']:<3} KV{cell['kv_head']}  "
            f"probe/weighted {r['probe_vs_weighted_resolution']:.4f}  "
            f"probe/unweighted {r['probe_vs_unweighted_overlap']:.4f}  "
            f"weighted/unweighted {r['weighted_vs_unweighted_overlap']:.4f}  "
            f"gap {r['closure_gap']:.4f}",
            flush=True,
        )

    if not rows:
        print("no cells graded")
        return 1

    pw = [r["probe_vs_weighted_resolution"] for r in rows]
    pu = [r["probe_vs_unweighted_overlap"] for r in rows]
    wu = [r["weighted_vs_unweighted_overlap"] for r in rows]
    gaps = [r["closure_gap"] for r in rows]
    med_pu = float(np.median(pu))

    bars = {
        "M1_probe_is_exact": bool(min(pw) >= PROBE_EXACT_BAR),
        "M2_replicates_published": bool(
            REPLICATION_BAND[0] <= med_pu <= REPLICATION_BAND[1]
        ),
        "M3_remainder_closes": bool(max(gaps) <= CLOSURE_BAR),
        "M4_reference_really_differs": bool(
            float(np.median(wu)) < REFERENCE_DIFFERS_BAR
        ),
        "M5_anti_vacuity": bool(
            all(
                r["rank_weighted"] >= R_SUB
                and r["rank_unweighted"] >= R_SUB
                and r["median_row_entropy_bits"] > ENTROPY_BAR
                for r in rows
            )
        ),
        "M6_census": bool(len(rows) + len(skipped) == len(manifest)),
    }
    verdict = "PASS" if all(bars.values()) else "FAIL"

    record = {
        "schema": "readscope-c10-source-closure-v1",
        "closes": "calibration/records/c4-reference-choice.json",
        "declared": {
            "r_sub": R_SUB,
            "n_probe": N_PROBE,
            "probe_keys": PROBE_KEYS,
            "h_fd": H_FD,
            "seed": SEED,
            "probe_exact_bar": PROBE_EXACT_BAR,
            "replication_band": list(REPLICATION_BAND),
            "closure_bar": CLOSURE_BAR,
            "reference_differs_bar": REFERENCE_DIFFERS_BAR,
            "published": "GO-P-2026-020 overlap 0.567, "
            "GO-P-2026-021 overlap 0.647",
            "protocol": "matched to gateB_llama_rematch.py",
        },
        "rows": rows,
        "skipped": skipped,
        "summary": {
            "n_cells": len(rows),
            "median_probe_vs_unweighted_overlap": med_pu,
            "mean_probe_vs_unweighted_overlap": float(np.mean(pu)),
            "median_weighted_vs_unweighted_overlap": float(np.median(wu)),
            "min_probe_vs_weighted_resolution": float(min(pw)),
            "max_closure_gap": float(max(gaps)),
            "median_closure_gap": float(np.median(gaps)),
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
        Path(__file__).resolve().parent / "records" / "c10-source-closure.json"
    )
    out.write_text(json.dumps(record, indent=2, sort_keys=True))

    print(
        f"\nmedian probe-vs-unweighted overlap {med_pu:.4f}   "
        f"published 0.567 and 0.647"
    )
    print(
        f"median weighted-vs-unweighted overlap "
        f"{float(np.median(wu)):.4f}   max closure gap {max(gaps):.4f}"
    )
    print(f"min probe-vs-weighted resolution {min(pw):.4f}")
    for k in sorted(bars):
        print(f"{k:<30} {'PASS' if bars[k] else 'FAIL'}")
    print("VERDICT", verdict)
    print(out)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
