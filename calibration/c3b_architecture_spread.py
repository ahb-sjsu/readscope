#!/usr/bin/env python3
"""C-3b, the architecture spread with real queries. Declared before it runs.

Supersedes c3_architecture_spread.py, which failed three of six bars because
its query set was drawn from the key stream. The self-match term saturated
the softmax, median attention entropy came out at 0.152 bits across 42 cells
where 192 positions allow 7.6, and eleven cells had analytic operators below
the graded rank. That was an artifact of the declared simplification and not
a property of any model, so it is not reported as one. C-3's record stays.

Here the queries are real. A forward hook on ``q_proj`` captures the pre-
rotary query projection and the model's own ``rotary_emb`` supplies the
cosines and sines, so the query tensor is the one attention actually
consumed. Grouped-query attention is resolved by mapping each attention head
to the key-value head it reads.

**The ground truth is exact.** For a softmax attention consumer reading one
key, ``d p_i / d k_s = p_i (e_s - p_{i,s}) q_i / sqrt(d)``, so

    M_true = sum_s sum_i a_{i,s} q_i q_i^T / d,
    a_{i,s} = || p_i * (e_s - p_{i,s} 1) ||^2

The read subspace of an attention head with respect to a key is spanned by
its queries. That is a fact about softmax attention, not a claim of this
program.

Bars are carried over from C-3 unchanged, with one added in front:

  A0  artifact cleared median attention row entropy across cells exceeds
                      1.0 bits. This is the check that C-3's failure really
                      was the query simplification. If A0 fails, the queries
                      are still wrong and nothing after it is interpretable.
  A1  instrument      resolution at k/d = 1.25 is at least 0.90 on every
                      cell.
  A2  budget cliff    resolution at k/d = 0.5 is strictly below that at
                      k/d = 1.25 on every cell.
  A3  spread reported per-family resolution and the across-family spread are
                      computed and reported, with no bar on their size.
  A4  rank floor      the analytic operator has rank at least the graded rank
                      on every cell.
  A5  anti-vacuity    median row entropy exceeds 0.1 bits on every cell.
  A6  census          every manifest cell is attempted and every skip is
                      recorded with its reason.

Gemma-3 is absent from this sweep. Its rotary embedding takes a per-layer
type argument that the extraction did not supply, so no real queries were
captured for it. That is a missing family, recorded here rather than papered
over, and it means this sweep covers three architectures and not four.

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

from readscope import jacobian_probe, subspace_overlap  # noqa: E402

ACTS = Path("/archive/readscope/activations_v2")
RATIOS = [0.5, 1.25]
GRADED_RANK = 16
PROBE_KEYS = 32
EPS = 1e-3
SEED = 0
INSTRUMENT_BAR = 0.90
ENTROPY_FLOOR = 0.1


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def attention(K, Q, d):
    return softmax((Q @ K.T) / np.sqrt(d))


def analytic_operator(K, Q, probe_idx, d):
    """Closed-form ``sum_s J_s^T J_s`` for the softmax-attention consumer."""
    P = attention(K, Q, d)
    M = np.zeros((d, d))
    for s in probe_idx:
        e = np.zeros(P.shape[1])
        e[s] = 1.0
        W = P * (e[None, :] - P[:, s : s + 1])
        a = (W**2).sum(axis=1)
        M += (Q * a[:, None]).T @ Q / d
    return 0.5 * (M + M.T)


def row_entropy_bits(P):
    q = np.clip(P, 1e-12, 1.0)
    return float(np.median(-(q * np.log2(q)).sum(axis=1)))


def main() -> int:
    manifest_path = ACTS / "manifest.json"
    if not manifest_path.exists():
        print("no manifest; run extract_activations.py first")
        return 2
    manifest = json.loads(manifest_path.read_text())

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
        if d < GRADED_RANK:
            skipped.append({**cell, "reason": f"head_dim {d} < graded rank"})
            continue

        probe_idx = rng.choice(S, size=min(PROBE_KEYS, S), replace=False)
        M_true = analytic_operator(K, Q, probe_idx, d)
        truth_rank = int(np.linalg.matrix_rank(M_true, tol=1e-10))
        w, V = np.linalg.eigh(M_true)
        truth_sub = np.ascontiguousarray(V[:, ::-1][:, :GRADED_RANK])
        ent = row_entropy_bits(attention(K, Q, d))

        per_ratio = {}
        for ratio in RATIOS:
            k = max(1, int(round(ratio * d)))
            acc = np.zeros((d, d))
            for s in probe_idx:

                def consumer(kv, _s=s, _K=K, _Q=Q, _d=d):
                    Kp = _K.copy()
                    Kp[_s] = kv
                    return attention(Kp, _Q, _d).ravel()

                pr = jacobian_probe(
                    consumer,
                    K[s][None, :],
                    n_directions=k,
                    eps=EPS,
                    rng=np.random.default_rng(SEED + int(_s_seed(s))),
                )
                acc += pr.S
            _, Vh = np.linalg.eigh(0.5 * (acc + acc.T))
            sub = np.ascontiguousarray(Vh[:, ::-1][:, :GRADED_RANK])
            per_ratio[str(ratio)] = subspace_overlap(sub, truth_sub).resolution

        rows.append(
            {
                **{
                    kk: cell[kk]
                    for kk in (
                        "tag",
                        "family",
                        "layer",
                        "layer_fraction",
                        "head",
                        "head_dim",
                        "n_layers",
                    )
                },
                "resolution": per_ratio,
                "truth_rank": truth_rank,
                "median_row_entropy_bits": ent,
                "probe_keys": int(len(probe_idx)),
            }
        )
        print(
            f"  {cell['tag']:<14} L{cell['layer']:<3} H{cell['head']} "
            f"res@0.5 {per_ratio['0.5']:.3f}  res@1.25 "
            f"{per_ratio['1.25']:.3f}  truth_rank {truth_rank}  "
            f"H {ent:.2f}",
            flush=True,
        )

    if not rows:
        print("no cells graded")
        return 1

    hi = [r["resolution"]["1.25"] for r in rows]
    lo = [r["resolution"]["0.5"] for r in rows]

    by_family = {}
    for r in rows:
        by_family.setdefault(r["family"], []).append(r["resolution"]["1.25"])
    family_means = {k: float(np.mean(v)) for k, v in by_family.items()}
    spread = (
        max(family_means.values()) - min(family_means.values())
        if len(family_means) > 1
        else 0.0
    )

    median_entropy = float(
        np.median([r["median_row_entropy_bits"] for r in rows])
    )
    bars = {
        "A0_artifact_cleared": bool(median_entropy > 1.0),
        "A1_instrument": bool(min(hi) >= INSTRUMENT_BAR),
        "A2_budget_cliff": bool(
            all(
                r["resolution"]["0.5"] < r["resolution"]["1.25"] - 1e-9
                for r in rows
            )
        ),
        "A3_spread_reported": bool(len(family_means) >= 1),
        "A4_rank_floor": bool(
            all(r["truth_rank"] >= GRADED_RANK for r in rows)
        ),
        "A5_anti_vacuity": bool(
            all(r["median_row_entropy_bits"] > ENTROPY_FLOOR for r in rows)
        ),
        "A6_census": bool(len(rows) + len(skipped) == len(manifest)),
    }
    verdict = "PASS" if all(bars.values()) else "FAIL"

    record = {
        "schema": "readscope-c3b-architecture-spread-v1",
        "supersedes": "calibration/records/c3-architecture-spread.json",
        "declared": {
            "ratios": RATIOS,
            "graded_rank": GRADED_RANK,
            "probe_keys": PROBE_KEYS,
            "eps": EPS,
            "seed": SEED,
            "instrument_bar": INSTRUMENT_BAR,
            "entropy_floor": ENTROPY_FLOOR,
            "artifact_entropy_bar": 1.0,
            "truth": "closed-form sum_s J_s^T J_s for softmax attention "
            "on the model's own post-RoPE keys and queries",
        },
        "rows": rows,
        "skipped": skipped,
        "summary": {
            "n_cells": len(rows),
            "median_row_entropy_bits": median_entropy,
            "families": sorted(by_family),
            "family_mean_resolution_at_1.25": family_means,
            "across_family_spread": spread,
            "min_resolution_at_1.25": float(min(hi)),
            "max_resolution_at_1.25": float(max(hi)),
            "mean_resolution_at_1.25": float(np.mean(hi)),
            "mean_resolution_at_0.5": float(np.mean(lo)),
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
        / "c3b-architecture-spread.json"
    )
    out.write_text(json.dumps(record, indent=2, sort_keys=True))

    print("\nfamily mean resolution at k/d = 1.25:")
    for k, v in sorted(family_means.items()):
        print(f"  {k:<10} {v:.4f}")
    print(f"across-family spread {spread:.4f}")
    print(
        f"mean resolution  k/d=0.5 {np.mean(lo):.4f}   "
        f"k/d=1.25 {np.mean(hi):.4f}"
    )
    for k in sorted(bars):
        print(f"{k:<22} {'PASS' if bars[k] else 'FAIL'}")
    print("VERDICT", verdict)
    print(out)
    return 0 if verdict == "PASS" else 1


def _s_seed(s):
    return int(s) * 7919 % 100003


if __name__ == "__main__":
    raise SystemExit(main())
