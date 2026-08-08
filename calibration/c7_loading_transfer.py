#!/usr/bin/env python3
"""C-7, does the loading correction transfer? Declared before it runs.

C-1b measured how recovery degrades with probe loading on one synthetic
gated consumer, and `CALIBRATION.md` has said since then that turning that
curve into a correction is the point of having it. A curve that only
describes the family it was fitted on is a warning. A curve that predicts
degradation on a family it never saw is a correction, and it is the
difference between an instrument you can trust off its calibration points
and one you cannot.

**So the fit and the test are deliberately different consumers.** The
correction is fitted on C-1b's synthetic gated consumer, where the true
resolution is 1.0 by construction. It is then applied, unchanged, to real
attention heads across four families, where the truth is the exact
closed-form Jacobian Gram and the loading is created by drawing probe points
from distributions increasingly far from the model's own key distribution.

Attention qualifies for this test. C-1's finding was that loading can only
bite when the consumer's read subspace varies over the input space, and for
softmax attention the operator is
``sum_i a_{i,s} q_i q_i^T / d`` with ``a`` depending on the softmax at the
probed key, so it does.

The model under test is multiplicative: ``reading ~ truth * g(loading)``.
That is an assumption of the correction and this sweep is what decides
whether it holds off its own family.

Declared bars:

  T1  loading bites      on real attention, mean resolution is strictly
                         decreasing across the declared loading ladder. If
                         it is flat there is nothing to correct and the rest
                         of the sweep is void.
  T2  the question       the correction fitted on the synthetic family
                         reduces mean absolute error against the known truth
                         by at least 50 percent, compared with reporting the
                         raw reading uncorrected. **This is falsifiable and
                         is the whole sweep.** If it fails, loading stays a
                         warning and the specification says so.
  T3  no harm            the correction increases absolute error on at most
                         one of the loading rungs. A correction that helps on
                         average while wrecking a rung is not usable.
  T4  bounded            no corrected value exceeds 1.0 by more than 1e-9,
                         so the correction cannot manufacture a reading
                         better than perfect.
  T5  anti-vacuity       every graded cell has an analytic operator of rank
                         at least the graded rank, and the loading measured
                         at the far rung exceeds that at the near rung by at
                         least a factor of two, so the ladder really spans a
                         range.
  T6  census             every attempted cell is recorded, skips with
                         reasons.

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
    fit_loading_correction,
    interpolate_distribution,
    jacobian_probe,
    probe_loading,
    subspace_overlap,
)

ACTS = Path("/archive/readscope/activations_v2")
C1B = Path(__file__).resolve().parent / "records" / "c1b-loading-curve.json"

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
GRADED_RANK = 16
PROBE_KEYS = 16
RATIO = 1.25
EPS = 1e-3
SEED = 0
ERROR_REDUCTION_BAR = 0.50
MAX_HARMED_RUNGS = 1
LADDER_SPAN_BAR = 2.0


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


def load_correction():
    rec = json.loads(C1B.read_text())
    curve = rec["curve"]
    alphas = rec["declared"]["alphas"]
    load = [curve["mean_loading_by_alpha"][str(a)] for a in alphas]
    read = [curve["mean_overlap_by_alpha"][str(a)] for a in alphas]
    return fit_loading_correction(
        load, read, truth=1.0, source="c1b-loading-curve.json"
    )


def main() -> int:
    if not C1B.exists():
        print("no C-1b record to fit the correction from")
        return 2
    corr = load_correction()
    print("correction fitted from", corr.source)
    print(
        "  attenuation at 1, 10, 25, 50, 92 nats: "
        + " ".join(
            f"{corr.expected_attenuation(x):.3f}" for x in (1, 10, 25, 50, 92)
        )
    )

    mpath = ACTS / "manifest.json"
    if not mpath.exists():
        print("no activation manifest")
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
        truth_sub = top(M_true, GRADED_RANK)

        # a probing distribution a careless user might reach for
        far = rng.standard_normal((S, d)) * 0.4 + 3.0 * K.mean(axis=0)
        k = max(1, int(round(RATIO * d)))

        rungs = []
        for alpha in ALPHAS:
            pts = interpolate_distribution(
                far,
                K,
                alpha,
                rng=np.random.default_rng(SEED + 7),
                n_samples=len(probe_idx),
            )
            load = probe_loading(pts, K).jeffreys
            acc = np.zeros((d, d))
            for j, s in enumerate(probe_idx):

                def consumer(kv, _s=s, _K=K, _Q=Q, _d=d):
                    Kp = _K.copy()
                    Kp[_s] = kv
                    return attention(Kp, _Q, _d).ravel()

                acc += jacobian_probe(
                    consumer,
                    pts[j][None, :],
                    n_directions=k,
                    eps=EPS,
                    rng=np.random.default_rng(SEED + int(s) * 7919 % 100003),
                ).S
            res = subspace_overlap(top(acc, GRADED_RANK), truth_sub).resolution
            rungs.append(
                {
                    "alpha": alpha,
                    "loading": float(load),
                    "reading": float(res),
                    "corrected": corr.correct(res, load),
                }
            )

        rows.append(
            {
                **{
                    kk: cell[kk]
                    for kk in ("tag", "family", "layer", "head", "head_dim")
                },
                "truth_rank": truth_rank,
                "rungs": rungs,
            }
        )
        print(
            f"  {cell['tag']:<14} L{cell['layer']:<3} H{cell['head']}  "
            + "  ".join(
                f"a{r['alpha']:.2f}:L{r['loading']:7.1f} "
                f"r{r['reading']:.3f}->{r['corrected']:.3f}"
                for r in rungs
            ),
            flush=True,
        )

    if not rows:
        print("no cells graded")
        return 1

    # truth for a cell probed on its own distribution is the alpha=1 rung's
    # own reading; the reference for error is perfect recovery, 1.0, since
    # the probe is exact at this budget when unloaded (C-3b)
    TRUTH = 1.0
    by_alpha = {}
    for a in ALPHAS:
        rd = [r["rungs"][ALPHAS.index(a)]["reading"] for r in rows]
        cd = [r["rungs"][ALPHAS.index(a)]["corrected"] for r in rows]
        ld = [r["rungs"][ALPHAS.index(a)]["loading"] for r in rows]
        by_alpha[str(a)] = {
            "mean_loading": float(np.mean(ld)),
            "mean_reading": float(np.mean(rd)),
            "mean_corrected": float(np.mean(cd)),
            "mae_raw": float(np.mean([abs(TRUTH - v) for v in rd])),
            "mae_corrected": float(np.mean([abs(TRUTH - v) for v in cd])),
        }

    mae_raw = float(np.mean([by_alpha[str(a)]["mae_raw"] for a in ALPHAS]))
    mae_cor = float(
        np.mean([by_alpha[str(a)]["mae_corrected"] for a in ALPHAS])
    )
    reduction = (mae_raw - mae_cor) / mae_raw if mae_raw > 0 else 0.0
    harmed = sum(
        1
        for a in ALPHAS
        if by_alpha[str(a)]["mae_corrected"]
        > by_alpha[str(a)]["mae_raw"] + 1e-12
    )
    readings_by_alpha = [by_alpha[str(a)]["mean_reading"] for a in ALPHAS]
    loads_by_alpha = [by_alpha[str(a)]["mean_loading"] for a in ALPHAS]

    bars = {
        "T1_loading_bites": bool(
            all(
                readings_by_alpha[i] < readings_by_alpha[i + 1] - 1e-9
                for i in range(len(ALPHAS) - 1)
            )
        ),
        "T2_correction_transfers": bool(reduction >= ERROR_REDUCTION_BAR),
        "T3_no_harm": bool(harmed <= MAX_HARMED_RUNGS),
        "T4_bounded": bool(
            all(
                r["corrected"] <= 1.0 + 1e-9
                for row in rows
                for r in row["rungs"]
            )
        ),
        "T5_anti_vacuity": bool(
            all(r["truth_rank"] >= GRADED_RANK for r in rows)
            and loads_by_alpha[0] >= LADDER_SPAN_BAR * loads_by_alpha[-1]
        ),
        "T6_census": bool(len(rows) + len(skipped) == len(manifest)),
    }
    verdict = "PASS" if all(bars.values()) else "FAIL"

    record = {
        "schema": "readscope-c7-loading-transfer-v1",
        "declared": {
            "alphas": ALPHAS,
            "graded_rank": GRADED_RANK,
            "probe_keys": PROBE_KEYS,
            "ratio": RATIO,
            "eps": EPS,
            "seed": SEED,
            "error_reduction_bar": ERROR_REDUCTION_BAR,
            "max_harmed_rungs": MAX_HARMED_RUNGS,
            "ladder_span_bar": LADDER_SPAN_BAR,
            "model": "reading ~ truth * g(loading), multiplicative",
            "fit_family": "C-1b synthetic gated consumer",
            "test_family": "real attention heads, four families",
        },
        "correction": corr.to_dict(),
        "rows": rows,
        "skipped": skipped,
        "by_alpha": by_alpha,
        "summary": {
            "n_cells": len(rows),
            "mae_raw": mae_raw,
            "mae_corrected": mae_cor,
            "error_reduction": reduction,
            "harmed_rungs": harmed,
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
        / "c7-loading-transfer.json"
    )
    out.write_text(json.dumps(record, indent=2, sort_keys=True))

    print("\nby loading rung:")
    print(
        f"  {'alpha':<7} {'loading':>9} {'reading':>9} {'corrected':>10} "
        f"{'mae_raw':>9} {'mae_cor':>9}"
    )
    for a in ALPHAS:
        b = by_alpha[str(a)]
        print(
            f"  {a:<7} {b['mean_loading']:9.2f} {b['mean_reading']:9.4f} "
            f"{b['mean_corrected']:10.4f} {b['mae_raw']:9.4f} "
            f"{b['mae_corrected']:9.4f}"
        )
    print(
        f"MAE raw {mae_raw:.4f} -> corrected {mae_cor:.4f}   "
        f"reduction {reduction * 100:.1f}%   harmed rungs {harmed}"
    )
    for k in sorted(bars):
        print(f"{k:<26} {'PASS' if bars[k] else 'FAIL'}")
    print("VERDICT", verdict)
    print(out)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
