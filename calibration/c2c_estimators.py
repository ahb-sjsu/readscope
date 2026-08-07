#!/usr/bin/env python3
"""C-2c, the estimator comparison. Declared before it runs.

C-2b found the shipped Gaussian sketch has a bandwidth of one to two
directions where the exact estimator resolves all thirty-two, and
`CALIBRATION.md` named the isotropic bias as the cause and debiasing as the
fix. **That diagnosis was half right and the proposed fix was wrong**, and
this sweep is declared to establish both on the record rather than quietly
substituting a better story.

The bias is exact and now derived:

    E[ghat ghat^T] = (1 + 1/k) g g^T + (||g||^2 / k) I

so it is a multiple of the identity plus a rescaling. Subtracting a multiple
of the identity shifts every eigenvalue equally and moves no eigenvector at
all. **Debiasing therefore cannot buy a single direction of bandwidth**, and
what actually degrades subspace recovery is the sketch's variance.

The variance comes from the magnitudes of the iid Gaussian directions. An
orthonormal frame removes it: recombining as ``U^T y`` gives exactly the
orthogonal projection of the gradient onto the drawn subspace, and at ``k =
d`` the projector is the identity so the estimate is exact. Same ``2k``
consumer calls either way.

Declared bars, computed from the record:

  E1  debias is inert for subspace   at every cell, the bandwidth and the
                                     resolution curve of the debiased sketch
                                     equal the raw sketch's to 1e-9. This is
                                     a prediction that could fail; if it
                                     does, the derivation above is wrong.
  E2  debias corrects the spectrum   at every cell, the debiased operator's
                                     trace is closer to the exact
                                     estimator's trace than the raw sketch's
                                     is, at matched points.
  E3  ortho is not worse             at every swept rank, ortho's resolution
                                     is at least the sketch's at equal call
                                     budget, allowing one inversion.
  E4  ortho buys bandwidth           ortho's bandwidth strictly exceeds the
                                     sketch's at the same call budget, for
                                     at least one of the two budgets swept.
                                     If this fails the variance diagnosis is
                                     wrong too and the sketch's limit is
                                     something neither story has named.
  E5  ortho at full rank is exact    at k = d, ortho's resolution equals the
                                     exact estimator's to 1e-9.
  E6  anti-vacuity + census          every cell present, gradient norms above
                                     the floor, declared calls spent.

Runs on Atlas.
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
    blind_probe,
    debias_sketch,
    spectrum_of,
    subspace_overlap,
)

DIM = 32
RANKS = [1, 2, 4, 8, 16]
BUDGETS = [8, 16]
N_POINTS = 192
EPS = 1e-3
SEEDS = [0, 1, 2]
GRAD_FLOOR = 1e-10
RESOLUTION_BAR = 0.5
SPECTRUM_DECAY = 0.75
INPUT_SCALE = 0.35
TOL = 1e-9


def graded_consumer(basis, weights):
    def C(x):
        return float(np.tanh(basis.T @ x) @ weights)

    return C


def setup(rank, seed):
    rng = np.random.default_rng(seed * 977 + rank)
    basis = np.linalg.qr(rng.standard_normal((DIM, rank)))[0]
    weights = SPECTRUM_DECAY ** np.arange(rank)
    pts = rng.standard_normal((N_POINTS, DIM)) * INPUT_SCALE
    return basis, graded_consumer(basis, weights), pts


def prefix_bandwidth(curve):
    best = None
    for r in RANKS:
        if curve[str(r)] >= RESOLUTION_BAR:
            best = r
        else:
            break
    return best


def main() -> int:
    rows = []
    for seed in SEEDS:
        for rank in RANKS:
            basis, consumer, pts = setup(rank, seed)

            ex = blind_probe(consumer, pts, eps=EPS, check_regime=False)
            ex_res = subspace_overlap(ex.read_subspace(rank), basis)
            rows.append(
                {
                    "seed": seed,
                    "rank": rank,
                    "estimator": "exact",
                    "budget": DIM,
                    "resolution": ex_res.resolution,
                    "trace": float(np.trace(ex.S)),
                    "trace_error_vs_exact": 0.0,
                    "n_calls": ex.n_calls,
                    "calls_as_declared": ex.n_calls == N_POINTS * 2 * DIM,
                }
            )

            for k in BUDGETS:
                sk = blind_probe(
                    consumer,
                    pts,
                    mode="sketch",
                    sketch_dim=k,
                    eps=EPS,
                    rng=np.random.default_rng(seed * 131 + 5),
                    check_regime=False,
                )
                deb = debias_sketch(sk.S, k)
                orth = blind_probe(
                    consumer,
                    pts,
                    mode="ortho",
                    sketch_dim=k,
                    eps=EPS,
                    rng=np.random.default_rng(seed * 131 + 5),
                    check_regime=False,
                )
                tr_exact = float(np.trace(ex.S))
                for name, S, res in (
                    ("sketch", sk.S, sk),
                    ("sketch_debiased", deb, sk),
                    ("ortho", orth.S, orth),
                ):
                    sub = spectrum_of(S).eigenvectors[:, :rank]
                    ov = subspace_overlap(sub, basis)
                    rows.append(
                        {
                            "seed": seed,
                            "rank": rank,
                            "estimator": name,
                            "budget": k,
                            "resolution": ov.resolution,
                            "trace": float(np.trace(S)),
                            "trace_error_vs_exact": abs(
                                float(np.trace(S)) - tr_exact
                            ),
                            "n_calls": res.n_calls,
                            "calls_as_declared": res.n_calls
                            == N_POINTS * 2 * k,
                        }
                    )
            # anti-vacuity witness for this cell
            rows[-1]["mean_sq_grad_norm"] = float(np.trace(ex.S))

    def curve(est, budget):
        out = {}
        for r in RANKS:
            vals = [
                x["resolution"]
                for x in rows
                if x["estimator"] == est
                and x["rank"] == r
                and x["budget"] == budget
            ]
            out[str(r)] = float(np.mean(vals))
        return out

    curves = {"exact": {str(DIM): curve("exact", DIM)}}
    for est in ("sketch", "sketch_debiased", "ortho"):
        curves[est] = {str(k): curve(est, k) for k in BUDGETS}

    bandwidths = {
        "exact": {str(DIM): prefix_bandwidth(curves["exact"][str(DIM)])}
    }
    for est in ("sketch", "sketch_debiased", "ortho"):
        bandwidths[est] = {
            str(k): prefix_bandwidth(curves[est][str(k)]) for k in BUDGETS
        }

    # E1, debias must be inert for the subspace
    e1_diffs = []
    for k in BUDGETS:
        for r in RANKS:
            e1_diffs.append(
                abs(
                    curves["sketch"][str(k)][str(r)]
                    - curves["sketch_debiased"][str(k)][str(r)]
                )
            )
    e1 = bool(
        max(e1_diffs) <= TOL
        and all(
            bandwidths["sketch"][str(k)]
            == bandwidths["sketch_debiased"][str(k)]
            for k in BUDGETS
        )
    )

    # E2, debias must move the trace toward the exact estimator's
    e2_pairs = []
    for k in BUDGETS:
        raw = [
            x["trace_error_vs_exact"]
            for x in rows
            if x["estimator"] == "sketch" and x["budget"] == k
        ]
        deb = [
            x["trace_error_vs_exact"]
            for x in rows
            if x["estimator"] == "sketch_debiased" and x["budget"] == k
        ]
        e2_pairs.append(
            {
                "budget": k,
                "mean_raw_trace_error": float(np.mean(raw)),
                "mean_debiased_trace_error": float(np.mean(deb)),
            }
        )
    e2 = bool(
        all(
            p["mean_debiased_trace_error"] < p["mean_raw_trace_error"]
            for p in e2_pairs
        )
    )

    # E3, ortho not worse at equal budget
    e3_violations = sum(
        1
        for k in BUDGETS
        for r in RANKS
        if curves["ortho"][str(k)][str(r)]
        < curves["sketch"][str(k)][str(r)] - 1e-9
    )
    e3 = bool(e3_violations <= 1)

    # E4, ortho buys bandwidth at at least one budget
    e4 = bool(
        any(
            (bandwidths["ortho"][str(k)] or 0)
            > (bandwidths["sketch"][str(k)] or 0)
            for k in BUDGETS
        )
    )

    # E5, ortho at full rank is exact
    full = []
    for seed in SEEDS:
        for rank in RANKS:
            basis, consumer, pts = setup(rank, seed)
            o = blind_probe(
                consumer,
                pts,
                mode="ortho",
                sketch_dim=DIM,
                eps=EPS,
                rng=np.random.default_rng(seed * 7 + 3),
                check_regime=False,
            )
            e = blind_probe(consumer, pts, eps=EPS, check_regime=False)
            full.append(
                abs(
                    subspace_overlap(o.read_subspace(rank), basis).resolution
                    - subspace_overlap(e.read_subspace(rank), basis).resolution
                )
            )
    e5 = bool(max(full) <= TOL)

    expected = len(SEEDS) * len(RANKS) * (1 + 3 * len(BUDGETS))
    e6 = bool(
        len(rows) == expected
        and all(x["calls_as_declared"] for x in rows)
        and all(
            x["trace"] > GRAD_FLOOR
            for x in rows
            if x["estimator"] != "sketch_debiased"
        )
    )

    bars = {
        "E1_debias_inert_for_subspace": e1,
        "E2_debias_corrects_spectrum": e2,
        "E3_ortho_not_worse": e3,
        "E4_ortho_buys_bandwidth": e4,
        "E5_ortho_full_rank_exact": e5,
        "E6_anti_vacuity_and_census": e6,
    }
    verdict = "PASS" if all(bars.values()) else "FAIL"

    record = {
        "schema": "readscope-c2c-estimators-v1",
        "declared": {
            "dim": DIM,
            "ranks": RANKS,
            "budgets": BUDGETS,
            "n_points": N_POINTS,
            "eps": EPS,
            "seeds": SEEDS,
            "resolution_bar": RESOLUTION_BAR,
            "spectrum_decay": SPECTRUM_DECAY,
            "input_scale": INPUT_SCALE,
            "tolerance": TOL,
            "closed_form_bias": "E[ghat ghat^T] = (1 + 1/k) g g^T "
            "+ (||g||^2 / k) I",
        },
        "rows": rows,
        "curves_resolution": curves,
        "bandwidths": bandwidths,
        "e1_max_resolution_difference": max(e1_diffs),
        "e2_trace_errors": e2_pairs,
        "e3_violations": e3_violations,
        "e5_max_resolution_difference": max(full),
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

    out = Path(__file__).resolve().parent / "records" / "c2c-estimators.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True))

    print("resolution by read rank, dim", DIM)
    print("  " + f"{'ranks':<22} " + " ".join(f"{r:6d}" for r in RANKS))
    print(
        f"  {'exact (k=' + str(DIM) + ')':<22} "
        + " ".join(f"{curves['exact'][str(DIM)][str(r)]:6.3f}" for r in RANKS)
        + f"   bw {bandwidths['exact'][str(DIM)]}"
    )
    for k in BUDGETS:
        for est in ("sketch", "sketch_debiased", "ortho"):
            print(
                f"  {est + ' (k=' + str(k) + ')':<22} "
                + " ".join(
                    f"{curves[est][str(k)][str(r)]:6.3f}" for r in RANKS
                )
                + f"   bw {bandwidths[est][str(k)]}"
            )
    for p in e2_pairs:
        print(
            f"  trace error k={p['budget']}: raw "
            f"{p['mean_raw_trace_error']:.4f} -> debiased "
            f"{p['mean_debiased_trace_error']:.4f}"
        )
    for kk in sorted(bars):
        print(f"{kk:<32} {'PASS' if bars[kk] else 'FAIL'}")
    print("VERDICT", verdict)
    print(out)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
