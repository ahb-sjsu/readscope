#!/usr/bin/env python3
"""C-1b, the loading curve, corrected. Declared before it runs.

Supersedes c1_loading_curve.py, which failed four of its five bars. That
script and its record stay in the repository. Three defects, named:

Defect one, a rank-one consumer.
    The planted consumer was ``tanh(sum(basis.T @ x))``. Summing the
    projections collapses the rank-four subspace into a single direction, so
    the read operator was rank one by construction and recovering a
    rank-four subspace was impossible.

Defect two, saturation.
    At the swept scale the tanh argument was large, so its derivative
    vanished and the recovered operator was numerically zero at some cells
    and pure sketch noise at others. The measured operator ranks were 0, 5
    and 32 against a declared rank of 4, which is what the anti-vacuity bar
    caught.

Defect three, and the one that mattered.
    **A consumer whose read subspace does not vary across the input space
    cannot exhibit probe loading at all.** If the Jacobian's row space is
    the same everywhere, the recovered subspace is that row space no matter
    where the probe points are drawn from, so the sweep would have measured
    nothing even with the first two defects fixed. Probe loading degrades
    subspace recovery only when the consumer's local sensitivity direction
    changes over the input space, and it is then the average over the wrong
    measure that goes wrong.

This script therefore uses a gated consumer whose read direction rotates
with position, and it grades recovery against the read operator **as seen on
the activation distribution**, which is what a user of the instrument
actually wants to know.

Declared bars, fixed here before any run and computed from the record:

  L1  monotone       overlap is non-decreasing as the probing distribution
                     approaches the activation distribution, allowing one
                     inversion for sampling noise.
  L2  self-consistent  probing on the activation distribution recovers the
                     truth at overlap at least 0.90.
  L3  separation     on-distribution overlap exceeds far-distribution
                     overlap by at least 0.05.
  L4  anti-vacuity   at every cell the loading is finite, the recovered
                     operator has rank at least the graded rank, the mean
                     squared gradient norm is above 1e-8 so nothing is
                     graded on a saturated consumer, and the probe spent the
                     declared number of calls.
  L5  census         every declared cell appears.

Failing L1 or L3 remains a result. It would mean loading as measured does
not track what degrades recovery, which would send the loading measure back
for redesign rather than the instrument.

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
    chance_overlap,
    interpolate_distribution,
    probe_loading,
    spectrum_of,
    subspace_overlap,
)

DIM = 24
GRADED_RANK = 3
N_POINTS = 384
SKETCH_DIM = 48
TRUTH_POINTS = 4096
EPS = 1e-3
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = [0, 1, 2, 3, 4]
MIN_SEPARATION = 0.05
SELF_CONSISTENCY = 0.90
GRAD_FLOOR = 1e-8


def gated_consumer(a: np.ndarray, b: np.ndarray, s: np.ndarray):
    """A consumer whose read direction rotates with position.

    ``C(x) = sig(s.x) (a.x) + (1 - sig(s.x)) (b.x)``. Its gradient depends
    on where ``x`` sits along the gate, so the read operator genuinely
    depends on the measure the probe averages over. That dependence is the
    thing probe loading acts on.
    """

    def C(x):
        t = float(s @ x)
        g = 1.0 / (1.0 + np.exp(-t))
        return float(g * (a @ x) + (1.0 - g) * (b @ x))

    return C


def main() -> int:
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        basis = np.linalg.qr(rng.standard_normal((DIM, 3)))[0]
        a, b, s = basis[:, 0], basis[:, 1], basis[:, 2]
        consumer = gated_consumer(a, b, s)

        activation = rng.standard_normal((TRUTH_POINTS, DIM))
        # the probing distribution a lazy user would reach for: shifted off
        # the gate and anisotropic in the wrong directions
        far = rng.standard_normal((TRUTH_POINTS, DIM)) * 0.4 + 3.0 * s

        truth = blind_probe(consumer, activation, eps=EPS)
        truth_sub = truth.read_subspace(GRADED_RANK)
        truth_spec = spectrum_of(truth.S)

        for alpha in ALPHAS:
            pts = interpolate_distribution(
                far,
                activation,
                alpha,
                rng=np.random.default_rng(seed * 100 + 7),
                n_samples=N_POINTS,
            )
            load = probe_loading(pts, activation)
            res = blind_probe(
                consumer,
                pts,
                mode="sketch",
                sketch_dim=SKETCH_DIM,
                eps=EPS,
                rng=np.random.default_rng(seed * 100 + 11),
            )
            ov = subspace_overlap(res.read_subspace(GRADED_RANK), truth_sub)
            rows.append(
                {
                    "seed": seed,
                    "alpha": alpha,
                    "loading": load.to_dict(),
                    "overlap": ov.to_dict(),
                    "n_calls": res.n_calls,
                    "mean_sq_grad_norm": float(np.trace(res.S)),
                    "operator_rank": int(
                        np.linalg.matrix_rank(res.S, tol=1e-10)
                    ),
                    "truth_effective_rank": truth_spec.effective_rank,
                }
            )

    def mean_over(alpha, key, sub):
        vals = [r[key][sub] for r in rows if r["alpha"] == alpha]
        return float(np.mean(vals))

    by_alpha = {a: mean_over(a, "overlap", "overlap") for a in ALPHAS}
    load_by_alpha = {a: mean_over(a, "loading", "jeffreys") for a in ALPHAS}

    ordered = [by_alpha[a] for a in ALPHAS]
    inversions = sum(
        1 for i in range(len(ordered) - 1) if ordered[i] > ordered[i + 1]
    )

    bars = {
        "L1_monotone_in_loading": bool(inversions <= 1),
        "L2_self_consistent": bool(by_alpha[1.0] >= SELF_CONSISTENCY),
        "L3_separation": bool(by_alpha[1.0] - by_alpha[0.0] >= MIN_SEPARATION),
        "L4_anti_vacuity": bool(
            all(
                np.isfinite(r["loading"]["jeffreys"])
                and r["operator_rank"] >= GRADED_RANK
                and r["mean_sq_grad_norm"] > GRAD_FLOOR
                and r["n_calls"] == N_POINTS * 2 * SKETCH_DIM
                for r in rows
            )
        ),
        "L5_census": bool(len(rows) == len(ALPHAS) * len(SEEDS)),
    }

    verdict = "PASS" if all(bars.values()) else "FAIL"
    record = {
        "schema": "readscope-c1b-loading-curve-v1",
        "supersedes": "calibration/records/c1-loading-curve.json",
        "declared": {
            "dim": DIM,
            "graded_rank": GRADED_RANK,
            "n_points": N_POINTS,
            "sketch_dim": SKETCH_DIM,
            "truth_points": TRUTH_POINTS,
            "eps": EPS,
            "alphas": ALPHAS,
            "seeds": SEEDS,
            "chance_overlap": chance_overlap(GRADED_RANK, DIM),
            "min_separation": MIN_SEPARATION,
            "self_consistency": SELF_CONSISTENCY,
            "grad_floor": GRAD_FLOOR,
            "truth": "the read operator recovered by the exact estimator "
            "on the activation distribution",
        },
        "rows": rows,
        "curve": {
            "mean_overlap_by_alpha": by_alpha,
            "mean_loading_by_alpha": load_by_alpha,
            "inversions": inversions,
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
        Path(__file__).resolve().parent / "records" / "c1b-loading-curve.json"
    )
    out.write_text(json.dumps(record, indent=2, sort_keys=True))

    for a in ALPHAS:
        print(
            f"alpha {a:.2f}  loading {load_by_alpha[a]:10.4f}  "
            f"overlap {by_alpha[a]:.4f}"
        )
    for k in sorted(bars):
        print(f"{k:<28} {'PASS' if bars[k] else 'FAIL'}")
    print("VERDICT", verdict)
    print(out)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
