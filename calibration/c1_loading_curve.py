#!/usr/bin/env python3
"""C-1, the loading curve. Declared before it runs.

Sweeps the probing distribution from itself toward the activation
distribution, holding the consumer, the rank, the dimension and the probe
budget fixed, and records recovered-subspace overlap against measured
loading at every step.

Declared bars, fixed here before any run and computed from the record:

  L1  monotone       overlap is non-increasing in loading, allowing one
                     inversion per sweep for sampling noise.
  L2  above floor    at zero loading, overlap exceeds four times the chance
                     value for the swept shape.
  L3  separation     overlap at maximum loading is strictly below overlap at
                     zero loading by at least 0.05.
  L4  anti-vacuity   at every point the measured loading is finite, the
                     recovered operator has rank at least the swept rank, and
                     the probe spent the declared number of calls.
  L5  census         every declared (alpha, seed) cell appears in the record.

An outcome that fails L1 or L3 is a result, not a bug. It would mean either
that loading as measured does not capture what degrades recovery, or that
recovery is insensitive to loading over this range. Both narrow the
instrument's specification and both are worth knowing before anyone relies
on it.

This runs on Atlas, never on the laptop.
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
    subspace_overlap,
)

DIM = 32
RANK = 4
N_POINTS = 512
SKETCH_DIM = 32
EPS = 1e-3
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = [0, 1, 2, 3, 4]
MIN_SEPARATION = 0.05


def planted_consumer(basis: np.ndarray):
    """A tanh consumer reading exactly the planted subspace."""

    def C(x):
        return float(np.tanh(np.sum(basis.T @ x)))

    return C


def main() -> int:
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        basis = np.linalg.qr(rng.standard_normal((DIM, RANK)))[0]
        consumer = planted_consumer(basis)

        scale = np.concatenate(
            [np.full(RANK, 3.0), np.full(DIM - RANK, 0.3)]
        )
        activation = rng.standard_normal((4096, DIM)) * scale
        far = rng.standard_normal((4096, DIM)) * scale[::-1] + 4.0

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
            ov = subspace_overlap(res.read_subspace(RANK), basis)
            rows.append(
                {
                    "seed": seed,
                    "alpha": alpha,
                    "loading": load.to_dict(),
                    "overlap": ov.to_dict(),
                    "n_calls": res.n_calls,
                    "operator_rank": int(
                        np.linalg.matrix_rank(res.S, tol=1e-10)
                    ),
                }
            )

    bars = {}
    by_alpha = {
        a: float(
            np.mean([r["overlap"]["overlap"] for r in rows if r["alpha"] == a])
        )
        for a in ALPHAS
    }
    load_by_alpha = {
        a: float(
            np.mean([r["loading"]["jeffreys"] for r in rows if r["alpha"] == a])
        )
        for a in ALPHAS
    }
    ordered = [by_alpha[a] for a in sorted(ALPHAS, key=lambda a: -a)]
    inversions = sum(
        1 for i in range(len(ordered) - 1) if ordered[i] > ordered[i + 1]
    )
    bars["L1_monotone_in_loading"] = bool(inversions <= 1)
    bars["L2_above_floor_unloaded"] = bool(
        by_alpha[1.0] > 4.0 * chance_overlap(RANK, DIM)
    )
    bars["L3_separation"] = bool(
        by_alpha[1.0] - by_alpha[0.0] >= MIN_SEPARATION
    )
    bars["L4_anti_vacuity"] = bool(
        all(
            np.isfinite(r["loading"]["jeffreys"])
            and r["operator_rank"] >= RANK
            and r["n_calls"] == N_POINTS * 2 * SKETCH_DIM
            for r in rows
        )
    )
    bars["L5_census"] = bool(len(rows) == len(ALPHAS) * len(SEEDS))

    verdict = "PASS" if all(bars.values()) else "FAIL"
    record = {
        "schema": "readscope-c1-loading-curve-v1",
        "declared": {
            "dim": DIM,
            "rank": RANK,
            "n_points": N_POINTS,
            "sketch_dim": SKETCH_DIM,
            "eps": EPS,
            "alphas": ALPHAS,
            "seeds": SEEDS,
            "chance_overlap": chance_overlap(RANK, DIM),
            "min_separation": MIN_SEPARATION,
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

    out = Path(__file__).resolve().parent / "records" / "c1-loading-curve.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True))

    for a in ALPHAS:
        print(
            "alpha %.2f  loading %10.4f  overlap %.4f"
            % (a, load_by_alpha[a], by_alpha[a])
        )
    for k in sorted(bars):
        print("%-28s %s" % (k, "PASS" if bars[k] else "FAIL"))
    print("VERDICT", verdict)
    print(out)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
