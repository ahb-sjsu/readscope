#!/usr/bin/env python3
"""C-2b, the bandwidth sweep, corrected. Declared before it runs.

Supersedes c2_bandwidth.py, which failed its monotonicity bar. That script
and its record stay in the repository.

**Disclosure.** C-2's numbers were seen before this document was written, so
the discipline that protects the other calibrations does not protect this
one, and saying so is the only remedy available. The two repairs below are
structural, meaning they change *which quantity* is barred and *how a cutoff
is defined*. No threshold has been moved to fit an observed value, and the
monotonicity allowance is carried over unchanged at one inversion.

Defect one, a bar on a statistic with a moving floor.
    B2 required raw overlap to be non-increasing in read rank. Chance
    overlap is ``rank / dim``, so the floor rises with rank, and a reading
    can climb in raw terms while falling further behind the floor. The sketch
    curves did exactly that, turning up after rank 16 purely because chance
    was overtaking them.

    Repair: bar the resolution, ``(overlap - chance) / (1 - chance)``, which
    is zero at the floor and one at perfect recovery whatever the shape. It
    now lives in `readscope.metrics` because it belongs to the instrument
    rather than to this harness.

Defect two, a cutoff that was unsatisfiable by construction.
    Bandwidth was the largest swept value clearing twice chance. At rank
    ``dim / 2`` twice chance is 1.0, so no finite instrument can clear it and
    the criterion expires before the sweep ends. It also took the largest
    passing value rather than the largest passing prefix, so it could report
    a bandwidth beyond a region that had already failed.

    Repair: bandwidth is the largest swept value at which resolution is at
    least ``RESOLUTION_BAR`` **and** is at least that at every smaller swept
    value. A cutoff is a prefix, not a scatter of survivors.

Sweep B is reported and not barred for monotonicity, because raising the
ambient dimension at fixed rank is not unambiguously harder: the task grows
while the floor falls, and the two effects oppose. Declaring a direction for
it would be declaring a guess.

Declared bars, computed from the record:

  B1  easiest point   exact estimator at rank one reaches resolution 0.99.
  B2  monotone        in sweep A, within each estimator, resolution is
                      non-increasing in rank, allowing one inversion.
  B3  bandwidth       every estimator has a bandwidth of at least one in
                      sweep A, defined by the prefix rule above.
  B4  budget ordering at matched cells the larger sketch is no worse than
                      the smaller in resolution, allowing one inversion.
  B5  anti-vacuity    at every cell the mean squared gradient norm clears
                      the floor, the recovered operator has rank at least
                      the graded rank, and the declared calls were spent.
  B6  census          every declared cell appears.

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
    subspace_overlap,
)

N_POINTS = 192
EPS = 1e-3
SEEDS = [0, 1, 2]
GRAD_FLOOR = 1e-10
RESOLUTION_BAR = 0.5
SPECTRUM_DECAY = 0.75
INPUT_SCALE = 0.35

SWEEP_A_DIM = 64
SWEEP_A_RANKS = [1, 2, 4, 8, 16, 24, 32]
SWEEP_B_RANK = 4
SWEEP_B_DIMS = [16, 32, 64, 128]

ESTIMATORS = [
    {"name": "exact", "mode": "exact", "sketch_dim": None},
    {"name": "sketch16", "mode": "sketch", "sketch_dim": 16},
    {"name": "sketch64", "mode": "sketch", "sketch_dim": 64},
]


def graded_consumer(basis: np.ndarray, weights: np.ndarray):
    """Reads a planted subspace with a declared graded spectrum."""

    def C(x):
        return float(np.tanh(basis.T @ x) @ weights)

    return C


def run_cell(dim, rank, est, seed):
    rng = np.random.default_rng(seed * 977 + dim * 31 + rank)
    basis = np.linalg.qr(rng.standard_normal((dim, rank)))[0]
    weights = SPECTRUM_DECAY ** np.arange(rank)
    consumer = graded_consumer(basis, weights)
    pts = rng.standard_normal((N_POINTS, dim)) * INPUT_SCALE

    res = blind_probe(
        consumer,
        pts,
        mode=est["mode"],
        sketch_dim=est["sketch_dim"],
        eps=EPS,
        rng=np.random.default_rng(seed * 131 + 5),
    )
    ov = subspace_overlap(res.read_subspace(rank), basis)
    expected = (
        N_POINTS * 2 * (dim if est["mode"] == "exact" else est["sketch_dim"])
    )
    row = ov.to_dict()
    row.update(
        {
            "dim": dim,
            "rank": rank,
            "estimator": est["name"],
            "seed": seed,
            "n_calls": res.n_calls,
            "calls_as_declared": bool(res.n_calls == expected),
            "mean_sq_grad_norm": float(np.trace(res.S)),
            "operator_rank": int(np.linalg.matrix_rank(res.S, tol=1e-10)),
        }
    )
    return row


def mean_resolution(rows, **sel):
    vals = [
        r["resolution"] for r in rows if all(r[k] == v for k, v in sel.items())
    ]
    return float(np.mean(vals)) if vals else float("nan")


def prefix_bandwidth(curve, values):
    """Largest value whose whole prefix clears the resolution bar."""
    best = None
    for v in values:
        if curve[str(v)] >= RESOLUTION_BAR:
            best = v
        else:
            break
    return best


def inversions(seq):
    return sum(1 for i in range(len(seq) - 1) if seq[i] < seq[i + 1] - 1e-9)


def main() -> int:
    rows = []
    for seed in SEEDS:
        for est in ESTIMATORS:
            for rank in SWEEP_A_RANKS:
                rows.append(run_cell(SWEEP_A_DIM, rank, est, seed))
            for dim in SWEEP_B_DIMS:
                rows.append(run_cell(dim, SWEEP_B_RANK, est, seed))

    curves = {"sweep_a_rank": {}, "sweep_b_dim": {}}
    for est in ESTIMATORS:
        n = est["name"]
        curves["sweep_a_rank"][n] = {
            str(r): mean_resolution(rows, estimator=n, rank=r, dim=SWEEP_A_DIM)
            for r in SWEEP_A_RANKS
        }
        curves["sweep_b_dim"][n] = {
            str(d): mean_resolution(
                rows, estimator=n, dim=d, rank=SWEEP_B_RANK
            )
            for d in SWEEP_B_DIMS
        }

    bandwidths = {
        n: prefix_bandwidth(curves["sweep_a_rank"][n], SWEEP_A_RANKS)
        for n in curves["sweep_a_rank"]
    }
    inv_a = {
        n: inversions(
            [curves["sweep_a_rank"][n][str(r)] for r in SWEEP_A_RANKS]
        )
        for n in curves["sweep_a_rank"]
    }

    ordering = sum(
        1
        for r in SWEEP_A_RANKS
        if curves["sweep_a_rank"]["sketch64"][str(r)]
        < curves["sweep_a_rank"]["sketch16"][str(r)] - 1e-9
    )

    vac = [
        {
            k: r[k]
            for k in ("dim", "rank", "estimator", "seed", "operator_rank")
        }
        for r in rows
        if not (
            r["mean_sq_grad_norm"] > GRAD_FLOOR
            and r["operator_rank"] >= r["rank"]
            and r["calls_as_declared"]
        )
    ]

    expected = (
        len(SEEDS) * len(ESTIMATORS) * (len(SWEEP_A_RANKS) + len(SWEEP_B_DIMS))
    )

    bars = {
        "B1_easiest_point": bool(curves["sweep_a_rank"]["exact"]["1"] >= 0.99),
        "B2_monotone_in_rank": bool(all(v <= 1 for v in inv_a.values())),
        "B3_bandwidth_defined": bool(
            all(b is not None and b >= 1 for b in bandwidths.values())
        ),
        "B4_budget_ordering": bool(ordering <= 1),
        "B5_anti_vacuity": bool(not vac),
        "B6_census": bool(len(rows) == expected),
    }

    verdict = "PASS" if all(bars.values()) else "FAIL"
    record = {
        "schema": "readscope-c2b-bandwidth-v1",
        "supersedes": "calibration/records/c2-bandwidth.json",
        "disclosure": "C-2's numbers were seen before this was declared. "
        "The repairs are structural, changing the barred statistic and the "
        "cutoff rule. No threshold was moved to fit an observed value.",
        "declared": {
            "statistic": "resolution = (overlap - chance) / (1 - chance)",
            "n_points": N_POINTS,
            "eps": EPS,
            "seeds": SEEDS,
            "grad_floor": GRAD_FLOOR,
            "resolution_bar": RESOLUTION_BAR,
            "spectrum_decay": SPECTRUM_DECAY,
            "input_scale": INPUT_SCALE,
            "sweep_a": {"dim": SWEEP_A_DIM, "ranks": SWEEP_A_RANKS},
            "sweep_b": {"rank": SWEEP_B_RANK, "dims": SWEEP_B_DIMS},
            "estimators": [e["name"] for e in ESTIMATORS],
        },
        "rows": rows,
        "curves_resolution": curves,
        "bandwidth_sweep_a": bandwidths,
        "inversions_sweep_a": inv_a,
        "budget_ordering_violations": ordering,
        "anti_vacuity_failures": vac,
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

    out = Path(__file__).resolve().parent / "records" / "c2b-bandwidth.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True))

    print(f"sweep A, dim {SWEEP_A_DIM}, resolution by read rank")
    print("  " + f"{'ranks':<9} " + " ".join(f"{r:6d}" for r in SWEEP_A_RANKS))
    for n in curves["sweep_a_rank"]:
        vals = " ".join(
            f"{curves['sweep_a_rank'][n][str(r)]:6.3f}" for r in SWEEP_A_RANKS
        )
        print(f"  {n:<9} {vals}   bandwidth {bandwidths[n]}")
    print(f"sweep B, rank {SWEEP_B_RANK}, resolution by dimension (reported)")
    print("  " + f"{'dims':<9} " + " ".join(f"{d:6d}" for d in SWEEP_B_DIMS))
    for n in curves["sweep_b_dim"]:
        vals = " ".join(
            f"{curves['sweep_b_dim'][n][str(d)]:6.3f}" for d in SWEEP_B_DIMS
        )
        print(f"  {n:<9} {vals}")
    for k in sorted(bars):
        print(f"{k:<24} {'PASS' if bars[k] else 'FAIL'}")
    print("VERDICT", verdict)
    print(out)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
