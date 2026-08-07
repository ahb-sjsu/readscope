#!/usr/bin/env python3
"""C-2, the bandwidth sweep. Declared before it runs.

How many eigendirections can this instrument resolve before the reading is
chance? That number is the bandwidth, and quoting an accuracy without it is
quoting a scope's accuracy without saying at what frequency.

Two sweeps, both graded against a planted subspace with a declared graded
spectrum, because a real consumer's sensitivity is not flat and the tail
directions are the ones that go first.

  Sweep A, rank      fixed ambient dimension, rising read rank.
  Sweep B, dimension fixed read rank, rising ambient dimension.

Three estimators, since bandwidth is a property of the estimator and not only
of the consumer: the exact coordinate estimator, and the sketch at two
budgets. The sketch's ``k`` is the sample-rate knob that trades cost against
bandwidth, and the whole point of measuring here is to price that trade.

Bandwidth is defined, before any run, as the largest graded rank at which
mean overlap stays at or above ``BANDWIDTH_MULTIPLE`` times the chance value
for that shape. Chance is ``rank / dim`` and is recomputed per cell.

Declared bars, computed from the record:

  B1  easiest point   the exact estimator at rank one reaches overlap 0.99.
                      If the instrument cannot do the easy case it has no
                      bandwidth to report.
  B2  monotone        within each estimator and each sweep, overlap is
                      non-increasing as the task gets harder, allowing one
                      inversion for sampling noise.
  B3  bandwidth       every estimator has a defined bandwidth of at least
                      one in each sweep, and it is recorded.
  B4  budget ordering at matched cells the larger sketch is no worse than
                      the smaller, allowing one inversion. Spending more
                      calls must not buy a worse reading.
  B5  anti-vacuity    at every cell the mean squared gradient norm is above
                      the floor, the recovered operator has rank at least
                      the graded rank, and the probe spent the declared
                      number of calls.
  B6  census          every declared cell appears.

B2 failing would mean the reading is not ordered by task difficulty, which
would make a single bandwidth number meaningless. B4 failing would mean the
sketch is not converging to the exact estimator. Both are results.

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
    subspace_overlap,
)

N_POINTS = 192
EPS = 1e-3
SEEDS = [0, 1, 2]
GRAD_FLOOR = 1e-10
BANDWIDTH_MULTIPLE = 2.0
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
    """Reads a planted subspace with a declared graded spectrum.

    ``C(x) = sum_j w_j tanh(b_j . x)``. Distinct weights keep every planted
    direction genuinely read, which C-1 got wrong by summing the projections
    into one scalar and so planting rank one while grading rank four. The
    input scale keeps the tanh off its saturated tails, which C-1 also got
    wrong.
    """

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
    expected_calls = (
        N_POINTS * 2 * (dim if est["mode"] == "exact" else est["sketch_dim"])
    )
    return {
        "dim": dim,
        "rank": rank,
        "estimator": est["name"],
        "seed": seed,
        "overlap": ov.overlap,
        "chance": ov.chance,
        "ratio_to_chance": ov.ratio,
        "n_calls": res.n_calls,
        "calls_as_declared": bool(res.n_calls == expected_calls),
        "mean_sq_grad_norm": float(np.trace(res.S)),
        "operator_rank": int(np.linalg.matrix_rank(res.S, tol=1e-10)),
    }


def mean_overlap(rows, **sel):
    vals = [
        r["overlap"] for r in rows if all(r[k] == v for k, v in sel.items())
    ]
    return float(np.mean(vals)) if vals else float("nan")


def bandwidth_of(rows, est_name, axis, values, fixed):
    """Largest swept value whose mean overlap clears the declared multiple."""
    best = None
    for v in values:
        sel = {"estimator": est_name, axis: v, **fixed}
        ov = mean_overlap(rows, **sel)
        rank = v if axis == "rank" else fixed["rank"]
        dim = v if axis == "dim" else fixed["dim"]
        if ov >= BANDWIDTH_MULTIPLE * chance_overlap(rank, dim):
            best = v
    return best


def inversions(seq):
    return sum(1 for i in range(len(seq) - 1) if seq[i] < seq[i + 1])


def main() -> int:
    rows = []
    for seed in SEEDS:
        for est in ESTIMATORS:
            for rank in SWEEP_A_RANKS:
                rows.append(run_cell(SWEEP_A_DIM, rank, est, seed))
            for dim in SWEEP_B_DIMS:
                rows.append(run_cell(dim, SWEEP_B_RANK, est, seed))

    curves = {"sweep_a_rank": {}, "sweep_b_dim": {}}
    bandwidths = {"sweep_a_rank": {}, "sweep_b_dim": {}}
    for est in ESTIMATORS:
        n = est["name"]
        curves["sweep_a_rank"][n] = {
            str(r): mean_overlap(rows, estimator=n, rank=r, dim=SWEEP_A_DIM)
            for r in SWEEP_A_RANKS
        }
        curves["sweep_b_dim"][n] = {
            str(d): mean_overlap(rows, estimator=n, dim=d, rank=SWEEP_B_RANK)
            for d in SWEEP_B_DIMS
        }
        bandwidths["sweep_a_rank"][n] = bandwidth_of(
            rows, n, "rank", SWEEP_A_RANKS, {"dim": SWEEP_A_DIM}
        )
        bandwidths["sweep_b_dim"][n] = bandwidth_of(
            rows, n, "dim", SWEEP_B_DIMS, {"rank": SWEEP_B_RANK}
        )

    inv_a = {
        n: inversions(
            [curves["sweep_a_rank"][n][str(r)] for r in SWEEP_A_RANKS]
        )
        for n in curves["sweep_a_rank"]
    }
    inv_b = {
        n: inversions([curves["sweep_b_dim"][n][str(d)] for d in SWEEP_B_DIMS])
        for n in curves["sweep_b_dim"]
    }

    ordering_violations = []
    for r in SWEEP_A_RANKS:
        small = curves["sweep_a_rank"]["sketch16"][str(r)]
        large = curves["sweep_a_rank"]["sketch64"][str(r)]
        if large < small - 1e-9:
            ordering_violations.append({"sweep": "a", "rank": r})
    for d in SWEEP_B_DIMS:
        small = curves["sweep_b_dim"]["sketch16"][str(d)]
        large = curves["sweep_b_dim"]["sketch64"][str(d)]
        if large < small - 1e-9:
            ordering_violations.append({"sweep": "b", "dim": d})

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
        "B2_monotone": bool(
            all(v <= 1 for v in inv_a.values())
            and all(v <= 1 for v in inv_b.values())
        ),
        "B3_bandwidth_defined": bool(
            all(
                bandwidths[s][n] is not None and bandwidths[s][n] >= 1
                for s in bandwidths
                for n in bandwidths[s]
            )
        ),
        "B4_budget_ordering": bool(len(ordering_violations) <= 1),
        "B5_anti_vacuity": bool(not vac),
        "B6_census": bool(len(rows) == expected),
    }

    verdict = "PASS" if all(bars.values()) else "FAIL"
    record = {
        "schema": "readscope-c2-bandwidth-v1",
        "declared": {
            "n_points": N_POINTS,
            "eps": EPS,
            "seeds": SEEDS,
            "grad_floor": GRAD_FLOOR,
            "bandwidth_multiple": BANDWIDTH_MULTIPLE,
            "spectrum_decay": SPECTRUM_DECAY,
            "input_scale": INPUT_SCALE,
            "sweep_a": {"dim": SWEEP_A_DIM, "ranks": SWEEP_A_RANKS},
            "sweep_b": {"rank": SWEEP_B_RANK, "dims": SWEEP_B_DIMS},
            "estimators": [e["name"] for e in ESTIMATORS],
        },
        "rows": rows,
        "curves": curves,
        "bandwidths": bandwidths,
        "inversions": {"sweep_a": inv_a, "sweep_b": inv_b},
        "budget_ordering_violations": ordering_violations,
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

    out = Path(__file__).resolve().parent / "records" / "c2-bandwidth.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True))

    print(f"sweep A, dim {SWEEP_A_DIM}, overlap by read rank")
    for n in curves["sweep_a_rank"]:
        vals = " ".join(
            f"{curves['sweep_a_rank'][n][str(r)]:.3f}" for r in SWEEP_A_RANKS
        )
        print(f"  {n:<9} {vals}   bandwidth {bandwidths['sweep_a_rank'][n]}")
    print(f"  {'ranks':<9} " + " ".join(f"{r:5d}" for r in SWEEP_A_RANKS))
    print(f"sweep B, rank {SWEEP_B_RANK}, overlap by ambient dimension")
    for n in curves["sweep_b_dim"]:
        vals = " ".join(
            f"{curves['sweep_b_dim'][n][str(d)]:.3f}" for d in SWEEP_B_DIMS
        )
        print(f"  {n:<9} {vals}   bandwidth {bandwidths['sweep_b_dim'][n]}")
    for k in sorted(bars):
        print(f"{k:<24} {'PASS' if bars[k] else 'FAIL'}")
    print("VERDICT", verdict)
    print(out)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
