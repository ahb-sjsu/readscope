#!/usr/bin/env python3
"""C-2e, the budget law. Declared before it runs.

Reading the source program's own probe settled where the accuracy gap came
from. It runs ``N_PROBE = 160`` directions in a ``d = 128`` head space, so
``k/d = 1.25``. **It is not a cheaper estimator, it is an overdetermined
one.** Everything C-2b and C-2d measured was the sub-dimensional regime,
which nothing in the source program ever relied on. The premise that the
sketch was "the affordable estimator" was mine.

That makes the useful specification a **budget law**: how does bandwidth
depend on the direction budget as a fraction of ambient dimension, and is
there any way to buy a discount below one?

Sweep A, the law itself. Least-squares recovery at ``k/d`` from a quarter to
one and a half, against the exact estimator.

Sweep B, the discount question. A vector-valued consumer returns ``m``
numbers per direction instead of one, so a single direction carries ``m``
times the information. If that substitutes for directions, a sub-dimensional
budget becomes usable and the instrument gets cheap. If it does not, the
budget law is a hard floor and the specification has to say the probe costs
``2d`` calls per operating point, full stop. **Neither answer is known and
both are useful.**

Declared bars, computed from the record:

  S1  law is monotone      bandwidth is non-decreasing in k/d.
  S2  saturation at one    at k/d >= 1 the least-squares bandwidth equals the
                           exact estimator's.
  S3  sub-dimensional cost at k/d = 0.5 the scalar least-squares bandwidth is
                           strictly below the exact estimator's. This is the
                           deficit the discount would have to close, and if
                           it is absent there is nothing to test in sweep B.
  S4  discount direction   at k/d = 0.5, bandwidth is non-decreasing in the
                           consumer's output dimension m.
  S5  discount sufficiency at k/d = 0.5 and m >= 4, bandwidth reaches the
                           exact estimator's. The strong form. Failing it
                           while S4 passes means vector output helps and does
                           not rescue a sub-dimensional budget.
  S6  anti-vacuity+census  every cell present, operator rank at least the
                           graded rank, declared calls spent.

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
    jacobian_probe,
    subspace_overlap,
)

DIM = 32
RANKS = [1, 2, 4, 8, 16]
RATIOS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
OUT_DIMS = [1, 2, 4, 8]
DISCOUNT_RATIO = 0.5
N_POINTS = 96
EPS = 1e-3
SEEDS = [0, 1, 2, 3, 4]
RESOLUTION_BAR = 0.5
SPECTRUM_DECAY = 0.75
INPUT_SCALE = 0.35


def scalar_consumer(basis, weights):
    def C(x):
        return float(np.tanh(basis.T @ x) @ weights)

    return C


def vector_consumer(basis, weights, mix):
    """Vector output whose Jacobian spans the same planted subspace.

    ``C(x) = mix @ (w * tanh(B^T x))``, so ``J = mix diag(w sech^2) B^T`` and
    the row space of ``J`` is inside ``span(B)`` for every ``x``. Changing
    ``m`` changes only how many independent readings come back per direction,
    not what is there to be read, which is what makes the discount question
    well posed.
    """

    def C(x):
        return mix @ (weights * np.tanh(basis.T @ x))

    return C


def setup(rank, seed, out_dim):
    rng = np.random.default_rng(seed * 977 + rank * 13 + out_dim)
    basis = np.linalg.qr(rng.standard_normal((DIM, rank)))[0]
    weights = SPECTRUM_DECAY ** np.arange(rank)
    mix = rng.standard_normal((out_dim, rank))
    pts = rng.standard_normal((N_POINTS, DIM)) * INPUT_SCALE
    return basis, weights, mix, pts


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
            basis, weights, _mix, pts = setup(rank, seed, 1)
            cons = scalar_consumer(basis, weights)

            ex = blind_probe(cons, pts, eps=EPS, check_regime=False)
            rows.append(
                {
                    "sweep": "A",
                    "seed": seed,
                    "rank": rank,
                    "arm": "exact",
                    "ratio": 1.0,
                    "out_dim": 1,
                    "resolution": subspace_overlap(
                        ex.read_subspace(rank), basis
                    ).resolution,
                    "n_calls": ex.n_calls,
                    "calls_ok": ex.n_calls == N_POINTS * 2 * DIM,
                    "operator_rank": int(
                        np.linalg.matrix_rank(ex.S, tol=1e-10)
                    ),
                }
            )

            for ratio in RATIOS:
                k = max(1, int(round(ratio * DIM)))
                pr = blind_probe(
                    cons,
                    pts,
                    mode="lstsq",
                    sketch_dim=k,
                    eps=EPS,
                    rng=np.random.default_rng(seed * 131 + 5),
                    check_regime=False,
                )
                rows.append(
                    {
                        "sweep": "A",
                        "seed": seed,
                        "rank": rank,
                        "arm": "lstsq",
                        "ratio": ratio,
                        "out_dim": 1,
                        "resolution": subspace_overlap(
                            pr.read_subspace(rank), basis
                        ).resolution,
                        "n_calls": pr.n_calls,
                        "calls_ok": pr.n_calls == N_POINTS * 2 * k,
                        "operator_rank": int(
                            np.linalg.matrix_rank(pr.S, tol=1e-10)
                        ),
                    }
                )

    k_disc = max(1, int(round(DISCOUNT_RATIO * DIM)))
    for seed in SEEDS:
        for rank in RANKS:
            for m in OUT_DIMS:
                basis, weights, mix, pts = setup(rank, seed, m)
                cons = vector_consumer(basis, weights, mix)
                pr = jacobian_probe(
                    cons,
                    pts,
                    n_directions=k_disc,
                    eps=EPS,
                    rng=np.random.default_rng(seed * 131 + 5),
                )
                rows.append(
                    {
                        "sweep": "B",
                        "seed": seed,
                        "rank": rank,
                        "arm": "jacobian",
                        "ratio": DISCOUNT_RATIO,
                        "out_dim": m,
                        "resolution": subspace_overlap(
                            pr.read_subspace(rank), basis
                        ).resolution,
                        "n_calls": pr.n_calls,
                        "calls_ok": pr.n_calls == N_POINTS * 2 * k_disc,
                        "operator_rank": int(
                            np.linalg.matrix_rank(pr.S, tol=1e-10)
                        ),
                    }
                )

    def curve(**sel):
        out = {}
        for r in RANKS:
            vals = [
                x["resolution"]
                for x in rows
                if x["rank"] == r and all(x[k] == v for k, v in sel.items())
            ]
            out[str(r)] = float(np.mean(vals)) if vals else float("nan")
        return out

    exact_curve = curve(arm="exact")
    exact_bw = prefix_bandwidth(exact_curve)

    law = {}
    law_bw = {}
    for ratio in RATIOS:
        c = curve(arm="lstsq", ratio=ratio)
        law[str(ratio)] = c
        law_bw[str(ratio)] = prefix_bandwidth(c)

    disc = {}
    disc_bw = {}
    for m in OUT_DIMS:
        c = curve(arm="jacobian", out_dim=m)
        disc[str(m)] = c
        disc_bw[str(m)] = prefix_bandwidth(c)

    def as_num(b):
        return -1 if b is None else b

    s1 = all(
        as_num(law_bw[str(RATIOS[i])]) <= as_num(law_bw[str(RATIOS[i + 1])])
        for i in range(len(RATIOS) - 1)
    )
    s2 = all(law_bw[str(r)] == exact_bw for r in RATIOS if r >= 1.0)
    s3 = as_num(law_bw[str(DISCOUNT_RATIO)]) < as_num(exact_bw)
    s4 = all(
        as_num(disc_bw[str(OUT_DIMS[i])])
        <= as_num(disc_bw[str(OUT_DIMS[i + 1])])
        for i in range(len(OUT_DIMS) - 1)
    )
    s5 = any(disc_bw[str(m)] == exact_bw for m in OUT_DIMS if m >= 4)
    expected = len(SEEDS) * len(RANKS) * (1 + len(RATIOS) + len(OUT_DIMS))
    s6 = bool(
        len(rows) == expected
        and all(x["calls_ok"] for x in rows)
        and all(x["operator_rank"] >= min(x["rank"], DIM) for x in rows)
    )

    bars = {
        "S1_law_monotone": bool(s1),
        "S2_saturates_at_one": bool(s2),
        "S3_subdimensional_deficit": bool(s3),
        "S4_discount_direction": bool(s4),
        "S5_discount_sufficient": bool(s5),
        "S6_anti_vacuity_and_census": bool(s6),
    }
    verdict = "PASS" if all(bars.values()) else "FAIL"

    record = {
        "schema": "readscope-c2e-budget-law-v1",
        "declared": {
            "dim": DIM,
            "ranks": RANKS,
            "ratios": RATIOS,
            "out_dims": OUT_DIMS,
            "discount_ratio": DISCOUNT_RATIO,
            "n_points": N_POINTS,
            "eps": EPS,
            "seeds": SEEDS,
            "resolution_bar": RESOLUTION_BAR,
            "source_probe_ratio": "geometric-observation ran "
            "N_PROBE=160 in d=128, so k/d = 1.25",
        },
        "rows": rows,
        "exact_curve": exact_curve,
        "exact_bandwidth": exact_bw,
        "budget_law": law,
        "budget_law_bandwidth": law_bw,
        "discount_curves": disc,
        "discount_bandwidth": disc_bw,
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

    out = Path(__file__).resolve().parent / "records" / "c2e-budget-law.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True))

    hdr = "  " + f"{'ranks':<18} " + " ".join(f"{r:6d}" for r in RANKS)
    print("sweep A, the budget law, dim", DIM)
    print(hdr)
    print(
        f"  {'exact':<18} "
        + " ".join(f"{exact_curve[str(r)]:6.3f}" for r in RANKS)
        + f"   bw {exact_bw}"
    )
    for ratio in RATIOS:
        print(
            f"  {'lstsq k/d=' + str(ratio):<18} "
            + " ".join(f"{law[str(ratio)][str(r)]:6.3f}" for r in RANKS)
            + f"   bw {law_bw[str(ratio)]}"
        )
    print(f"sweep B, the discount question, k/d = {DISCOUNT_RATIO}")
    print(hdr)
    for m in OUT_DIMS:
        print(
            f"  {'jacobian m=' + str(m):<18} "
            + " ".join(f"{disc[str(m)][str(r)]:6.3f}" for r in RANKS)
            + f"   bw {disc_bw[str(m)]}"
        )
    for k in sorted(bars):
        print(f"{k:<30} {'PASS' if bars[k] else 'FAIL'}")
    print("VERDICT", verdict)
    print(out)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
