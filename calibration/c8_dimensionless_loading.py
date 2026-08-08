#!/usr/bin/env python3
"""C-8, the dimensionless loading axis. Declared before it runs.

C-7b established that a correction fitted against raw Jeffreys divergence
cannot transfer, and that the obstacle was the axis rather than the consumer.
Raw Jeffreys fails as a datasheet quantity in two ways at once. It grows with
dimension for a mismatch that is identical in every direction, so the same
physical situation read 0.89 to 92 nats at dimension 24 and billions at
dimension 256. And it is positive when there is no mismatch at all, because
two finite samples of one law have different fitted Gaussians.

The proposed axis fixes both:

    loading = max(0, jeffreys - null_floor(n_p, n_a, d)) / d

The null floor is simulated for the shape and is distribution free, since
Jeffreys between fitted Gaussians depends on the moments only through
products invariant to a shared affine map.

**This sweep tests the axis first and the correction second, in that order,
because a correction fitted on a bad axis is what produced C-7b's false
pass.**

Part A, the axis. A per-direction mismatch of fixed size is constructed at
five dimensions and the readings compared.

Part B, the correction. A correction is fitted at one dimension and applied
at the others, which is precisely what raw Jeffreys made impossible.

Declared bars:

  D1  floor is real       the null floor is strictly positive at every
                          declared shape and rises as samples thin. If it
                          were zero there would be nothing to subtract.
  D2  null reads zero     two independent samples of one law read a
                          normalised loading below 0.05 at every dimension.
  D3  the axis transfers  a fixed per-direction mismatch reads within 0.20
                          relative spread across dimensions 16 to 256.
                          **Falsifiable, and the point of the sweep.** Raw
                          Jeffreys on the same configurations must exceed
                          1.0 relative spread, which is the contrast that
                          makes D3 mean something.
  D4  monotone            normalised loading is strictly increasing in
                          mismatch magnitude at every dimension.
  D5  correction transfers a correction fitted at dimension 32 and applied
                          at 16, 64, 128 and 256 predicts the attenuation
                          with mean absolute error below 0.10, **evaluated
                          only on readings inside its fitted domain**, and
                          the in-domain fraction is reported. This is the
                          bar C-7b was missing.
  D6  census              every declared cell computed and reported.

Runs on Atlas. Numpy only, synthetic by construction, since the question is
about the axis and not about any model.
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
    fit_loading_correction,
    loading_null_floor,
    probe_loading,
    subspace_overlap,
)

DIMS = [16, 32, 64, 128, 256]
SAMPLES_PER_DIM = 200
MISMATCHES = [1.0, 1.25, 1.5, 2.0, 3.0]
FIT_DIM = 32
SEED = 0

NULL_ZERO_BAR = 0.05
AXIS_SPREAD_BAR = 0.20
RAW_SPREAD_FLOOR = 1.0
CORRECTION_MAE_BAR = 0.10

# part B probes a planted consumer whose read subspace moves with position,
# the only regime where loading can bite at all (C-1's finding)
PROBE_POINTS = 128
PROBE_RANK = 3
PROBE_BUDGET = 1.25


def gated_consumer(a, b, s):
    def C(x):
        t = float(s @ x)
        g = 1.0 / (1.0 + np.exp(-t))
        return float(g * (a @ x) + (1.0 - g) * (b @ x))

    return C


def main() -> int:
    rng = np.random.default_rng(SEED)
    report = {"axis": [], "null": [], "correction": []}

    # ---- D1 and D2, the floor -------------------------------------------
    for d in DIMS:
        n = SAMPLES_PER_DIM * d
        floor = loading_null_floor(n, n, d, trials=16)
        thin = loading_null_floor(max(2 * d, 8), max(2 * d, 8), d, trials=16)
        a = rng.standard_normal((n, d))
        b = rng.standard_normal((n, d))
        r = probe_loading(a, b)
        report["null"].append(
            {
                "dim": d,
                "n": n,
                "floor": floor,
                "floor_thin_sample": thin,
                "null_reading": r.loading,
                "null_raw_jeffreys": r.jeffreys,
            }
        )
        print(
            f"  null d={d:<4} floor {floor:10.3f}  thin {thin:12.1f}  "
            f"reading {r.loading:.4f}  raw {r.jeffreys:10.2f}",
            flush=True,
        )

    # ---- D3 and D4, the axis --------------------------------------------
    for d in DIMS:
        n = SAMPLES_PER_DIM * d
        base = rng.standard_normal((n, d))
        row = {"dim": d, "n": n, "points": []}
        for m in MISMATCHES:
            other = rng.standard_normal((n, d)) * m
            r = probe_loading(other, base)
            row["points"].append(
                {
                    "mismatch": m,
                    "loading": r.loading,
                    "raw_jeffreys": r.jeffreys,
                }
            )
        report["axis"].append(row)
        print(
            f"  axis d={d:<4} "
            + " ".join(
                f"m{p['mismatch']}:{p['loading']:.3f}" for p in row["points"]
            )
            + "   raw "
            + " ".join(f"{p['raw_jeffreys']:.1f}" for p in row["points"]),
            flush=True,
        )

    def at(m):
        return [
            next(p for p in r["points"] if p["mismatch"] == m)
            for r in report["axis"]
        ]

    def rel_spread(v):
        return (max(v) - min(v)) / max(abs(float(np.mean(v))), 1e-12)

    probe_m = 2.0
    axis_vals = [p["loading"] for p in at(probe_m)]
    raw_vals = [p["raw_jeffreys"] for p in at(probe_m)]

    d1 = all(
        x["floor"] > 0.0 and x["floor_thin_sample"] > x["floor"]
        for x in report["null"]
    )
    d2 = all(x["null_reading"] < NULL_ZERO_BAR for x in report["null"])
    d3 = bool(
        rel_spread(axis_vals) < AXIS_SPREAD_BAR
        and rel_spread(raw_vals) > RAW_SPREAD_FLOOR
    )
    d4 = all(
        all(
            r["points"][i]["loading"] < r["points"][i + 1]["loading"]
            for i in range(len(MISMATCHES) - 1)
        )
        for r in report["axis"]
    )

    # ---- D5, does a correction fitted at one dimension transfer ----------
    def sweep_consumer(d, seed):
        g = np.random.default_rng(seed)
        basis = np.linalg.qr(g.standard_normal((d, 3)))[0]
        a, b, s = basis[:, 0], basis[:, 1], basis[:, 2]
        cons = gated_consumer(a, b, s)
        act = g.standard_normal((SAMPLES_PER_DIM * d, d))
        truth = blind_probe(
            cons, act[:PROBE_POINTS], eps=1e-3, check_regime=False
        ).read_subspace(PROBE_RANK)
        out = []
        for m in MISMATCHES:
            far = g.standard_normal((SAMPLES_PER_DIM * d, d)) * m + (m - 1.0)
            load = probe_loading(far, act).loading
            k = max(1, int(round(PROBE_BUDGET * d)))
            res = blind_probe(
                cons,
                far[:PROBE_POINTS],
                mode="lstsq",
                sketch_dim=k,
                eps=1e-3,
                rng=np.random.default_rng(seed + 3),
                check_regime=False,
            ).read_subspace(PROBE_RANK)
            out.append(
                {
                    "mismatch": m,
                    "loading": load,
                    "reading": subspace_overlap(res, truth).resolution,
                }
            )
        return out

    fit_pts = sweep_consumer(FIT_DIM, SEED + 100)
    corr = fit_loading_correction(
        [p["loading"] for p in fit_pts],
        [p["reading"] for p in fit_pts],
        truth=1.0,
        source=f"c8 fit at dim {FIT_DIM}",
    )
    print(
        f"  correction fitted at dim {FIT_DIM} over loading "
        f"[{min(p['loading'] for p in fit_pts):.3f}, "
        f"{max(p['loading'] for p in fit_pts):.3f}]",
        flush=True,
    )

    errs, in_domain, total = [], 0, 0
    for d in DIMS:
        if d == FIT_DIM:
            continue
        pts = sweep_consumer(d, SEED + 200 + d)
        for p in pts:
            total += 1
            inside = corr.in_domain(p["loading"])
            in_domain += int(inside)
            entry = {
                "dim": d,
                "mismatch": p["mismatch"],
                "loading": p["loading"],
                "reading": p["reading"],
                "in_domain": inside,
            }
            if inside:
                pred = corr.expected_attenuation(p["loading"])
                entry["predicted_attenuation"] = pred
                entry["abs_error"] = abs(p["reading"] - pred)
                errs.append(entry["abs_error"])
            report["correction"].append(entry)
        print(
            f"  transfer d={d:<4} "
            + " ".join(
                f"m{p['mismatch']}:L{p['loading']:.2f} r{p['reading']:.3f}"
                for p in pts
            ),
            flush=True,
        )

    frac_in = in_domain / max(total, 1)
    mae = float(np.mean(errs)) if errs else float("nan")
    d5 = bool(errs and mae < CORRECTION_MAE_BAR)

    bars = {
        "D1_floor_is_real": bool(d1),
        "D2_null_reads_zero": bool(d2),
        "D3_axis_transfers": bool(d3),
        "D4_monotone": bool(d4),
        "D5_correction_transfers": d5,
        "D6_census": bool(
            len(report["null"]) == len(DIMS)
            and len(report["axis"]) == len(DIMS)
            and total == (len(DIMS) - 1) * len(MISMATCHES)
        ),
    }
    verdict = "PASS" if all(bars.values()) else "FAIL"

    record = {
        "schema": "readscope-c8-dimensionless-loading-v1",
        "declared": {
            "dims": DIMS,
            "samples_per_dim": SAMPLES_PER_DIM,
            "mismatches": MISMATCHES,
            "fit_dim": FIT_DIM,
            "seed": SEED,
            "null_zero_bar": NULL_ZERO_BAR,
            "axis_spread_bar": AXIS_SPREAD_BAR,
            "raw_spread_floor": RAW_SPREAD_FLOOR,
            "correction_mae_bar": CORRECTION_MAE_BAR,
            "axis": "max(0, jeffreys - null_floor(n_p, n_a, d)) / d",
        },
        "null": report["null"],
        "axis": report["axis"],
        "correction": {
            "fitted": corr.to_dict(),
            "fit_points": fit_pts,
            "transfer": report["correction"],
            "in_domain_fraction": frac_in,
            "mae_in_domain": mae,
            "n_in_domain": in_domain,
            "n_total": total,
        },
        "summary": {
            "axis_relative_spread_at_mismatch_2": rel_spread(axis_vals),
            "raw_relative_spread_at_mismatch_2": rel_spread(raw_vals),
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
        / "c8-dimensionless-loading.json"
    )
    out.write_text(json.dumps(record, indent=2, sort_keys=True))

    print(
        f"\naxis relative spread at mismatch {probe_m}: "
        f"{rel_spread(axis_vals):.4f}   raw: {rel_spread(raw_vals):.4f}"
    )
    print(
        f"correction in-domain {in_domain}/{total} "
        f"({frac_in * 100:.0f}%)   MAE in domain {mae:.4f}"
    )
    for k in sorted(bars):
        print(f"{k:<26} {'PASS' if bars[k] else 'FAIL'}")
    print("VERDICT", verdict)
    print(out)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
