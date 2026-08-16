#!/usr/bin/env python3
"""C-13: backend equivalence and scaling. Declared in DECLARATION-C13.md,
sealed before any run; bars live there, verdicts are computed here.

    python c13_backend_suite.py --device-label atlas-gv100 [--max-cpu-d 8192]

Requires cupy for the GPU cells; CPU-only invocation runs the numpy
reference cells and records them (useful for cross-host comparison).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from readscope import (blind_probe, jacobian_probe, spectrum_of,  # noqa: E402
                       top_spectrum)

SEED = 20260816
DIMS = [128, 1024, 4096, 8192]
N_PTS = 4
RANK = 8
TOP_R = 16


def make_consumers(d, rng):
    """Planted rank-8 consumers, namespace-agnostic: the closure works on
    whatever array type it is handed."""
    v = np.linalg.qr(rng.standard_normal((d, RANK)))[0]
    w = np.sqrt(np.linspace(1.0, 0.1, RANK))

    def scalar(x):
        xp = type(x).__module__.split(".")[0]
        if xp == "cupy":
            import cupy as cp

            vv, ww = cp.asarray(v), cp.asarray(w)
            return float((cp.tanh(vv.T @ x) * ww).sum())
        return float((np.tanh(v.T @ x) * w).sum())

    def vector(x):
        xp = type(x).__module__.split(".")[0]
        if xp == "cupy":
            import cupy as cp

            vv, ww = cp.asarray(v), cp.asarray(w)
            return cp.tanh(vv.T @ x) * ww
        return np.tanh(v.T @ x) * w

    return scalar, vector


def to_np(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def run_cells(xp_label, as_backend, dims):
    out = {}
    for d in dims:
        rng = np.random.default_rng(SEED)
        pts_np = rng.standard_normal((N_PTS, d))
        scalar, vector = make_consumers(d, np.random.default_rng(SEED + 1))
        pts = as_backend(pts_np)
        cell = {}

        t0 = time.time()
        rb = blind_probe(scalar, pts, mode="lstsq", sketch_dim=d,
                         rng=np.random.default_rng(SEED + 2),
                         check_regime=False)
        cell["blind_s"] = round(time.time() - t0, 3)
        t0 = time.time()
        rj = jacobian_probe(vector, pts, n_directions=d,
                            rng=np.random.default_rng(SEED + 2))
        cell["jac_s"] = round(time.time() - t0, 3)

        t0 = time.time()
        spec = spectrum_of(rj.S)
        cell["eigh_s"] = round(time.time() - t0, 3)
        t0 = time.time()
        top = top_spectrum(rj.S, TOP_R)
        cell["top_s"] = round(time.time() - t0, 3)

        ev_full = to_np(spec.eigenvalues)
        ev_top = to_np(top.eigenvalues)
        scale = max(float(ev_full[0]), 1e-300)
        e3_vals = float(np.max(np.abs(ev_top - ev_full[:TOP_R])) / scale)
        vec_f = to_np(spec.eigenvectors)[:, :TOP_R]
        vec_t = to_np(top.eigenvectors)
        # overlap graded on non-degenerate directions (planted rank 8 +
        # noise floor; degenerate tail pairs are basis-ambiguous)
        ovl = [abs(float(vec_f[:, i] @ vec_t[:, i])) for i in range(RANK)]
        er_full = float(spec.effective_rank)
        er_top = float(top.effective_rank)
        cell.update({
            "S_blind": to_np(rb.S), "S_jac": to_np(rj.S),
            "eigvals": ev_full[:TOP_R].tolist(),
            "e3_val_dev": e3_vals,
            "e3_min_overlap": min(ovl),
            "e3_er_reldev": abs(er_top - er_full) / max(er_full, 1e-300),
        })
        out[d] = cell
        print(f"[{xp_label}] d={d}: blind {cell['blind_s']}s  "
              f"jac {cell['jac_s']}s  eigh {cell['eigh_s']}s  "
              f"top {cell['top_s']}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-label", required=True)
    ap.add_argument("--max-cpu-d", type=int, default=8192)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cpu_dims = [d for d in DIMS if d <= args.max_cpu_d]
    cpu = run_cells("numpy", lambda a: a, cpu_dims)

    gpu = None
    try:
        import cupy as cp

        gpu = run_cells("cupy", cp.asarray, DIMS)
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except ImportError:
        gpu_name = None
        print("cupy not available; CPU reference only", flush=True)

    cells, ok = {}, True
    for d in DIMS:
        row = {}
        for label, res in (("cpu", cpu.get(d)), ("gpu", gpu.get(d) if gpu else None)):
            if res is None:
                continue
            row[label] = {k: v for k, v in res.items()
                          if not k.startswith("S_")}
            e3 = (res["e3_val_dev"] <= 1e-8
                  and res["e3_min_overlap"] >= 1 - 1e-8
                  and res["e3_er_reldev"] <= 1e-10)
            row[label]["E3_pass"] = bool(e3)
            ok &= e3
        if gpu and d in cpu:
            for probe in ("S_blind", "S_jac"):
                num = np.linalg.norm(cpu[d][probe] - gpu[d][probe])
                den = max(np.linalg.norm(cpu[d][probe]), 1e-300)
                row[f"E1_{probe}_reldev"] = float(num / den)
                ok &= row[f"E1_{probe}_reldev"] <= 1e-9
            ev_c = np.asarray(cpu[d]["eigvals"])
            ev_g = np.asarray(gpu[d]["eigvals"])
            row["E2_eig_reldev"] = float(
                np.max(np.abs(ev_c - ev_g)) / max(ev_c[0], 1e-300))
            ok &= row["E2_eig_reldev"] <= 1e-9
        cells[str(d)] = row

    record = {
        "schema": "readscope-c13-backend-v1",
        "declaration": "DECLARATION-C13.md",
        "device_label": args.device_label,
        "gpu": gpu_name,
        "seed": SEED,
        "cells": cells,
        "verdict": "PASS" if ok else "FAIL",
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
    }
    out = args.out or str(Path(__file__).parent / "records" /
                          f"c13-backend-{args.device_label}.json")
    Path(out).write_text(json.dumps(record, indent=1, sort_keys=True))
    print(f"\nC-13 [{args.device_label}]: {record['verdict']} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
