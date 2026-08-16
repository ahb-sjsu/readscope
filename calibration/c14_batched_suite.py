#!/usr/bin/env python3
"""C-14: the batched-consumer probe variant. Declared in
DECLARATION-C14.md, sealed before this ran.

    python c14_batched_suite.py --device-label cpu            # B1-B3 + timing
    python c14_batched_suite.py --device-label nrp-3090 --gpu # batched GPU timing
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from readscope import blind_probe, jacobian_probe  # noqa: E402

SEED = 20260816
RANK = 8
B1_DIMS = [128, 1024, 4096]
GPU_DIMS = [8192, 4096, 1024]
N_PTS = 4


def consumers(d, rng):
    v = np.linalg.qr(rng.standard_normal((d, RANK)))[0]
    w = np.sqrt(np.linspace(1.0, 0.1, RANK))

    def xp_of(x):
        if type(x).__module__.split(".")[0] == "cupy":
            import cupy
            return cupy
        return np

    def scalar(x):
        xp = xp_of(x)
        return float((xp.tanh(xp.asarray(v).T @ x) * xp.asarray(w)).sum())

    def scalar_batch(X):
        xp = xp_of(X)
        return (xp.tanh(X @ xp.asarray(v)) * xp.asarray(w)).sum(axis=1)

    def vector(x):
        xp = xp_of(x)
        return xp.tanh(xp.asarray(v).T @ x) * xp.asarray(w)

    def vector_batch(X):
        xp = xp_of(X)
        return xp.tanh(X @ xp.asarray(v)) * xp.asarray(w)

    return scalar, scalar_batch, vector, vector_batch


def rel(a, b):
    a = a.get() if hasattr(a, "get") else a
    b = b.get() if hasattr(b, "get") else b
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-label", required=True)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    record = {"schema": "readscope-c14-batched-v1",
              "declaration": "DECLARATION-C14.md",
              "device_label": args.device_label, "seed": SEED,
              "cells": {}, "runtime": {
                  "python": sys.version.split()[0],
                  "numpy": np.__version__,
                  "platform": platform.platform(),
                  "hostname": platform.node()}}
    ok = True

    if args.gpu:
        import cupy as cp

        record["gpu"] = cp.cuda.runtime.getDeviceProperties(0)[
            "name"].decode()
        for d in GPU_DIMS:
            rng = np.random.default_rng(SEED)
            pts = cp.asarray(rng.standard_normal((N_PTS, d)))
            _, sb, _, vb = consumers(d, np.random.default_rng(SEED + 1))
            t0 = time.time()
            blind_probe(sb, pts, mode="lstsq", sketch_dim=d,
                        rng=np.random.default_rng(SEED + 2),
                        check_regime=False, batched=True)
            tb = time.time() - t0
            t0 = time.time()
            jacobian_probe(vb, pts, n_directions=d,
                           rng=np.random.default_rng(SEED + 2),
                           batched=True)
            tj = time.time() - t0
            record["cells"][f"gpu_d{d}"] = {
                "blind_batched_s": round(tb, 3),
                "jac_batched_s": round(tj, 3)}
            print(f"[gpu-batched] d={d}: blind {tb:.3f}s  jac {tj:.3f}s",
                  flush=True)
        record["verdict"] = "DESCRIPTIVE"
    else:
        # B1: identity with the serial instrument
        for d in B1_DIMS:
            rng = np.random.default_rng(SEED)
            pts = rng.standard_normal((N_PTS, d))
            sc, sb, vc, vb = consumers(d, np.random.default_rng(SEED + 1))
            cell = {}
            t0 = time.time()
            r_ser = blind_probe(sc, pts, mode="lstsq", sketch_dim=d,
                                rng=np.random.default_rng(SEED + 2),
                                check_regime=False)
            cell["blind_serial_s"] = round(time.time() - t0, 3)
            t0 = time.time()
            r_bat = blind_probe(sb, pts, mode="lstsq", sketch_dim=d,
                                rng=np.random.default_rng(SEED + 2),
                                check_regime=False, batched=True)
            cell["blind_batched_s"] = round(time.time() - t0, 3)
            cell["blind_reldev"] = rel(r_ser.S, r_bat.S)
            t0 = time.time()
            j_ser = jacobian_probe(vc, pts, n_directions=d,
                                   rng=np.random.default_rng(SEED + 2))
            cell["jac_serial_s"] = round(time.time() - t0, 3)
            t0 = time.time()
            j_bat = jacobian_probe(vb, pts, n_directions=d,
                                   rng=np.random.default_rng(SEED + 2),
                                   batched=True)
            cell["jac_batched_s"] = round(time.time() - t0, 3)
            cell["jac_reldev"] = rel(j_ser.S, j_bat.S)
            cell["invocations"] = int(r_bat.n_calls)
            cell["observations"] = int(r_bat.meta["observations"])
            b1 = cell["blind_reldev"] <= 1e-10 and cell["jac_reldev"] <= 1e-10
            cell["B1_pass"] = bool(b1)
            ok &= b1
            record["cells"][f"d{d}"] = cell
            print(f"[cpu] d={d}: blind dev {cell['blind_reldev']:.2e} "
                  f"({cell['blind_serial_s']}s -> {cell['blind_batched_s']}s)"
                  f"  jac dev {cell['jac_reldev']:.2e}  B1 "
                  f"{'PASS' if b1 else 'FAIL'}", flush=True)

        # B2: the cliff stands under batching
        d, k, hits = 128, 64, 0
        for t in range(50):
            rng = np.random.default_rng(SEED + 100 + t)
            basis = np.linalg.qr(rng.standard_normal((d, d)))[0]
            u_true = basis[:, -1]
            w = 3.0

            def sb2(X, u=u_true, w=w):
                return np.tanh(w * (X @ u))

            span = basis[:, :k]
            pts = rng.standard_normal((2, d)) @ span @ span.T
            res = blind_probe(sb2, pts, mode="lstsq", sketch_dim=k,
                              rng=rng, check_regime=False, batched=True)
            s_np = res.S
            top = np.linalg.eigh(s_np)[1][:, -1]
            hidden = 1 - float((span.T @ u_true) @ (span.T @ u_true))
            if hidden >= 1e-3 and float((top @ u_true) ** 2) >= 0.999:
                hits += 1
        b2 = hits == 0
        ok &= b2
        record["cells"]["B2_cliff"] = {"hidden_recoveries": hits,
                                       "trials": 50, "pass": bool(b2)}
        print(f"B2 cliff under batching: {hits}/50 hidden recoveries "
              f"-> {'PASS' if b2 else 'FAIL'}")

        # B3: the consistency gate fires on a row-coupled consumer
        d = 64
        rng = np.random.default_rng(SEED)
        pts = rng.standard_normal((2, d))
        _, sb3, _, _ = consumers(d, np.random.default_rng(SEED + 1))

        def coupled(X):
            return sb3(X) + X.mean()          # cross-row coupling

        try:
            blind_probe(coupled, pts, mode="lstsq", sketch_dim=d,
                        rng=np.random.default_rng(SEED),
                        check_regime=False, batched=True)
            b3 = False
        except ValueError as e:
            b3 = "row-independent" in str(e)
        ok &= b3
        record["cells"]["B3_gate"] = {"pass": bool(b3)}
        print(f"B3 consistency gate: {'PASS' if b3 else 'FAIL'}")
        record["verdict"] = "PASS" if ok else "FAIL"

    out = args.out or str(Path(__file__).parent / "records" /
                          f"c14-batched-{args.device_label}.json")
    Path(out).write_text(json.dumps(record, indent=1, sort_keys=True))
    print(f"\nC-14 [{args.device_label}]: {record['verdict']} -> {out}")
    return 0 if record["verdict"] in ("PASS", "DESCRIPTIVE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
